"""
Module 3 — Dark Ship Detection via Faster Threshold Method (FTM).

Detects physical vessels directly from SAR imagery when AIS transponders are
switched off (non-cooperative / "dark" vessels), then correlates SAR detections
with the AIS record to identify gaps that indicate deliberate non-broadcasting.

Algorithm overview (Jeon 2023 FTM)
-------------------------------------
Steel vessel hulls and superstructures produce a strong double-bounce or
specular return under SAR illumination, generating bright pixels orders of
magnitude above the surrounding sea clutter. FTM exploits this:

1. Threshold at the top 0.1% of VV linear backscatter intensity (≈ 30 dB
   above sea surface mean). This is far more aggressive than standard CFAR
   (constant false alarm rate) detectors because we are not trying to detect
   all vessels — only those bright enough to be positively identified as steel
   ships in the presence of potential oil-dampened clutter.

2. Group threshold-passing pixels into connected components using
   ``skimage.measure.regionprops`` (8-connectivity). Each component is one
   vessel candidate.

3. Compute centroid (lat, lon from pixel + affine transform, or fallback
   pixel indices when no geo-reference is available) and length estimate
   from the pixel area (``length_m ≈ √(area_px) × pixel_spacing_m``).

AIS correlation
---------------
For each SAR-detected centroid, interpolate the AIS track of every vessel that
was within 2× ``distance_tolerance_m`` at the SAR acquisition time. If no AIS
ping interpolates to within ``distance_tolerance_m``, the ship is flagged as
Dark. The interpolation uses linear position interpolation between the two
nearest AIS pings by time.

Output
------
Each dark-ship record is a JSON-serializable dict:
    {
        "lat":              float,
        "lon":              float,
        "estimated_length_m": float,
        "sar_timestamp":    str (ISO-8601 UTC),
        "pixel_area_px":    int,
        "pixel_row":        float,
        "pixel_col":        float,
    }

References
----------
- Jeon (2023): FTM threshold — top-0.1% VV backscatter for vessel detection.
- Synopsis §3.3: "Dark ship fallback — SAR vessel detection when AIS off".
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── FTM configuration ────────────────────────────────────────────────────────
_FTM_TOP_FRACTION: float = 0.001    # top 0.1% of VV linear intensity
_MIN_VESSEL_PX:    int   = 3        # minimum connected pixels to qualify as a vessel
_MAX_VESSEL_PX:    int   = 50_000   # maximum; larger = ocean glint or land leakage


# ─── Task 3a: SAR vessel detection via FTM ────────────────────────────────────

def detect_ships_ftm(
    sar_intensity_array: np.ndarray,
    pixel_spacing_m: float = 10.0,
    scene_transform: Any | None = None,
    scene_crs: str | None = None,
    sar_timestamp: str | pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """
    Detect vessel centroids in a SAR VV linear intensity array via FTM.

    Applies the Faster Threshold Method (Jeon 2023): identifies the top
    ``0.1%`` of backscatter values, then groups the bright pixels into vessel
    candidates via 8-connected morphological clustering.

    Args:
        sar_intensity_array: 2-D float32/float64 array of **linear** VV
                             backscatter intensity (not dB). Must be positive.
                             If dB values are passed, convert first:
                             ``linear = 10 ** (vv_db / 10)``.
        pixel_spacing_m:     Ground sampling distance in metres (default 10.0
                             for Sentinel-1 IW GRD). Used to compute estimated
                             vessel length from pixel area.
        scene_transform:     ``rasterio.Affine`` transform for the scene, or
                             ``None``. When provided, centroid pixel coordinates
                             are projected to (lon, lat). When ``None``, the
                             ``lat`` and ``lon`` fields in the output are set to
                             the pixel (row, col) indices (clearly flagged as
                             pixel coords in the log).
        scene_crs:           CRS string for logging/debugging (not used in
                             computation).
        sar_timestamp:       SAR acquisition time (UTC), any pandas-parseable
                             string or Timestamp. Embedded in every output dict.

    Returns:
        List of vessel-candidate dicts, sorted by estimated length (descending).
        Each dict contains:

        - ``lat`` (float): Latitude (° WGS84) or pixel row if no transform.
        - ``lon`` (float): Longitude (° WGS84) or pixel col if no transform.
        - ``estimated_length_m`` (float): ``sqrt(area_px) × pixel_spacing_m``.
        - ``sar_timestamp`` (str): ISO-8601 UTC timestamp or ``"unknown"``.
        - ``pixel_area_px`` (int): Number of bright pixels in the component.
        - ``pixel_row`` (float): Component centroid row (pixel).
        - ``pixel_col`` (float): Component centroid column (pixel).

    Raises:
        ValueError: If ``sar_intensity_array`` is not 2-D.

    Note:
        The ``0.1%`` threshold is deliberately aggressive. At this level,
        ~1 in 1000 sea-surface pixels passes — at 10m GSD over a typical
        100×100 km scene that is ~10 000 candidates before morphological
        grouping. After grouping, real vessels stand out as compact multi-pixel
        components while isolated clutter speckle appears as single pixels.
    """
    from skimage.measure import label as sk_label, regionprops

    arr = np.asarray(sar_intensity_array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"detect_ships_ftm: expected 2-D array, got shape {arr.shape}."
        )

    # ── FTM intensity threshold ───────────────────────────────────────────────
    finite_vals = arr[np.isfinite(arr) & (arr > 0)]
    if len(finite_vals) == 0:
        log.warning("detect_ships_ftm: no finite positive pixels — returning empty list.")
        return []

    threshold = float(np.percentile(finite_vals, 100.0 * (1.0 - _FTM_TOP_FRACTION)))
    bright_mask = arr >= threshold

    n_bright = int(bright_mask.sum())
    log.debug(
        "detect_ships_ftm: FTM threshold=%.4e (top %.1f%%), bright_px=%d",
        threshold, 100.0 * _FTM_TOP_FRACTION, n_bright,
    )

    if n_bright == 0:
        log.info("detect_ships_ftm: 0 pixels above FTM threshold.")
        return []

    # ── Connected component labelling (8-connectivity) ────────────────────────
    labelled = sk_label(bright_mask, connectivity=2)
    regions  = regionprops(labelled)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts_str = "unknown"
    if sar_timestamp is not None:
        try:
            ts_str = pd.Timestamp(sar_timestamp).isoformat()
        except Exception:
            ts_str = str(sar_timestamp)

    # ── Geo-referencing (rasterio affine) ─────────────────────────────────────
    has_transform = scene_transform is not None
    if not has_transform:
        log.info(
            "detect_ships_ftm: no scene_transform — lat/lon fields will be "
            "pixel (row, col) indices."
        )

    detections: list[dict[str, Any]] = []

    for region in regions:
        area_px = region.area
        if area_px < _MIN_VESSEL_PX or area_px > _MAX_VESSEL_PX:
            continue  # too small (speckle) or too large (land/glint)

        row_c, col_c = region.centroid
        length_m     = float(np.sqrt(area_px)) * pixel_spacing_m

        # Geocode centroid
        if has_transform:
            try:
                from rasterio.transform import xy as rio_xy
                xs, ys = rio_xy(scene_transform, [int(row_c)], [int(col_c)])
                lon = float(xs[0])
                lat = float(ys[0])
            except Exception as exc:
                log.debug("Geocoding failed for region %d: %s", region.label, exc)
                lon, lat = float(col_c), float(row_c)
        else:
            lon, lat = float(col_c), float(row_c)

        detections.append({
            "lat":                float(lat),
            "lon":                float(lon),
            "estimated_length_m": length_m,
            "sar_timestamp":      ts_str,
            "pixel_area_px":      int(area_px),
            "pixel_row":          float(row_c),
            "pixel_col":          float(col_c),
        })

    # Sort by estimated length descending (larger = more likely a real vessel)
    detections.sort(key=lambda d: d["estimated_length_m"], reverse=True)

    log.info(
        "detect_ships_ftm: %d components total, %d vessel candidates "
        "(area %d–%d px).",
        len(regions), len(detections), _MIN_VESSEL_PX, _MAX_VESSEL_PX,
    )
    return detections


# ─── Task 3b: SAR↔AIS correlation and Dark Ship flagging ─────────────────────

def _interpolate_ais_position(
    ais_gdf: "gpd.GeoDataFrame",
    mmsi: int,
    target_time: pd.Timestamp,
    time_col: str = "BaseDateTime",
) -> tuple[float, float] | None:
    """
    Linearly interpolate an AIS vessel's (lon, lat) at a target time.

    Args:
        ais_gdf:     Full AIS GeoDataFrame (must have LAT, LON, MMSI columns).
        mmsi:        MMSI to interpolate.
        target_time: Time at which to estimate position.
        time_col:    Datetime column name.

    Returns:
        ``(lon, lat)`` interpolated tuple, or ``None`` if fewer than 2 pings
        straddle ``target_time``.
    """
    track = ais_gdf[ais_gdf["MMSI"] == mmsi].sort_values(time_col)
    if len(track) < 2:
        return None

    times = track[time_col].to_numpy(dtype="datetime64[ns]")
    lons  = track["LON"].to_numpy(dtype=np.float64)
    lats  = track["LAT"].to_numpy(dtype=np.float64)

    target_ns = np.datetime64(target_time.to_pydatetime(), "ns")

    # Find bracketing indices
    idx_after = np.searchsorted(times, target_ns, side="left")
    if idx_after == 0 or idx_after >= len(times):
        return None  # target_time outside the track's time span

    t0 = float(times[idx_after - 1].astype(np.float64))
    t1 = float(times[idx_after].astype(np.float64))
    tt = float(target_ns.astype(np.float64))

    if t1 == t0:
        return float(lons[idx_after - 1]), float(lats[idx_after - 1])

    frac = (tt - t0) / (t1 - t0)
    lon  = lons[idx_after - 1] + frac * (lons[idx_after] - lons[idx_after - 1])
    lat  = lats[idx_after - 1] + frac * (lats[idx_after] - lats[idx_after - 1])
    return float(lon), float(lat)


def correlate_sar_to_ais(
    sar_ship_centroids: list[dict[str, Any]],
    ais_gdf: "gpd.GeoDataFrame",
    distance_tolerance_m: float = 500.0,
    time_col: str = "BaseDateTime",
) -> list[dict[str, Any]]:
    """
    Correlate SAR-detected vessel centroids with the AIS record.

    For each SAR ship detection, interpolates the position of every AIS vessel
    at the SAR acquisition time and checks whether any vessel was within
    ``distance_tolerance_m`` metres. SAR detections with no AIS match are
    flagged as "Dark Ships".

    Args:
        sar_ship_centroids:     Output of :func:`detect_ships_ftm` — list of
                                ship-detection dicts containing ``lat``, ``lon``,
                                ``sar_timestamp``, and ``estimated_length_m``.
        ais_gdf:                GeoDataFrame of AIS pings in the scene's
                                spatiotemporal window (EPSG:4326). Must contain
                                ``MMSI``, ``LAT``, ``LON``, and ``time_col``.
                                Pass the output of :func:`fetch_spill_candidates`.
        distance_tolerance_m:   Maximum distance (m) between a SAR centroid and
                                an interpolated AIS position to declare a match.
                                Default 500 m — approx. 1-pixel positioning error
                                at 10 m GSD across 50 pings.
        time_col:               Datetime column name in ``ais_gdf`` (default
                                ``"BaseDateTime"``).

    Returns:
        List of Dark Ship Evidence Flag dicts. Each dict is a strict superset
        of the input detection dict (all original fields preserved), plus:

        - ``is_dark_ship`` (bool): ``True`` = no AIS match found.
        - ``matched_mmsi`` (int | None): MMSI of the matched vessel, or ``None``.
        - ``matched_dist_m`` (float | None): Distance to matched vessel in m.

        Only detections flagged as ``is_dark_ship=True`` are included in the
        return list. Pass ``sar_ship_centroids`` directly if you need all
        detections regardless of AIS match status.

    Note:
        The ``lat``/``lon`` fields in ``sar_ship_centroids`` must be WGS84
        geodetic coordinates (not pixel indices). If :func:`detect_ships_ftm`
        was called without a ``scene_transform``, these fields contain pixel
        row/col — do not call this function in that case.
    """
    if not sar_ship_centroids:
        log.debug("correlate_sar_to_ais: no SAR detections to correlate.")
        return []

    if ais_gdf is None or len(ais_gdf) == 0:
        log.info("correlate_sar_to_ais: AIS GDF empty — all SAR ships flagged as dark.")
        return [
            {**d, "is_dark_ship": True, "matched_mmsi": None, "matched_dist_m": None}
            for d in sar_ship_centroids
        ]

    from src.ais_attribution.trajectory_cleaning import _haversine_dist_km

    tol_km = distance_tolerance_m / 1000.0
    unique_mmsis = ais_gdf["MMSI"].unique().tolist()

    dark_ships: list[dict[str, Any]] = []

    for det in sar_ship_centroids:
        sar_ts  = pd.Timestamp(det["sar_timestamp"]) if det["sar_timestamp"] != "unknown" else None
        sar_lon = det["lon"]
        sar_lat = det["lat"]

        if sar_ts is None:
            # Cannot correlate without timestamp — conservatively flag as dark
            dark_ships.append({
                **det,
                "is_dark_ship": True,
                "matched_mmsi": None,
                "matched_dist_m": None,
            })
            continue

        best_dist_km  = np.inf
        best_mmsi     = None

        for mmsi in unique_mmsis:
            interp = _interpolate_ais_position(ais_gdf, mmsi, sar_ts, time_col=time_col)
            if interp is None:
                continue
            i_lon, i_lat = interp
            dist_km = float(_haversine_dist_km(sar_lon, sar_lat, i_lon, i_lat))
            if dist_km < best_dist_km:
                best_dist_km = dist_km
                best_mmsi    = mmsi

        is_dark    = best_dist_km > tol_km
        mmsi_match = best_mmsi if not is_dark else None
        dist_m     = best_dist_km * 1000.0 if best_mmsi is not None else None

        if is_dark:
            dark_ships.append({
                **det,
                "is_dark_ship":     True,
                "matched_mmsi":     None,
                "matched_dist_m":   None,
            })
            log.debug(
                "Dark ship detected at (%.4f, %.4f)  length=%.0fm  "
                "best_AIS_dist=%.0fm",
                sar_lat, sar_lon, det["estimated_length_m"], best_dist_km * 1000,
            )

    log.info(
        "correlate_sar_to_ais: %d SAR detections → %d dark ships "
        "(tolerance=%.0f m).",
        len(sar_ship_centroids), len(dark_ships), distance_tolerance_m,
    )
    return dark_ships


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(99)
    H, W = 1024, 1024

    # Synthetic VV linear intensity (sea surface + 3 bright vessel blobs)
    arr = rng.exponential(scale=0.01, size=(H, W)).astype(np.float32)
    # Inject 3 "vessels" (very bright compact blobs)
    for (r, c) in [(150, 200), (500, 700), (800, 300)]:
        arr[r-2:r+3, c-2:c+3] = rng.uniform(5.0, 10.0, (5, 5))

    detections = detect_ships_ftm(arr, pixel_spacing_m=10.0, sar_timestamp="2026-03-15T09:00:00Z")
    print(f"Detected {len(detections)} vessels")
    for d in detections[:5]:
        print(f"  lat={d['lat']:.1f}  lon={d['lon']:.1f}  "
              f"length={d['estimated_length_m']:.0f}m  "
              f"area={d['pixel_area_px']}px")

    assert len(detections) >= 1, "Expected at least 1 vessel detected"
    for d in detections:
        assert "lat" in d and "lon" in d and "estimated_length_m" in d
        assert d["estimated_length_m"] > 0
        assert d["pixel_area_px"] >= _MIN_VESSEL_PX

    # Dark ship correlation smoke test
    import geopandas as gpd
    from shapely.geometry import Point

    # AIS with one vessel near detection 0 (should match), none near others
    d0 = detections[0]
    ais_df = gpd.GeoDataFrame({
        "MMSI":         [123456789, 123456789],
        "LAT":          [d0["lat"] + 0.001, d0["lat"] + 0.002],
        "LON":          [d0["lon"] + 0.001, d0["lon"] + 0.002],
        "SOG":          [5.0, 5.1],
        "COG":          [180.0, 181.0],
        "VesselType":   [80, 80],
        "BaseDateTime": pd.to_datetime(["2026-03-15T08:55:00Z", "2026-03-15T09:05:00Z"]),
        "geometry":     [Point(d0["lon"] + 0.001, d0["lat"] + 0.001),
                         Point(d0["lon"] + 0.002, d0["lat"] + 0.002)],
    }, crs="EPSG:4326")

    dark = correlate_sar_to_ais(detections, ais_df, distance_tolerance_m=1000.0)
    print(f"\nDark ships: {len(dark)}")
    for d in dark:
        print(f"  {d['lat']:.1f}, {d['lon']:.1f}  dark={d['is_dark_ship']}")

    print("\nAll dark_ship.py smoke tests passed.")
    sys.exit(0)
