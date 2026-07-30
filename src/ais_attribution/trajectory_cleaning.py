"""
Module 3 — AIS Trajectory Cleaning and 3D DBSCAN De-spoofing.

Resolves the "mixed MMSI" problem (Jeon 2023): multiple physical vessels
briefly sharing or reusing an MMSI, or GPS-spoofing artifacts that create
implausible position jumps within a single MMSI stream. Per-MMSI 3D DBSCAN
separates interleaved trajectories on (Lat, Lon, Time) without ever conflating
position reports from *different* vessels — which would happen if DBSCAN were
run on the whole multi-MMSI table at once.

Integration with existing codebase
------------------------------------
- `ais_noaa.load_ais_window` / `pair_sar_to_ais` — upstream data providers
- `ais_noaa.filter_to_bilge_relevant_types` — IMO-type pre-filter
- `crs_utils.normalize_longitude_convention` — 0-360 vs ±180 safety
- `sentinel1_cdse.SHIPPING_LANE_BBOXES` — shipping-lane proximity (passive)

References
----------
- Jeon (2023): 3D DBSCAN per-MMSI trajectory de-spoofing
- Synopsis §3.1: ±50 km radius, ±6 h window around spill centroid
- NOAA AIS schema: MMSI, BaseDateTime, LAT, LON, SOG, COG, VesselType, Length
"""
from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from src.data_access.ais_noaa import (
    load_ais_window,
    filter_to_bilge_relevant_types,
    to_geodataframe,
)

log = logging.getLogger(__name__)

# ─── IMO-relevant vessel type codes (NOAA AIS schema) ────────────────────────
# 70–79 = Cargo  |  80–89 = Tanker  |  30–39 = Fishing
# Extend with 20-29 (Wing-in-ground) or 60-69 (Passenger) if scope changes.
_BILGE_RELEVANT_TYPES: frozenset[int] = frozenset(
    set(range(70, 90)) | set(range(30, 40))
)

# Minimum vessel length (metres) for fishing vessels (synopsis: ">50 m")
_MIN_FISHING_LENGTH_M: float = 50.0

# ─── Haversine helper ─────────────────────────────────────────────────────────
_EARTH_RADIUS_KM: float = 6371.0


def _haversine_bbox(
    center_lon: float,
    center_lat: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """
    Compute an axis-aligned bounding box (WGS84) around a centre point.

    The latitude offset is exact (1° lat ≈ 111.32 km everywhere).
    The longitude offset uses the cosine correction at ``center_lat`` to
    account for meridian convergence.

    Args:
        center_lon: Longitude of the centre point (degrees, –180 to 180).
        center_lat: Latitude of the centre point (degrees, –90 to 90).
        radius_km:  Half-width of the bounding box in kilometres.

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 degrees.
    """
    d_lat = radius_km / _EARTH_RADIUS_KM * (180.0 / np.pi)
    d_lon = radius_km / (
        _EARTH_RADIUS_KM * np.cos(np.radians(center_lat))
    ) * (180.0 / np.pi)
    return (
        center_lon - d_lon,
        center_lat - d_lat,
        center_lon + d_lon,
        center_lat + d_lat,
    )


def _haversine_dist_km(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """
    Vectorised haversine distance (km) between two WGS84 points or arrays.

    Significantly more accurate than flat-earth approximations near the poles
    or at distances > 100 km.

    Args:
        lon1, lat1: Source point(s) in degrees.
        lon2, lat2: Destination point(s) in degrees.

    Returns:
        Great-circle distance(s) in kilometres.
    """
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ─── Task 1a: Fetch spill candidates ─────────────────────────────────────────

def fetch_spill_candidates(
    spill_centroid_lon: float,
    spill_centroid_lat: float,
    spill_time: str | pd.Timestamp,
    ais_csv_path: str,
    radius_km: float = 50.0,
    window_hours: float = 6.0,
    chunksize: int = 500_000,
) -> gpd.GeoDataFrame:
    """
    Load AIS position reports within a spatiotemporal window around a verified
    spill centroid.

    Implements the synopsis §3.1 search geometry:
    - **Spatial**: ±``radius_km`` km circular radius (approximated as a WGS84
      bounding box, then filtered precisely via Haversine distance).
    - **Temporal**: ±``window_hours`` hours around the SAR acquisition time.
    - **Vessel types**: IMO-relevant tankers, bulk carriers, container ships,
      and fishing vessels > 50 m (NOAA VesselType codes 70–89, 30–39).

    Args:
        spill_centroid_lon: Longitude of the verified spill centroid (°, WGS84).
        spill_centroid_lat: Latitude of the verified spill centroid (°, WGS84).
        spill_time:         SAR acquisition timestamp (UTC). Any
                            pandas-parseable string or ``pd.Timestamp``.
        ais_csv_path:       Path to the NOAA AIS monthly CSV file. The function
                            reads in chunks via :func:`ais_noaa.load_ais_window`
                            so multi-GB files are handled safely.
        radius_km:          Spatial search radius in km (default 50.0 per synopsis).
        window_hours:       Temporal half-window in hours (default 6.0 per synopsis).
        chunksize:          CSV reading chunk size (rows). Default 500 000.

    Returns:
        GeoDataFrame (EPSG:4326) with one row per AIS ping passing all filters.
        Columns inherited from NOAA schema plus ``geometry`` (Point).
        Returns an empty GeoDataFrame if no candidates are found.

    Note:
        Length filtering for fishing vessels (>50 m) is applied when the
        ``Length`` column is present. If it is absent, all fishing vessels
        pass the type filter (conservative — prefer false positives over misses).
    """
    spill_ts = pd.Timestamp(spill_time)
    if spill_ts.tzinfo is None:
        spill_ts = spill_ts.tz_localize("UTC")

    start_time = spill_ts - pd.Timedelta(hours=window_hours)
    end_time   = spill_ts + pd.Timedelta(hours=window_hours)
    bbox       = _haversine_bbox(spill_centroid_lon, spill_centroid_lat, radius_km)

    log.info(
        "fetch_spill_candidates: spill=(%.4f, %.4f) t=%s  "
        "bbox=%s  window=[%s, %s]",
        spill_centroid_lon, spill_centroid_lat, spill_ts,
        bbox, start_time, end_time,
    )

    raw_df = load_ais_window(
        csv_path   = ais_csv_path,
        bbox       = bbox,
        start_time = start_time,
        end_time   = end_time,
        chunksize  = chunksize,
    )

    if raw_df.empty:
        log.info("fetch_spill_candidates: no AIS pings in bbox/window.")
        return gpd.GeoDataFrame(columns=["MMSI", "BaseDateTime", "LAT", "LON",
                                          "SOG", "COG", "VesselType", "geometry"])

    # ── IMO-type filter ───────────────────────────────────────────────────────
    typed_df = filter_to_bilge_relevant_types(raw_df)

    # ── Fishing vessel length filter (>50 m) ──────────────────────────────────
    if "Length" in typed_df.columns and "VesselType" in typed_df.columns:
        fishing_mask = typed_df["VesselType"].between(30, 39)
        short_fishing = fishing_mask & (
            typed_df["Length"].fillna(0.0) < _MIN_FISHING_LENGTH_M
        )
        typed_df = typed_df[~short_fishing].copy()
        log.debug("Length filter: removed %d short fishing vessels.", short_fishing.sum())

    # ── Precise circular distance filter ──────────────────────────────────────
    # The bbox is a rectangular approximation; a vessel in the bbox corner
    # could be >50 km away. Filter precisely using Haversine.
    dist_km = _haversine_dist_km(
        typed_df["LON"].to_numpy(),
        typed_df["LAT"].to_numpy(),
        spill_centroid_lon,
        spill_centroid_lat,
    )
    in_radius = dist_km <= radius_km
    typed_df  = typed_df[in_radius].copy()
    typed_df["dist_to_spill_km"] = dist_km[in_radius]

    if typed_df.empty:
        log.info("fetch_spill_candidates: 0 candidates after all filters.")
        return gpd.GeoDataFrame(columns=typed_df.columns.tolist() + ["geometry"])

    gdf = to_geodataframe(typed_df, src_crs="EPSG:4326")
    log.info(
        "fetch_spill_candidates: %d AIS pings, %d unique MMSIs.",
        len(gdf), gdf["MMSI"].nunique(),
    )
    return gdf


# ─── Task 1b: Per-MMSI 3D DBSCAN trajectory cleaning ─────────────────────────

def apply_3d_dbscan(
    ais_gdf: gpd.GeoDataFrame,
    spatial_eps_km: float = 2.0,
    temporal_eps_hr: float = 0.5,
    min_samples: int = 5,
    time_col: str = "BaseDateTime",
) -> gpd.GeoDataFrame:
    """
    Separate mixed or spoofed AIS trajectories within each MMSI using 3D DBSCAN.

    **Why per-MMSI?**
    Running DBSCAN across the entire multi-MMSI table would cluster by spatial
    proximity *between* vessels, not by identity continuity *within* one MMSI.
    The problem being solved (Jeon 2023) is that two physical ships briefly
    share the same MMSI, interleaving their pings into one stream. Only a
    per-MMSI run separates those sub-streams.

    **3D feature space**: ``(lat_km, lon_km, time_scaled_km)``

    - Latitude → km using 1° ≈ 111.32 km.
    - Longitude → km using the cosine correction at the MMSI centroid latitude.
    - Time → hours from the track's first ping, then scaled so that
      ``temporal_eps_hr`` has the same metric weight as ``spatial_eps_km``::

          time_feature = elapsed_hours × (spatial_eps_km / temporal_eps_hr)

      This makes ``eps`` in DBSCAN correspond simultaneously to
      ``spatial_eps_km`` in space **and** ``temporal_eps_hr`` in time.

    **Noise handling**: DBSCAN label –1 (noise) points are dropped. They
    represent isolated pings that cannot be assigned to any coherent trajectory.

    Args:
        ais_gdf:         GeoDataFrame of AIS pings (output of
                         :func:`fetch_spill_candidates`). Must contain columns
                         ``MMSI``, ``LAT``, ``LON``, and ``time_col``.
        spatial_eps_km:  DBSCAN epsilon in kilometres (default 2.0 km — roughly
                         the distance a vessel travels in 4 min at 15 kn).
        temporal_eps_hr: Temporal component of epsilon in hours (default 0.5 h).
                         Together with ``spatial_eps_km``, defines the combined
                         3D neighbourhood threshold.
        min_samples:     Minimum points to form a core cluster (default 5).
                         Lower values allow sparser trajectory fragments through;
                         higher values reject noisy sporadic pings.
        time_col:        Name of the datetime column (default ``"BaseDateTime"``).

    Returns:
        GeoDataFrame identical to ``ais_gdf`` but with two added columns:

        - ``track_id`` (``str``): Unique trajectory identifier formatted as
          ``"<MMSI>_<ClusterID>"``, e.g. ``"123456789_0"``.
        - ``cluster_label`` (``int``): Raw DBSCAN label (≥ 0; noise –1 dropped).

        Noise points (DBSCAN label –1) are **excluded** from the return value.

    Raises:
        ValueError: If required columns are missing from ``ais_gdf``.
    """
    required = {"MMSI", "LAT", "LON", time_col}
    missing  = required - set(ais_gdf.columns)
    if missing:
        raise ValueError(
            f"apply_3d_dbscan: missing columns {missing} in ais_gdf."
        )

    if ais_gdf.empty:
        log.debug("apply_3d_dbscan: input is empty — returning empty GeoDataFrame.")
        return ais_gdf.assign(track_id=pd.Series(dtype=str),
                              cluster_label=pd.Series(dtype=int))

    # Time-to-space scaling factor: maps 1 hour → spatial_eps_km/temporal_eps_hr km
    _time_scale = spatial_eps_km / max(temporal_eps_hr, 1e-9)

    processed: list[pd.DataFrame] = []
    n_noise_total = 0

    for mmsi, group in ais_gdf.groupby("MMSI"):
        g = group.sort_values(time_col).copy()
        n = len(g)

        if n < min_samples:
            # Fewer pings than min_samples → all would be noise → skip MMSI
            log.debug("MMSI %s: only %d pings (< min_samples=%d) — skipped.",
                      mmsi, n, min_samples)
            continue

        # ── Centroid latitude for cosine correction ──────────────────────────
        lat_centre = g["LAT"].mean()
        cos_lat    = np.cos(np.radians(lat_centre))

        # ── 3D feature array [lat_km, lon_km, time_km_equiv] ─────────────────
        t_elapsed_hr = (
            g[time_col] - g[time_col].iloc[0]
        ).dt.total_seconds() / 3600.0

        coords = np.column_stack([
            g["LAT"].to_numpy()  * 111.32,          # degrees → km
            g["LON"].to_numpy()  * 111.32 * cos_lat, # degrees → km (corrected)
            t_elapsed_hr.to_numpy() * _time_scale,   # hours → km-equivalent
        ])

        labels = DBSCAN(
            eps        = spatial_eps_km,
            min_samples= min_samples,
            n_jobs     = 1,             # per-MMSI: 1 job; parallelism is at MMSI level
        ).fit_predict(coords)

        n_noise = int((labels == -1).sum())
        n_noise_total += n_noise

        g["cluster_label"] = labels
        g["track_id"]      = [
            f"{mmsi}_{lbl}" if lbl >= 0 else None
            for lbl in labels
        ]

        # Drop noise points
        g = g[g["cluster_label"] >= 0].copy()
        if not g.empty:
            processed.append(g)

    if not processed:
        log.warning("apply_3d_dbscan: all pings classified as noise. "
                    "Consider lowering min_samples or increasing spatial_eps_km.")
        return ais_gdf.iloc[0:0].assign(
            track_id=pd.Series(dtype=str),
            cluster_label=pd.Series(dtype=int),
        )

    result = gpd.GeoDataFrame(
        pd.concat(processed, ignore_index=True),
        crs=ais_gdf.crs,
    )
    n_tracks = result["track_id"].nunique()
    log.info(
        "apply_3d_dbscan: %d clean pings, %d unique tracks  "
        "(dropped %d noise pings).",
        len(result), n_tracks, n_noise_total,
    )
    return result


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(42)
    n   = 600

    # Synthetic AIS: 10 MMSIs, each with a small trajectory + 1 "spoof" burst
    mmsi_pool = [100_000_000 + i for i in range(10)]
    df = pd.DataFrame({
        "MMSI":         np.repeat(mmsi_pool, n // 10),
        "BaseDateTime": pd.date_range("2026-03-15 06:00", periods=n, freq="3min", tz="UTC"),
        "LAT":          28.5 + rng.normal(0, 0.05, n),
        "LON":         -90.0 + rng.normal(0, 0.05, n),
        "SOG":          rng.uniform(0, 14, n),
        "COG":          rng.uniform(0, 360, n),
        "VesselType":   rng.choice(list(range(70, 90)), n),
    })
    # Inject a "spoof" burst — same MMSI, far-away position
    df.loc[df["MMSI"] == 100_000_000, "LON"] = np.where(
        np.arange(n // 10) < 10, -95.0, df.loc[df["MMSI"] == 100_000_000, "LON"]
    )

    gdf = to_geodataframe(df)
    result = apply_3d_dbscan(gdf, spatial_eps_km=2.0, temporal_eps_hr=0.5, min_samples=3)

    print(f"Input : {len(gdf)} pings,  {gdf['MMSI'].nunique()} MMSIs")
    print(f"Output: {len(result)} pings, {result['track_id'].nunique()} tracks")

    assert "track_id" in result.columns, "track_id missing"
    assert (result["cluster_label"] >= 0).all(), "Noise points not dropped"
    assert result["track_id"].str.match(r"^\d+_\d+$").all(), "track_id format wrong"

    print("All trajectory_cleaning smoke tests passed.")
    sys.exit(0)
