"""
Module 2 — 12-Feature Extraction Engine for look-alike discrimination.

Computes exactly 12 features per connected component (dark patch candidate)
from the Module 1 segmentation output, organised into 4 categories per the
project synopsis:

    Category        Features (4 each)
    ─────────────── ──────────────────────────────────────────────────
    Polarimetric    mean_H, anisotropy_A, mean_rvi_dp, copol_ratio_VV_VH
    Geometric       area_km2, elongation, perimeter_area_ratio, compactness
    Contextual      wind_speed_ms, proximity_shipping_lane_km, is_night
    Temporal        morphology_change_km2

IMPORTANT — RVI_dp instead of Alpha
-----------------------------------
The synopsis specifies Cloude-Pottier alpha (α) angle. However, α requires
the eigenvectors of the coherency matrix, which in turn require complex phase
information. Sentinel-1 GRD products are phase-discarded amplitude products
(phase is destroyed during SNAP's detection step regardless of polarization).
Thus, alpha is uncomputable from this dataset. In its place, we use the
dual-pol Radar Vegetation Index (RVI_dp = 4*VH/(VV+VH)), a phase-free
descriptor used for dual-pol surface discrimination, applied here as a proxy
for surface-roughness/depolarization contrast around oil slicks.

IMPORTANT — Anisotropy A degenerate case
-----------------------------------------
Sentinel-1 IW GRD is dual-polarisation (VV + VH). The Cloude-Pottier
Anisotropy A requires **three eigenvalues** and is therefore undefined for
dual-pol data. `anisotropy_A` is always 0.0 in this pipeline (via
`polsar_decomp.compute_anisotropy_placeholder()`). The Random Forest will
assign near-zero importance to this column; it is kept for:
  (a) API compatibility with future quad-pol products (RADARSAT-2, ALOS-2),
  (b) explicit documentation that A was considered and found degenerate.

References
----------
- Song et al. (2024)       — polarimetric features H, A, α, co-pol ratio
- Chen & Wang (2022)       — H/A/α validated for oil-spill discrimination
- Yang et al. (2022)       — geometric shape descriptors for slick morphology
- Liao et al. (2023)       — contextual: wind, shipping lane proximity, time-of-day
- Li et al. (2023)         — temporal morphology change across acquisitions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.polsar_decomp import (
    db_to_linear,
    dual_pol_entropy_rvi,
    compute_anisotropy_placeholder,
)
from src.data_access.sentinel1_cdse import SHIPPING_LANE_BBOXES

log = logging.getLogger(__name__)


# ─── Feature name registry ────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Polarimetric (Song et al. 2024; Chen & Wang 2022)
    "mean_H",                   # Entropy H ∈ [0, 1]
    "anisotropy_A",             # Always 0.0 for dual-pol (see module docstring)
    "mean_rvi_dp",              # Dual-pol Radar Vegetation Index (RVI_dp)
    "copol_ratio_VV_VH",        # VV_lin / VH_lin (linear scale, per-patch mean)
    # Geometric (Yang et al. 2022)
    "area_km2",                 # Component area in km²
    "elongation",               # major_axis / minor_axis (≥ 1)
    "perimeter_area_ratio",     # perimeter / area  (shape complexity)
    "compactness",              # 4π × area / perimeter²  (1.0 = circle)
    # Contextual (Liao et al. 2023)
    "wind_speed_ms",            # ERA5 10-m neutral wind speed in m/s
    "proximity_shipping_lane_km",  # Geodetic distance to nearest lane bbox
    "is_night",                 # 1 if 20:00 ≤ hour_local < 6:00 else 0
    # Temporal (Li et al. 2023)
    "morphology_change_km2",    # |Δarea| from prior acquisition; 0.0 if none
]

#: Metadata columns carried in the DataFrame but NOT used as RF features.
META_COLUMNS: list[str] = [
    "scene_id",
    "component_label",
    "centroid_row",
    "centroid_col",
    "has_temporal_pair",   # bool: whether morphology_change_km2 is non-degenerate
    "label",               # ground-truth class: 1=oil, 0=lookalike (if available)
]


# ─── Shipping-lane proximity (Liao et al. 2023) ────────────────────────────────

def _scene_centroid_lonlat(
    centroid_row: float,
    centroid_col: float,
    scene_transform: Any | None,
) -> tuple[float, float] | None:
    """
    Convert (row, col) centroid to (lon, lat) using the scene's affine transform.

    Parameters
    ----------
    centroid_row, centroid_col : pixel coordinates (from regionprops.centroid)
    scene_transform            : rasterio Affine transform, or None

    Returns
    -------
    (lon, lat) tuple or None if transform unavailable.
    """
    if scene_transform is None:
        return None
    try:
        from rasterio.transform import xy as rio_xy
        row_int = int(centroid_row)
        col_int = int(centroid_col)
        # rasterio.transform.xy returns (xs, ys) in the native CRS
        xs, ys = rio_xy(scene_transform, [row_int], [col_int])
        return float(xs[0]), float(ys[0])
    except Exception as exc:
        log.debug("_scene_centroid_lonlat: transform failed — %s", exc)
        return None


def _proximity_to_shipping_lanes_km(
    lon: float | None,
    lat: float | None,
    lane_bboxes: dict[str, tuple[float, float, float, float]] = SHIPPING_LANE_BBOXES,
) -> float:
    """
    Compute the minimum geodetic distance (km) from a point to any shipping
    lane bounding box defined in sentinel1_cdse.SHIPPING_LANE_BBOXES.

    Uses Shapely for efficient bbox distance computation. Falls back to
    999.0 km (a large sentinel value) if coordinates are unavailable or
    Shapely is not installed.

    Parameters
    ----------
    lon, lat    : WGS84 longitude / latitude of the patch centroid (degrees).
                  Pass None to get the fallback sentinel value.
    lane_bboxes : Dict of (min_lon, min_lat, max_lon, max_lat) tuples.
                  Defaults to the 4 canonical shipping lanes in the project.

    Returns
    -------
    min_dist_km : float — minimum distance to any lane bbox, in kilometres.
                  0.0 if the point is inside a lane bbox.
                  999.0 if coordinates are unavailable.
    """
    if lon is None or lat is None:
        return 999.0

    try:
        from shapely.geometry import Point, box

        point = Point(lon, lat)
        min_dist_deg = min(
            point.distance(box(min_lon, min_lat, max_lon, max_lat))
            for min_lon, min_lat, max_lon, max_lat in lane_bboxes.values()
        )
        # 1° ≈ 111.32 km at equator — adequate approximation for lane proximity
        return float(min_dist_deg) * 111.32

    except ImportError:
        log.warning(
            "Shapely not installed — proximity_shipping_lane_km set to 999.0. "
            "Install with: pip install shapely"
        )
        return 999.0
    except Exception as exc:
        log.debug("_proximity_to_shipping_lanes_km: failed — %s", exc)
        return 999.0


# ─── Per-component feature extraction ─────────────────────────────────────────

def _extract_region_features(
    region,                         # skimage RegionProperties
    vv_db: np.ndarray,              # (H, W) float32, Sigma0 VV in dB
    vh_db: np.ndarray,              # (H, W) float32, Sigma0 VH in dB
    H_map: np.ndarray,              # (H, W) float32, Entropy H in [0, 1]
    rvi_dp_map: np.ndarray,         # (H, W) float32, dual-pol Radar Vegetation Index (RVI_dp), unnormalised >= 0. Substitutes Cloude-Pottier alpha, which is uncomputable from phase-discarded Sentinel-1 GRD amplitude data (see module docstring).
    gsd_m: float,                   # Ground sampling distance in metres (10.0 for S1 IW GRD)
    wind_speed_ms: float,           # ERA5 U10, m/s
    proximity_km: float,            # Pre-computed shipping-lane proximity
    hour_local: int,                # Local hour 0–23 at scene acquisition
    morphology_change_km2: float,   # |Δarea| from temporal pair; 0.0 if none
    has_temporal_pair: bool,        # Whether temporal feature is non-degenerate
    scene_id: str,
    label: int | None,
) -> dict[str, Any]:
    """
    Extract the 12 canonical features for one connected component (region).

    This is the inner loop function called once per RegionProperties entry.
    All 12 FEATURE_NAMES columns are populated here, plus META_COLUMNS.

    Parameters
    ----------
    region          : skimage.measure.RegionProperties instance
    vv_db, vh_db    : full-scene dB arrays (NOT patch-cropped); indexing via
                      region.coords for per-pixel values inside the component
    H_map, rvi_dp_map : full-scene polsar decomp arrays (linear-scale H,
                      unnormalised RVI_dp). These are pre-computed once per scene
                      for efficiency and passed in.
    gsd_m           : GSD in metres (10.0 default for Sentinel-1 IW GRD 10m)
    wind_speed_ms   : Scene-level ERA5 wind speed (scalar). Per-patch centroid
                      ERA5 would be more accurate but requires ERA5 spatial query;
                      scene-mean is the standard offline approximation.
    proximity_km    : Pre-computed proximity_to_shipping_lane for this centroid.
    hour_local      : Local solar hour at scene acquisition centre.
    morphology_change_km2 : |Δarea_km2| from a temporally-paired prior scene.
                      Set to 0.0 if no temporal pair exists.
    has_temporal_pair : bool — whether the temporal feature is meaningful.
    scene_id        : Scene identifier string (for grouping in GroupKFold).
    label           : Ground-truth label (1=oil, 0=lookalike) or None.

    Returns
    -------
    row : dict with exactly len(FEATURE_NAMES) + len(META_COLUMNS) entries.
    """
    ys, xs = region.coords[:, 0], region.coords[:, 1]

    # ── Polarimetric features (Song et al. 2024; Chen & Wang 2022) ──────────
    vv_lin_vals = np.power(10.0, np.clip(vv_db[ys, xs], -50.0, None) / 10.0)
    vh_lin_vals = np.power(10.0, np.clip(vh_db[ys, xs], -50.0, None) / 10.0)

    mean_H            = float(H_map[ys, xs].mean())
    anisotropy_A      = 0.0          # Degenerate: dual-pol, see module docstring
    mean_rvi_dp       = float(rvi_dp_map[ys, xs].mean())
    copol_ratio_VV_VH = float(vv_lin_vals.mean() / max(vh_lin_vals.mean(), 1e-9))

    # ── Geometric features (Yang et al. 2022) ───────────────────────────────
    px_area_km2         = (gsd_m / 1000.0) ** 2          # km² per pixel
    area_km2            = float(region.area) * px_area_km2

    minor              = getattr(region, 'axis_minor_length', None) or region.minor_axis_length
    maj                = getattr(region, 'axis_major_length', None) or region.major_axis_length
    elongation         = float(maj / max(minor, 1e-6))

    perimeter          = max(region.perimeter, 1e-6)
    perimeter_area_ratio = float(perimeter / max(region.area, 1e-6))

    # Compactness = 4π·A / P² → 1.0 for a circle, → 0 for very spiky shapes.
    # (Note: the scaffold used P²/A — we use the ISO definition here.)
    compactness = float((4.0 * np.pi * region.area) / max(perimeter ** 2, 1e-9))

    # ── Contextual features (Liao et al. 2023) ──────────────────────────────
    is_night = int(hour_local < 6 or hour_local >= 20)

    # ── Temporal feature (Li et al. 2023) ───────────────────────────────────
    # morphology_change_km2 is supplied by the caller; 0.0 for single-pass.

    # ── Assemble row ────────────────────────────────────────────────────────
    cr, cc = region.centroid
    row: dict[str, Any] = {
        # Meta
        "scene_id":            scene_id,
        "component_label":     int(region.label),
        "centroid_row":        float(cr),
        "centroid_col":        float(cc),
        "has_temporal_pair":   has_temporal_pair,
        "label":               label,
        # Polarimetric
        "mean_H":              mean_H,
        "anisotropy_A":        anisotropy_A,
        "mean_rvi_dp":         mean_rvi_dp,
        "copol_ratio_VV_VH":   copol_ratio_VV_VH,
        # Geometric
        "area_km2":            area_km2,
        "elongation":          elongation,
        "perimeter_area_ratio": perimeter_area_ratio,
        "compactness":         compactness,
        # Contextual
        "wind_speed_ms":       wind_speed_ms,
        "proximity_shipping_lane_km": proximity_km,
        "is_night":            is_night,
        # Temporal
        "morphology_change_km2": morphology_change_km2,
    }
    return row


# ─── Scene-level extraction (public API) ─────────────────────────────────────

def extract_scene_features(
    regions: list,
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    scene_id: str,
    gsd_m: float = 10.0,
    wind_speed_ms: float = 7.0,
    hour_local: int = 12,
    scene_transform: Any | None = None,
    scene_crs: str | None = None,
    prior_area_km2_map: dict[int, float] | None = None,
    label_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Extract all 12 features for every component in a scene and return a DataFrame.

    This is the primary public entry point. Call once per scene after
    `morphology.close_and_extract()`.

    Parameters
    ----------
    regions          : list[RegionProperties] — output of morphology.extract_components()
    vv_db, vh_db     : (H, W) float32 — Sigma0 VV / VH in dB for the full scene
    scene_id         : str — unique scene identifier (used as GroupKFold group key)
    gsd_m            : Ground sampling distance in metres. Default 10.0 m (S1 IW GRD).
    wind_speed_ms    : ERA5 U10 wind speed at scene acquisition time, m/s.
                       Default 7.0 m/s = open-ocean climatological mean (same
                       fallback as band_stack.py and wind_ratio.py).
    hour_local       : Integer hour (0-23) of local solar time at scene acquisition.
                       Used for the night-time prior (feature 11).
    scene_transform  : rasterio.Affine transform for the scene, or None.
                       Required to compute proximity_shipping_lane_km accurately.
                       If None, proximity defaults to 999.0 km.
    scene_crs        : CRS string (e.g. 'EPSG:32636') for context; not used directly
                       in feature computation but available for logging / debugging.
    prior_area_km2_map : Optional dict mapping component_label → area_km2 from a
                       temporally-paired prior acquisition. Used to compute
                       morphology_change_km2. If None or key missing → 0.0.
    label_map        : Optional dict mapping component_label → ground-truth label
                       (1=oil, 0=lookalike). If None, label column = None (inference mode).

    Returns
    -------
    df : pd.DataFrame
        Columns: META_COLUMNS + FEATURE_NAMES
        One row per region. Empty DataFrame if regions is empty.
    """
    if not regions:
        log.debug("extract_scene_features: no regions for scene %s", scene_id)
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES)

    # ── Pre-compute polsar decomp once for the full scene ───────────────────
    vv_db_arr = np.asarray(vv_db, dtype=np.float32)
    vh_db_arr = np.asarray(vh_db, dtype=np.float32)
    vv_lin    = np.power(10.0, np.maximum(vv_db_arr, -50.0) / 10.0)
    vh_lin    = np.power(10.0, np.maximum(vh_db_arr, -50.0) / 10.0)
    H_map, rvi_dp_map = dual_pol_entropy_rvi(vv_lin, vh_lin)
    # rvi_dp_map is RAW RVI_dp (not the /2.0-clipped [0,1] version used for
    # Band 3 of the CNN input in band_stack.py / zenodo_sos_dataset.py).
    # RVI_dp replaces alpha here because alpha requires complex SLC phase
    # that Sentinel-1 GRD amplitude products do not retain (see module
    # docstring). Raw scale is kept for RF feature-importance interpretability.

    rows = []
    for region in regions:
        # ── Centroid → lonlat → proximity ───────────────────────────────────
        cr, cc = region.centroid
        lonlat = _scene_centroid_lonlat(cr, cc, scene_transform)
        if lonlat is not None:
            lon, lat = lonlat
        else:
            lon, lat = None, None
        proximity_km = _proximity_to_shipping_lanes_km(lon, lat)

        # ── Temporal feature ─────────────────────────────────────────────────
        has_temporal = (
            prior_area_km2_map is not None
            and region.label in prior_area_km2_map
        )
        if has_temporal:
            px_area_km2 = (gsd_m / 1000.0) ** 2
            curr_area   = float(region.area) * px_area_km2
            morph_change = abs(curr_area - prior_area_km2_map[region.label])
        else:
            morph_change = 0.0

        # ── Label ────────────────────────────────────────────────────────────
        lbl = label_map.get(region.label) if label_map is not None else None

        row = _extract_region_features(
            region              = region,
            vv_db               = vv_db_arr,
            vh_db               = vh_db_arr,
            H_map               = H_map,
            rvi_dp_map          = rvi_dp_map,
            gsd_m               = gsd_m,
            wind_speed_ms       = wind_speed_ms,
            proximity_km        = proximity_km,
            hour_local          = hour_local,
            morphology_change_km2 = morph_change,
            has_temporal_pair   = has_temporal,
            scene_id            = scene_id,
            label               = lbl,
        )
        rows.append(row)

    df = pd.DataFrame(rows, columns=META_COLUMNS + FEATURE_NAMES)
    log.info(
        "extract_scene_features: scene=%s  components=%d  shape=%s",
        scene_id, len(df), df.shape,
    )
    return df


def build_feature_dataframe(
    scene_dicts: list[dict],
    gsd_m: float = 10.0,
) -> pd.DataFrame:
    """
    Batch-extract features across a list of pre-processed scene dictionaries.

    Each dict in scene_dicts must contain:
        "scene_id"       : str
        "regions"        : list[RegionProperties]  (from morphology.extract_components)
        "vv_db"          : (H, W) float32 array
        "vh_db"          : (H, W) float32 array
        "wind_speed_ms"  : float
        "hour_local"     : int (0–23)
    Optional keys:
        "scene_transform"      : rasterio Affine (for proximity feature)
        "prior_area_km2_map"   : dict[int, float] (for temporal feature)
        "label_map"            : dict[int, int]   (for supervised training)

    Parameters
    ----------
    scene_dicts : list of scene-spec dicts (see above)
    gsd_m       : GSD in metres, applied uniformly (all scenes assumed same resolution)

    Returns
    -------
    df : pd.DataFrame — concatenated feature rows from all scenes
    """
    all_dfs = []
    for d in scene_dicts:
        df = extract_scene_features(
            regions          = d["regions"],
            vv_db            = d["vv_db"],
            vh_db            = d["vh_db"],
            scene_id         = d["scene_id"],
            gsd_m            = gsd_m,
            wind_speed_ms    = d.get("wind_speed_ms", 7.0),
            hour_local       = d.get("hour_local", 12),
            scene_transform  = d.get("scene_transform", None),
            prior_area_km2_map = d.get("prior_area_km2_map", None),
            label_map        = d.get("label_map", None),
        )
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_NAMES)

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "build_feature_dataframe: %d scenes → %d total components",
        len(scene_dicts), len(combined),
    )
    return combined


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.lookalike.morphology import close_and_extract

    rng = np.random.default_rng(7)
    H, W = 512, 512

    # Synthetic scene
    vv_db = rng.uniform(-25.0, -5.0,  (H, W)).astype(np.float32)
    vh_db = rng.uniform(-30.0, -10.0, (H, W)).astype(np.float32)

    # 2 synthetic oil streaks
    mask = np.zeros((H, W), dtype=bool)
    mask[50:62, 30:280]  = True
    mask[300:310, 100:380] = True

    _, regions = close_and_extract(mask)
    print(f"Components: {len(regions)}")

    scene = {
        "scene_id":      "smoke_test_001",
        "regions":       regions,
        "vv_db":         vv_db,
        "vh_db":         vh_db,
        "wind_speed_ms": 8.5,
        "hour_local":    22,   # Night-time → is_night=1
        "label_map":     {r.label: 1 for r in regions},  # all labelled oil
    }
    df = build_feature_dataframe([scene], gsd_m=10.0)

    print(f"DataFrame shape: {df.shape}")
    print(df[FEATURE_NAMES].to_string())

    # ── Assertions ──────────────────────────────────────────────────────────
    assert set(FEATURE_NAMES).issubset(df.columns), "Missing feature columns"
    assert len(FEATURE_NAMES) == 12, f"Expected 12 features, got {len(FEATURE_NAMES)}"
    assert len(df) == len(regions), f"Row count mismatch: {len(df)} vs {len(regions)}"
    assert df["anisotropy_A"].eq(0.0).all(), "anisotropy_A must be zero for dual-pol"
    assert df["is_night"].eq(1).all(), "hour=22 should produce is_night=1"
    assert (df["area_km2"] > 0).all(), "area_km2 must be positive"
    assert (df["elongation"] >= 1.0).all(), "elongation must be ≥ 1"
    assert (df["compactness"] > 0).all(), "compactness must be positive"
    assert (df["label"] == 1).all(), "labels should be 1 (oil)"

    print(f"\nAll {len(FEATURE_NAMES)} features present and validated.")
    print("All features.py smoke tests passed.")
    sys.exit(0)
