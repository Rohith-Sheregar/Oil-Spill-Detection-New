# Module 3 Audit — AIS Vessel Candidate Filtering
## Oil Spill Detection Pipeline | Rohith Sheregar

**Date:** 2026-07-30 (implementation) · **Updated:** 2026-08-01 (data-source verification)
**Status:** ✅ COMPLETE — All synopsis requirements implemented and smoke-tested
**Scope:** `src/ais_attribution/` (4 files) + `src/data_access/ais_noaa.py`

---

## 0. Data Source Summary

> [!IMPORTANT]
> Module 3 does **NOT** require any Copernicus portal accounts. AIS data comes from NOAA Marine Cadastre — free, public, no login required.

| Data | Source | Account? | URL |
|------|--------|---------|-----|
| AIS vessel positions | **NOAA Marine Cadastre** | ❌ No account | [hub.marinecadastre.gov/pages/vesseltraffic](https://hub.marinecadastre.gov/pages/vesseltraffic) |
| Sentinel-1 VV array (dark-ship FTM) | Already downloaded by M1 | ❌ No extra account | From the same GRD TIFF used by Module 1 |

**How to get AIS data:**
1. Go to [hub.marinecadastre.gov/pages/vesseltraffic](https://hub.marinecadastre.gov/pages/vesseltraffic)
2. Select year + UTM zone matching your SAR scene (Gulf of Mexico → Zone 14/15/16)
3. Download the monthly CSV — chunked reading in `ais_noaa.py` handles GB-scale files safely

**File format:** `AIS_YYYY_MM_ZoneNN.csv` — columns: `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType`

---

## 1. Synopsis Requirements vs. Implementation Status

### 3.1 AIS Data Acquisition and Cleaning

| # | Synopsis Requirement | Status | Implementation Detail |
|---|---|---|---|
| 3.1.1 | Download AIS records ±6 hours around SAR time | ✅ | `fetch_spill_candidates` applies ±6h temporal window (`window_hours=6.0`) when querying `ais_noaa.load_ais_window()`. |
| 3.1.2 | Search ±50 km around detected spill centroid | ✅ | `fetch_spill_candidates` uses `_haversine_bbox` + exact Haversine distance filtering to precisely cut a 50km radius circle. |
| 3.1.3 | Apply 3D DBSCAN trajectory cleaning (Jeon 2023) | ✅ | `apply_3d_dbscan()` groups by MMSI and applies 3D DBSCAN clustering on `(lat_km, lon_km, time_scaled_km)` space to separate interleaved tracks and drop noise. |
| 3.1.4 | Filter to vessel types likely to carry bilge water | ✅ | `_BILGE_RELEVANT_TYPES` includes Cargo (70-79), Tanker (80-89), and Fishing (30-39). |
| 3.1.5 | Fishing vessels over 50 m | ✅ | Enforced in `fetch_spill_candidates()` via `_MIN_FISHING_LENGTH_M = 50.0`. Short fishing vessels are dropped. |

### 3.2 AIS Anomaly Detection

| # | Synopsis Requirement | Status | Implementation Detail |
|---|---|---|---|
| 3.2.1 | Extract behavioral features per vessel | ✅ | `extract_trajectory_features()` extracts 6 features per track. |
| 3.2.2 | SOG variance | ✅ | Extracted as `sog_variance`. |
| 3.2.3 | Course deviation index | ✅ | Extracted as `course_deviation_std` (circular difference to handle 0/360 wraparound). |
| 3.2.4 | Stopping events | ✅ | Extracted as `n_stops` (count of pings where SOG < 0.5 kn). |
| 3.2.5 | Unusual slowdown patterns | ✅ | Extracted as `max_sudden_sog_drop` (most negative consecutive SOG delta). |
| 3.2.6 | Proximity to detected spill zone | ✅ | Extracted as `min_proximity_km` (minimum Haversine distance to spill centroid). |
| 3.2.7 | Train Isolation Forest + Random Forest hybrid | ✅ | `AISAnomalyDetector` class implements the Balsaraf 2025 hybrid approach with IsoForest (unsupervised) and optional RF (supervised). |
| 3.2.8 | Flag vessels with anomaly score above threshold as Tier-1 | ✅ | `score_candidates()` flags vessels ≥ 85th percentile anomaly score (`_TIER1_PERCENTILE = 85.0`). |

### 3.3 Dark Ship Handling

| # | Synopsis Requirement | Status | Implementation Detail |
|---|---|---|---|
| 3.3.1 | Satellite-detected vessel layer from Sentinel-1 | ✅ | `detect_ships_ftm()` detects vessels from the linear VV SAR array. |
| 3.3.2 | FTM ship detection approach (Jeon 2023) | ✅ | Implemented via `_FTM_TOP_FRACTION = 0.001` (top 0.1% of VV pixels) + skimage `regionprops`. |
| 3.3.3 | Flag detections with no matching AIS as "dark ship" | ✅ | `correlate_sar_to_ais()` interpolates AIS tracks and flags SAR detections > `distance_tolerance_m` (default 500m) from any AIS ping as dark ships. |

---

## 2. File-by-File Implementation Details

### 2A. `src/data_access/ais_noaa.py` (128 lines)

**Purpose:** Memory-safe bulk reading of NOAA AIS monthly CSV files.

| Function | Signature | Purpose |
|---|---|---|
| `load_ais_window()` | `(csv_path, bbox, start_time, end_time) → DataFrame` | Chunked read (500k rows/chunk) — safe on multi-GB files. Filters bbox + time per chunk. |
| `filter_to_bilge_relevant_types()` | `(df) → df` | Keeps VesselType 70-89 (cargo/tanker) and 30-39 (fishing) only. |
| `to_geodataframe()` | `(df) → GeoDataFrame` | Attaches Point geometry in WGS84. |
| `pair_sar_to_ais()` | `(csv_path, scene_bbox, scene_acquisition_time, window_hours=6.0) → GeoDataFrame` | Convenience wrapper: auto-computes ±6h bracket, calls load_ais_window, optionally filters to bilge types. |

### 2B. `src/ais_attribution/trajectory_cleaning.py` (290 lines)

**Purpose:** Spatial/temporal slicing of AIS data and per-MMSI spoofing removal.

| Function | Signature | Purpose |
|---|---|---|
| `_haversine_bbox()` | `(center_lon, center_lat, radius_km) → bbox` | Exact WGS84 bounding box (cosine corrected) |
| `_haversine_dist_km()` | `(lon1, lat1, lon2, lat2) → dist` | Vectorised spherical distance |
| `fetch_spill_candidates()` | `(lon, lat, time, path, ...) → GeoDataFrame` | Extracts ±50km/±6h slice, filters to IMO types (>50m fishing) |
| `apply_3d_dbscan()` | `(ais_gdf, eps_km, eps_hr, ...) → GeoDataFrame` | 3D DBSCAN per MMSI to decouple mixed trajectories. Re-scales time to km-equivalents so spatial `eps` applies uniformly. Drops noise (-1). |

### 2C. `src/ais_attribution/anomaly_detection.py` (390 lines)

**Purpose:** Behavioral feature extraction and scoring.

| Component | Purpose / Detail |
|---|---|
| `FEATURE_NAMES` | `["sog_mean", "sog_variance", "course_deviation_std", "n_stops", "max_sudden_sog_drop", "min_proximity_km"]` |
| `extract_trajectory_features()` | Computes above features per `track_id` (not MMSI, ensuring cleaned sub-tracks are evaluated independently). |
| `AISAnomalyDetector` | Implements IsoForest (weight 0.6) + RF (weight 0.4) hybrid scorer. Maps IsoForest `decision_function` to monotonic [0, 1] range. |
| `fit()` | Trains IsoForest. Trains RF only if ≥5 positive and ≥5 negative labels are provided. |
| `score_candidates()` | Returns anomaly score and boolean `is_tier1_candidate` flag (≥85th percentile). |

### 2D. `src/ais_attribution/dark_ship.py` (330 lines)

**Purpose:** Detect physical ships in SAR imagery (FTM) and correlate with AIS.

| Function | Signature | Purpose |
|---|---|---|
| `detect_ships_ftm()` | `(vv_array, pixel_spacing_m, ...) → list[dict]` | Detects top 0.1% VV pixels, clusters via 8-connectivity. Computes `estimated_length_m` from area. Geocodes centroid via rasterio Affine. |
| `_interpolate_ais_position()` | `(ais_gdf, mmsi, target_time) → (lon, lat)` | Linear interpolation of an MMSI's position at the exact SAR acquisition time. |
| `correlate_sar_to_ais()` | `(sar_centroids, ais_gdf, tol_m) → list[dict]` | Compares SAR centroids to interpolated AIS. Flags non-matching ships as `is_dark_ship=True`. |

### 2E. `src/ais_attribution/pipeline.py` (400 lines)

**Purpose:** End-to-end Module 3 orchestration.

| Class/Method | Purpose / Detail |
|---|---|
| `Module3Pipeline` | Stateful pipeline object (holds `AISAnomalyDetector`). |
| `run()` | 1. Fetch AIS<br>2. 3D DBSCAN<br>3. Extract features<br>4. Score anomalies<br>5. Trigger Dark Ship fallback if 0 Tier-1 candidates (or forced)<br>Returns dict with candidates and dark-ship flags. |

---

## 3. Critical Engineering Decisions

1. **3D DBSCAN Metric Unification**: The temporal axis in DBSCAN is scaled by `spatial_eps_km / temporal_eps_hr`. This aligns the time units with spatial units so the single `eps` parameter correctly enforces a cylindrical space-time neighbourhood.
2. **Circular COG Wrapping**: Course deviation correctly wraps `[-180°, +180°]` so a ship oscillating between 359° and 001° does not trigger a massive variance spike.
3. **IsoForest Score Normalisation**: `decision_function` is mapped via `1 - clip(score + 0.5, 0, 1)` into a monotonic `[0, 1]` probability-like range, rather than relying on raw arbitrary margins.
4. **Exact Interpolation for Correlation**: `_interpolate_ais_position()` uses precise linear temporal interpolation for SAR-AIS correlation. Simply taking the "nearest" ping can be wildly inaccurate for fast-moving vessels traversing large distances between reports.
5. **No-Transform Fallback**: If `scene_transform` is omitted in dark-ship detection, centroids gracefully fall back to pixel row/col indices (flagged in logs) rather than corrupting WGS84 outputs.
6. **Chunked AIS Reading**: `load_ais_window()` reads the multi-GB NOAA CSV in 500k-row chunks, filtering bbox + time per chunk. Peak RAM stays bounded by chunk size, not file size.

---

## 4. Smoke Test Results

All 4 files passed smoke tests successfully locally:

```
Input : 600 pings,  10 MMSIs
Output: 590 pings, 11 tracks
All trajectory_cleaning smoke tests passed.

Scored 80 tracks
Tier-1 candidates: 12
Score range: [0.0000, 0.2319]
All anomaly_detection smoke tests passed.

Detected 3 vessels
  lat=150.0  lon=200.0  length=50m  area=25px
  lat=500.0  lon=700.0  length=50m  area=25px
  lat=800.0  lon=300.0  length=50m  area=25px
Dark ships: 2
  150.0, 200.0  dark=False (Matched!)
  500.0, 700.0  dark=True
  800.0, 300.0  dark=True
All dark_ship.py smoke tests passed.

tier1_candidates : 0 rows
dark_ship_flags  : 3
pipeline_meta    : {'spill_lon': -90.0, 'spill_lat': 28.5, 'sar_time': '2026-03-15T09:00:00Z',
                    'n_raw_ais_pings': 0, 'n_raw_mmsis': 0, 'n_tier1_candidates': 0,
                    'n_sar_detections': 3, 'n_dark_ships': 3, 'elapsed_s': 0.08}
All pipeline.py smoke tests passed.
```

---

## 5. How to Run Module 3

```bash
python -m src.pipeline.run_full_pipeline \
    --sar-tiff data/test/scene.tif \
    --ais-csv data/ais/AIS_2026_03_Zone15.csv \
    --m1-weights results/module1/checkpoints/best_model.pt \
    --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
    --sar-time 2026-03-15T09:00:00Z \
    --output-dir results/forensic_reports
```

Module 3 runs automatically as part of the full pipeline. To use it standalone:

```python
from src.ais_attribution.pipeline import Module3Pipeline

pipeline = Module3Pipeline()
result = pipeline.run(
    spill_lon=-90.5,
    spill_lat=28.3,
    sar_time="2026-03-15T09:00:00Z",
    ais_csv_path="data/ais/AIS_2026_03_Zone15.csv",
    sar_vv_array=vv_array,           # optional, for dark-ship detection
    scene_transform=affine_transform, # optional, for geocoded centroids
)
print(result["tier1_candidates"])
print(result["dark_ship_flags"])
```
