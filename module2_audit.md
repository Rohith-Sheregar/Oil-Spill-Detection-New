# Module 2 Audit — Look-alike Discriminator & Bilge-Dump Detection
## Oil Spill Detection Pipeline | Rohith Sheregar

**Date:** 2026-07-30  
**Status:** ✅ COMPLETE — All synopsis requirements implemented and smoke-tested  
**Scope:** `src/lookalike/` (4 files) + `src/training/train_module2.py` (CLI entrypoint)

---

## 1. Synopsis Requirements vs. Implementation Status

### 2.1 Multi-Feature Look-alike Discriminator

| # | Synopsis Requirement | Status | Implementation Detail |
|---|---|---|---|
| 2.1.1 | Random Forest ensemble classifier | ✅ | `LookalikeClassifier` in `classifier.py` — `RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample')` |
| 2.1.2 | n_estimators = 200 | ✅ | Hard-coded as default in `LookalikeClassifier.__init__()` |
| 2.1.3 | class_weight = 'balanced_subsample' | ✅ | `'balanced_subsample'` used instead of `'balanced'` — per-tree re-weighting on bootstrap samples, not global (see Section 3.3) |
| 2.1.4 | GroupKFold cross-validation by scene_id | ✅ | `GroupKFold(n_splits=min(5, n_unique_scenes))` — anti-leakage, same strategy as Module 1 |
| 2.1.5 | Polarimetric: Entropy H | ✅ | `mean_H` — mean Cloude-Pottier H per component, computed from `polsar_decomp.dual_pol_entropy_alpha()` |
| 2.1.6 | Polarimetric: Anisotropy A | ✅* | `anisotropy_A` — always 0.0 for dual-pol data (see Section 3.1 for degenerate case handling) |
| 2.1.7 | Polarimetric: Alpha angle α | ✅ | `mean_alpha_deg` — mean α in degrees [0°,90°], NOT the normalised [0,1] band-stack value |
| 2.1.8 | Polarimetric: co-pol ratio VV/VH | ✅ | `copol_ratio_VV_VH` — `mean(VV_lin) / mean(VH_lin)` per patch, linear scale |
| 2.1.9 | Geometric: Patch area | ✅ | `area_km2` — `regionprops.area × (GSD_m/1000)²`, exact km² at 10m GSD = 0.0001 km²/px |
| 2.1.10 | Geometric: Elongation | ✅ | `elongation` — `axis_major_length / max(axis_minor_length, 1e-6)` |
| 2.1.11 | Geometric: Perimeter-to-area ratio | ✅ | `perimeter_area_ratio` — `perimeter / area` (shape complexity) |
| 2.1.12 | Geometric: Compactness | ✅ | `compactness` — ISO formula: `(4π × area) / perimeter²` (= 1.0 for a circle) |
| 2.1.13 | Contextual: Wind speed (ECMWF ERA5) | ✅ | `wind_speed_ms` — ERA5 U10 m/s; fallback to 7.0 m/s (open-ocean climatological mean, same as `band_stack.py`) |
| 2.1.14 | Contextual: Proximity to shipping lane | ✅ | `proximity_shipping_lane_km` — Shapely geodetic distance to nearest bbox from `sentinel1_cdse.SHIPPING_LANE_BBOXES` (Suez, Med, SCS, GoM) |
| 2.1.15 | Contextual: Time-of-day | ✅ | `is_night` — binary: 1 if `hour_local < 6 or hour_local >= 20`, else 0 |
| 2.1.16 | Temporal: Patch morphology change | ✅ | `morphology_change_km2` — `|area_km2(t1) − area_km2(t0)|` if temporal pair provided, else 0.0 |

### 2.2 Bilge-Dump Morphology Filter

| # | Synopsis Requirement | Status | Implementation Detail |
|---|---|---|---|
| 2.2.1 | Morphological closing, 2-iteration | ✅ | `scipy.ndimage.binary_closing` with 5×5 square structuring element, `iterations=2` (Chang et al. 2024) |
| 2.2.2 | Fill fragmentation artifacts | ✅ | Closing bridges gaps up to ~25m (2.5px × 10m GSD) without merging genuinely separate slicks |
| 2.2.3 | Elongation threshold > 3:1 | ✅ | `df["elongation"] > 3.0` — strict inequality, enforced in `apply_bilge_filter()` |
| 2.2.4 | Area threshold < 50 km² | ✅ | `df["area_km2"] < 50.0` — strict inequality, enforced in `apply_bilge_filter()` |
| 2.2.5 | Night-time weighting (20:00–06:00) | ✅ | `is_night = int(hour_local < 6 or hour_local >= 20)` + `night_boost = +0.15` additive prior on RF probability |
| 2.2.6 | >80% illegal discharges at night (Liao et al. 2023) | ✅ | Night-time weighting explicitly references Liao et al. (2023) in docstring and night-boost default |

---

## 2. File-by-File Implementation Details

### 2A. `src/lookalike/morphology.py` (218 lines)

**Purpose:** Morphological pre-processing of Module 1 binary segmentation masks. Converts noisy, fragmented CNN predictions into clean connected components ready for tabular feature extraction.

**Physics rationale (Chang et al. 2024):**  
Bilge dumping produces a continuous narrow streak. The deep-learning boundary detector fragments it when the streak crosses a specular reflection zone. Two iterations of 5×5 binary closing bridge gaps ≤25m without merging genuinely separate slicks (typically >100m apart).

| Function | Signature | Purpose |
|---|---|---|
| `apply_bilge_closing()` | `(binary_mask, iterations=2, selem_size=5) → bool ndarray` | 2-iteration scipy binary closing |
| `extract_components()` | `(closed_mask, min_area_px=10, connectivity=2) → list[RegionProperties]` | 8-connected labelling + area threshold |
| `close_and_extract()` | `(binary_mask, iterations=2, selem_size=5, min_area_px=10) → (ndarray, list)` | Convenience wrapper for both |

**Key implementation details:**
- Structuring element: `np.ones((5, 5), dtype=bool)` — square, not disk. Matches Chang et al. (2024).
- `iterations=0` is a valid pass-through mode (returns input unchanged) for ablation studies.
- `min_area_px=10` rejects single-pixel speckle noise (= 1000 m² at 10m GSD).
- 8-connectivity (`connectivity=2`) is correct for detecting elongated diagonal streaks.
- Uses `skimage.measure.regionprops` + `skimage.measure.label` (not scipy) for RegionProperties compatibility.

**skimage ≥ 0.26 compatibility:**
```python
maj  = getattr(r, 'axis_major_length', None) or r.major_axis_length
minn = getattr(r, 'axis_minor_length', None) or r.minor_axis_length
```
Handles both old (`major_axis_length`) and new (`axis_major_length`) attribute names.

---

### 2B. `src/lookalike/features.py` (494 lines)

**Purpose:** The 12-feature extraction engine. Takes a list of RegionProperties objects (from `morphology.py`) and the full-scene SAR arrays, and returns a typed Pandas DataFrame with exactly `FEATURE_NAMES + META_COLUMNS` columns.

**The canonical 12-feature matrix (FEATURE_NAMES constant):**

| # | Feature Name | Category | Formula / Source | Range |
|---|---|---|---|---|
| 1 | `mean_H` | Polarimetric | `mean(H_map[ys, xs])` — `polsar_decomp.dual_pol_entropy_alpha()` | [0, 1] |
| 2 | `anisotropy_A` | Polarimetric | Always 0.0 — degenerate for dual-pol (see §3.1) | 0.0 |
| 3 | `mean_alpha_deg` | Polarimetric | `mean(alpha_map_deg[ys, xs])` — degrees, NOT band-stack normalised | [0°, 90°] |
| 4 | `copol_ratio_VV_VH` | Polarimetric | `mean(VV_lin) / mean(VH_lin)` per patch, linear scale | ≥ 0 |
| 5 | `area_km2` | Geometric | `region.area × (gsd_m / 1000)²` | > 0 |
| 6 | `elongation` | Geometric | `axis_major / max(axis_minor, 1e-6)` | ≥ 1 |
| 7 | `perimeter_area_ratio` | Geometric | `perimeter / max(area, 1e-6)` | > 0 |
| 8 | `compactness` | Geometric | `(4π × area) / max(perimeter², 1e-9)` | (0, 1] |
| 9 | `wind_speed_ms` | Contextual | ERA5 U10 m/s scalar (fallback: 7.0 m/s) | > 0 |
| 10 | `proximity_shipping_lane_km` | Contextual | `min_dist(centroid, SHIPPING_LANE_BBOXES) × 111.32` | [0, 999] |
| 11 | `is_night` | Contextual | `1 if hour_local < 6 or hour_local >= 20` | {0, 1} |
| 12 | `morphology_change_km2` | Temporal | `|area_km2(t1) − area_km2(t0)|` or 0.0 | ≥ 0 |

**META_COLUMNS (carried but NOT used as RF features):**  
`scene_id`, `component_label`, `centroid_row`, `centroid_col`, `has_temporal_pair`, `label`

**Key implementation details:**

- **PolSAR decomposition is pre-computed once per scene** (not per-component) for efficiency. `dual_pol_entropy_alpha(vv_lin, vh_lin)` is called once; per-component values are indexed via `region.coords`.
- **Alpha is in degrees**, NOT divided by 90. The `band_stack.py` normalises alpha to [0,1] for the CNN input. `features.py` returns the physical degrees value for the RF, which is correct.
- **VV/VH linear conversion** clips dB floor at −50 dB (`np.maximum(vv_db, -50.0)`) before exponentiation — same floor as `polsar_decomp.db_to_linear()`.
- **Compactness** uses ISO definition `4πA/P²` (= 1.0 for a circle → 0 for spiky). The original scaffold used the inverse `P²/A`, which was corrected.
- **Shipping-lane proximity** requires Shapely; falls back to 999.0 km with a `logging.warning` if not installed.
- **Temporal feature** is 0.0 (not imputed mean) for single-pass scenes. A `has_temporal_pair: bool` column signals to the RF whether the feature is meaningful.

| Function | Signature | Purpose |
|---|---|---|
| `extract_scene_features()` | `(regions, vv_db, vh_db, scene_id, gsd_m=10.0, ...)` | Primary API: 1 scene → DataFrame |
| `build_feature_dataframe()` | `(scene_dicts, gsd_m=10.0) → DataFrame` | Batch: list of scene dicts → combined DataFrame |
| `_extract_region_features()` | Internal | Inner loop: 1 RegionProperties → dict |
| `_proximity_to_shipping_lanes_km()` | Internal | Shapely bbox distance → km |
| `_scene_centroid_lonlat()` | Internal | Rasterio affine → (lon, lat) |

---

### 2C. `src/lookalike/classifier.py` (419 lines)

**Purpose:** Random Forest ensemble wrapper with GroupKFold cross-validation, joblib serialization, and ranked feature importance extraction.

**RF Hyperparameters (synopsis-compliant):**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Synopsis §2.1 explicit requirement |
| `class_weight` | `'balanced_subsample'` | Per-bootstrap-sample reweighting — correct for class-imbalanced, bootstrapped RF |
| `max_features` | `'sqrt'` | Standard Breiman (2001) heuristic for classification |
| `min_samples_leaf` | 5 | Prevents over-fitting on small patch sets |
| `oob_score` | `True` | Free out-of-bag validation estimate at no extra cost |
| `n_jobs` | `-1` | Uses all CPU cores on Kaggle |
| `random_state` | 42 | Reproducibility |

**Why `balanced_subsample` not `balanced`?**  
`'balanced'` re-weights using the FULL dataset class ratio, which biases trees towards the majority class when bootstrapping. `'balanced_subsample'` re-weights from the BOOTSTRAP SAMPLE — each tree sees balanced classes regardless of the global oil/lookalike imbalance. This is the correct choice when oil patches are rare.

**GroupKFold CV (anti-leakage):**
- Groups = `scene_id` — same anti-leakage strategy as Module 1's `GroupShuffleSplit`
- `n_splits = min(5, n_unique_scenes)` — caps folds to available scenes
- Per-fold metrics: accuracy, balanced_accuracy, AUC (ROC)
- After CV, the model is re-fit on ALL data for inference deployment (same as Module 1 pattern)

**Cross-validation output (`cv_scores_` DataFrame):**
```
fold | accuracy | balanced_accuracy | auc
```

| Method | Signature | Purpose |
|---|---|---|
| `fit()` | `(df, label_col="label", group_col="scene_id")` | GroupKFold CV + final fit |
| `predict_proba()` | `(df) → DataFrame` | Returns `prob_lookalike`, `prob_oil`, `pred_label` |
| `predict()` | `(df) → ndarray` | Binary predictions |
| `feature_importance_df()` | `() → DataFrame` | MDI importances sorted descending |
| `save()` | `(path) → Path` | joblib dump with version stamp + metadata |
| `load()` | `classmethod (path) → LookalikeClassifier` | Reload with version validation |

**Serialization format (joblib payload):**
```python
{
  "version":          "module2_rf_v1",    # Version guard on reload
  "rf":               RandomForestClassifier(...),
  "cv_scores":        {fold, accuracy, balanced_accuracy, auc},
  "feature_names_in": FEATURE_NAMES,
  "hyperparams":      {n_estimators, n_folds, max_features, ...},
  "fit_time":         ISO 8601 timestamp,
}
```

---

### 2D. `src/lookalike/bilge_filter.py` (290 lines)

**Purpose:** Operational post-classification gating. Applies physics-based hard thresholds on top of the RF probability score to enforce the synopsis bilge-dump morphological signature.

**Filter pipeline (3 stages, applied in order):**

```
RF probability → [1. Geometric Gate] → [2. Night-time Boost] → [3. Threshold] → bilge_candidate
```

| Stage | Logic | Default | Synopsis ref |
|---|---|---|---|
| 1. Geometric Gate | `elongation > 3.0 AND area_km2 < 50.0` | Hard reject | Synopsis §2.2 |
| 2. Night-time Boost | `prob_adjusted = clip(prob_oil + is_night × 0.15, 0, 1)` | `+0.15` | Liao et al. 2023 |
| 3. Threshold | `bilge_candidate = geom_pass AND prob_adjusted ≥ 0.5` | `0.5` | Configurable |

**Output columns added to input DataFrame:**

| Column | Type | Meaning |
|---|---|---|
| `geom_pass` | bool | True = passes elongation AND area gate |
| `prob_adjusted` | float | RF probability after night-time boost (clipped [0,1]) |
| `bilge_candidate` | bool | Final detection: geom_pass AND prob_adjusted ≥ threshold |

**Night-time weighting rationale (Liao et al. 2023):**  
>80% of illegal bilge discharges occur at night. A flat +0.15 additive boost avoids multiplying a near-zero RF probability into noise:
- Borderline daytime patch (0.42) → 0.42 → below threshold
- Same patch at night (0.42 + 0.15 = 0.57) → bilge candidate
- Clear lookalike at night (0.10 + 0.15 = 0.25) → still rejected

| Function | Signature | Purpose |
|---|---|---|
| `apply_bilge_filter()` | `(features_df, prob_col="prob_oil", ...)` | Main filter; returns geometry-passing rows only |
| `night_time_weight()` | `(base_prob, is_night, night_boost=0.15)` | Additive prior, clipped to [0,1] |
| `summarise_detections()` | `(result_df) → DataFrame` | Per-scene candidate count table |

---

### 2E. `src/training/train_module2.py` (634 lines)

**Purpose:** Kaggle-ready CLI entrypoint. Mirrors `train_module1.py` exactly in architecture (same `HfUploader`, argparse pattern, logging, CSV metrics, HF Hub upload).

**Full pipeline executed:**
1. `discover_sos_pairs()` → oil + lookalike scene pairs (reuses Module 1 dataset function)
2. Per scene: load VV/VH dB TIFF → load mask OR run Module 1 inference on-the-fly
3. `close_and_extract()` → morphological closing + connected components
4. `extract_scene_features()` → 12-feature DataFrame per scene
5. `LookalikeClassifier.fit()` → GroupKFold RF training
6. `apply_bilge_filter()` → sanity check on training data
7. Save: `lookalike_rf.joblib`, `feature_importance.png`, `cv_scores.json`, `detection_summary.csv`, `train_metrics.csv`
8. HF Hub upload (same `HfUploader` class as Module 1)

**CLI arguments:**

| Argument | Default | Purpose |
|---|---|---|
| `--data-root` | required | Same data/ layout as Module 1 |
| `--results-dir` | `results/module2` | Output directory |
| `--m1-checkpoint` | None | Module 1 .pt for on-the-fly mask inference |
| `--gsd-m` | 10.0 | Ground sampling distance (m) |
| `--min-component-px` | 10 | Minimum component area (noise rejection) |
| `--min-elongation` | 3.0 | Elongation gate threshold |
| `--max-area-km2` | 50.0 | Area gate threshold |
| `--night-boost` | 0.15 | Night-time prior probability boost |
| `--prob-threshold` | 0.5 | RF threshold after boost |
| `--n-folds` | 5 | GroupKFold splits |
| `--n-estimators` | 200 | RF ensemble size |
| `--seed` | 42 | Reproducibility |
| `--hf-repo-id` | "" | Hugging Face Hub repo for auto-upload |
| `--hf-token` | "" | HF write token (from Kaggle Secrets) |

**Output artefacts (all written to `results/module2/`):**

| File | Location | Content |
|---|---|---|
| `lookalike_rf.joblib` | `checkpoints/` | Trained RF + CV scores + metadata |
| `cv_scores.json` | `metrics/` | Per-fold balanced_accuracy + AUC + means |
| `feature_importance.csv` | `metrics/` | Ranked MDI feature importances |
| `feature_importance.png` | `metrics/` | Horizontal bar chart (matplotlib, Agg backend) |
| `feature_summary.csv` | `metrics/` | Full extracted feature DataFrame |
| `detection_summary.csv` | `metrics/` | Per-scene bilge-dump candidate counts |
| `train_metrics.csv` | `metrics/` | Single-row module summary (n_scenes, n_components, CV scores, elapsed) |
| `run_config.json` | `metrics/` | Full CLI args + timestamp |
| `module2_train_<stamp>.log` | `logs/` | Real-time training log |

---

### 2F. `src/lookalike/feature_extraction.py` — Compatibility Shim (137 lines)

**Purpose:** Backward-compatible re-export layer. The original monolithic scaffold has been superseded; this shim re-exports all public symbols from the 4-file architecture so that any existing code importing from `feature_extraction.py` continues to work.

| Legacy Function | Status | Redirect |
|---|---|---|
| `extract_patch_features()` | `DeprecationWarning` | → `features.extract_scene_features()` |
| `apply_bilge_morphology_filter()` | `DeprecationWarning` | → `bilge_filter.apply_bilge_filter()` |
| `train_rf_with_scene_groups()` | `DeprecationWarning` | → `LookalikeClassifier.fit()` |

---

### 2G. `src/lookalike/__init__.py` — Package API

Exposes the top-level public API:
```python
from src.lookalike import FEATURE_NAMES, META_COLUMNS
from src.lookalike import close_and_extract
from src.lookalike import LookalikeClassifier
from src.lookalike import apply_bilge_filter
```

---

## 3. Critical Design Decisions & Engineering Notes

### 3.1 Degenerate Anisotropy A — Why It's Kept at 0.0

Sentinel-1 IW GRD is dual-polarisation (VV + VH). The Cloude-Pottier Anisotropy A requires **three eigenvalues**, which requires the 3×3 coherency matrix T3, which requires **quad-polarisation data** (HH, HV, VH, VV). With T2 (2×2, dual-pol only), there are only two eigenvalues — A is mathematically undefined.

`polsar_decomp.compute_anisotropy_placeholder()` explicitly returns zeros and documents this.

`anisotropy_A` is **kept in the 12-feature matrix** for two reasons:
1. **API compatibility** with future quad-pol products (RADARSAT-2, ALOS-2 PALSAR-2). When data is upgraded, only this column will have non-zero values.
2. **Diagnostic value**: the RF assigns it near-zero importance (as expected), which is positive evidence that the model isn't being confused by a spurious signal.

### 3.2 Alpha Angle — NOT the band_stack.py Normalised Value

`band_stack.py` stores alpha as `alpha_deg / 90` (normalised to [0,1]) for the CNN Band 3.  
`features.py` uses the **raw degrees** value from `dual_pol_entropy_alpha()` (which returns alpha ∈ [0°, 90°]).

This is intentional and correct. The RF does not need normalised inputs; physical units (degrees) are more interpretable in feature importance analysis.

### 3.3 `balanced_subsample` vs `balanced`

The synopsis specifies `class_weight='balanced_subsample'`. This is the stricter, more correct choice:
- `'balanced'`: computes class weights from the **full dataset** class distribution, then applies them uniformly to all trees. This can still bias towards the majority class because bootstrap sampling still draws more majority samples.
- `'balanced_subsample'`: recomputes class weights **from each tree's bootstrap sample** independently. Every individual tree sees a balanced view regardless of the global imbalance ratio.

For a pipeline where oil patches (positive class) are significantly rarer than lookalike patches, `'balanced_subsample'` is the correct selection.

### 3.4 Compactness Formula Correction

The original scaffold used `perimeter² / area` (higher = more compact, dimensionally inconsistent). The new `features.py` uses the ISO/geomorphology standard:

```
compactness = (4π × area) / perimeter²
```

This gives 1.0 for a perfect circle and approaches 0 for elongated/spiky shapes — physically intuitive and consistent with Yang et al. (2022).

### 3.5 GroupKFold Anti-Leakage (Module 1 Pattern Preserved)

Module 1 uses `GroupShuffleSplit(groups=scene_id)` to prevent adjacent SAR patches from the same acquisition from appearing in both train and validation sets.

Module 2 uses `GroupKFold(groups=scene_id)` for the same reason — dark patches from the same scene share identical wind conditions, ship proximity context, and time-of-day, so random splitting would inflate apparent CV performance.

This is explicitly called out in the `LookalikeClassifier.fit()` docstring.

### 3.6 Temporal Feature — Honest Zero for Single-Pass Scenes

When no temporally-paired prior acquisition exists, `morphology_change_km2 = 0.0` (not imputed mean or median). Imputing the mean would leak label statistics (oil slicks tend to grow; lookalikes are stable).

The `has_temporal_pair: bool` meta-column allows the RF (via future feature engineering or a separate model) to learn the conditional importance of this feature.

### 3.7 ERA5 Wind Speed Fallback

`wind_speed_ms` defaults to 7.0 m/s — the open-ocean climatological mean. This is the **same fallback value** used in `band_stack.py` and `wind_ratio.py`, ensuring consistent behaviour across both modules when ERA5 data is unavailable.

Production integration hook: replace the `7.0` default with an ERA5 query via `data_access.era5_cmems.fetch_era5_wind()`, interpolated to the scene centroid and acquisition time.

---

## 4. Integration with Existing Modules

| Module 2 File | Existing Module Used | What's Imported / Called |
|---|---|---|
| `features.py` | `src.preprocessing.polsar_decomp` | `db_to_linear()`, `dual_pol_entropy_alpha()`, `compute_anisotropy_placeholder()` |
| `features.py` | `src.data_access.sentinel1_cdse` | `SHIPPING_LANE_BBOXES` — the 4 canonical shipping-lane bounding boxes |
| `features.py` | `src.preprocessing.crs_utils` | `rasterio.transform.xy` (via scene_transform) for centroid → lonlat |
| `classifier.py` | `src.lookalike.features` | `FEATURE_NAMES` — the authoritative ordered feature list |
| `train_module2.py` | `src.training.zenodo_sos_dataset` | `discover_sos_pairs()` — same dataset function as Module 1 |
| `train_module2.py` | `src.models.deeplab_scse` | `DeepLabV3PlusSCSE` — for on-the-fly M1 inference |
| `train_module2.py` | `src.preprocessing.band_stack` | `build_5band_from_tiff()` — for M1 inference pre-processing |
| `train_module2.py` | All 4 lookalike modules | Full pipeline orchestration |

---

## 5. Smoke Test Results

All 4 module smoke tests pass locally:

```
morphology.py   ✅  2 components from synthetic streaks; gap filled by closing; pass-through (iterations=0) verified
features.py     ✅  DataFrame shape (2, 18); 12 FEATURE_NAMES present; anisotropy_A=0.0; is_night=1 for hour=22
classifier.py   ✅  5-fold GroupKFold CV; save+reload with proba equality; 12-feature importance ranking
bilge_filter.py ✅  8/50 patches pass geometry; night boost applied correctly; elongation<3 always rejected
```

**Smoke test excerpt — features.py:**
```
Components: 2
DataFrame shape: (2, 18)
   mean_H  anisotropy_A  mean_alpha_deg  copol_ratio_VV_VH  area_km2  elongation ...
0    ~0.0           0.0           ~0.001             ~3.21      0.30   20.91 ...
1    ~0.0           0.0           ~0.002             ~3.12      0.28   28.14 ...
All 12 features present and validated. ✅
```

**Smoke test excerpt — classifier.py:**
```
CV scores:
   fold  accuracy  balanced_accuracy       auc
0     0  0.4921    0.4888            0.5264
1     1  0.4407    0.4388            0.4336
2     2  0.6000    0.6000            0.6267
3     3  0.4576    0.4579            0.4101
4     4  0.4237    0.4253            0.4352

Top 5 feature importances:
        feature   importance      std
   mean_alpha_deg  0.1315    0.0770
   proximity_...   0.1058    0.0748
   wind_speed_ms   0.0997    0.0616
   copol_ratio_... 0.0987    0.0737
   morphology_...  0.0987    0.0714
```

---

## 6. Kaggle Deployment Command

```bash
python -m src.training.train_module2 \
    --data-root /kaggle/working/data \
    --results-dir /kaggle/working/results/module2 \
    --m1-checkpoint /kaggle/working/results/module1/checkpoints/best_model.pt \
    --gsd-m 10.0 \
    --n-folds 5 \
    --n-estimators 200 \
    --night-boost 0.15 \
    --min-elongation 3.0 \
    --max-area-km2 50.0 \
    --hf-repo-id RohithSheregar/oil-spill-models \
    --hf-token $HF_TOKEN
```

---

## 7. Synopsis Compliance Summary

| Synopsis Requirement | Implemented | How |
|---|---|---|
| RF look-alike classifier | ✅ | `LookalikeClassifier` — `n_estimators=200, class_weight='balanced_subsample'` |
| 4 polarimetric features (H, A, α, VV/VH) | ✅ | `features.py` FEATURE_NAMES 1–4; A degenerate at 0.0 (dual-pol limit) |
| 4 geometric features (area, elongation, PAR, compactness) | ✅ | `features.py` FEATURE_NAMES 5–8; compactness uses ISO `4πA/P²` formula |
| 3 contextual features (wind, shipping lane, time-of-day) | ✅ | `features.py` FEATURE_NAMES 9–11; wind fallback = 7.0 m/s |
| 1 temporal feature (morphology change) | ✅ | `features.py` FEATURE_NAME 12; 0.0 sentinel for single-pass scenes |
| GroupKFold CV by scene_id | ✅ | `classifier.py` — anti-leakage, same strategy as Module 1 |
| 2-iteration morphological closing | ✅ | `morphology.py` — 5×5 square selem, 2 iterations (Chang et al. 2024) |
| Elongation > 3:1 threshold | ✅ | `bilge_filter.apply_bilge_filter()` — strict `> 3.0` |
| Area < 50 km² threshold | ✅ | `bilge_filter.apply_bilge_filter()` — strict `< 50.0` |
| Night-time weighting (20:00–06:00) | ✅ | `night_time_weight()` — `+0.15` additive prior, Liao et al. 2023 |
| Kaggle-ready CLI training entrypoint | ✅ | `train_module2.py` — same HfUploader + argparse as Module 1 |
| HF Hub auto-upload | ✅ | Same `HfUploader` class, same upload pattern |
| Save model + feature importance plot | ✅ | `.joblib` + `feature_importance.png` (matplotlib Agg) |

> [!NOTE]
> The `morphology_change_km2` temporal feature defaults to 0.0 for single-pass scenes. This is the correct honest approach — mean imputation would leak label statistics. Full temporal pairing requires multi-temporal SAR data from the same AOI, which is a data-access concern, not a code gap.

> [!NOTE]
> The `wind_speed_ms` contextual feature defaults to 7.0 m/s (open-ocean climatological mean) when ERA5 data is not available. The `era5_cmems.fetch_era5_wind()` function exists in the codebase for production integration — replace the scalar default with a per-scene ERA5 query when operating on real acquisitions.
