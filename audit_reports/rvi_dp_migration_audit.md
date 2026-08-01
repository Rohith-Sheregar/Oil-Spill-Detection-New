> NOTE: module1_audit.md and module2_audit.md predate the RVI_dp
> migration and still describe "Alpha angle" / "mean_alpha_deg" as
> current behavior. They are stale as of 2026-08-01. The synopsis's
> Cloude-Pottier alpha angle is uncomputable from this project's data
> source: Sentinel-1 GRD products are phase-discarded amplitude data
> (phase is destroyed at SNAP's detection step regardless of dual-pol
> vs quad-pol), and alpha requires complex phase to derive the
> coherency-matrix eigenvectors. RVI_dp (dual-pol Radar Vegetation
> Index, Mullissa et al. 2021) is substituted as a phase-free,
> physically-grounded proxy for surface roughness / depolarization
> contrast. These docs should be regenerated after the next successful
> training run, and this justification should be carried into the
> final report/thesis methodology section.

# Dual-Pol RVI_dp Migration Audit

## 1. Stale alpha/alpha_deg references

### src/lookalike/feature_extraction.py
- **Line 53:** `def extract_patch_features(mask, vv, vh, H, alpha, wind_speed, hour_local, scene_id)`
  - *Snippet:* Legacy wrapper parameter named `alpha`.
- **Verdict:** BROKEN (Legacy wrapper expects the old naming convention).

### src/lookalike/features.py
- **Line 175:** `alpha_map_deg: np.ndarray,      # (H, W) float32, Alpha angle in degrees [0°, 90°]`
  - *Snippet:* Argument incorrectly named and documented to be in degrees `[0°, 90°]`.
- **Line 223:** `mean_rvi_dp       = float(alpha_map_deg[ys, xs].mean())`
  - *Snippet:* Variable correctly feeds into `mean_rvi_dp` but is confusingly named `alpha_map_deg`.
- **Line 335:** `H_map, alpha_map_deg = dual_pol_entropy_alpha(vv_lin, vh_lin)`
  - *Snippet:* Still calls the old `dual_pol_entropy_alpha` function alias instead of `dual_pol_entropy_rvi`, expecting degrees.
- **Line 336:** `# alpha from dual_pol_entropy_alpha is already in [0°, 90°] (NOT normalised)`
  - *Snippet:* Docstring still describes the old alpha 0-90° bounds.
- **Line 371:** `alpha_map_deg       = alpha_map_deg,`
  - *Snippet:* Passing the misnamed variable down to `_extract_region_features`.
- **Verdict:** BROKEN (Live code and docstrings still assume the 2nd polarimetric return value is an alpha angle in degrees).

## 2. Band Order Agreement

- **src/preprocessing/band_stack.py:** `[band_vv, band_vh, band_H, band_rvi, band_wind]` (Order OK)
- **src/training/zenodo_sos_dataset.py:** `[vv_norm, vh_norm, band_H, band_rvi, band_w]` (Order OK)
- **src/models/deeplab_scse.py:**
  - *Snippet (Line 44):* `Band 3: Cloude-Pottier Alpha angle / 90 [0, 1]`
- **src/lookalike/features.py:** Uses polarimetric feature extraction manually (see Section 1).
- **Verdict:** BROKEN (Docstring in `deeplab_scse.py` disagrees; `features.py` has stale implementation logic).

## 3. Anisotropy/alpha as a real quantity

### src/lookalike/features.py
- **Line 10:** `Polarimetric    mean_H, anisotropy_A, mean_alpha_deg, copol_ratio_VV_VH`
  - *Snippet:* Still describes `mean_alpha_deg` instead of `mean_rvi_dp`.
- **Verdict:** BROKEN (Stale naming in feature layout description).

## 4. Pipeline E2E (run_full_pipeline.py)

- **Investigation:** `run_full_pipeline.py` orchestrates the pipeline, calling `extract_scene_features()` at Line 789. It extracts the DataFrame but does not explicitly reference or parse any alpha/rvi polarimetric columns downstream for Module 4 (Module 4 relies on morphological `slick_major_axis_angle_deg`).
- **Verdict:** OK.

## 5. Report Text (module1_report.py)

- **src/reporting/module1_report.py (Line 399):**
  - *Snippet:* `<li><b>Band 3 (Alpha angle):</b> Mean scattering mechanism angle. α&lt;45° = surface scattering (Bragg resonance); α&gt;45° = volume/double-bounce (oil layer multipaths).</li>`
- **Verdict:** NEEDS REVIEW (Hardcoded HTML report text needs to be updated to describe RVI_dp).

## 6. RVI_dp Normalization Consistency

- **src/preprocessing/band_stack.py (Line 85):**
  - *Snippet:* `band_rvi = np.clip(band_rvi_raw / 2.0, 0.0, 1.0).astype(np.float32)`
- **src/training/zenodo_sos_dataset.py (Line 292):**
  - *Snippet:* `band_rvi = np.clip(band_rvi_raw / 2.0, 0.0, 1.0).astype(np.float32)`
- **Verdict:** OK (Constants and logic match perfectly).

## 7. Hardcoded `in_channels=4`

- **Investigation:** Grep across `src/` and `notebooks/` yielded zero live-code usages of `in_channels=4` hardcoding. (The only match was in the documentation block of `deeplab_scse.py` explaining fallback modes).
- **Verdict:** OK.

## 8. Clean Fresh Start (module-1-training.ipynb)

- **notebooks/module-1-training.ipynb (Line 109):**
  - *Snippet:* `FRESH_START = False`
- **Verdict:** WARNING — resume path is reachable, will crash / load stale checkpoint (Since it's False, it will query Hugging Face Hub, find the `.pt` file, and download it to resume).

## 9. `train_metrics.csv` Leak

- **src/training/train_module1.py (Line 389):**
  - *Snippet:* `csv_mode = "a" if (args.resume and csv_path.exists()) else "w"`
- **Investigation:** Because `FRESH_START = False` will trigger `args.resume`, and `results/module1/metrics/train_metrics.csv` currently exists, the script will open the CSV in append (`"a"`) mode.
- **Verdict:** BROKEN — will append old-band metrics to new run instead of overwriting cleanly.

## 10. Audit Docs Drift

- **module2_audit.md (Line 22):** `| 2.1.7 | Polarimetric: Alpha angle α | ✅ | mean_alpha_deg — mean α in degrees [0°,90°], NOT the normalised [0,1] band-stack value |`
- **module2_audit.md (Line 87):** `| 3 | mean_alpha_deg | Polarimetric | mean(alpha_map_deg[ys, xs]) — degrees, NOT band-stack normalised | [0°, 90°] |`
- **module2_audit.md (Line 296):** `### 3.2 Alpha Angle — NOT the band_stack.py Normalised Value`
- **module2_audit.md (Line 373):** `mean_H  anisotropy_A  mean_alpha_deg  copol_ratio_VV_VH  area_km2  elongation ...`
- **module2_audit.md (Line 391):** `mean_alpha_deg  0.1315    0.0770`
- **module1_audit.md (Line 112):** `| **Band 3** | Alpha angle α / 90° (normalised to [0,1]) | dB → **linear** → α | ✅ |`
- **Verdict:** NEEDS REVIEW (Docs heavily reference `mean_alpha_deg`).

---

## SUMMARY — BLOCKING ISSUES

The following items must be fixed before starting the training run:
1. **Item 1:** Fix the stale alpha extraction logic and function calls in `src/lookalike/features.py`.
2. **Item 8:** Change `FRESH_START = False` to `FRESH_START = True` in `notebooks/module-1-training.ipynb` Cell 3 to prevent resuming from the HF Hub checkpoint.
3. **Item 9:** Prevent `train_metrics.csv` from leaking by ensuring it is overwritten (will be implicitly fixed when Item 8 sets `args.resume = False`, but good to be aware).


## POST-FIX VERIFICATION
grep -rn "alpha" src/lookalike/features.py src/models/deeplab_scse.py src/reporting/module1_report.py
src/reporting/module1_report.py:52:    "grid.alpha":       0.3,
src/reporting/module1_report.py:228:    ax1.bar(cycles, n_pseudo, color=BRAND_BLUE, alpha=0.8)
src/reporting/module1_report.py:399:            <li><b>Band 3 (RVI_dp):</b> Dual-pol Radar Vegetation Index, 4·VH/(VV+VH). Substitutes the synopsis's Cloude-Pottier alpha angle, which requires complex SLC phase unavailable in Sentinel-1 GRD amplitude products. Low values indicate depolarisation-suppressed, smooth (potentially oil-dampened) surfaces; higher values indicate rougher, more depolarising ocean backscatter.</li>
src/models/deeplab_scse.py:45:                via clip(RVI_dp/2.0, 0, 1). Substitutes Cloude-Pottier alpha, which is
src/models/deeplab_scse.py:46:                uncomputable from phase-discarded Sentinel-1 GRD amplitude data (alpha
src/models/deeplab_scse.py:50:        Pass in_channels=4 for the vv_vh_h_alpha mode or in_channels=2/3
src/lookalike/features.py:17:The synopsis specifies Cloude-Pottier alpha (α) angle. However, α requires
src/lookalike/features.py:21:Thus, alpha is uncomputable from this dataset. In its place, we use the
src/lookalike/features.py:186:    rvi_dp_map: np.ndarray,         # (H, W) float32, dual-pol Radar Vegetation Index (RVI_dp), unnormalised >= 0. Substitutes Cloude-Pottier alpha, which is uncomputable from phase-discarded Sentinel-1 GRD amplitude data (see module docstring).
src/lookalike/features.py:349:    # RVI_dp replaces alpha here because alpha requires complex SLC phase

Smoke test:
python -m src.lookalike.features
<frozen runpy>:128: RuntimeWarning: 'src.lookalike.features' found in sys.modules after import of package 'src.lookalike', but prior to execution of 'src.lookalike.features'; this may result in unpredictable behaviour
Components: 2
DataFrame shape: (2, 18)
     mean_H  anisotropy_A  mean_rvi_dp  copol_ratio_VV_VH  area_km2  elongation  perimeter_area_ratio  compactness  wind_speed_ms  proximity_shipping_lane_km  is_night  morphology_change_km2
0  0.611606           0.0     1.306371           3.210666      0.30   20.905883              0.173333     0.139420            8.5                       999.0         1                    0.0
1  0.618069           0.0     1.330766           3.115586      0.28   28.140879              0.205714     0.106053            8.5                       999.0         1                    0.0

All 12 features present and validated.
All features.py smoke tests passed.
VERDICT: PASS
