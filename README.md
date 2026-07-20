# Bilge Dump Detection — Implementation Plan (10–12 week window)

This is the reference doc for the project. Code scaffolding lives in `src/`.

## 1. The three data blockers — resolve in week 1, before model code

| Dependency | Status | Action this week |
|---|---|---|
| **Sentinel-1 via CDSE** | ✅ Live, low risk | `sentinelsat`/SciHub is dead — archived, won't work against CDSE. Use the OData + Keycloak pattern in `src/data_access/sentinel1_cdse.py`. Register a free account at dataspace.copernicus.eu today. |
| **AIS (Gulf of Mexico)** | ✅ Free, bulk, low risk | NOAA Marine Cadastre publishes free historical AIS as CSV (no login). `src/data_access/ais_noaa.py` parses/filters it. Locking the AOI to the Gulf of Mexico is what makes AIS free — it also matches the SOS dataset's geography and gives you a real confirmed incident (see below) for free. |
| **EMSA CleanSeaNet "confirmed incidents"** | ⚠️ Unresolved — do this first | The live interactive CleanSeaNet portal is restricted to coastal-state authorities; external academic requests are handled case-by-case and are not a 2-week turnaround. **But** EMSA publishes a public, no-login "Detections and Feedback Data" zip archive annually back to 2015: `https://www.emsa.europa.eu/csn-menu/detections-feedback-data.html`. Download the most recent one **today** and check what's inside: |

**Two outcomes, two different validation plans:**
- **Per-incident lat/lon + confirmation status present** → use it directly as your Tier-3 ("real confirmed incident") validation set. Best case, costs you 20 minutes to find out.
- **PDF/aggregate-only, no per-incident geometry** → fall back to the named, already-published, geometry-bearing case studies your own lit review cites: Zakzouk et al. 2025 (30 Suez Canal test cases), the OSDTAU-Net paper's USCG-confirmed Nov 2023 Gulf of Mexico pipeline spill, Chang et al. 2024's Taiwan cases (2023 Kaohsiung shipwreck, 2021 Xiaoliuqiu rupture). This is a legitimate substitution — two of your cited papers used exactly this approach — not a downgrade in rigor.

**Why this can't wait:** if you discover in week 7 that the archive is PDF-only and you haven't lined up the fallback case studies, you lose the validation deliverable, not just time. If you discover it in week 1, you have the full 10–12 weeks to chase down geometry for the fallback cases.

## 2. MATLAB data — decision tree

No `.mat` file has actually been shared in this conversation, so the triage below is generic. Upload one (or just describe what's in it — array shapes, whether it includes a georeferencing/mapping object, whether it's data or scripts) and I'll give exact loading code instead of a decision tree.

| What's likely in the file | How to check | Path forward |
|---|---|---|
| Plain numeric arrays (backscatter patches, polarimetric decomposition outputs, feature matrices) | `scipy.io.loadmat('file.mat')` works directly | Leverage as-is — load once, re-save as `.npy`/GeoTIFF so it's not a recurring dependency |
| MATLAB v7.3 file (HDF5-based — common for large arrays) | `scipy.io.loadmat` raises an error mentioning HDF5 | Use `h5py` or `mat73`/`pymatreader` instead |
| Mapping Toolbox geospatial objects (raster + referencing object, not a plain array) | File loads but contains nested structs you can't parse cleanly | Don't try to parse MATLAB's georeferencing object model in Python. Re-export from MATLAB side using `geotiffwrite()` into a proper GeoTIFF, then read that with `rasterio`. One-time export, not a pipeline dependency. |
| Pre-existing labeled masks/annotations from earlier work | Visual inspection | Definitely leverage — export as GeoTIFF/PNG mask + a CSV/JSON sidecar with scene_id, date, region |
| Custom signal-processing code (e.g., a hand-written polarimetric decomposition) | N/A | Either port the logic to numpy, or call it via `matlab.engine` for one-time *local* preprocessing only. **Do not** build a training-time dependency on MATLAB — Colab/Kaggle have no MATLAB installed, so anything that needs to run during GPU training must be pure Python/GeoTIFF/NetCDF by week 2. |

**Bottom line:** MATLAB is a one-time export step if you use it at all, never something the training or inference pipeline calls live.

## 3. Non-negotiable vs. adaptable

| Choice | Status | Rationale | When to build it |
|---|---|---|---|
| DeepLabV3+ / MobileNetV2 / scSE | **Non-negotiable** | Quantified architectural spine (Zhang et al. 2024: +34.42% mIoU vs. Xception, 9× fewer params) | v0 without scSE weeks 2–3; scSE bolted on week 3–4 |
| BCE+Dice loss + label smoothing (ε=0.1) | **Non-negotiable** | Cheap, directly targets class imbalance, already quantified in your lit review | Week 2, alongside v0 |
| Random Forest look-alike rejection | **Non-negotiable** | Lowest-risk, highest-payoff module — fast to train/iterate | Week 4–5, can overlap with Module 1 refinement |
| Bidirectional Lagrangian drift | **Non-negotiable (concept)**, adaptable (build method) | It's your actual novelty claim. Build method: use OpenDrift/OpenOil, don't hand-roll particle-drift physics | Week 6–7 |
| 5th band (wind-corrected VV/VH ratio) | Adaptable — sequencing, not deletion | Needs ERA5/CMOD5 pipeline proven first | 4 bands weeks 1–3; 5th band week 3–4 |
| 10 pseudo-labeling cycles | Adaptable — scope cut backed by your own citation | Li et al. 2023: F1 0.8432→0.8896 over 21 cycles, front-loaded returns | Cycles 1–3 weeks 3–4 (core deliverable); cycles 4–10 weeks 9–12 (stretch, only if ahead of schedule) |
| Composite score weights (w1–w4) fit by regression | **Don't do this** | ~5 confirmed incidents ⇒ fitting 4 free weights is overfitting by construction | Use equal weights or coarse grid search, validate by rank correlation (see `src/validation/metrics.py`) |

## 4. 10–12 week sequence

| Weeks | Focus | Exit criterion |
|---|---|---|
| 1 | Data access triage (§1) + one end-to-end integration test: one Sentinel-1 scene + one day of GoM AIS + one ERA5 wind file + one CMEMS current snapshot, all reprojected into one UTM zone, plotted together | You can see all four data sources overlaid correctly on one map, in the same CRS |
| 2–3 | Module 1 v0: 4-band DeepLabV3+/MobileNetV2, no scSE, trained on public MKLab + SOS datasets | Trains end-to-end on Colab/Kaggle without OOM; some non-trivial mIoU on held-out scenes |
| 3–4 | Module 1 v1: add scSE, 5th wind-corrected band, label smoothing, pseudo-label cycles 1–3 | mIoU improves over v0; train/val divergence checked (scene-level split, not random) |
| 4–5 | Module 2: RF look-alike rejection (overlaps with Module 1 polish) | RF trained with GroupKFold by scene; feature importances sanity-checked |
| 5–6 | Module 3: AIS cleaning, anomaly detection, dark-ship fallback | Tier-1 candidate vessels correctly flagged for at least one known case |
| 6–7 | Module 4: bidirectional drift + composite score | Forward/backward OpenDrift runs complete on real ERA5/CMEMS data for at least one incident |
| 7–8 | **Primary buffer.** Validation against confirmed incidents (§1 outcome), integration debugging, write-up | mIoU/F1 numbers computed on all three validation tiers; rank-correlation check on attribution |
| 9–12 | Stretch only, in this order: pseudo-label cycles 4–10, hyperparameter tuning, ensemble methods | Don't start these until weeks 1–8's exit criteria are all met |

## 5. Top gotchas

1. **CRS mismatches are the #1 silent bug.** Standardize everything into the SAR scene's UTM zone in one place (`src/preprocessing/crs_utils.py`) — never reproject ad hoc inline elsewhere.
2. **`snappy` is unreliable on recent SNAP versions.** Use `gpt` via subprocess (`src/preprocessing/snap_pipeline.py`) or pyroSAR, not the Python bridge.
3. **Random patch splits leak information.** Adjacent patches from one SAR pass share speckle/weather — split by scene (`src/training/splits.py`), always.
4. **ERA5/CMEMS longitude convention (0–360 vs. −180/180) silently differs by source** — normalize before any spatial join, or your wind field is shifted.
5. **OOM on Colab/Kaggle mid-run kills unattended training.** Probe max batch size once at start, and wrap train steps with a halving retry (`src/training/gpu_utils.py`).
6. **OpenOil's weathering model is NOAA ADIOS/PyGnome, not literally Fay (1971) spreading.** Fine to use — arguably stronger — but document the substitution if your writeup names Fay specifically.
7. **Lazy-built scSE silently never trains.** If the scSE module's channel count is only known/built on the first *training* forward pass, and your optimizer was already constructed from `model.parameters()` before that pass (the normal order), scSE's weights are never registered with the optimizer — it trains, the loss looks normal, and scSE quietly does nothing. `src/models/deeplab_scse.py` forces a dummy forward pass inside `__init__` to avoid this; don't refactor that away.
8. **Don't fit the composite score's 4 weights by regression.** ~5 confirmed incidents means 4 free parameters against ~5 points, with no held-out set possible — overfitting by construction. Use equal weights or a coarse grid search, and validate by whether the correct vessel ranks #1 (top-1 hit rate), not a formal rank-correlation coefficient, which also needs more data than you'll have to be meaningful (`src/validation/metrics.py::rank_correlation_check`).

## 6. Environment

Colab/Kaggle (pip-only):
```bash
!pip install -q -r requirements.txt
```
Check before reinstalling torch — Colab/Kaggle ship a CUDA-matched build already:
```python
import torch; print(torch.__version__, torch.cuda.is_available())
```

Local (conda/mamba, avoids GDAL/rasterio pip version mismatches):
```bash
mamba create -n bilge python=3.10 gdal rasterio geopandas shapely -c conda-forge
mamba activate bilge
pip install -r requirements.txt
```

## 7. Code map

```
src/data_access/sentinel1_cdse.py     CDSE OData + Keycloak auth, search + download
src/data_access/ais_noaa.py           NOAA AIS CSV parse/filter/project
src/data_access/era5_cmems.py         ERA5 (cdsapi) + CMEMS (copernicusmarine) retrieval
src/preprocessing/snap_pipeline.py    SNAP gpt-graph subprocess pattern
src/preprocessing/crs_utils.py        CRS standardization + error handling
src/models/deeplab_scse.py            DeepLabV3+/MobileNetV2 + bolt-on scSE
src/models/losses.py                  BCE+Dice + label smoothing
src/training/splits.py                Scene/region-level split (anti-leakage)
src/training/gpu_utils.py             OOM probing + retry-with-halving
src/lookalike/feature_extraction.py   RF feature engineering + GroupKFold training
src/ais_attribution/trajectory_cleaning.py   DBSCAN cleaning + IsolationForest anomaly scoring
src/drift/lagrangian_drift.py         OpenDrift forward/backward + match score
src/validation/metrics.py             mIoU, pixel F1, composite score, rank correlation
```
