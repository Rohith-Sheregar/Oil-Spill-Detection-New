# Oil Spill Detection & Vessel Attribution Pipeline

> **Sentinel-1 SAR → Deep Learning Segmentation → Look-alike Discrimination → AIS Vessel Attribution → Lagrangian Drift Simulation**

A complete forensic pipeline for detecting illegal bilge-dump oil spills in Sentinel-1 SAR imagery, discriminating them from natural look-alikes, and attributing them to suspect vessels using AIS trajectory analysis and physics-based drift modelling.

---

## Project Status

| Module | Description | Status |
|--------|-------------|--------|
| **Module 1** | DeepLabV3+ SAR segmentation | ✅ Trained — epoch 8, mIoU 0.8135 |
| **Module 2** | Random Forest look-alike discriminator | ✅ Trained — 1,200 scenes |
| **Module 3** | AIS vessel candidate filtering + anomaly detection | ✅ Implemented & smoke-tested |
| **Module 4** | Lagrangian drift + composite attribution scoring | ✅ Implemented & smoke-tested |

---

## Architecture

```
Sentinel-1 IW GRD TIFF
        │
        ▼
┌─────────────────────────────────────┐
│  MODULE 1 — SAR Segmentation        │
│  DeepLabV3+ / SCSe decoder          │
│  5-band input: VV, VH, H, α, wind   │
│  Output: binary oil/no-oil mask     │
└────────────────┬────────────────────┘
                 │ binary mask
                 ▼
┌─────────────────────────────────────┐
│  MODULE 2 — Look-alike Discriminator│
│  Morphological closing (5×5, ×2)    │
│  12-feature extraction per patch    │
│  Random Forest (n=200, GroupKFold)  │
│  Bilge-dump filter (elong>3, <50km²)│
│  Output: oil vs look-alike label    │
└────────────────┬────────────────────┘
                 │ confirmed oil patches + centroids
                 ▼
┌─────────────────────────────────────┐
│  MODULE 3 — AIS Attribution         │
│  NOAA AIS CSV ±6h / ±50km window    │
│  3D DBSCAN trajectory cleaning      │
│  IsoForest + RF anomaly scoring     │
│  Dark-ship FTM detection (SAR)      │
│  Output: Tier-1 vessel candidates   │
└────────────────┬────────────────────┘
                 │ suspect vessel list
                 ▼
┌─────────────────────────────────────┐
│  MODULE 4 — Drift Attribution       │
│  Forward simulation: vessel → SAR   │
│  Backward simulation: slick → origin│
│  Stokes drift: 1.5% of wind         │
│  Composite score: 0.4S_drift        │
│                 + 0.3S_AIS          │
│                 + 0.2S_morphology   │
│                 + 0.1S_temporal     │
│  Output: confidence score per vessel│
└─────────────────────────────────────┘
```

---

## Directory Structure

```
Oil_spill_detection/
├── notebooks/                            # Kaggle training notebooks
│   ├── module-1-training.ipynb           # M1: DeepLabV3+ SAR segmentation
│   └── module-2-training.ipynb           # M2: Random Forest look-alike training
│
├── src/
│   ├── __init__.py
│   ├── data_access/
│   │   ├── credentials.py                # Auto-loads outputs/.env into os.environ
│   │   ├── sentinel1_cdse.py             # CDSE Sentinel-1 search + download (M3/M4)
│   │   ├── era5_cmems.py                 # ERA5 wind + CMEMS ocean currents (M4)
│   │   └── ais_noaa.py                   # NOAA AIS CSV reader — free, no login (M3)
│   │
│   ├── preprocessing/
│   │   ├── band_stack.py                 # 5-band input tensor (VV, VH, H, α, wind)
│   │   ├── polsar_decomp.py              # Cloude-Pottier H/A/α decomposition
│   │   ├── wind_ratio.py                 # CMOD5.N wind-normalised backscatter ratio
│   │   ├── crs_utils.py                  # CRS reprojection helpers
│   │   └── snap_pipeline.py              # SNAP graph XML builder (optional)
│   │
│   ├── models/
│   │   ├── deeplab_scse.py               # DeepLabV3+ with SCSe decoder (Module 1)
│   │   └── losses.py                     # BCE + Dice composite loss
│   │
│   ├── lookalike/                        # Module 2
│   │   ├── morphology.py                 # Binary closing + connected components
│   │   ├── features.py                   # 12-feature extraction engine
│   │   ├── feature_extraction.py         # Backward-compat shim
│   │   ├── classifier.py                 # LookalikeClassifier (RF wrapper)
│   │   └── bilge_filter.py               # Bilge-dump morphology filter
│   │
│   ├── ais_attribution/                  # Module 3
│   │   ├── trajectory_cleaning.py        # DBSCAN cleaning + spatiotemporal slicing
│   │   ├── anomaly_detection.py          # IsoForest + RF hybrid anomaly scorer
│   │   ├── dark_ship.py                  # FTM ship detection + SAR-AIS correlation
│   │   └── pipeline.py                   # Module3Pipeline orchestrator
│   │
│   ├── drift/                            # Module 4
│   │   ├── lagrangian_drift.py           # Forward/backward drift simulation
│   │   └── scoring.py                    # Composite attribution score
│   │
│   ├── training/
│   │   ├── train_module1.py              # Module 1 CLI training script
│   │   ├── train_module2.py              # Module 2 CLI training script
│   │   ├── zenodo_sos_dataset.py         # Zenodo SOS dataset loader
│   │   ├── splits.py                     # GroupShuffleSplit helpers
│   │   ├── gpu_utils.py                  # GPU diagnostics
│   │   └── pseudo_label_trainer.py       # Semi-supervised pseudo-label training
│   │
│   ├── validation/
│   │   └── metrics.py                    # mIoU, F1, precision, recall
│   │
│   ├── reporting/
│   │   └── module1_report.py             # Training report generator
│   │
│   └── pipeline/
│       └── run_full_pipeline.py          # End-to-end CLI orchestrator (M1→M4)
│
├── results/
│   ├── module1/
│   │   ├── checkpoints/
│   │   │   └── best_model.pt             # Best M1 checkpoint (epoch 8, mIoU 0.8135)
│   │   └── metrics/
│   │       └── train_metrics.csv         # Full epoch history
│   └── module2/
│       ├── checkpoints/
│       │   └── lookalike_rf.joblib       # Trained RF classifier (download from HF)
│       └── metrics/                      # CV scores, feature importance (download from HF)
│
├── outputs/                              # Git-ignored — credentials only
│   ├── .env                              # Copernicus portal credentials
│   └── cdsapirc.txt                      # CDS API key (copy to ~/.cdsapirc)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Rohith-Sheregar/Oil-Spill-Detection-New.git
cd Oil-Spill-Detection-New
pip install -r requirements.txt
```

### 2. Copernicus credentials

Create `outputs/.env` (already provided — this file is git-ignored):

```bash
# outputs/.env
CDSE_USER="rohithraghu3228@gmail.com"
CDSE_PASS="Rohith@12345"

COPERNICUSMARINE_SERVICE_USERNAME="rraghu"
COPERNICUSMARINE_SERVICE_PASSWORD="Rohith@12345"

CDSAPI_URL="https://cds.climate.copernicus.eu/api"
CDSAPI_KEY="<your-cds-api-key>"
```

Install the CDS API key for ERA5 wind data:

```bash
cp outputs/cdsapirc.txt ~/.cdsapirc
```

### 3. Download trained models from Hugging Face Hub

```python
from huggingface_hub import hf_hub_download

# Module 1 — SAR segmentation model
hf_hub_download(
    repo_id="RohithSheregar/oil-spill-models",
    filename="best_model.pt",
    local_dir="results/module1/checkpoints"
)

# Module 2 — Look-alike RF classifier
hf_hub_download(
    repo_id="RohithSheregar/oil-spill-models",
    filename="module2/lookalike_rf.joblib",
    local_dir="results/module2/checkpoints"
)
```

---

## Training

### Module 1 — SAR Segmentation (Kaggle, T4 GPU)

Open [`notebooks/module-1-training.ipynb`](notebooks/module-1-training.ipynb) on Kaggle and run all cells.

- **Dataset:** Zenodo SOS Sentinel-1 (oil + no-oil + lookalike, ~37 GB)
- **Model:** DeepLabV3+ with SCSe decoder, 5-band input
- **Training:** GroupShuffleSplit by scene_id, cosine LR schedule, BCE+Dice loss
- **Auto-save:** Best checkpoint uploaded to HF Hub after every improvement

```bash
# Equivalent CLI command
python -m src.training.train_module1 \
    --data-root /kaggle/working/data \
    --results-dir /kaggle/working/results/module1 \
    --input-mode full_5band \
    --epochs 20 \
    --lr 1e-3 \
    --batch-size 16 \
    --hf-repo-id RohithSheregar/oil-spill-models \
    --hf-token $HF_TOKEN
```

### Module 1 Training Progress

| Epoch | Train Loss | Val Loss | Val mIoU | Val F1 | Notes |
|-------|-----------|---------|---------|-------|-------|
| 1 | 0.2863 | 0.5320 | 0.7529 | 0.8590 | |
| 2 | 0.2480 | 0.5463 | 0.7385 | 0.8496 | |
| 3 | 0.2434 | 0.5229 | 0.7835 | 0.8786 | |
| 4 | 0.2400 | 0.5142 | 0.7547 | 0.8602 | |
| 5 | 0.2327 | 0.5221 | 0.7752 | 0.8734 | |
| 6 | 0.2417 | 0.5010 | 0.7782 | 0.8753 | |
| 7 | 0.2409 | 0.4937 | 0.7757 | 0.8737 | |
| **8** | **0.2352** | **0.4836** | **0.8135** | **0.8971** | ⭐ **Best model** |
| 9 | 0.2314 | 0.5134 | 0.7874 | 0.8810 | |
| 10 | 0.2267 | 0.5105 | 0.7839 | 0.8789 | |
| 11 | 0.2295 | 0.5057 | 0.7638 | 0.8661 | |
| 12 | *(in progress)* | | | | |

> **Training convergence note:** Best val mIoU achieved at epoch 8. Subsequent epochs show mild oscillation around 0.77-0.79 — the model has converged. Recommend stopping after epoch 15 if no new best is set.

<!-- PLACEHOLDER: Add training loss curve graph here -->
<!-- ![Module 1 Training Curves](results/module1/metrics/training_curves.png) -->

### Module 2 — Look-alike RF Training (Kaggle, T4 GPU)

Open [`notebooks/module-2-training.ipynb`](notebooks/module-2-training.ipynb) on Kaggle and run all cells.

- **Dataset:** Oil (label=1) + Lookalike (label=0) TIFFs from Zenodo SOS
- **Features:** 12 tabular features per dark patch (4 polarimetric, 4 geometric, 3 contextual, 1 temporal)
- **Model:** RandomForestClassifier (n_estimators=200, class_weight='balanced_subsample')
- **CV:** GroupKFold (n_splits=5, grouped by scene_id — anti-leakage)
- **Runtime:** ~12 minutes total on T4 GPU

```bash
# Equivalent CLI command
python -m src.training.train_module2 \
    --data-root /kaggle/working/data \
    --results-dir /kaggle/working/results/module2 \
    --m1-checkpoint results/module1/checkpoints/best_model.pt \
    --n-estimators 200 \
    --n-folds 5 \
    --night-boost 0.15 \
    --hf-repo-id RohithSheregar/oil-spill-models \
    --hf-token $HF_TOKEN
```

### Module 2 Results

<!-- PLACEHOLDER: Add feature importance bar chart here -->
<!-- ![Feature Importance](results/module2/metrics/feature_importance.png) -->

| Metric | Value |
|--------|-------|
| CV Balanced Accuracy | *(see cv_scores.json)* |
| CV AUC | *(see cv_scores.json)* |
| OOB Score | *(see cv_scores.json)* |

---

## Running the Full Pipeline

### Prerequisites

- `results/module1/checkpoints/best_model.pt` — M1 segmentation model
- `results/module2/checkpoints/lookalike_rf.joblib` — M2 RF classifier
- A Sentinel-1 IW GRD TIFF (see data sources below)
- *(Optional)* NOAA AIS monthly CSV for vessel attribution
- *(Optional)* MetOcean NetCDF for drift simulation

### Single-command execution

```bash
python -m src.pipeline.run_full_pipeline \
    --sar-tiff data/test/scene.tif \
    --m1-weights results/module1/checkpoints/best_model.pt \
    --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
    --ais-csv data/ais/AIS_2026_03_Zone15.csv \
    --sar-time 2026-03-15T09:00:00Z \
    --output-dir results/forensic_reports
```

### Output

Each run produces a JSON forensic report in `results/forensic_reports/`:

```json
{
  "scene_id": "S1A_IW_GRDH_...",
  "n_dark_patches": 3,
  "n_bilge_candidates": 1,
  "vessel_attributions": [
    {
      "mmsi": 123456789,
      "vessel_name": "...",
      "composite_score": 0.87,
      "s_drift": 0.92,
      "s_ais_anomaly": 0.81,
      "s_morphology": 0.79,
      "s_temporal": 1.0
    }
  ]
}
```

<!-- PLACEHOLDER: Add example pipeline output screenshot here -->

---

## Data Sources

| Data | Source | Access | Used By |
|------|--------|--------|---------|
| Sentinel-1 IW GRD TIFFs (training) | [Zenodo SOS](https://zenodo.org/records/8346860) | Free, no login | M1 + M2 training |
| Sentinel-1 IW GRD (live inference) | [CDSE](https://dataspace.copernicus.eu) | Free account | M1 inference |
| AIS vessel positions | [NOAA Marine Cadastre](https://hub.marinecadastre.gov/pages/vesseltraffic) | Free, no login (monthly CSV) | M3 |
| ERA5 10m winds | [ECMWF CDS](https://cds.climate.copernicus.eu) | Free account + API key | M4 drift |
| Ocean surface currents | [CMEMS](https://data.marine.copernicus.eu) | Free account | M4 drift |

### Getting AIS Data (Module 3)

1. Go to [hub.marinecadastre.gov/pages/vesseltraffic](https://hub.marinecadastre.gov/pages/vesseltraffic)
2. Select year + UTM zone matching your SAR scene (Gulf of Mexico → Zone 14/15/16)
3. Download the monthly CSV (can be several GB — reading is chunked automatically)

### Getting a Sentinel-1 Test Scene (from CDSE)

```python
from src.data_access.credentials import load_env
from src.data_access.sentinel1_cdse import search_sentinel1_grd, get_access_token, download_product
import os

load_env()  # loads CDSE_USER, CDSE_PASS from outputs/.env

products = search_sentinel1_grd(
    bbox=(-94.0, 27.0, -88.0, 30.0),  # Gulf of Mexico
    start_date="2026-03-01",
    end_date="2026-03-07",
)
print(f"Found {len(products)} scenes")

token = get_access_token(os.environ["CDSE_USER"], os.environ["CDSE_PASS"])
download_product(products[0]["Id"], "scene_001.zip", token)
```

---

## The 12 Look-alike Discriminator Features (Module 2)

| # | Feature | Category | Physics Rationale |
|---|---------|----------|-------------------|
| 1 | `mean_H` | Polarimetric | Cloude-Pottier entropy — high H = volume scattering (biogenic film) |
| 2 | `anisotropy_A` | Polarimetric | Always 0.0 (dual-pol limit; reserved for future quad-pol) |
| 3 | `mean_alpha_deg` | Polarimetric | Scattering mechanism angle [0°–90°] |
| 4 | `copol_ratio_VV_VH` | Polarimetric | VV/VH linear ratio; oil suppresses VV more than VH |
| 5 | `area_km2` | Geometric | Bilge dumps < 50 km² |
| 6 | `elongation` | Geometric | Bilge streaks > 3:1 axis ratio |
| 7 | `perimeter_area_ratio` | Geometric | Shape complexity |
| 8 | `compactness` | Geometric | ISO formula: 4πA/P² = 1.0 for a circle |
| 9 | `wind_speed_ms` | Contextual | ERA5 U10; wind > 14 m/s suppresses surface films |
| 10 | `proximity_shipping_lane_km` | Contextual | Distance to nearest shipping lane bbox |
| 11 | `is_night` | Contextual | 1 if 20:00–06:00 local; >80% illegal dumps at night |
| 12 | `morphology_change_km2` | Temporal | Area change vs. prior acquisition (0.0 for single-pass) |

---

## Module 3: AIS Anomaly Features

Six behavioural features are extracted per vessel track:

| Feature | Meaning |
|---------|---------|
| `sog_mean` | Average speed over ground (knots) |
| `sog_variance` | SOG variability — irregular speed = suspicious |
| `course_deviation_std` | Circular standard deviation of COG |
| `n_stops` | Count of pings where SOG < 0.5 kn |
| `max_sudden_sog_drop` | Most negative consecutive SOG delta |
| `min_proximity_km` | Closest approach to spill centroid |

---

## Module 4: Composite Attribution Score

```
C = 0.4 × S_drift  +  0.3 × S_AIS_anomaly  +  0.2 × S_morphology  +  0.1 × S_temporal
```

| Component | Weight | Meaning |
|-----------|--------|---------|
| S_drift | 40% | Lagrangian drift similarity (forward ∩ backward particles) |
| S_AIS | 30% | Normalised IsoForest + RF anomaly score |
| S_morphology | 20% | Cosine alignment between slick elongation and vessel COG |
| S_temporal | 10% | 1.0 if nighttime (20:00–06:00), 0.5 otherwise |

---

## References

| Paper | Usage |
|-------|-------|
| Balsaraf et al. (2025) | Isolation Forest + RF hybrid anomaly detection (Module 3) |
| Chang et al. (2024) | Bilge-dump morphological signature — narrow streak, 5×5 closing |
| Chen & Wang (2022) | H/A/α validated for oil-spill discrimination |
| Jeon et al. (2023) | 3D DBSCAN trajectory cleaning; FTM dark-ship detection |
| Li et al. (2023) | Temporal morphology change feature |
| Liao et al. (2023) | Night-time weighting: >80% illegal discharges at night |
| Song et al. (2024) | Polarimetric features for SAR oil discrimination |
| Yang et al. (2022) | Geometric shape descriptors for slick morphology |
| Zakzouk et al. (2025) | Shipping-lane bounding boxes for proximity feature |
| Zenodo SOS Dataset | Training data — [DOI 10.5281/zenodo.8346860](https://zenodo.org/records/8346860) |

---

## Hugging Face Hub

Trained models are automatically uploaded to:
[https://huggingface.co/RohithSheregar/oil-spill-models](https://huggingface.co/RohithSheregar/oil-spill-models)

```
RohithSheregar/oil-spill-models/
├── best_model.pt          ← Module 1 (epoch 8, mIoU 0.8135)
├── last_model.pt          ← Module 1 last checkpoint
├── train_metrics.csv      ← Module 1 full epoch history
└── module2/
    ├── lookalike_rf.joblib      ← Module 2 Random Forest
    ├── cv_scores.json           ← 5-fold CV results
    ├── feature_importance.csv   ← Ranked MDI importances
    ├── feature_importance.png   ← Bar chart
    ├── detection_summary.csv    ← Per-scene bilge counts
    └── train_metrics.csv        ← Module 2 summary row
```

---

## License

Academic / research use. Sentinel-1 data © Copernicus Programme / ESA.
AIS data © NOAA Marine Cadastre.
ERA5 © ECMWF, licensed under CC-BY 4.0.
