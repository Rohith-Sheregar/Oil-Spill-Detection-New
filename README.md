# Automated Detection and Vessel Attribution of Illegal Bilge Dumping

This repository contains the complete, end-to-end pipeline for detecting illegal vessel bilge dumps using Sentinel-1 Synthetic Aperture Radar (SAR) imagery and attributing them to specific vessels using AIS (Automatic Identification System) tracking data and Lagrangian ocean drift modeling.

## 🏗️ Project Architecture

The system is divided into four distinct forensic modules:

### **Module 1: SAR Oil Slick Segmentation**
- A deep learning computer vision model (DeepLabV3+ / MobileNetV2 with scSE attention) that segments dark patches in Sentinel-1 VV/VH intensity imagery.
- Trained on the MKLab and SOS (Spill or Source) datasets.

### **Module 2: Look-alike Rejection & Bilge Filter**
- A Random Forest classifier that filters out environmental "look-alikes" (low wind zones, biogenic films, upwelling).
- Evaluates 12 morphological and contextual features (elongation, area, wind speed) to isolate deliberate bilge dumps from accidental spills or natural phenomena.

### **Module 3: AIS Vessel Candidate Filtering**
- Fetches NOAA AIS historical data within a ±6h / ±50km spatiotemporal window of the spill.
- Employs **3D DBSCAN** to clean and de-spoof vessel trajectories.
- Uses an Isolation Forest + Random Forest hybrid to score behavioral anomalies (sudden slowdowns, erratic heading changes).
- **Dark Ship Fallback**: Detects non-cooperative vessels (AIS off) directly from SAR using the Faster Threshold Method (FTM) and flags them for enforcement.

### **Module 4: Bidirectional Drift Attribution & Forensic Scoring**
- Runs physics-based Lagrangian drift modeling (incorporating ERA5 10m winds, CMEMS ocean currents, and wave Stokes drift).
- Calculates a final bounded `[0.0, 1.0]` composite confidence score per vessel by weighting drift similarity, AIS anomalies, morphological alignment, and temporal (night-time) likelihood.

---

## 🚀 Getting Started

### 1. Environment Setup

It is highly recommended to use Conda/Mamba to avoid `gdal` and `rasterio` version mismatches.

```bash
mamba create -n bilge python=3.10 gdal rasterio geopandas shapely -c conda-forge
mamba activate bilge
pip install -r requirements.txt
```

### 2. End-to-End Inference

The entire 4-module pipeline is orchestrated via a single unified CLI script. It accepts raw SAR imagery, AIS records, and MetOcean forcing data, and outputs a structured forensic JSON report.

```bash
python -m src.pipeline.run_full_pipeline \
    --sar-tiff /path/to/sentinel1_scene.tiff \
    --ais-csv /path/to/noaa_ais_monthly.csv \
    --m1-weights results/module1/checkpoints/best_model.pt \
    --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
    --metocean-nc /path/to/era5_cmems_forcing.nc \
    --sar-time 2026-03-15T09:00:00Z \
    --output-dir results/forensic_reports/
```

### 3. Training the Models

**Module 1 (Segmentation)**
Training is optimized for Kaggle/Colab environments to leverage free GPUs.
- Open `notebooks/kaggle_module1_zenodo_training.ipynb` in Kaggle.
- Run the preset `test_smoke` configuration to verify the pipeline.
- Switch to `module1_balanced` for the full training loop on Zenodo archive data.

**Module 2 (Look-alike RF)**
Train the morphological filter locally using extracted feature datasets:
```bash
python -m src.training.train_module2 \
    --features-csv data/features.csv \
    --output-model results/module2/checkpoints/lookalike_rf.joblib
```

---

## 📂 Code Map

```text
src/
├── data_access/            # Data fetchers (CDSE OData, NOAA AIS, ERA5/CMEMS)
├── preprocessing/          # SNAP gpt wrappers, CRS standardization, PolSAR decomp
├── models/                 # DeepLabV3+ with scSE, custom BCE+Dice losses
├── training/               # Training scripts, GPU utilities, spatial K-Fold splits
├── lookalike/              # Module 2: Feature extraction, morphology, RF classifier
├── ais_attribution/        # Module 3: 3D DBSCAN tracking, FTM Dark Ship, IsoForest anomalies
├── drift/                  # Module 4: 2D Lagrangian drift solver, forensic composite scoring
├── validation/             # mIoU, F1, and rank correlation metrics
└── pipeline/               # Unified run_full_pipeline.py orchestrator
```
