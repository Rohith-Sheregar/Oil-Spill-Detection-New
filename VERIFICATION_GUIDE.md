# Project Verification Guide — Testing the Full Pipeline on Real-World Data

## Can I verify my project end-to-end with a real historical case?

**Short answer:** Yes, but not with a single "perfect" open dataset — you have to assemble it yourself from 3 separate free sources. Here's exactly how.

---

## The Hard Truth About "Verified" Cases

> [!IMPORTANT]
> The **Zenodo SOS dataset does NOT include AIS data**. It only has Sentinel-1 TIFF images and binary oil/no-oil masks. It was built for segmentation training — no vessel identities, no AIS tracks, no enforcement records.

The reason a "Sentinel-1 + AIS + fined vessel" bundle doesn't exist publicly is legal: once a vessel is prosecuted, authorities seal the forensic evidence. Academic papers that use real enforcement cases always anonymise the MMSI.

**This is normal and not a problem for your project.** Here is how professional research groups do it instead:

---

## Verification Strategy: 3-Level Testing

| Level | What you test | Data source | Difficulty |
|-------|--------------|-------------|-----------|
| **Level 1** | M1 + M2 accuracy on held-out test scenes | Zenodo SOS test split (already available) | ✅ Easy |
| **Level 2** | Full pipeline end-to-end on a real SAR scene + real AIS | CDSE (new S1 scene) + NOAA AIS CSV (same area/month) | ⚠️ Medium |
| **Level 3** | Attribution on a *known* spill event | Gulf of Mexico or Mediterranean case below | ⚠️ Medium |

---

## Level 1 — Testing M1 + M2 on Unseen Data (Zenodo Test Split)

The Zenodo SOS dataset has a dedicated **test/** folder that was never used in training or validation.

```
data/
  test/       ← ~200 oil scenes + ~200 lookalike scenes
              ← these are UNSEEN by the model (different scene IDs)
```

This is your primary accuracy benchmark.

**Expected metrics for a correctly trained model:**
- M1 segmentation mIoU: **~0.75–0.85** (yours: epoch 8 mIoU = 0.8135 ✅)
- M2 balanced accuracy: **~0.80–0.90**
- M2 AUC: **~0.85–0.92**

**How to run:**
```bash
# Run M1 + M2 on the entire test split
python -m src.pipeline.run_full_pipeline \
    --sar-tiff data/test/oil/SCENE_ID.tif \
    --m1-weights results/module1/checkpoints/best_model.pt \
    --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
    --output-dir results/test_evaluation
```

---

## Level 2 — Full Pipeline on a Real New SAR Scene + Real AIS

Pick a **recent Sentinel-1 scene from the Gulf of Mexico** (a well-known bilge-dump hotspot) and the corresponding NOAA AIS monthly CSV.

### Step 1: Download a Sentinel-1 scene from CDSE

```python
from src.data_access.credentials import load_env
from src.data_access.sentinel1_cdse import search_sentinel1_grd, get_access_token, download_product
import os

load_env()

# Search Gulf of Mexico (shipping lane between Galveston and Yucatan Channel)
products = search_sentinel1_grd(
    bbox=(-97.0, 25.0, -90.0, 30.0),    # Gulf of Mexico
    start_date="2024-03-01",
    end_date="2024-03-07",
)
print(f"Found {len(products)} scenes")
for p in products[:5]:
    print(p["Name"], p["ContentDate"]["Start"])

# Download the first one
token = get_access_token(os.environ["CDSE_USER"], os.environ["CDSE_PASS"])
download_product(products[0]["Id"], "data/test/gulf_scene.zip", token)
```

### Step 2: Download NOAA AIS for the same month

1. Go to: **https://marinecadastre.gov/ais/**
2. Click **"Data"** → select year=2024, month=March
3. Download UTM Zone 14 or 15 (covers Gulf of Mexico)
4. Unzip → you get `AIS_2024_03_Zone14.csv` (~3–8 GB)

This file contains all vessel positions in the Gulf of Mexico for March 2024.

### Step 3: Run the full pipeline

```bash
python -m src.pipeline.run_full_pipeline \
    --sar-tiff data/test/gulf_scene.tif \
    --ais-csv data/ais/AIS_2024_03_Zone14.csv \
    --m1-weights results/module1/checkpoints/best_model.pt \
    --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
    --sar-time 2024-03-04T09:00:00Z \
    --output-dir results/forensic_reports
```

**What you'll see in the output JSON:**
- Dark patches detected (if any)
- Whether they passed the look-alike filter
- Vessel candidates within ±50km / ±6h
- Composite attribution scores

> [!NOTE]
> There may be **no detected spill** in any single random scene — that's correct! Oil spills don't happen in every frame. Try 5–10 scenes until you find one with dark patches.

---

## Level 3 — Using a Documented Historical Spill Event

These are **real, publicly documented** incidents where the location, SAR time, and general circumstances are known. The AIS is available from NOAA.

### Case A: Gulf of Mexico — Chronic Shipping Lane Bilge Dumps

**Background:** The Gulf of Mexico shipping corridor (Galveston to Mexico) is one of the highest-frequency bilge-dump areas in the world. EMSA CleanSeaNet documented dozens of confirmed slicks here between 2018–2023.

| Parameter | Value |
|-----------|-------|
| **Area** | 27°N–29°N, 93°W–96°W (offshore Texas/Louisiana) |
| **Best months to check** | Winter months (Nov–Feb) — calm sea state makes slicks visible |
| **AIS source** | NOAA MarineCadastre — Zone 15 |
| **SAR source** | CDSE search in the bbox above |

**Verification approach:** No single vessel MMSI is public. Instead, you validate by:
1. Detecting a slick
2. Finding the set of vessels the pipeline nominates
3. Checking those MMSIs against MarineTraffic (free web lookup) — do they look like tankers or cargo vessels? Were they in the area?

---

### Case B: Mediterranean — Eastern Basin (Highest Slick Density Globally)

**Background:** The Eastern Mediterranean (Greece–Turkey–Cyprus triangle) has the world's highest density of satellite-detected oil slicks, per EMSA 2019–2023 CleanSeaNet reports.

| Parameter | Value |
|-----------|-------|
| **Area** | 34°N–37°N, 20°E–28°E |
| **AIS source** | MarineTraffic historical API (free tier: last 12 months), OR Spire Maritime (paid) |
| **SAR source** | CDSE |
| **Academic reference** | Yang & Singha 2025 PANGAEA dataset below |

> [!TIP]
> For Mediterranean AIS, NOAA covers only US waters. For international waters use: **https://www.marinetraffic.com/en/ais/home** (free historical AIS up to 1 year) or the **Global Fishing Watch** API (free with account).

---

### Case C: North Sea — PANGAEA Dataset (Best Option for Accuracy Benchmarking)

**This is your best option for an independent test set with real SAR imagery.**

**Dataset:** Yang & Singha (2025) — North Sea oil slick SAR patches on PANGAEA
- **DOI:** https://doi.org/10.1594/PANGAEA.980773
- **What it contains:** 512×512 Sentinel-1 GRD patches with expert-labelled masks — oil spills and look-alikes — from the North Sea
- **Key advantage:** These scenes were collected AFTER the Zenodo SOS dataset (different years, different region) — completely unseen by your model
- **No AIS included**, but you can pair it yourself with NOAA/MarineTraffic data

**Steps:**
1. Download from PANGAEA DOI above
2. Run M1 + M2 on these scenes
3. Compare mask predictions vs expert labels → this gives you a **true generalisation accuracy** (unseen region + unseen year)

---

## What Metrics to Report for Your Project

| Metric | Where computed | How to get it |
|--------|---------------|---------------|
| **M1 Test mIoU** | Zenodo SOS test/ split | Run `train_module1.py` with `--eval-only` flag on test split |
| **M2 CV AUC** | Module 2 training | Already in `results/module2/metrics/cv_scores.json` |
| **M2 Balanced Accuracy** | Module 2 training | Already in `results/module2/metrics/cv_scores.json` |
| **Pipeline False Positive Rate** | PANGAEA test set | Count look-alikes classified as oil by M1+M2 |
| **Attribution Plausibility** | Gulf of Mexico / Mediterranean | Manually verify: do nominated vessels match expected ship types? |

---

## Open Dataset Summary Table

| Dataset | Contains SAR | Contains AIS | Has GT vessel | Unseen by model | URL |
|---------|-------------|-------------|--------------|-----------------|-----|
| **Zenodo SOS (test split)** | ✅ | ❌ | ❌ | ✅ | [doi.org/10.5281/zenodo.8346860](https://zenodo.org/records/8346860) |
| **PANGAEA Yang & Singha 2025** | ✅ | ❌ | ❌ | ✅ | [doi.org/10.1594/PANGAEA.980773](https://doi.org/10.1594/PANGAEA.980773) |
| **NOAA MarineCadastre AIS** | ❌ | ✅ | ❌ | N/A | [marinecadastre.gov/ais](https://marinecadastre.gov/ais/) |
| **Global Fishing Watch** | Vessel detections | ✅ | Partial | N/A | [globalfishingwatch.org](https://globalfishingwatch.org/data-download/) |
| **SkyTruth Cerulean API** | Slick detections | Vessel near-slick | Partial | N/A | [api.cerulean.skytruth.org](https://api.cerulean.skytruth.org) |
| **CDSE (live Sentinel-1)** | ✅ | ❌ | ❌ | ✅ | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) |
| **EMSA CleanSeaNet reports** | ❌ | Anonymised | Anonymised | ✅ | Reports only (PDF) — not raw data |

> [!WARNING]
> EMSA CleanSeaNet's underlying raw data (exact MMSI of fined vessels) is **not publicly available** anywhere. This is by design — the legal chain of custody for prosecution requires data to be held by authorities, not posted online.

---

## Recommended Verification Plan (Practical)

1. **Day 1 — Accuracy on Zenodo test split**
   - Run M1+M2 on `data/test/` (100–200 scenes)
   - Compute mIoU, balanced accuracy, AUC, F1
   - These are your primary reported metrics

2. **Day 2 — Generalisation on PANGAEA**
   - Download PANGAEA.980773 (~5 GB)
   - Run M1+M2 without retraining
   - Compare metrics: if mIoU stays >0.70, model generalises well

3. **Day 3 — Full pipeline smoke test**
   - Pick one real CDSE scene from the Gulf of Mexico
   - Download corresponding NOAA AIS CSV
   - Run the complete pipeline
   - Verify the JSON output looks reasonable (vessel candidates, scores)
   - Use MarineTraffic to look up the nominated MMSI(s) and check if they are cargo/tanker vessels

4. **For the report/thesis** — cite:
   - Test mIoU, M2 CV AUC (measured)
   - Case study: "Pipeline nominates vessels of correct type in Gulf of Mexico bilge-dump hotspot"
   - Limitation: "Ground-truth MMSI for fined vessels is not publicly available per MARPOL enforcement protocol"
