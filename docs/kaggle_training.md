# Kaggle Training Guide: Module 1 Segmentation

This repository can train the Module 1 oil-spill segmentation model on Kaggle,
but the full Zenodo package should be treated carefully.

## What is feasible on Kaggle

- Your Zenodo URL is valid for the Part I oil-spill image archive:
  `https://zenodo.org/records/8346860/files/01_Train_Val_Oil_Spill_images.7z?download=1`.
- The complete Zenodo SOS set is split across three records: Part I oil
  train/validation, Part II no-oil/look-alike train/validation, and Part III
  test.
- The compressed archives are roughly 90 GB total before extraction. Extracted
  TIFFs can exceed typical Kaggle working/output space, so the notebook defaults
  to a subset and uses `/kaggle/temp` for scratch data.
- For a first run, train on a balanced subset of oil, no-oil, and look-alike
  scenes. Use the full data only after a subset run finishes and you know the
  session has enough disk, time, and GPU quota.

## Module 1 scope

The synopsis Module 1 has four parts:

1. Acquire Sentinel-1 SAR and aligned AIS windows.
2. Preprocess SAR: thermal noise removal, border noise removal, calibration,
   terrain correction, speckle filtering, and Sigma0 dB conversion.
3. Train an improved DeepLabV3+ segmentation model with MobileNetV2, BCE+Dice
   loss, label smoothing, and optional scSE attention.
4. Validate with scene-level splits and report mIoU/F1.

The Zenodo SOS TIFFs already contain Sentinel-1 Sigma0 VV/VH images and masks,
so the Kaggle notebook starts at step 3. It does not replace the raw Sentinel-1
preprocessing pipeline needed for your final end-to-end system.

## Important band correction

The synopsis describes a five-band model: VV, VH, H, alpha, and a wind-corrected
VV/VH ratio. Zenodo SOS does not contain all five bands. The Kaggle notebook
therefore trains a core model with:

- `vv_vh`: two channels from Zenodo, or
- `vv_vh_diff`: VV, VH, and the VV minus VH dB difference as a lightweight
  derived channel.

When you later generate H/alpha/wind-corrected features from raw scenes, update
the dataset loader to read those extra bands and set the model input channels
accordingly.

## Recommended Kaggle run order

1. Run the notebook with `ARCHIVE_PRESET = "test_smoke"` to test download,
   extraction, model construction, and one short training job. This uses the
   smaller Zenodo Part III archive only as a pipeline test, not as final
   reportable training evidence.
2. Run `ARCHIVE_PRESET = "module1_balanced"` for oil, no-oil, and look-alike
   train/validation data with `MAX_SCENES_PER_CLASS` between 50 and 150.
3. Increase `MAX_SCENES_PER_CLASS`, `EPOCHS`, and `SAMPLES_PER_SCENE`.
4. Only try `ARCHIVE_PRESET = "all"` if Kaggle shows enough free disk and you
   are comfortable with a long session.

The best checkpoint is written to `module1_outputs/best_model.pt`, with
`metrics.csv`, split CSVs, and `run_config.json` beside it.
