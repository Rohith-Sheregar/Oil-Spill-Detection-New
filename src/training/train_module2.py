"""
Module 2 — Kaggle-ready CLI Entrypoint.

Trains the LookalikeClassifier (Random Forest) on dark-patch features extracted
from the Module 1 segmentation output, using GroupKFold cross-validation grouped
by scene_id to prevent data leakage.

Pipeline executed by this script
---------------------------------
1. Discover oil + lookalike scene pairs via zenodo_sos_dataset.discover_sos_pairs()
2. For each scene:
   a. Load SAR TIFF (VV, VH dB arrays)
   b. Load ground-truth mask OR run Module 1 inference on-the-fly
   c. Apply 2-iteration morphological closing (bilge_closing)
   d. Extract connected components (regionprops)
   e. Compute 12 tabular features per component
3. Concatenate all scene features into a single DataFrame
4. Train LookalikeClassifier with GroupKFold CV (grouped by scene_id)
5. Save: lookalike_rf.joblib, feature_importance.png, cv_scores.json,
         feature_summary.csv, train_metrics.csv
6. Optionally upload results to Hugging Face Hub (same HfUploader as Module 1)

Usage (Kaggle)
--------------
python -m src.training.train_module2 \\
    --data-root /kaggle/working/data \\
    --results-dir /kaggle/working/results/module2 \\
    --m1-checkpoint /kaggle/working/results/module1/checkpoints/best_model.pt \\
    --gsd-m 10.0 \\
    --n-folds 5 \\
    --hf-repo-id RohithSheregar/oil-spill-models \\
    --hf-token $HF_TOKEN

CLI Arguments
-------------
--data-root         Path to organized data/ directory (same layout as Module 1)
--results-dir       Output directory (default: results/module2/)
--m1-checkpoint     Module 1 checkpoint (.pt). If masks are available on disk,
                    inference is skipped and saved masks are used directly.
--gsd-m             Ground sampling distance in metres (default: 10.0 for S1 IW GRD)
--prob-threshold    RF probability threshold for bilge-dump classification (default: 0.5)
--night-boost       Night-time prior boost (default: 0.15 per Liao et al. 2023)
--min-elongation    Minimum elongation ratio for bilge filter (default: 3.0)
--max-area-km2      Maximum patch area for bilge filter (default: 50.0)
--n-folds           GroupKFold cross-validation folds (default: 5)
--n-estimators      RF ensemble size (default: 200)
--min-component-px  Minimum component area in pixels (default: 10)
--seed              Random seed (default: 42)
--hf-repo-id        Hugging Face repo ID for model upload
--hf-token          Hugging Face write access token
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Make project root importable on Kaggle (same pattern as train_module1.py) ─
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tifffile

from src.lookalike.morphology  import close_and_extract
from src.lookalike.features    import (
    FEATURE_NAMES, META_COLUMNS,
    extract_scene_features, build_feature_dataframe,
)
from src.lookalike.classifier  import LookalikeClassifier
from src.lookalike.bilge_filter import (
    apply_bilge_filter, summarise_detections,
    DEFAULT_MIN_ELONGATION, DEFAULT_MAX_AREA_KM2,
    DEFAULT_NIGHT_BOOST, DEFAULT_PROB_THRESHOLD,
)
from src.training.zenodo_sos_dataset import discover_sos_pairs

log = logging.getLogger(__name__)


# ─── Hugging Face Hub uploader (reused from Module 1) ─────────────────────────

class HfUploader:
    """
    Silently uploads result files to Hugging Face Hub.
    Identical design to the HfUploader in train_module1.py.
    If authentication or any upload fails, training continues uninterrupted.
    """

    def __init__(self, repo_id: str, token: str) -> None:
        self.repo_id = repo_id
        self.enabled = False
        self._api    = None
        self._log    = logging.getLogger(self.__class__.__name__)

        if not repo_id or not token:
            self._log.info("HfUploader: disabled (no repo_id or token).")
            return
        try:
            from huggingface_hub import HfApi
            self._api = HfApi(token=token)
            self._api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
            self.enabled = True
            self._log.info("🤗 HF uploader ready — repo: %s", repo_id)
        except Exception as exc:
            self._log.warning("HfUploader init failed (%s). HF disabled.", exc)

    def upload(self, local_path: str | Path, label: str = "") -> bool:
        if not self.enabled or self._api is None:
            return False
        try:
            local_path = Path(local_path)
            size_mb    = local_path.stat().st_size / (1024 ** 2)
            self._api.upload_file(
                path_or_fileobj = str(local_path),
                path_in_repo    = local_path.name,
                repo_id         = self.repo_id,
                commit_message  = f"Module2 auto-save {local_path.name} {label}".strip(),
            )
            self._log.info("  🤗 HF uploaded: %s  (%.1f MB)", local_path.name, size_mb)
            return True
        except Exception as exc:
            self._log.warning("  HF upload failed for %s: %s", local_path.name, exc)
            return False

    def upload_many(self, paths: list, label: str = "") -> None:
        for p in paths:
            if Path(p).exists():
                self.upload(p, label=label)


# ─── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"module2_train_{stamp}.log"
    fmt      = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logging.info("Log file: %s", log_path)


# ─── Module 1 inference helper ────────────────────────────────────────────────

def _run_m1_inference(
    tiff_path: str | Path,
    m1_checkpoint: str | Path | None,
    device: Any,
) -> np.ndarray | None:
    """
    Run Module 1 inference on a TIFF to produce a binary mask.

    Returns (H, W) uint8 array with values 0/1, or None on failure.

    Falls back to None (skip scene) if the checkpoint is unavailable
    or if the TIFF cannot be loaded — this allows feature extraction to
    proceed using ground-truth masks in supervised mode.
    """
    if m1_checkpoint is None:
        return None
    try:
        import torch
        from src.models.deeplab_scse import DeepLabV3PlusSCSE
        from src.preprocessing.band_stack import build_5band_from_tiff

        ckpt_path = Path(m1_checkpoint)
        if not ckpt_path.exists():
            log.warning("M1 checkpoint not found: %s", ckpt_path)
            return None

        stack = build_5band_from_tiff(str(tiff_path))   # (5, H, W)
        in_ch, H, W = stack.shape

        # Load model lazily (will be shared across scenes via closure in caller)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = DeepLabV3PlusSCSE(in_channels=in_ch, classes=1, input_size=256)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        tensor = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, 5, H, W)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        return (probs >= 0.5).astype(np.uint8)

    except Exception as exc:
        log.warning("M1 inference failed for %s: %s", tiff_path, exc)
        return None


# ─── Scene loading and feature extraction ─────────────────────────────────────

def _load_scene_arrays(
    image_path: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load a Zenodo-style 2048×2048×2 TIFF and return (vv_db, vh_db) float32 arrays.

    Returns None if the file cannot be read.
    """
    try:
        arr = tifffile.imread(str(image_path)).astype(np.float32)
        if arr.ndim == 2:
            return arr, arr.copy()    # single-band fallback
        if arr.ndim == 3:
            if arr.shape[0] <= 8 and arr.shape[1] > 8:
                arr = np.moveaxis(arr, 0, -1)    # (C, H, W) → (H, W, C)
            return arr[..., 0], arr[..., 1]      # VV, VH
        log.warning("Unexpected TIFF shape %s for %s", arr.shape, image_path)
        return None
    except Exception as exc:
        log.warning("Failed to load TIFF %s: %s", image_path, exc)
        return None


def _load_mask(mask_path: str | Path | None) -> np.ndarray | None:
    """Load a binary mask TIFF. Returns None on failure."""
    if mask_path is None:
        return None
    try:
        mask = tifffile.imread(str(mask_path)).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.uint8)
    except Exception as exc:
        log.warning("Failed to load mask %s: %s", mask_path, exc)
        return None


def _build_scene_dicts(
    df_pairs: pd.DataFrame,
    m1_checkpoint: str | Path | None,
    args: argparse.Namespace,
    device: Any,
    label_value: int,
) -> list[dict]:
    """
    Build a list of scene_dicts ready for build_feature_dataframe().

    For each row in df_pairs (one scene = one image_path + mask_path):
      1. Load VV, VH dB arrays from TIFF
      2. Load binary mask from mask_path OR run M1 inference
      3. Apply morphological closing + extract components
      4. Assemble scene_dict

    Parameters
    ----------
    df_pairs      : DataFrame from discover_sos_pairs()
    m1_checkpoint : path to Module 1 .pt checkpoint (used if mask_path missing)
    args          : parsed CLI args (gsd_m, min_component_px, etc.)
    device        : torch device for M1 inference
    label_value   : ground-truth label for all components in this split (1=oil, 0=lookalike)

    Returns
    -------
    list of scene_dicts
    """
    scene_dicts = []
    n_scenes = len(df_pairs)
    log.info("Building scene dicts for %d scenes (label=%d)...", n_scenes, label_value)

    for i, row in df_pairs.iterrows():
        scene_id   = row["scene_id"]
        image_path = row["image_path"]
        mask_path  = row.get("mask_path", None)

        # ── Load SAR arrays ──────────────────────────────────────────────────
        arrays = _load_scene_arrays(image_path)
        if arrays is None:
            log.warning("Skipping scene %s — could not load TIFF", scene_id)
            continue
        vv_db, vh_db = arrays

        # ── Get binary mask ──────────────────────────────────────────────────
        mask = _load_mask(mask_path)
        if mask is None:
            log.info("  No mask for %s — running M1 inference...", scene_id)
            mask = _run_m1_inference(image_path, m1_checkpoint, device)
        if mask is None:
            log.warning("  Skipping scene %s — no mask and M1 inference failed.", scene_id)
            continue

        # ── Morphological closing + connected components ─────────────────────
        _, regions = close_and_extract(
            binary_mask   = mask,
            iterations    = 2,
            selem_size    = 5,
            min_area_px   = args.min_component_px,
        )
        if not regions:
            log.debug("  Scene %s: no components after closing — skipping.", scene_id)
            continue

        # ── Ground-truth label map (all components in this scene get the scene label) ─
        label_map = {r.label: label_value for r in regions}

        scene_dict: dict = {
            "scene_id":      scene_id,
            "regions":       regions,
            "vv_db":         vv_db,
            "vh_db":         vh_db,
            "wind_speed_ms": 7.0,    # ERA5 fallback; integrate era5_cmems for production
            "hour_local":    12,     # Scene acquisition hour; replace with scene metadata
            "label_map":     label_map,
        }
        scene_dicts.append(scene_dict)

        if (i % 50 == 0) or (i == n_scenes - 1):
            log.info("  Processed %d/%d scenes...", i + 1, n_scenes)

    log.info("Total usable scenes: %d → %d scene_dicts",
             n_scenes, len(scene_dicts))
    return scene_dicts


# ─── Feature importance plot ───────────────────────────────────────────────────

def _plot_feature_importance(
    fi_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Save a horizontal bar chart of feature importances to out_path (PNG).

    Falls back silently if matplotlib is unavailable (e.g. headless Kaggle runs).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # non-interactive backend for headless Kaggle
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = ["#e63946" if imp > 0.08 else "#457b9d" for imp in fi_df["importance"]]
        ax.barh(
            fi_df["feature"],
            fi_df["importance"],
            xerr=fi_df["std"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "#aaaaaa"},
        )
        ax.invert_yaxis()
        ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
        ax.set_title("Module 2 — RF Feature Importances", fontsize=13, fontweight="bold")
        ax.axvline(x=1 / len(fi_df), color="grey", linestyle="--",
                   linewidth=1.0, label="Uniform baseline")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Feature importance plot saved: %s", out_path)
    except ImportError:
        log.warning("matplotlib not available — skipping feature importance plot.")
    except Exception as exc:
        log.warning("Feature importance plot failed: %s", exc)


# ─── Main training routine ────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Full Module 2 training pipeline.
    """
    data_root   = Path(args.data_root)
    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "checkpoints"
    metrics_dir = results_dir / "metrics"
    log_dir     = results_dir / "logs"

    for d in [ckpt_dir, metrics_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_dir)
    np.random.seed(args.seed)

    # ── Device (for optional M1 inference) ───────────────────────────────────
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)
    except ImportError:
        device = None
        log.warning("PyTorch not found — M1 on-the-fly inference disabled.")

    # ── HF Uploader ───────────────────────────────────────────────────────────
    hf_uploader = HfUploader(
        repo_id = getattr(args, "hf_repo_id", "") or "",
        token   = getattr(args, "hf_token", "") or "",
    )

    # ── Discover scene pairs ─────────────────────────────────────────────────
    train_root = data_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_root}\n"
            "Use the same --data-root as Module 1."
        )

    all_dfs: list[pd.DataFrame] = []
    class_labels = {"oil": 1, "lookalike": 0}

    for cls_name, lbl in class_labels.items():
        cls_dir = train_root / cls_name
        if not cls_dir.exists():
            log.warning("Class dir missing: %s — skipping", cls_dir)
            continue
        df_cls = discover_sos_pairs(cls_dir, include_classes=[cls_name])
        log.info("Class %s: %d scene pairs discovered", cls_name, len(df_cls))
        df_cls["_label_value"] = lbl
        all_dfs.append(df_cls)

    if not all_dfs:
        raise RuntimeError(f"No TIFF pairs found under {train_root}")

    full_pairs_df = pd.concat(all_dfs, ignore_index=True)
    log.info("Total scene pairs: %d", len(full_pairs_df))

    # ── Build feature DataFrames by class ────────────────────────────────────
    t0 = time.time()
    feature_dfs: list[pd.DataFrame] = []

    for cls_name, lbl in class_labels.items():
        subset = full_pairs_df[full_pairs_df["_label_value"] == lbl].reset_index(drop=True)
        if len(subset) == 0:
            continue
        log.info("Extracting features for class '%s' (%d scenes)...", cls_name, len(subset))
        scene_dicts = _build_scene_dicts(
            df_pairs      = subset,
            m1_checkpoint = args.m1_checkpoint,
            args          = args,
            device        = device,
            label_value   = lbl,
        )
        feat_df = build_feature_dataframe(scene_dicts, gsd_m=args.gsd_m)
        log.info("  → %d component rows for class '%s'", len(feat_df), cls_name)
        feature_dfs.append(feat_df)

    if not feature_dfs:
        raise RuntimeError("No features extracted. Check data directory and masks.")

    all_features = pd.concat(feature_dfs, ignore_index=True)
    log.info("Total feature rows: %d  (oil=%d, lookalike=%d)",
             len(all_features),
             int((all_features["label"] == 1).sum()),
             int((all_features["label"] == 0).sum()))

    # Save feature summary CSV
    feat_summary_path = metrics_dir / "feature_summary.csv"
    all_features.to_csv(feat_summary_path, index=False)
    log.info("Feature summary CSV: %s", feat_summary_path)

    # ── Train classifier ──────────────────────────────────────────────────────
    log.info("Training LookalikeClassifier (n_estimators=%d, n_folds=%d)...",
             args.n_estimators, args.n_folds)

    # Drop rows with no label (inference-mode scenes — shouldn't happen in training)
    train_df = all_features.dropna(subset=["label"]).copy()
    train_df["label"] = train_df["label"].astype(int)

    clf = LookalikeClassifier(
        n_estimators    = args.n_estimators,
        n_folds         = args.n_folds,
        random_state    = args.seed,
    )
    clf.fit(train_df, label_col="label", group_col="scene_id")

    elapsed = time.time() - t0
    log.info("Training complete in %.1f s", elapsed)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = ckpt_dir / "lookalike_rf.joblib"
    clf.save(model_path)

    # ── CV scores JSON ────────────────────────────────────────────────────────
    cv_json_path = metrics_dir / "cv_scores.json"
    if clf.cv_scores_ is not None:
        cv_records = clf.cv_scores_.to_dict(orient="records")
        cv_records.append({
            "mean_balanced_accuracy": float(clf.cv_scores_["balanced_accuracy"].mean()),
            "std_balanced_accuracy":  float(clf.cv_scores_["balanced_accuracy"].std()),
            "mean_auc":               float(clf.cv_scores_["auc"].mean()),
            "std_auc":                float(clf.cv_scores_["auc"].std()),
        })
        cv_json_path.write_text(json.dumps(cv_records, indent=2), encoding="utf-8")
        log.info("CV scores: %s", cv_json_path)

    # ── Feature importance plot ───────────────────────────────────────────────
    fi_df     = clf.feature_importance_df()
    fi_path   = metrics_dir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    _plot_feature_importance(fi_df, metrics_dir / "feature_importance.png")
    log.info("Feature importances:\n%s", fi_df.to_string(index=False))

    # ── Apply bilge filter on training data (sanity check metrics) ───────────
    log.info("Applying bilge filter to training data (sanity check)...")
    proba_df  = clf.predict_proba(train_df)
    train_aug = pd.concat([train_df.reset_index(drop=True), proba_df], axis=1)
    filter_result = apply_bilge_filter(
        train_aug,
        prob_col        = "prob_oil",
        min_elongation  = args.min_elongation,
        max_area_km2    = args.max_area_km2,
        night_boost     = args.night_boost,
        prob_threshold  = args.prob_threshold,
    )
    summary = summarise_detections(filter_result)
    summary_path = metrics_dir / "detection_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Detection summary (train):\n%s", summary.to_string(index=False))

    # ── train_metrics.csv (module-level summary row) ──────────────────────────
    metrics_csv = metrics_dir / "train_metrics.csv"
    n_bilge     = int(filter_result["bilge_candidate"].sum())
    n_geom_pass = len(filter_result)
    row = {
        "module":             "module2",
        "n_scenes":           len(full_pairs_df),
        "n_components":       len(all_features),
        "n_geom_pass":        n_geom_pass,
        "n_bilge_candidates": n_bilge,
        "mean_cv_bacc":       round(float(clf.cv_scores_["balanced_accuracy"].mean()), 4)
                              if clf.cv_scores_ is not None else float("nan"),
        "mean_cv_auc":        round(float(clf.cv_scores_["auc"].mean()), 4)
                              if clf.cv_scores_ is not None else float("nan"),
        "elapsed_s":          round(elapsed, 2),
        "timestamp":          datetime.now().isoformat(),
    }
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    log.info("Metrics CSV: %s", metrics_csv)

    # ── Run config JSON ───────────────────────────────────────────────────────
    run_config = {**vars(args), "elapsed_s": elapsed, "started": datetime.now().isoformat()}
    (metrics_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )

    # ── HF upload ────────────────────────────────────────────────────────────
    hf_uploader.upload_many(
        [
            model_path,
            cv_json_path,
            fi_path,
            metrics_dir / "feature_importance.png",
            metrics_csv,
            summary_path,
        ],
        label="module2",
    )

    log.info("=" * 60)
    log.info("Module 2 training complete.")
    log.info("  Model         : %s", model_path)
    log.info("  CV balanced_acc: %.4f ± %.4f",
             clf.cv_scores_["balanced_accuracy"].mean() if clf.cv_scores_ is not None else float("nan"),
             clf.cv_scores_["balanced_accuracy"].std()  if clf.cv_scores_ is not None else float("nan"))
    log.info("  CV AUC        : %.4f ± %.4f",
             clf.cv_scores_["auc"].mean() if clf.cv_scores_ is not None else float("nan"),
             clf.cv_scores_["auc"].std()  if clf.cv_scores_ is not None else float("nan"))
    log.info("  Bilge detections (train): %d / %d geometry-passing patches",
             n_bilge, n_geom_pass)
    log.info("  All results   : %s", results_dir)
    log.info("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module 2 — Look-alike Discriminator Training (Oil Spill Detection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data-root",   required=True,
                   help="Path to organized data/ directory (same layout as Module 1)")
    p.add_argument("--results-dir", default="results/module2",
                   help="Output directory for model, metrics, and plots")
    p.add_argument("--m1-checkpoint", default=None,
                   help="Module 1 .pt checkpoint for on-the-fly inference when masks "
                        "are unavailable. Optional if masks exist on disk.")

    # Feature extraction
    p.add_argument("--gsd-m",           type=float, default=10.0,
                   help="Ground sampling distance in metres (10.0 for S1 IW GRD)")
    p.add_argument("--min-component-px",type=int,   default=10,
                   help="Minimum component area in pixels (smaller = noise)")

    # Bilge filter
    p.add_argument("--min-elongation", type=float, default=DEFAULT_MIN_ELONGATION,
                   help="Minimum elongation ratio (>3:1 per synopsis)")
    p.add_argument("--max-area-km2",   type=float, default=DEFAULT_MAX_AREA_KM2,
                   help="Maximum patch area in km² (<50 per synopsis)")
    p.add_argument("--night-boost",    type=float, default=DEFAULT_NIGHT_BOOST,
                   help="Night-time probability prior boost (Liao et al. 2023)")
    p.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD,
                   help="RF probability threshold for bilge candidate classification")

    # Classifier
    p.add_argument("--n-folds",       type=int, default=5,
                   help="GroupKFold cross-validation folds (grouped by scene_id)")
    p.add_argument("--n-estimators",  type=int, default=200,
                   help="Random Forest ensemble size (synopsis: 200)")
    p.add_argument("--seed",          type=int, default=42)

    # HF Hub
    p.add_argument("--hf-repo-id", default="",
                   help="HF Hub dataset repo ID for auto-upload")
    p.add_argument("--hf-token",   default="",
                   help="HF Hub write access token (from Kaggle Secrets)")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
