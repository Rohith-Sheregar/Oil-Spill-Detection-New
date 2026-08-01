"""
Module 1 — On-device training entrypoint.

Trains DeepLabV3+/MobileNetV2/scSE on the Zenodo SOS dataset with a
5-band feature stack (VV, VH, Entropy H, Alpha, Wind-corrected ratio).

Usage (from project root):
    python -m src.training.train_module1 --data-root /path/to/data

Arguments:
    --data-root     Path to the organized data/ directory from the download notebook.
                    Must contain train/oil/images, train/oil/masks, etc.
    --results-dir   Where to save checkpoints, metrics, and the HTML report.
                    Default: results/module1/
    --input-mode    Band mode: vv_vh | vv_vh_diff | vv_vh_h_alpha | full_5band
                    Default: full_5band
    --epochs        Total training epochs to run in THIS session. Default: 50
    --patch-size    Crop size for patches. Default: 256
    --batch-size    Override auto-probed batch size. Default: auto
    --lr            Initial learning rate. Default: 1e-3
    --no-pseudo     Disable pseudo-labelling self-evolution cycles.
    --pseudo-cycles Max pseudo-label cycles. Default: 5 (resource-friendly on-device default)
    --num-workers   DataLoader workers. Default: 2
    --seed          Random seed. Default: 42
    --resume        Path to a checkpoint (.pt) to resume training from.
                    Restores model weights, optimizer, scheduler, best_val_loss,
                    and starts from (saved_epoch + 1). Use this to continue
                    training across multiple Kaggle sessions.
                    Example: --resume /kaggle/input/oil-spill-checkpoints/last_model.pt
    --gdrive-folder-id   Google Drive folder ID to auto-upload checkpoints.
                    Upload happens after every best_model.pt save and every 5 epochs.
                    Get the ID from the Drive folder URL:
                    https://drive.google.com/drive/folders/<THIS_PART>
    --gdrive-credentials Path to service account JSON file for Drive authentication.
                    Write credentials to this file from Kaggle Secrets in Cell 1.
                    Example: --gdrive-credentials /kaggle/working/gdrive_sa.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make project root importable regardless of working directory
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.deeplab_scse import DeepLabV3PlusSCSE
from src.models.losses import BCEDiceLoss
from src.training.zenodo_sos_dataset import (
    PatchConfig,
    SOSTiffPatchDataset,
    discover_sos_pairs,
    balanced_scene_subset,
)
from src.training.splits import scene_level_split
from src.training.gpu_utils import probe_max_batch_size
from src.validation.metrics import compute_miou, compute_pixel_f1
from src.reporting.module1_report import generate_module1_report


# ─── Hugging Face Hub auto-uploader ──────────────────────────────────────────────

class HfUploader:
    """
    Silently uploads checkpoint / metrics files to Hugging Face Hub after every
    best-epoch save. Designed for headless Kaggle Save & Run sessions.

    If authentication fails or any upload errors, a warning is logged and
    training continues uninterrupted.
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
            # Verify token and repo exist (or create repo)
            self._api.create_repo(repo_id=repo_id, exist_ok=True, private=True)
            self.enabled = True
            self._log.info("🤗 Hugging Face uploader ready — repo: %s", repo_id)
        except Exception as exc:
            self._log.warning(
                "HfUploader: init failed (%s). HF uploads disabled.", exc
            )

    # ------------------------------------------------------------------ #
    def upload(self, local_path: str | Path, label: str = "") -> bool:
        """Upload *local_path* to HF Hub. Returns True on success."""
        if not self.enabled or self._api is None:
            return False
        try:
            local_path = Path(local_path)
            name       = local_path.name
            size_mb    = local_path.stat().st_size / (1024 ** 2)

            self._api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=name,
                repo_id=self.repo_id,
                commit_message=f"Auto-save {name} {label}".strip()
            )

            suffix = f"  [{label}]" if label else ""
            self._log.info(
                "  🤗 HF Hub uploaded: %s  (%.1f MB)%s", name, size_mb, suffix
            )
            return True
        except Exception as exc:
            self._log.warning("  HF upload failed for %s: %s", local_path.name, exc)
            return False

    # ------------------------------------------------------------------ #
    def upload_many(self, paths: list, label: str = "") -> None:
        """Upload a list of files; continue even if individual uploads fail."""
        for p in paths:
            if Path(p).exists():
                self.upload(p, label=label)


# ─── Logging setup ───────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"module1_train_{stamp}.log"
    fmt       = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"
    handlers  = [logging.FileHandler(log_path, encoding="utf-8"),
                 logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logging.info("Log file: %s", log_path)


log = logging.getLogger(__name__)


# ─── Dataset building ─────────────────────────────────────────────────────────

def _build_dataframes(data_root: Path, seed: int):
    """
    Discover all image/mask pairs, perform scene-level split, return
    (train_df, val_df, unlabelled_df, metadata).
    """
    data_root = Path(data_root)
    train_root = data_root / "train"
    test_root  = data_root / "test"

    if not train_root.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_root}\n"
            "Run dataset download/prep script first to populate data/."
        )

    all_dfs = []
    class_counts = {}

    for cls in ["oil", "lookalike", "no_oil"]:
        cls_dir = train_root / cls
        if not cls_dir.exists():
            log.warning("Class directory missing: %s — skipping", cls_dir)
            continue
        df = discover_sos_pairs(cls_dir, include_classes=[cls])
        class_counts[cls] = len(df)
        all_dfs.append(df)
        log.info("Class %s: %d scene pairs discovered", cls, len(df))

    if not all_dfs:
        raise RuntimeError(f"No TIFF pairs found under {train_root}")

    import pandas as pd
    full_df = pd.concat(all_dfs, ignore_index=True)
    log.info("Total scenes: %d", len(full_df))

    # Scene-level split (anti-leakage: GroupShuffleSplit on scene_id)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(gss.split(full_df, groups=full_df["scene_id"]))

    # Oil scenes for supervised training only; negatives go to pseudo-label pool
    oil_mask   = full_df["class_name"] == "oil"
    train_oil  = full_df.iloc[train_idx][oil_mask].reset_index(drop=True)
    val_oil    = full_df.iloc[val_idx][oil_mask].reset_index(drop=True)
    unlabelled = full_df[full_df["class_name"].isin(["lookalike", "no_oil"])].reset_index(drop=True)

    # For val, use the full val split (oil + negatives) to compute realistic class balance metrics
    val_all = full_df.iloc[val_idx].reset_index(drop=True)

    log.info("Train oil scenes : %d", len(train_oil))
    log.info("Val scenes       : %d", len(val_all))
    log.info("Unlabelled pool  : %d (pseudo-label candidates)", len(unlabelled))

    metadata = {
        "n_train":      len(train_oil),
        "n_val":        len(val_all),
        "n_unlabelled": len(unlabelled),
        "class_counts": class_counts,
        "seed":         seed,
    }
    return train_oil, val_all, unlabelled, metadata


# ─── Validation loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_miou, all_f1 = [], []
    cm = np.zeros((2, 2), dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        probs    = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        gt       = masks.squeeze(1).cpu().numpy().astype(int)
        pred_bin = (probs >= 0.5).astype(int)

        for i in range(len(images)):
            miou = compute_miou(pred_bin[i], gt[i])
            f1   = compute_pixel_f1(pred_bin[i], gt[i])
            if not np.isnan(miou):
                all_miou.append(miou)
            if not np.isnan(f1):
                all_f1.append(f1)
            # Accumulate confusion matrix
            tp = int(np.logical_and(pred_bin[i] == 1, gt[i] == 1).sum())
            tn = int(np.logical_and(pred_bin[i] == 0, gt[i] == 0).sum())
            fp = int(np.logical_and(pred_bin[i] == 1, gt[i] == 0).sum())
            fn = int(np.logical_and(pred_bin[i] == 0, gt[i] == 1).sum())
            cm[0, 0] += tn; cm[0, 1] += fp
            cm[1, 0] += fn; cm[1, 1] += tp

    n = max(len(loader), 1)
    return {
        "val_loss"  : total_loss / n,
        "val_miou"  : float(np.mean(all_miou)) if all_miou else float("nan"),
        "val_f1"    : float(np.mean(all_f1))   if all_f1   else float("nan"),
        "cm"        : cm,
    }


# ─── Training entrypoint ──────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    data_root   = Path(args.data_root)
    results_dir = Path(args.results_dir)
    ckpt_dir    = results_dir / "checkpoints"
    metrics_dir = results_dir / "metrics"
    log_dir     = results_dir / "logs"

    for d in [ckpt_dir, metrics_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1024**3)
    else:
        log.warning(
            "⚠️  No CUDA GPU detected — training on CPU will be very slow!\n"
            "    On Kaggle: Settings → Accelerator → GPU T4 x1, then restart."
        )

    # ── Build datasets ────────────────────────────────────────────────────
    train_df, val_df, unlabelled_df, metadata = _build_dataframes(data_root, args.seed)

    cfg = PatchConfig(
        patch_size         = args.patch_size,
        input_mode         = args.input_mode,
        samples_per_scene  = args.samples_per_scene,
        positive_crop_prob = 0.70,
        normalize          = True,
        augment            = True,
        seed               = args.seed,
    )
    metadata["input_mode"] = cfg.input_mode
    metadata["patch_size"] = cfg.patch_size

    train_ds = SOSTiffPatchDataset(train_df, cfg, train=True)
    val_ds   = SOSTiffPatchDataset(val_df,   cfg, train=False)
    in_ch    = train_ds.in_channels
    log.info("in_channels=%d  train samples=%d  val samples=%d",
             in_ch, len(train_ds), len(val_ds))

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = DeepLabV3PlusSCSE(in_channels=in_ch, classes=1, input_size=args.patch_size)
    model.to(device)
    log.info("Model: DeepLabV3+/MobileNetV2+scSE  params=%d",
             sum(p.numel() for p in model.parameters()))

    criterion = BCEDiceLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    scaler = (torch.cuda.amp.GradScaler()
              if device.type == "cuda" and args.use_amp else None)

    # ── Hugging Face uploader (headless auto-save for Save & Run sessions) ──
    hf_uploader = HfUploader(
        repo_id = getattr(args, "hf_repo_id", "") or "",
        token   = getattr(args, "hf_token", "") or "",
    )

    # ── Resume from checkpoint (multi-session support) ────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            log.error("--resume path not found: %s", resume_path)
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        log.info("▶ Resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "val_loss" in ckpt:
            best_val_loss = ckpt["val_loss"]
        resumed_epoch = ckpt.get("epoch", 0)
        start_epoch   = resumed_epoch + 1
        log.info(
            "  Resumed: epoch=%d  best_val_loss=%.4f  "
            "will train epochs %d → %d",
            resumed_epoch, best_val_loss,
            start_epoch, start_epoch + args.epochs - 1,
        )

    # Probe batch size
    if args.batch_size:
        batch_size = args.batch_size
    else:
        def _make_batch(bs):
            return torch.zeros(bs, in_ch, args.patch_size, args.patch_size)
        try:
            batch_size = probe_max_batch_size(model, _make_batch, start_bs=16, device=device)
        except Exception:
            batch_size = 4
    log.info("Batch size: %d", batch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, batch_size // 2),
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True)

    # ── Save run config ───────────────────────────────────────────────────
    run_config = {**vars(args), **metadata,
                  "batch_size": batch_size,
                  "device": str(device),
                  "started": datetime.now().isoformat()}
    (metrics_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )

    # ── Metrics CSV setup ─────────────────────────────────────────────────
    csv_path = metrics_dir / "train_metrics.csv"
    csv_mode = "a" if (args.resume and csv_path.exists()) else "w"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "train_loss", "val_loss", "val_miou", "val_f1",
                    "val_precision", "val_recall", "lr", "epoch_s"],
    )
    if csv_mode == "w":  # only write header for new files
        csv_writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_f1": [], "lr": []}
    # NOTE: best_val_loss and start_epoch are set above (either fresh or from --resume)
    samples_for_report = []

    end_epoch = start_epoch + args.epochs - 1
    log.info("Training epochs %d → %d", start_epoch, end_epoch)
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        step_lr = []

        for batch in train_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            optimizer.zero_grad()

            if scaler is not None:
                from torch.cuda.amp import autocast
                with autocast():
                    logits = model(images)
                    loss   = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss   = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            step_lr.append(optimizer.param_groups[0]["lr"])

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics    = _validate(model, val_loader, criterion, device)
        epoch_s        = time.time() - t0
        current_lr     = float(np.mean(step_lr))

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "mIoU=%.4f  F1=%.4f  lr=%.2e  t=%.1fs",
            epoch, end_epoch,
            avg_train_loss, val_metrics["val_loss"],
            val_metrics["val_miou"], val_metrics["val_f1"],
            current_lr, epoch_s,
        )

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_miou"].append(val_metrics["val_miou"])
        history["val_f1"].append(val_metrics["val_f1"])
        history["lr"].extend(step_lr)

        # CSV row
        cm = val_metrics["cm"]
        tp = int(cm[1, 1]); fp = int(cm[0, 1])
        fn = int(cm[1, 0]); tn = int(cm[0, 0])
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        csv_writer.writerow({
            "epoch"        : epoch,
            "train_loss"   : round(avg_train_loss, 6),
            "val_loss"     : round(val_metrics["val_loss"], 6),
            "val_miou"     : round(val_metrics["val_miou"], 6),
            "val_f1"       : round(val_metrics["val_f1"], 6),
            "val_precision": round(precision, 6),
            "val_recall"   : round(recall, 6),
            "lr"           : f"{current_lr:.6e}",
            "epoch_s"      : round(epoch_s, 2),
        })
        csv_file.flush()

        # Collect sample predictions for report (first epoch only)
        if epoch == start_epoch and not samples_for_report:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                imgs_cpu = sample_batch["image"].cpu().numpy()
                masks_cpu = sample_batch["mask"].squeeze(1).cpu().numpy()
                logits_cpu = torch.sigmoid(
                    model(sample_batch["image"].to(device))
                ).squeeze(1).cpu().numpy()
            for i in range(min(4, len(imgs_cpu))):
                samples_for_report.append({
                    "image": imgs_cpu[i],
                    "mask" : masks_cpu[i],
                    "pred" : logits_cpu[i],
                })

        # Save best checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "val_loss"    : best_val_loss,
                "val_miou"    : val_metrics["val_miou"],
                "config"      : run_config,
            }, ckpt_dir / "best_model.pt")
            log.info("  ★ New best model saved (epoch=%d  val_loss=%.4f)",
                     epoch, best_val_loss)
            # ── AUTO-UPLOAD best to HF immediately after save ──
            hf_uploader.upload(
                ckpt_dir / "best_model.pt",
                label=f"epoch={epoch} val_loss={best_val_loss:.4f}",
            )

        # Save last checkpoint every 5 epochs (always resumable)
        if epoch % 5 == 0 or epoch == end_epoch:
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "scheduler"   : scheduler.state_dict(),
                "val_loss"    : best_val_loss,
            }, ckpt_dir / "last_model.pt")
            log.info("  💾 last_model.pt saved at epoch %d", epoch)
            # ── AUTO-UPLOAD last + metrics CSV to HF every 5 epochs ──
            hf_uploader.upload_many(
                [
                    ckpt_dir / "last_model.pt",
                    metrics_dir / "train_metrics.csv",
                ],
                label=f"epoch={epoch}",
            )

    csv_file.close()
    log.info("Training complete. Best val_loss=%.4f", best_val_loss)

    # Save split scene IDs
    import pandas as pd
    train_df[["scene_id", "class_name", "image_path"]].to_csv(
        metrics_dir / "train_scenes.csv", index=False)
    val_df[["scene_id", "class_name", "image_path"]].to_csv(
        metrics_dir / "val_scenes.csv", index=False)

    # ── Pseudo-labelling ──────────────────────────────────────────────────
    pseudo_history = None
    if not args.no_pseudo and len(unlabelled_df) > 0 and not args.skip_pseudo_on_resume:
        log.info("Starting pseudo-label self-evolution (%d cycles)...", args.pseudo_cycles)
        # Load best checkpoint before pseudo-labelling
        best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        from src.training.pseudo_label_trainer import run_pseudo_label_cycles
        pseudo_history = run_pseudo_label_cycles(
            model          = model,
            supervised_df  = train_df,
            unlabelled_df  = unlabelled_df,
            config         = cfg,
            device         = device,
            results_dir    = results_dir,
            n_cycles       = args.pseudo_cycles,
            epochs_per_cycle = 3,
            lr             = args.lr * 0.1,    # lower LR for fine-tune
            num_workers    = args.num_workers,
        )
        log.info("Pseudo-labelling done. %d cycles completed.", len(pseudo_history))

    # ── Generate HTML report ──────────────────────────────────────────────
    log.info("Generating HTML report...")
    final_cm = None
    if "cm" in val_metrics:
        final_cm = val_metrics["cm"]
    report_path = generate_module1_report(
        results_dir    = results_dir,
        history        = history,
        metadata       = metadata,
        samples        = samples_for_report,
        cm             = final_cm,
        pseudo_history = pseudo_history,
    )
    log.info("HTML report: %s", report_path)
    log.info("Done. All results saved to: %s", results_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module 1 on-device training — Oil Spill Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root",   required=True,
                   help="Path to organized data/ directory (from download notebook)")
    p.add_argument("--results-dir", default="results/module1",
                   help="Directory for checkpoints, metrics, and report")
    p.add_argument("--input-mode",  default="full_5band",
                   choices=["vv_vh", "vv_vh_diff", "vv_vh_h_alpha", "full_5band"],
                   help="Feature band mode")
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--patch-size",       type=int,   default=256)
    p.add_argument("--batch-size",       type=int,   default=None,
                   help="Override auto-probed batch size")
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--samples-per-scene",type=int,   default=8,
                   help="Patches sampled from each scene per epoch")
    p.add_argument("--no-pseudo",        action="store_true",
                   help="Disable pseudo-labelling (faster, for quick runs)")
    p.add_argument("--pseudo-cycles",    type=int,   default=5,
                   help="Max pseudo-label self-evolution cycles")
    p.add_argument("--num-workers",      type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--no-amp",           action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a .pt checkpoint to resume training from. "
            "Restores model, optimizer, scheduler, and best_val_loss. "
            "Training continues from (checkpoint_epoch + 1). "
            "Use 'last_model.pt' to continue a session, or "
            "'best_model.pt' to fine-tune from the best weights."
        ),
    )
    p.add_argument(
        "--skip-pseudo-on-resume",
        action="store_true",
        help="Skip pseudo-labelling when resuming mid-training (use on intermediate sessions).",
    )
    p.add_argument(
        "--hf-repo-id",
        default="",
        metavar="REPO_ID",
        help=(
            "Hugging Face dataset repo ID (e.g., 'username/oil-spill-checkpoints') "
            "for automatic checkpoint uploads."
        ),
    )
    p.add_argument(
        "--hf-token",
        default="",
        metavar="TOKEN",
        help=(
            "Hugging Face write access token. "
            "Usually fetched from Kaggle Secrets."
        ),
    )
    args = p.parse_args()
    args.use_amp = not args.no_amp
    return args


if __name__ == "__main__":
    train(_parse_args())
