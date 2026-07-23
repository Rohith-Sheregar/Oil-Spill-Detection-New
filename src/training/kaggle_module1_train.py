"""
Kaggle-ready Module 1 training entrypoint.

Example:
    python -m src.training.kaggle_module1_train \
        --data-root /kaggle/temp/sos_extracted \
        --output-dir /kaggle/working/module1_outputs \
        --epochs 5 --batch-size 8 --use-scse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.deeplab_scse import DeepLabV3PlusSCSE
from src.models.losses import BCEDiceLoss
from src.training.zenodo_sos_dataset import (
    PatchConfig,
    SOSTiffPatchDataset,
    balanced_scene_subset,
    discover_sos_pairs,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(in_channels: int, use_scse: bool, patch_size: int, encoder_weights: str | None):
    if use_scse:
        return DeepLabV3PlusSCSE(
            in_channels=in_channels,
            classes=1,
            input_size=patch_size,
            encoder_weights=encoder_weights,
        )
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
    )


def binary_segmentation_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    truth = targets > 0.5
    tp = torch.logical_and(preds, truth).sum().item()
    fp = torch.logical_and(preds, ~truth).sum().item()
    fn = torch.logical_and(~preds, truth).sum().item()
    intersection = tp
    union = torch.logical_or(preds, truth).sum().item()
    iou = intersection / union if union else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return iou, f1


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, train: bool):
    model.train(train)
    losses, ious, f1s = [], [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, desc="train" if train else "valid"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            iou, f1 = binary_segmentation_scores(logits.detach(), y.detach())
            losses.append(loss.item())
            if not np.isnan(iou):
                ious.append(iou)
            if not np.isnan(f1):
                f1s.append(f1)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
    }


def make_scene_split(df: pd.DataFrame, val_fraction: float, seed: int):
    if df["scene_id"].nunique() < 2:
        raise ValueError("Need at least two scenes for a scene-level train/validation split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["scene_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing extracted Zenodo TIFFs.")
    parser.add_argument("--output-dir", default="module1_outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--samples-per-scene", type=int, default=8)
    parser.add_argument("--max-scenes-per-class", type=int, default=120)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=["vv_vh", "vv_vh_diff"], default="vv_vh_diff")
    parser.add_argument("--use-scse", action="store_true", help="Enable the synopsis v1 scSE attention block.")
    parser.add_argument("--no-imagenet", action="store_true", help="Do not download ImageNet encoder weights.")
    parser.add_argument("--include-classes", nargs="+", default=["oil", "no_oil", "lookalike"])
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = discover_sos_pairs(args.data_root, include_classes=args.include_classes)
    if df.empty:
        raise FileNotFoundError(f"No TIFF image/mask pairs found under {args.data_root}")
    df = balanced_scene_subset(df, args.max_scenes_per_class, seed=args.seed)
    df.to_csv(output_dir / "discovered_scenes.csv", index=False)
    train_df, val_df = make_scene_split(df, args.val_fraction, args.seed)
    train_df.to_csv(output_dir / "train_scenes.csv", index=False)
    val_df.to_csv(output_dir / "val_scenes.csv", index=False)

    config = PatchConfig(
        patch_size=args.patch_size,
        input_mode=args.input_mode,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
        augment=True,
    )
    train_ds = SOSTiffPatchDataset(train_df, config=config, train=True)
    val_ds = SOSTiffPatchDataset(val_df, config=config, train=False)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.no_imagenet else "imagenet"
    try:
        model = build_model(train_ds.in_channels, args.use_scse, args.patch_size, encoder_weights)
    except Exception as exc:
        if encoder_weights is None:
            raise
        print(f"Could not initialize ImageNet weights ({exc}); retrying with encoder_weights=None")
        encoder_weights = None
        model = build_model(train_ds.in_channels, args.use_scse, args.patch_size, encoder_weights)
    model.to(device)

    loss_fn = BCEDiceLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run_config = vars(args) | {
        "in_channels": train_ds.in_channels,
        "device": str(device),
        "encoder_weights": encoder_weights,
        "n_train_scenes": int(len(train_df)),
        "n_val_scenes": int(len(val_df)),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_scores = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True)
        val_scores = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_scores.items()},
            **{f"val_{k}": v for k, v in val_scores.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
        torch.save({"model": model.state_dict(), "config": run_config, "epoch": epoch}, output_dir / "last_model.pt")
        if val_scores["miou"] > best_miou:
            best_miou = val_scores["miou"]
            torch.save({"model": model.state_dict(), "config": run_config, "epoch": epoch}, output_dir / "best_model.pt")
        print(row)

    print(f"Done. Best validation mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()

