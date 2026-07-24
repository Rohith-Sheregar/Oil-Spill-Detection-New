"""
Self-evolving pseudo-label training loop — Module 1, synopsis Section H.1.4.

Methodology (Li et al., 2023)
------------------------------
After supervised training on the labelled Zenodo SOS dataset, the model is
run in inference mode on the unlabelled negative scenes (lookalike + no-oil).
A scene is accepted into the pseudo-label pool if and only if >80% of its
pixels are HIGH-CONFIDENCE: sigmoid score > 0.85 (confident oil) or < 0.15
(confident background). This dual-threshold scheme ensures the pseudo-label
is reliable for BOTH the positive AND negative classes.

The binary pseudo-mask is then assigned as:
  - Pixels with sigmoid > 0.85  →  1  (pseudo oil)
  - Pixels with sigmoid < 0.15  →  0  (pseudo background)
  - Pixels between 0.15–0.85   →  masked out (ignored in loss)

The model is fine-tuned for exactly 1 epoch on the expanded dataset per cycle,
then a new checkpoint is saved. This repeats for up to 10 cycles (synopsis
maximum), allowing the model to self-improve without manual annotation.

References
----------
- Li et al. (2023) — self-evolving data generation algorithm (Section H.1.4)
- Synopsis Section 1.4: "iterating for up to 10 cycles to continuously improve
  model performance without manual annotation"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.zenodo_sos_dataset import (
    PatchConfig,
    SOSTiffPatchDataset,
    read_image_channels,
    robust_normalize,
)
from src.models.losses import BCEDiceLoss

log = logging.getLogger(__name__)


# ─── Synopsis-specified constants ─────────────────────────────────────────────

MAX_CYCLES           = 10     # synopsis maximum self-evolution iterations
CONF_HIGH            = 0.85   # sigmoid > 0.85  → confident oil  (pseudo-positive)
CONF_LOW             = 0.15   # sigmoid < 0.15  → confident background (pseudo-negative)
MIN_CONFIDENT_FRAC   = 0.80   # >80 % of pixels must be high-confidence to accept scene
EPOCHS_PER_CYCLE     = 1      # 1 epoch per cycle (synopsis: Li et al. 2023)


# ─── Patch-based tiled inference ─────────────────────────────────────────────

class _InferenceDataset(Dataset):
    """Tile a full SAR image into overlapping patches for dense prediction."""

    def __init__(self, chw: np.ndarray, patch_size: int = 256, stride: int = 192) -> None:
        self.chw        = chw
        self.patch_size = patch_size
        self.stride     = stride
        h, w = chw.shape[1], chw.shape[2]
        self.positions  = [
            (y, x)
            for y in range(0, max(1, h - patch_size + 1), stride)
            for x in range(0, max(1, w - patch_size + 1), stride)
        ]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        y, x = self.positions[idx]
        ps   = self.patch_size
        crop = self.chw[:, y : y + ps, x : x + ps]
        # zero-pad at borders if the tile is smaller than patch_size
        if crop.shape[1] < ps or crop.shape[2] < ps:
            pad_h = max(0, ps - crop.shape[1])
            pad_w = max(0, ps - crop.shape[2])
            crop  = np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        return torch.from_numpy(crop.astype(np.float32)), torch.tensor([y, x])


def _predict_full_image(
    model:      torch.nn.Module,
    chw:        np.ndarray,
    device:     torch.device,
    patch_size: int = 256,
    stride:     int = 192,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Tiled dense inference → (H, W) sigmoid confidence map in [0, 1].
    Overlapping tiles are averaged.
    """
    _, H, W  = chw.shape
    acc      = np.zeros((H, W), dtype=np.float32)
    cnt      = np.zeros((H, W), dtype=np.float32)
    ds       = _InferenceDataset(chw, patch_size=patch_size, stride=stride)
    loader   = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    with torch.no_grad():
        for patches, coords in loader:
            patches = patches.to(device)
            logits  = model(patches)
            probs   = torch.sigmoid(logits).squeeze(1).cpu().numpy()   # (B, ps, ps)
            for b_i, (y, x) in enumerate(coords):
                y, x   = int(y), int(x)
                tile_h = min(patch_size, H - y)
                tile_w = min(patch_size, W - x)
                acc[y : y + tile_h, x : x + tile_w] += probs[b_i, :tile_h, :tile_w]
                cnt[y : y + tile_h, x : x + tile_w] += 1.0

    return acc / np.maximum(cnt, 1.0)


# ─── Pseudo-label generation ───────────────────────────────────────────────────

def generate_pseudo_labels(
    model:          torch.nn.Module,
    unlabelled_df:  pd.DataFrame,
    config:         PatchConfig,
    device:         torch.device,
    pseudo_dir:     Path,
    patch_size:     int = 256,
    stride:         int = 192,
) -> pd.DataFrame:
    """
    Run inference on all rows in unlabelled_df. Accept a scene if and only if
    >80% of its pixels are high-confidence (sigmoid >0.85 or <0.15).

    For accepted scenes, save:
      - binary pseudo-mask:  0/1 float32 (uncertain pixels get value = -1,
                             meaning they are excluded from training loss)
      - confidence map:      raw sigmoid scores (for debugging/inspection)

    Parameters
    ----------
    model         : trained DeepLabV3+/scSE model in eval mode
    unlabelled_df : DataFrame with columns image_path, scene_id, class_name
    config        : PatchConfig (input_mode, normalize, wind params)
    device        : torch device
    pseudo_dir    : directory to write .npz files into
    patch_size    : tile size for inference
    stride        : stride between overlapping tiles

    Returns
    -------
    DataFrame of accepted scenes with mask_path pointing to the .npz file,
    plus a column 'pseudo'=True for identification in the training loop.
    """
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    total    = len(unlabelled_df)

    for i, (_, row) in enumerate(unlabelled_df.iterrows()):
        scene_id = row.scene_id
        npz_path = pseudo_dir / f"{scene_id}.npz"
        log.info("[%d/%d] Inferring scene %s ...", i + 1, total, scene_id)

        # ── Load image as feature stack ────────────────────────────────────
        try:
            chw = read_image_channels(
                row.image_path,
                config.input_mode,
                wind_speed_ms=config.wind_speed_ms,
                incidence_deg=config.incidence_deg,
                wind_dir_deg =config.wind_dir_deg,
            )
        except Exception as exc:
            log.warning("Scene %s: read failed (%s), skipping.", scene_id, exc)
            continue

        if config.normalize:
            chw = robust_normalize(chw)

        # ── Dense inference ────────────────────────────────────────────────
        conf_map = _predict_full_image(
            model, chw, device, patch_size=patch_size, stride=stride, batch_size=4
        )

        # ── Synopsis acceptance criterion ──────────────────────────────────
        # A pixel is 'high-confidence' if sigmoid > 0.85 (oil) or < 0.15 (bg)
        high_conf_mask = (conf_map > CONF_HIGH) | (conf_map < CONF_LOW)
        conf_frac      = float(high_conf_mask.mean())

        if conf_frac < MIN_CONFIDENT_FRAC:
            log.debug(
                "Scene %s rejected: high-conf fraction=%.3f < %.2f",
                scene_id, conf_frac, MIN_CONFIDENT_FRAC,
            )
            continue

        # ── Build pseudo-mask ──────────────────────────────────────────────
        # Confident oil pixels → 1.0
        # Confident background → 0.0
        # Uncertain pixels     → -1.0  (excluded from loss in __getitem__)
        pseudo_mask = np.full_like(conf_map, fill_value=-1.0, dtype=np.float32)
        pseudo_mask[conf_map > CONF_HIGH] = 1.0
        pseudo_mask[conf_map < CONF_LOW]  = 0.0

        np.savez_compressed(npz_path, mask=pseudo_mask, conf=conf_map)
        accepted.append({**row.to_dict(), "mask_path": str(npz_path), "pseudo": True})
        log.info(
            "Scene %s accepted: conf_frac=%.3f  pos=%.4f  neg=%.4f",
            scene_id, conf_frac,
            float((pseudo_mask == 1.0).mean()),
            float((pseudo_mask == 0.0).mean()),
        )

    log.info("Pseudo-label generation done: %d / %d scenes accepted.", len(accepted), total)
    return pd.DataFrame(accepted)


# ─── Training helpers ──────────────────────────────────────────────────────────

def _train_one_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device:    torch.device,
    scaler=None,
) -> float:
    """Run one training epoch, returning average loss."""
    model.train()
    running_loss = 0.0
    n_batches    = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        # Ignore uncertain pseudo-pixels (mask == -1) in the loss
        valid = masks >= 0.0
        if not valid.any():
            continue

        optimizer.zero_grad()

        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(images)
                loss   = criterion(logits[valid], masks[valid])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits[valid], masks[valid])
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches    += 1

    return running_loss / max(n_batches, 1)


# ─── Main public entry point ───────────────────────────────────────────────────

def run_pseudo_label_cycle(
    model:          torch.nn.Module,
    labeled_df:     pd.DataFrame,
    unlabeled_df:   pd.DataFrame,
    loss_fn:        torch.nn.Module,
    optimizer:      torch.optim.Optimizer,
    device:         torch.device,
    config:         PatchConfig,
    results_dir:    Path,
    max_cycles:     int = MAX_CYCLES,
    num_workers:    int = 2,
    use_amp:        bool = True,
) -> list[dict]:
    """
    Self-evolving pseudo-label loop (synopsis Section H.1.4, Li et al. 2023).

    Algorithm per cycle:
        a. Run dense inference on all unlabelled SAR scenes.
        b. Identify scenes where >80% of pixels have sigmoid >0.85 or <0.15.
        c. Save confident binary predictions as pseudo-masks (.npz).
        d. Add pseudo-labelled scenes to the labelled training DataFrame.
        e. Retrain the DeepLabV3+ model for 1 epoch on the expanded dataset.
        f. Save checkpoint `pseudo_cycle_{cycle:02d}.pt`.

    Parameters
    ----------
    model         : DeepLabV3PlusSCSE model (already supervised-trained)
    labeled_df    : DataFrame of labelled scenes (image_path, mask_path, scene_id, class_name)
    unlabeled_df  : DataFrame of unlabelled scenes (lookalike, no_oil — no masks)
    loss_fn       : BCEDiceLoss instance
    optimizer     : torch.optim.Optimizer already constructed for model
    device        : torch device (cpu or cuda)
    config        : PatchConfig for the dataset loader
    results_dir   : Base results directory (checkpoints/ and pseudo_labels/ created inside)
    max_cycles    : Maximum self-evolution cycles (synopsis: 10)
    num_workers   : DataLoader workers
    use_amp       : Enable torch.cuda.amp on CUDA

    Returns
    -------
    history : list of per-cycle metric dicts
    """
    ckpt_dir   = Path(results_dir) / "checkpoints"
    pseudo_dir = Path(results_dir) / "pseudo_labels"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler  = (
        torch.cuda.amp.GradScaler()
        if use_amp and device.type == "cuda"
        else None
    )
    history = []

    for cycle in range(1, max_cycles + 1):
        log.info("=" * 60)
        log.info("Pseudo-label cycle %d / %d", cycle, max_cycles)
        log.info("=" * 60)
        t0 = time.time()

        # ── a+b+c: infer + filter + save pseudo-labels ────────────────────
        cycle_pseudo_dir = pseudo_dir / f"cycle_{cycle:02d}"
        pseudo_df = generate_pseudo_labels(
            model=model,
            unlabelled_df=unlabeled_df,
            config=config,
            device=device,
            pseudo_dir=cycle_pseudo_dir,
        )

        if pseudo_df.empty:
            log.warning(
                "Cycle %d: no scenes met the acceptance criterion "
                "(>%.0f%% high-confidence pixels). Stopping early.",
                cycle, MIN_CONFIDENT_FRAC * 100,
            )
            break

        # ── d: expand training set ─────────────────────────────────────────
        combined_df = pd.concat([labeled_df, pseudo_df], ignore_index=True)
        combined_ds = SOSTiffPatchDataset(combined_df, config, train=True)

        # Auto-pick safe batch size (start at 8, halve on OOM)
        batch_size = 8
        while batch_size > 1:
            try:
                dummy = torch.zeros(batch_size, config.in_channels if hasattr(config, "in_channels") else 5,
                                    config.patch_size, config.patch_size, device=device)
                with torch.no_grad():
                    model(dummy)
                del dummy
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                batch_size //= 2

        loader = DataLoader(
            combined_ds,
            batch_size=max(1, batch_size),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        # ── e: retrain for exactly 1 epoch (synopsis spec) ────────────────
        epoch_loss = _train_one_epoch(model, loader, optimizer, loss_fn, device, scaler)
        log.info("Cycle %d — 1-epoch fine-tune loss: %.4f", cycle, epoch_loss)

        # ── f: save checkpoint ────────────────────────────────────────────
        ckpt_path = ckpt_dir / f"pseudo_cycle_{cycle:02d}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "cycle":       cycle,
                "epoch_loss":  epoch_loss,
                "n_pseudo":    len(pseudo_df),
            },
            ckpt_path,
        )
        log.info("Checkpoint saved: %s", ckpt_path)

        # ── Record metrics ─────────────────────────────────────────────────
        record = {
            "cycle":           cycle,
            "n_pseudo_scenes": len(pseudo_df),
            "epoch_loss":      round(epoch_loss, 6),
            "elapsed_s":       round(time.time() - t0, 1),
        }
        history.append(record)
        log.info("Cycle %d summary: %s", cycle, record)

    # Persist history to disk
    hist_path = Path(results_dir) / "pseudo_label_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Pseudo-label history: %s", hist_path)
    return history


# ─── Backward-compat alias ───────────────────────────────────────────────────

def run_pseudo_label_cycles(
    model:          torch.nn.Module,
    supervised_df:  pd.DataFrame,
    unlabelled_df:  pd.DataFrame,
    config:         PatchConfig,
    device:         torch.device,
    results_dir:    Path,
    n_cycles:       int = MAX_CYCLES,
    lr:             float = 1e-4,
    num_workers:    int = 2,
    use_amp:        bool = True,
    **_kwargs,                    # absorb any deprecated kwargs silently
) -> list[dict]:
    """
    Backward-compatible alias for `run_pseudo_label_cycle()`.
    Called by train_module1.py.
    """
    criterion = BCEDiceLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return run_pseudo_label_cycle(
        model         = model,
        labeled_df    = supervised_df,
        unlabeled_df  = unlabelled_df,
        loss_fn       = criterion,
        optimizer     = optimizer,
        device        = device,
        config        = config,
        results_dir   = results_dir,
        max_cycles    = n_cycles,
        num_workers   = num_workers,
        use_amp       = use_amp,
    )
