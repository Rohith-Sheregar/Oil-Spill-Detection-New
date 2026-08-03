"""
Dataset helpers for the Zenodo SOS Sentinel-1 oil-spill TIFF archives.

Band modes (input_mode):
  "vv_vh"         — 2 bands:  VV, VH
  "vv_vh_diff"    — 3 bands:  VV, VH, VV-VH difference
  "vv_vh_h_alpha" — 4 bands:  VV, VH, Entropy H, Alpha angle  [synopsis v0]
  "full_5band"    — 5 bands:  VV, VH, H, Alpha, Wind-corrected ratio  [synopsis v1]

The 4- and 5-band modes use dual-pol Cloude-Pottier decomposition
(src/preprocessing/polsar_decomp) and CMOD5.N wind normalisation
(src/preprocessing/wind_ratio). Both work on the Zenodo Sigma0 dB TIFFs
without requiring raw .SAFE products.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import shutil
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


ZENODO_ARCHIVES = {
    "oil_images": {
        "record": "8346860",
        "filename": "01_Train_Val_Oil_Spill_images.7z",
        "url": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_images.7z/content",
        "size_gb": 37.92,
        "split": "train_val",
        "class_name": "oil",
    },
    "oil_masks": {
        "record": "8346860",
        "filename": "01_Train_Val_Oil_Spill_mask.7z",
        "url": "https://zenodo.org/api/records/8346860/files/01_Train_Val_Oil_Spill_mask.7z/content",
        "size_gb": 0.01,
        "split": "train_val",
        "class_name": "oil",
    },
    "no_oil_images": {
        "record": "8253899",
        "filename": "01_Train_Val_No_Oil_Images.7z",
        "url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_No_Oil_Images.7z/content",
        "size_gb": 21.36,
        "split": "train_val",
        "class_name": "no_oil",
    },
    "no_oil_masks": {
        "record": "8253899",
        "filename": "01_Train_Val_No_Oil_mask.7z",
        "url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_No_Oil_mask.7z/content",
        "size_gb": 0.001,
        "split": "train_val",
        "class_name": "no_oil",
    },
    "lookalike_images": {
        "record": "8253899",
        "filename": "01_Train_Val_Lookalike_images.7z",
        "url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_images.7z/content",
        "size_gb": 21.41,
        "split": "train_val",
        "class_name": "lookalike",
    },
    "lookalike_masks": {
        "record": "8253899",
        "filename": "01_Train_Val_Lookalike_mask.7z",
        "url": "https://zenodo.org/api/records/8253899/files/01_Train_Val_Lookalike_mask.7z/content",
        "size_gb": 0.001,
        "split": "train_val",
        "class_name": "lookalike",
    },
    "test_all": {
        "record": "13761290",
        "filename": "02_Test_images_and_ground_truth.7z",
        "url": "https://zenodo.org/api/records/13761290/files/02_Test_images_and_ground_truth.7z/content",
        "size_gb": 9.18,
        "split": "test",
        "class_name": "mixed",
    },
}

TIFF_EXTENSIONS = {".tif", ".tiff"}


def available_space_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def download_file(url: str, destination: str | Path, chunk_mb: int = 32) -> Path:
    """Download with resume support, useful for large Zenodo archives."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "oil-spill-ondevice-training/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with part.open(mode) as f, tqdm(
            total=total,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    part.replace(destination)
    return destination


def download_zenodo_archives(
    archive_keys: Iterable[str],
    archive_dir: str | Path,
    min_free_gb_after_download: float = 10.0,
) -> list[Path]:
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key in archive_keys:
        spec = ZENODO_ARCHIVES[key]
        destination = archive_dir / spec["filename"]
        if destination.exists():
            downloaded.append(destination)
            continue
        required = float(spec["size_gb"]) + min_free_gb_after_download
        free = available_space_gb(archive_dir)
        if free < required:
            raise RuntimeError(
                f"Not enough free space for {key}: need about {required:.1f} GB, "
                f"but {archive_dir} has {free:.1f} GB free."
            )
        downloaded.append(download_file(spec["url"], destination))
    return downloaded


def extract_7z(archive_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a 7z archive using system 7z when available, else py7zr."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if seven_zip:
        subprocess.run(
            [seven_zip, "x", str(archive_path), f"-o{output_dir}", "-y"],
            check=True,
        )
    else:
        import py7zr

        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=output_dir)
    return output_dir


def _numeric_id(path: Path) -> str:
    nums = re.findall(r"\d+", path.stem)
    return nums[-1].zfill(5) if nums else path.stem.lower()


def _class_from_path(path: Path) -> str:
    text = str(path).lower().replace("-", "_").replace(" ", "_")
    if "lookalike" in text or "look_alike" in text or "looklike" in text:
        return "lookalike"
    if "no_oil" in text or "nooil" in text or "non_oil" in text:
        return "no_oil"
    if "oil" in text:
        return "oil"
    return "unknown"


def _is_mask_path(path: Path) -> bool:
    text = str(path).lower()
    return "mask" in text or "ground_truth" in text or "groundtruth" in text


def discover_sos_pairs(root: str | Path, include_classes: Iterable[str] | None = None) -> pd.DataFrame:
    """Return image/mask pairs with scene-level IDs for leakage-free splits."""
    root = Path(root)
    include = set(include_classes) if include_classes else None
    masks = {}
    images = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in TIFF_EXTENSIONS:
            continue
        class_name = _class_from_path(path)
        if include and class_name not in include:
            continue
        key = (class_name, _numeric_id(path))
        if _is_mask_path(path):
            masks[key] = path
        else:
            images.append((key, path))

    rows = []
    for (class_name, num), image_path in sorted(images, key=lambda x: (x[0][0], x[0][1], str(x[1]))):
        mask_path = masks.get((class_name, num))
        rows.append(
            {
                "scene_id": f"{class_name}_{num}",
                "class_name": class_name,
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path else "",
                "has_mask": bool(mask_path),
                "label_is_oil": class_name == "oil",
            }
        )
    return pd.DataFrame(rows)


def balanced_scene_subset(df: pd.DataFrame, max_scenes_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    if not max_scenes_per_class:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby("class_name"):
        parts.append(group.sample(n=min(max_scenes_per_class, len(group)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _as_hwc(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D TIFF array, got shape {array.shape}")
    if array.shape[0] <= 8 and array.shape[1] > 32 and array.shape[2] > 32:
        return np.moveaxis(array, 0, -1)
    return array


def read_image_channels(
    path: str | Path,
    input_mode: str = "vv_vh_diff",
    wind_speed_ms: float = 7.0,
    incidence_deg: float = 35.0,
    wind_dir_deg:  float = 0.0,
) -> np.ndarray:
    """Read a TIFF and return C,H,W float32 channels.

    Band layout for full_5band (synopsis-compliant):
      Band 0: VV_norm   — Sigma0 VV dB, robust-percentile normalised to [0, 1]
      Band 1: VH_norm   — Sigma0 VH dB, robust-percentile normalised to [0, 1]
      Band 2: H         — Cloude-Pottier entropy [0, 1] (from linear VV/VH)
      Band 3: alpha_norm — Cloude-Pottier alpha angle / 90, in [0, 1]
      Band 4: wind_ratio — CMOD5.N wind-corrected VV/VH ratio, in [0, 1]

    Supported input_mode values:
      "vv_vh"         -> 2 bands (VV dB, VH dB, un-normalised)
      "vv_vh_diff"    -> 3 bands (VV dB, VH dB, VV-VH dB difference)
      "vv_vh_h_alpha" -> 4 bands (VV_norm, VH_norm, H, alpha_norm)
      "full_5band"    -> 5 bands — synopsis Band 1-5 (default)
    """
    hwc = _as_hwc(tifffile.imread(path)).astype(np.float32)
    if hwc.shape[-1] < 2:
        hwc = np.repeat(hwc, 2, axis=-1)
    vv = hwc[..., 0]   # Sigma0 VV in dB  (Zenodo TIFFs are pre-calibrated dB)
    vh = hwc[..., 1]   # Sigma0 VH in dB

    if input_mode == "vv_vh":
        return np.stack([vv, vh], axis=0)

    if input_mode == "vv_vh_diff":
        return np.stack([vv, vh, vv - vh], axis=0)

    if input_mode in ("vv_vh_h_alpha", "full_5band"):
        # ── Step 1: dB → linear power (required by Cloude-Pottier) ─────────
        from src.preprocessing.polsar_decomp import db_to_linear, dual_pol_entropy_alpha
        vv_lin = db_to_linear(vv)    # σ₀_VV in linear power
        vh_lin = db_to_linear(vh)    # σ₀_VH in linear power

        # ── Step 2: Cloude-Pottier H and α ───────────────────────────────────
        # dual_pol_entropy_alpha() expects LINEAR-scale inputs (not dB).
        # Returns H ∈ [0, 1] and alpha ∈ [0°, 90°] (dual-pol range ×2 convention).
        # This matches exactly what best_model.pt (epoch 8) was trained with.
        band_H, band_alpha = dual_pol_entropy_alpha(vv_lin, vh_lin)
        # Normalise alpha to [0, 1] by dividing by 90° (training convention)
        band_a = (band_alpha / 90.0).astype(np.float32)

        # ── Step 3: robust percentile-stretch VV and VH (dB → [0, 1]) ────
        def _pstretch(arr: np.ndarray) -> np.ndarray:
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                return np.zeros_like(arr, dtype=np.float32)
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if hi <= lo:
                return np.clip(arr - lo, 0.0, None).astype(np.float32)
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        vv_norm = _pstretch(vv)
        vh_norm = _pstretch(vh)

        if input_mode == "vv_vh_h_alpha":
            return np.stack([vv_norm, vh_norm, band_H, band_a], axis=0)

        # ── Step 4: Band 4 — CMOD5.N wind-corrected ratio ─────────────────
        # compute_wind_corrected_ratio() handles its own dB→linear internally
        from src.preprocessing.wind_ratio import compute_wind_corrected_ratio
        band_w = compute_wind_corrected_ratio(
            vv, vh,
            wind_speed_ms=wind_speed_ms,
            incidence_deg=incidence_deg,
            wind_dir_deg=wind_dir_deg,
        )

        # ── Stack: [VV_norm, VH_norm, H, alpha_norm, wind_ratio] ─────────
        return np.stack([vv_norm, vh_norm, band_H, band_a, band_w], axis=0)

    raise ValueError(f"Unsupported input_mode={input_mode!r}")


def read_mask(path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape_hw, dtype=np.float32)
    if isinstance(path, float) and np.isnan(path):
        return np.zeros(shape_hw, dtype=np.float32)
    if str(path).strip() == "" or str(path).lower() == "nan":
        return np.zeros(shape_hw, dtype=np.float32)
    arr = _as_hwc(tifffile.imread(path))[..., 0]
    return (arr > 0).astype(np.float32)


def robust_normalize(chw: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    out = np.empty_like(chw, dtype=np.float32)
    for i, band in enumerate(chw):
        finite = np.isfinite(band)
        if not finite.any():
            out[i] = 0
            continue
        lo, hi = np.percentile(band[finite], [low, high])
        if hi <= lo:
            out[i] = np.nan_to_num(band, nan=lo) - lo
            continue
        clipped = np.clip(np.nan_to_num(band, nan=lo), lo, hi)
        out[i] = (clipped - lo) / (hi - lo)
    return out


def _stable_seed(text: str, sample_index: int, seed: int) -> int:
    raw = f"{seed}:{sample_index}:{text}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _pad_to_patch(chw: np.ndarray, mask: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h or pad_w:
        chw = np.pad(chw, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
    return chw, mask


def _choose_crop(mask: np.ndarray, patch_size: int, rng: np.random.Generator, positive_prob: float) -> tuple[int, int]:
    h, w = mask.shape
    max_y = max(0, h - patch_size)
    max_x = max(0, w - patch_size)
    positives = np.argwhere(mask > 0)
    if len(positives) and rng.random() < positive_prob:
        cy, cx = positives[rng.integers(0, len(positives))]
        y = int(np.clip(cy - patch_size // 2, 0, max_y))
        x = int(np.clip(cx - patch_size // 2, 0, max_x))
        return y, x
    return int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1))


def _augment(chw: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """SAR-specific augmentation pipeline (synopsis-compliant).

    Applies: horizontal flip, vertical flip, 90/180/270 rotation,
    Gaussian noise injection, Gaussian blur, and histogram equalization.
    Augmentations are applied only to the image (chw), not the mask,
    except for geometric transforms which are applied jointly.
    """
    # ── Geometric transforms (applied jointly to image + mask) ─────────
    if rng.random() < 0.5:
        chw  = chw[:, :, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        chw  = chw[:, ::-1, :]
        mask = mask[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        chw  = np.rot90(chw, k, axes=(1, 2))
        mask = np.rot90(mask, k)

    # ── Photometric transforms (image only) ────────────────────────────
    # Gaussian noise injection
    if rng.random() < 0.25:
        sigma_noise = float(rng.uniform(0.005, 0.025))
        noise = rng.normal(0, sigma_noise, size=chw.shape).astype(np.float32)
        chw   = np.clip(chw + noise, 0.0, 1.0)

    # Gaussian blur (per-band spatial smoothing, SAR speckle simulation)
    if rng.random() < 0.25:
        try:
            from scipy.ndimage import gaussian_filter
            sigma_blur = float(rng.uniform(0.5, 1.5))
            blurred    = np.stack(
                [gaussian_filter(chw[i], sigma=sigma_blur) for i in range(chw.shape[0])],
                axis=0,
            ).astype(np.float32)
            chw = np.clip(blurred, 0.0, 1.0)
        except ImportError:
            pass   # scipy optional; skip blur if unavailable

    # Histogram equalization (contrast normalisation, per-band)
    if rng.random() < 0.20:
        try:
            from skimage.exposure import equalize_hist
            chw = np.stack(
                [equalize_hist(chw[i]).astype(np.float32) for i in range(chw.shape[0])],
                axis=0,
            )
        except ImportError:
            pass   # scikit-image optional; skip if unavailable

    return np.ascontiguousarray(chw), np.ascontiguousarray(mask)


@dataclass
class PatchConfig:
    patch_size:         int   = 256
    input_mode:         str   = "full_5band"   # synopsis v1 default
    samples_per_scene:  int   = 8
    positive_crop_prob: float = 0.70
    normalize:          bool  = True
    augment:            bool  = True
    seed:               int   = 42
    # Wind parameters for Band 5 (used when input_mode="full_5band")
    wind_speed_ms:      float = 7.0    # climatological default; override with ERA5
    incidence_deg:      float = 35.0   # IW swath mean; override with scene metadata
    wind_dir_deg:       float = 0.0    # upwind direction (no azimuth data in GRD)


class SOSTiffPatchDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config: PatchConfig, train: bool):
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.train = train
        self.samples_per_scene = max(1, int(config.samples_per_scene))

    def __len__(self) -> int:
        return len(self.df) * self.samples_per_scene

    @property
    def in_channels(self) -> int:
        return {
            "vv_vh":         2,
            "vv_vh_diff":    3,
            "vv_vh_h_alpha": 4,
            "full_5band":    5,
        }.get(self.config.input_mode, 2)

    def __getitem__(self, index: int):
        row_index = index // self.samples_per_scene
        sample_index = index % self.samples_per_scene
        row = self.df.iloc[row_index]
        image = read_image_channels(
            row.image_path,
            self.config.input_mode,
            wind_speed_ms=self.config.wind_speed_ms,
            incidence_deg=self.config.incidence_deg,
            wind_dir_deg=self.config.wind_dir_deg,
        )
        if self.config.normalize:
            image = robust_normalize(image)
        mask = read_mask(row.mask_path, image.shape[1:])
        image, mask = _pad_to_patch(image, mask, self.config.patch_size)

        if self.train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(_stable_seed(row.scene_id, sample_index, self.config.seed))
        y, x = _choose_crop(mask, self.config.patch_size, rng, self.config.positive_crop_prob)
        ps = self.config.patch_size
        image = image[:, y : y + ps, x : x + ps]
        mask = mask[y : y + ps, x : x + ps]

        if self.train and self.config.augment:
            image, mask = _augment(image, mask, rng)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "scene_id": row.scene_id,
            "class_name": row.class_name,
        }
