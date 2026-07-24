"""
5-band stacking pipeline for Module 1.

Band layout (synopsis-compliant):
  Band 0  VV amplitude          (Sigma0 dB, from Zenodo TIFF channel 0)
  Band 1  VH amplitude          (Sigma0 dB, from Zenodo TIFF channel 1)
  Band 2  Entropy H             (Cloude-Pottier dual-pol approx)
  Band 3  Alpha angle alpha     (Cloude-Pottier dual-pol approx, degrees / 90)
  Band 4  Wind-corrected ratio  (VV/VH normalised by CMOD5.N)

All bands are normalised to [0, 1] before stacking so each band has equal
dynamic range going into the model.

Fallback modes
--------------
If ERA5 wind speed is unavailable, Band 4 uses the CMOD5.N
model with a climatological default wind speed (7 m/s). This is less accurate
but produces a calibrated ratio that still conveys wind-normalised contrast.

The Zenodo TIFFs are 2048×2048×2 (VV, VH). This module processes them into
2048×2048×5 float32 stacks that the DeepLabV3+/scSE model ingests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from src.preprocessing.polsar_decomp import db_to_linear, dual_pol_entropy_alpha
from src.preprocessing.wind_ratio import compute_wind_corrected_ratio


# ─── normalisation helpers ────────────────────────────────────────────────────

def _robust_norm(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Percentile-stretch a single 2D band to [0, 1], NaN-safe."""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_lo, p_hi])
    if hi <= lo:
        return np.clip(arr - lo, 0.0, None).astype(np.float32)
    clipped = np.clip(np.nan_to_num(arr, nan=lo), lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


# ─── public interface ─────────────────────────────────────────────────────────

def build_5band_stack(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    wind_speed_ms: float | np.ndarray = 7.0,
    incidence_deg: float | np.ndarray = 35.0,
    wind_dir_deg:  float | np.ndarray = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute and return the full 5-band feature stack as (5, H, W) float32.

    Parameters
    ----------
    vv_db         : (H, W) float32 — Sigma0 VV in dB
    vh_db         : (H, W) float32 — Sigma0 VH in dB
    wind_speed_ms : ERA5 10-m neutral wind speed, m/s (scalar or H×W grid)
    incidence_deg : SAR incidence angle, degrees (scalar or H×W grid)
    wind_dir_deg  : wind direction rel. to look, degrees
    normalize     : if True, robust percentile-stretch each band to [0, 1]

    Returns
    -------
    stack : (5, H, W) float32 array
    """
    vv_db = np.asarray(vv_db, dtype=np.float32)
    vh_db = np.asarray(vh_db, dtype=np.float32)

    # ── Band 0+1: VV and VH ───────────────────────────────────────────────
    band_vv = vv_db.copy()
    band_vh = vh_db.copy()

    # ── Band 2+3: H and alpha (Cloude-Pottier) ────────────────────────────
    # Convert dB → linear before calling dual_pol_entropy_alpha()
    vv_lin  = db_to_linear(vv_db)
    vh_lin  = db_to_linear(vh_db)
    band_H, band_alpha_deg = dual_pol_entropy_alpha(vv_lin, vh_lin)
    band_alpha = band_alpha_deg / 90.0   # normalise [0°, 90°] → [0, 1]

    # ── Band 4: wind-corrected ratio ──────────────────────────────────────
    band_wind = compute_wind_corrected_ratio(
        vv_db, vh_db,
        wind_speed_ms=wind_speed_ms,
        incidence_deg=incidence_deg,
        wind_dir_deg=wind_dir_deg,
    )

    bands = [band_vv, band_vh, band_H, band_alpha, band_wind]

    if normalize:
        # H, alpha, wind_ratio already in [0,1]; normalise VV/VH dB bands
        bands[0] = _robust_norm(bands[0])
        bands[1] = _robust_norm(bands[1])
        # H and alpha are already [0, 1] — re-clip for safety
        bands[2] = np.clip(bands[2], 0.0, 1.0)
        bands[3] = np.clip(bands[3], 0.0, 1.0)
        bands[4] = np.clip(bands[4], 0.0, 1.0)

    return np.stack(bands, axis=0).astype(np.float32)   # (5, H, W)


def build_5band_from_tiff(
    tiff_path: str | Path,
    wind_speed_ms: float = 7.0,
    incidence_deg: float = 35.0,
    wind_dir_deg:  float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load a Zenodo-style 2048×2048×2 (or ×1) TIFF and return the 5-band stack.

    This is the primary entry point for the dataset loader.
    """
    arr = tifffile.imread(str(tiff_path)).astype(np.float32)

    # Normalise axis ordering to (H, W, C)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[1] > 8:
        arr = np.moveaxis(arr, 0, -1)   # (C, H, W) → (H, W, C)

    # Zenodo TIFFs are (H, W, 2) — ch0=VV, ch1=VH
    if arr.shape[-1] < 2:
        arr = np.concatenate([arr, arr], axis=-1)   # single-band fallback

    vv_db = arr[..., 0]
    vh_db = arr[..., 1]

    return build_5band_stack(
        vv_db, vh_db,
        wind_speed_ms=wind_speed_ms,
        incidence_deg=incidence_deg,
        wind_dir_deg=wind_dir_deg,
        normalize=normalize,
    )


BAND_NAMES = ["VV (dB)", "VH (dB)", "Entropy H", "Alpha (norm)", "Wind-ratio"]


# ─── smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    h, w = 256, 256
    vv = rng.uniform(-25, -5,  (h, w)).astype(np.float32)
    vh = rng.uniform(-30, -10, (h, w)).astype(np.float32)

    stack = build_5band_stack(vv, vh, wind_speed_ms=8.0)
    print("Stack shape:", stack.shape)
    for i, name in enumerate(BAND_NAMES):
        b = stack[i]
        print(f"  Band {i} {name:<20} min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}")
    assert stack.shape == (5, h, w), "Shape mismatch"
    assert stack.min() >= 0.0 and stack.max() <= 1.0, "Values out of [0, 1]"
    print("All smoke tests passed.")
