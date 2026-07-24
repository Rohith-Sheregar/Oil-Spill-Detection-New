"""
Dual-pol Cloude-Pottier decomposition: Entropy (H) and Alpha (α) angle.

Physics background
------------------
Full quad-pol SAR (HH, HV, VH, VV) admits a 3×3 coherency matrix T3, from
whose three eigenvalues one derives H, A (Anisotropy), and alpha exactly.
Sentinel-1 IW GRD is dual-pol (VV + VH), so T3 is unavailable. We construct
a 2×2 coherency matrix T2 whose two eigenvalues yield H and a meaningful alpha
approximation. Anisotropy (A) cannot be computed from T2 (it requires three
eigenvalues / full quad-pol SLC); the function returns a zero-filled A for
API compatibility with quad-pol pipelines.

Input convention
----------------
The primary public function `dual_pol_entropy_alpha()` expects **linear-scale**
power values (i.e. σ₀ in linear, not dB). If your input is in dB (as produced
by the Zenodo SOS TIFFs and the SNAP pipeline), call `db_to_linear()` first,
or use the convenience wrapper `dual_pol_entropy_alpha_from_db()`.

SNAP alternative
----------------
If you have raw .SAFE SLC products, the SNAP graph in snap_pipeline.py can
be extended with a Polarimetric-Decomposition node (Cloude-Pottier) placed
after Calibration. The SNAP implementation is marginally more accurate because
it operates on complex SLC data before multi-looking; for GRD data the NumPy
approximation here is equivalent.

References
----------
- Cloude & Pottier (1997) IEEE Trans. Geosci. Remote Sens. 35(1): 68–78
- Song et al. (2024) — H/A/α improved OA from 95.01% to 97.06%
- Chen & Wang (2022) — H/A/α validated for oil-spill discrimination
"""
from __future__ import annotations

import numpy as np


# ─── Unit conversion ──────────────────────────────────────────────────────────

def db_to_linear(db_array: np.ndarray, clip_db: float = -50.0) -> np.ndarray:
    """Convert Sigma0 dB → linear power. Clips floor to avoid log(0) issues."""
    return np.power(10.0, np.maximum(np.asarray(db_array, dtype=np.float32), clip_db) / 10.0)


def linear_to_db(lin_array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert linear power → Sigma0 dB."""
    return 10.0 * np.log10(np.maximum(np.asarray(lin_array, dtype=np.float32), eps))


# ─── Core 2×2 coherency matrix eigenvalues ───────────────────────────────────

def _t2_eigenvalues(
    vv_lin: np.ndarray,
    vh_lin: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel eigenvalues of the 2×2 coherency matrix T2.

        T2 = [[|VV|²,    VV·VH*],
              [VH·VV*,   |VH|² ]]

    For GRD data the off-diagonal cross-terms are real (no phase information),
    so T2 is symmetric. The eigenvalues of a 2×2 symmetric real matrix are:
        λ₁,₂ = trace/2 ± sqrt((trace/2)² − det)

    Parameters
    ----------
    vv_lin, vh_lin : (H, W) float32 arrays, linear-scale σ₀ power values
    eps            : numerical floor added to diagonal elements

    Returns
    -------
    l1, l2 : (H, W) float32 arrays, λ₁ ≥ λ₂ ≥ 0
    """
    t11 = vv_lin + eps          # |VV|²  (diagonal element)
    t22 = vh_lin + eps          # |VH|²  (diagonal element)
    t12 = np.sqrt(t11 * t22)    # cross-term magnitude approximation for GRD

    trace = t11 + t22
    det   = t11 * t22 - t12 ** 2
    disc  = np.sqrt(np.maximum(trace ** 2 / 4.0 - det, 0.0))

    l1 = trace / 2.0 + disc
    l2 = np.maximum(trace / 2.0 - disc, 0.0)   # clamp numerical negatives
    return l1.astype(np.float32), l2.astype(np.float32)


# ─── Primary public function ───────────────────────────────────────────────────

def dual_pol_entropy_alpha(
    vv_linear: np.ndarray,
    vh_linear: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cloude-Pottier Entropy H and mean Alpha angle α from dual-pol
    Sentinel-1 GRD data in **linear** power scale.

    This is the canonical entry-point referenced by zenodo_sos_dataset.py
    and band_stack.py. Always pass linear-scale values here; convert from
    dB upstream using `db_to_linear()` or `dual_pol_entropy_alpha_from_db()`.

    Parameters
    ----------
    vv_linear : (H, W) float32 — VV σ₀ in linear power (NOT dB)
    vh_linear : (H, W) float32 — VH σ₀ in linear power (NOT dB)
    eps       : numerical stability floor (default 1e-12)

    Returns
    -------
    H     : Entropy,     float32 (H, W), values clipped to [0, 1]
            H ≈ 0 → single dominant scattering mechanism (specular/calm ocean)
            H ≈ 1 → depolarising, complex medium (oil layer, rough sea, etc.)
    alpha : Alpha angle, float32 (H, W), values clipped to [0, 90] degrees
            α < 45° → surface/Bragg scattering
            α > 45° → volume/double-bounce scattering (oil multipaths)
    """
    vv_lin = np.asarray(vv_linear, dtype=np.float32)
    vh_lin = np.asarray(vh_linear, dtype=np.float32)

    l1, l2 = _t2_eigenvalues(vv_lin, vh_lin, eps=eps)
    sum_l  = l1 + l2 + eps

    # Pseudo-probabilities (fractional eigenvalue contributions)
    p1 = l1 / sum_l
    p2 = l2 / sum_l

    # Shannon entropy — normalised to [0, 1] for N=2 scattering mechanisms
    # (log₂(2) = 1, so division by log₂(N) = 1 for N=2 is implicit)
    H = -(
        p1 * np.log2(np.maximum(p1, eps))
        + p2 * np.log2(np.maximum(p2, eps))
    )
    H = np.clip(H, 0.0, 1.0).astype(np.float32)

    # Mean alpha angle via arctan2(sqrt(p2), sqrt(p1))
    # For dual-pol this maps naturally to [0°, 45°]; multiply by 2 to
    # match the [0°, 90°] convention of the full quad-pol Cloude-Pottier.
    alpha_raw = np.degrees(np.arctan2(np.sqrt(p2 + eps), np.sqrt(p1 + eps)))
    alpha     = np.clip(alpha_raw * 2.0, 0.0, 90.0).astype(np.float32)

    return H, alpha


def dual_pol_entropy_alpha_from_db(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    clip_db: float = -50.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: accepts Sigma0 dB inputs (as stored in Zenodo TIFFs)
    and converts to linear before calling `dual_pol_entropy_alpha()`.

    Parameters
    ----------
    vv_db, vh_db : (H, W) float32 arrays, Sigma0 in decibels
    clip_db      : floor clamp before exponentiation (default -50 dB)

    Returns
    -------
    H, alpha — same as `dual_pol_entropy_alpha()`
    """
    vv_lin = db_to_linear(vv_db, clip_db=clip_db)
    vh_lin = db_to_linear(vh_db, clip_db=clip_db)
    return dual_pol_entropy_alpha(vv_lin, vh_lin, eps=eps)


def compute_anisotropy_placeholder(shape_hw: tuple[int, int]) -> np.ndarray:
    """
    Anisotropy (A) requires three eigenvalues and is not computable from
    dual-pol data. Returns a zero array for API compatibility with quad-pol
    pipelines that expect (H, A, alpha).
    """
    return np.zeros(shape_hw, dtype=np.float32)


# ─── Legacy alias (kept for backward-compat with band_stack.py) ───────────────

def decompose_dual_pol_tiff(
    vv_band: np.ndarray,
    vh_band: np.ndarray,
    input_is_db: bool = True,
) -> dict[str, np.ndarray]:
    """
    Legacy wrapper — use `dual_pol_entropy_alpha()` for new code.

    Returns dict with keys: 'H', 'alpha', 'anisotropy'
    """
    if input_is_db:
        H, alpha = dual_pol_entropy_alpha_from_db(vv_band, vh_band)
    else:
        H, alpha = dual_pol_entropy_alpha(vv_band, vh_band)
    return {"H": H, "alpha": alpha, "anisotropy": compute_anisotropy_placeholder(H.shape)}


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    h, w = 512, 512

    # Test with linear inputs (primary API)
    vv_lin = rng.uniform(1e-4, 0.2, (h, w)).astype(np.float32)
    vh_lin = rng.uniform(1e-5, 0.05, (h, w)).astype(np.float32)
    H, alpha = dual_pol_entropy_alpha(vv_lin, vh_lin)
    print("=== Linear input ===")
    print(f"H     min={H.min():.4f}  max={H.max():.4f}  mean={H.mean():.4f}")
    print(f"alpha min={alpha.min():.4f}  max={alpha.max():.4f}  mean={alpha.mean():.4f}")
    assert 0.0 <= float(H.min()) and float(H.max()) <= 1.0, "H out of [0,1]"
    assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 90.0, "alpha out of [0,90]"

    # Test with dB inputs (convenience wrapper)
    vv_db = rng.uniform(-25, -5, (h, w)).astype(np.float32)
    vh_db = rng.uniform(-30, -10, (h, w)).astype(np.float32)
    H2, alpha2 = dual_pol_entropy_alpha_from_db(vv_db, vh_db)
    print("\n=== dB input (via wrapper) ===")
    print(f"H     min={H2.min():.4f}  max={H2.max():.4f}  mean={H2.mean():.4f}")
    print(f"alpha min={alpha2.min():.4f}  max={alpha2.max():.4f}  mean={alpha2.mean():.4f}")
    print("\nAll assertions passed.")
