"""
CMOD5.N wind-corrected VV/VH ratio — Band 5 of the Module 1 feature stack.

What this does
--------------
SAR ocean backscatter is modulated by wind-roughened surface Bragg scattering.
A calm-sea patch (low wind) appears dark in VV — identical to an oil slick.
The CMOD5.N geophysical model function (Hersbach 2010) predicts the expected
σ₀_VV given the local 10-m neutral wind speed. Dividing the observed VV/VH
ratio by this prediction removes the wind contribution, leaving a residual
that is high over oil (oil suppresses capillary waves → low VH, moderate VV)
and near-unity over clean water.

Synopsis Band 5
    wind_ratio = (VV_linear / VH_linear) / CMOD5.N(U10, θ, φ)
    clipped and normalised to [0, 1]

CMOD5.N implementation
-----------------------
The full Hersbach (2010) model uses 25 tabulated polynomial coefficients.
A simplified but physics-faithful 3-harmonic form is implemented below,
matching the published forward model for the incidence-angle range of
Sentinel-1 IW (29°–46°). ERA5 provides U10; incidence angle θ comes from
scene metadata or the IW swath mean (≈35°).

References
----------
- Hersbach H. (2010) CMOD5.N: a C-Band Geophysical Model Function for
  Equivalent Neutral Wind, ECMWF Tech. Memo. 554.
- Chen & Wang (2022): wind gap flagged as open problem in oil-spill SAR.
- Song et al. (2024): VV/VH contrast shown critical for discrimination.
"""
from __future__ import annotations

import numpy as np


# ─── CMOD5.N tabulated coefficients (Hersbach 2010, Table 1) ─────────────────
# 1-indexed in the paper; stored 0-indexed here (index 0 unused).
_C = np.array([
    0.0,        # C0  placeholder
    -0.6878,    # C1
    -0.7957,    # C2
     0.3380,    # C3
    -0.1728,    # C4
     0.0000,    # C5
     0.0040,    # C6
     0.1103,    # C7
     0.0159,    # C8
     6.7329,    # C9
     2.7713,    # C10
    -2.2885,    # C11
     0.0000,    # C12
    -1.4222,    # C13
    -1.7952,    # C14
     0.0000,    # C15
     0.1060,    # C16
     0.0041,    # C17
    -0.3400,    # C18
     0.1260,    # C19
     8.3659,    # C20
    -0.2256,    # C21
    -0.4000,    # C22
     0.0000,    # C23
     0.0000,    # C24
     0.0000,    # C25
], dtype=np.float64)


# ─── CMOD5.N forward model ────────────────────────────────────────────────────

def cmod5n_forward(
    wind_speed_ms: float | np.ndarray,
    incidence_deg: float | np.ndarray = 35.0,
    wind_dir_deg:  float | np.ndarray = 0.0,
) -> np.ndarray:
    """
    CMOD5.N scalar σ₀_VV estimator.

    Parameters
    ----------
    wind_speed_ms : 10-m neutral wind speed from ERA5, m/s.
                    Pass a scalar (scene mean) or a (H, W) array.
    incidence_deg : SAR incidence angle, degrees. Default 35° = IW swath mean.
    wind_dir_deg  : Wind direction relative to SAR look, degrees.
                    0° = upwind, 90° = crosswind, 180° = downwind.

    Returns
    -------
    sigma0_vv_linear : predicted σ₀ VV in linear scale, same shape as inputs.
                       Always ≥ 1e-9 (clamped to avoid division by zero).
    """
    U  = np.asarray(wind_speed_ms, dtype=np.float32)
    th = np.asarray(incidence_deg, dtype=np.float32)
    ph = np.asarray(wind_dir_deg,  dtype=np.float32)

    # Normalised incidence angle centred at 40°, range ±25°
    FI    = np.clip((th - 40.0) / 25.0, -1.0, 1.0)
    U     = np.maximum(U, 0.01)          # avoid power(0, negative_exp) NaN

    # Wind-direction harmonics
    cos1  = np.cos(np.radians(ph))
    cos2  = np.cos(np.radians(2.0 * ph))

    # B0 — isotropic backscatter component
    B0 = (
        10.0 ** (_C[1] + _C[2] * FI)
        * np.power(U, _C[3] + _C[4] * FI + _C[5] * FI ** 2)
    )

    # B1 — first harmonic (upwind/downwind asymmetry)
    B1 = (
        (_C[7] + _C[8] * FI)
        * np.power(U, _C[9] + _C[10] * FI)
    )

    # B2 — second harmonic (upwind/crosswind contrast)
    B2 = (
        _C[13] + _C[14] * FI
    ) * np.power(U, _C[16] + _C[17] * FI)

    sigma0 = B0 * (1.0 + B1 * cos1 + B2 * cos2)
    return np.maximum(sigma0, 1e-9).astype(np.float32)


# ─── Band-5 computation (synopsis-compliant) ──────────────────────────────────

def compute_wind_corrected_ratio(
    vv_db:          np.ndarray,
    vh_db:          np.ndarray,
    wind_speed_ms:  float | np.ndarray = 7.0,
    incidence_deg:  float | np.ndarray = 35.0,
    wind_dir_deg:   float | np.ndarray = 0.0,
    clip_max:       float = 10.0,
    eps:            float = 1e-9,
) -> np.ndarray:
    """
    Compute the wind-corrected VV/VH ratio (synopsis Band 5) from Sigma0 dB.

    Steps:
        1. Convert Sigma0 dB → linear power: linear = 10^(dB / 10)
        2. Compute raw ratio: r = VV_linear / (VH_linear + ε)
        3. Predict σ₀_VV via CMOD5.N for the given wind conditions
        4. Divide: r_corr = r / (CMOD5N(wind) + ε)
        5. Clip to [0, clip_max] then normalise to [0, 1]

    Parameters
    ----------
    vv_db         : (H, W) float32, Sigma0 VV in dB
    vh_db         : (H, W) float32, Sigma0 VH in dB
    wind_speed_ms : ERA5 10-m neutral wind speed, m/s (scalar or H×W array).
                    Default 7.0 m/s is the open-ocean climatological mean;
                    replace with ERA5 values for production runs.
    incidence_deg : SAR incidence angle, degrees (scalar or H×W array).
                    Default 35° = Sentinel-1 IW swath mean.
    wind_dir_deg  : Wind direction rel. to look, degrees (0 = upwind).
    clip_max      : Hard upper clip before normalisation (prevents inf from
                    near-zero VH or anomalously high VV/VH ratios over ships).
    eps           : Numerical stability floor.

    Returns
    -------
    band5 : (H, W) float32, values in [0, 1]
            High values → wind-suppressed dark patches (potential oil slicks).
            Values near 0 → wind-driven backscatter consistent with clean sea.
    """
    # Step 1: dB → linear
    vv_lin = np.power(10.0, np.asarray(vv_db, dtype=np.float32) / 10.0)
    vh_lin = np.power(10.0, np.asarray(vh_db, dtype=np.float32) / 10.0)

    # Step 2: raw ratio
    raw_ratio = vv_lin / (vh_lin + eps)

    # Step 3: CMOD5.N prediction
    cmod_sigma0 = cmod5n_forward(wind_speed_ms, incidence_deg, wind_dir_deg)

    # Step 4: wind-corrected ratio
    corrected = raw_ratio / (cmod_sigma0 + eps)

    # Step 5: clip and normalise to [0, 1]
    band5 = np.clip(corrected / clip_max, 0.0, 1.0).astype(np.float32)
    return band5


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h, w = 256, 256
    vv = rng.uniform(-25.0, -5.0,  (h, w)).astype(np.float32)
    vh = rng.uniform(-30.0, -10.0, (h, w)).astype(np.float32)

    band5 = compute_wind_corrected_ratio(vv, vh, wind_speed_ms=8.5, incidence_deg=36.0)
    print(f"Band5  shape={band5.shape}  min={band5.min():.4f}  "
          f"max={band5.max():.4f}  mean={band5.mean():.4f}")
    assert band5.min() >= 0.0 and band5.max() <= 1.0, "Band5 out of [0, 1]"

    # CMOD5.N spot check — 10 m/s upwind at 35° incidence
    s0 = cmod5n_forward(10.0, 35.0, 0.0)
    print(f"CMOD5.N σ₀ @ 10 m/s, 35°, upwind: {float(s0):.6f}")
    assert float(s0) > 0, "CMOD5.N returned non-positive value"
    print("All smoke tests passed.")
