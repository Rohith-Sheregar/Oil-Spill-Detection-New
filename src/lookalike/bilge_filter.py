"""
Module 2 — Bilge-Dump Post-Processing Filter.

Applies operational gating logic on top of the Random Forest classifier output
to enforce the morphological and temporal constraints that define bilge-dump
signatures:

    1. Elongation  > 3:1   (narrow streak vs. large slick blob)
    2. Area        < 50 km² (bilge dumps are small; large slicks = blowouts)
    3. Night-time weighting  (>80% of illegal discharges occur at night,
                              per Liao et al. 2023; boost posterior probability)

These constraints are applied as a hard gate + soft weight:
  - Patches failing either geometric threshold are REMOVED (label = rejected).
  - Patches passing geometry get their oil probability boosted by night_boost
    if is_night == 1. The boosted probability can be re-thresholded by the
    caller (default 0.5).

Design intent
-------------
This filter runs AFTER the RF classifier, not instead of it. The RF's
learned probability is used as the base score; this module applies the
physics-based priors on top. A patch could have very high RF probability but
still be filtered if it is a 200 km² blob (clearly not a bilge dump).

References
----------
- Chang et al. (2024): bilge-dump morphological signature (narrow streak)
- Liao et al. (2023): night-time weighting for illegal discharge detection
- Synopsis Section 2.2: "elongation > 3:1 AND area < 50 km²"
- Synopsis Section 2.2: "night-time weighting: >80% of illegal discharges
  occur at night"
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── Default thresholds (synopsis 2.2) ────────────────────────────────────────
DEFAULT_MIN_ELONGATION: float = 3.0    # strictly greater than 3:1
DEFAULT_MAX_AREA_KM2:   float = 50.0   # strictly less than 50 km²
DEFAULT_NIGHT_BOOST:    float = 0.15   # additive prior for night-time detections
DEFAULT_PROB_THRESHOLD: float = 0.50   # classification threshold after boosting


# ─── Night-time probability weighting ─────────────────────────────────────────

def night_time_weight(
    base_prob: float | np.ndarray | pd.Series,
    is_night:  int   | np.ndarray | pd.Series,
    night_boost: float = DEFAULT_NIGHT_BOOST,
) -> np.ndarray:
    """
    Adjust the RF oil probability with a night-time additive prior.

    Illegal bilge dumping is strongly night-time biased (>80% of discharges
    occur between 20:00 and 06:00 local time, Liao et al. 2023). A flat
    additive boost avoids multiplying a near-zero RF probability into the
    noise regime.

    The boosted probability is clipped to [0, 1] before returning.

    Parameters
    ----------
    base_prob    : RF posterior P(oil | features), float or array in [0, 1]
    is_night     : binary indicator (1 = night-time, 0 = day-time); same shape as base_prob
    night_boost  : additive prior added when is_night == 1.
                   Default 0.15 — nudges a borderline 0.42 patch to 0.57
                   without overriding a definitive lookalike (0.1 + 0.15 = 0.25
                   still below 0.5 threshold).

    Returns
    -------
    adjusted_prob : np.ndarray, same shape as base_prob, values in [0, 1]
    """
    p   = np.asarray(base_prob,   dtype=np.float64)
    nit = np.asarray(is_night,    dtype=np.float64)
    return np.clip(p + nit * night_boost, 0.0, 1.0)


# ─── Geometric gate ───────────────────────────────────────────────────────────

def _apply_geometric_gate(
    df: pd.DataFrame,
    min_elongation: float,
    max_area_km2:   float,
) -> pd.Series:
    """
    Return a boolean Series: True where both geometric constraints are satisfied.

    Parameters
    ----------
    df             : DataFrame with "elongation" and "area_km2" columns
    min_elongation : strict lower bound on elongation ratio (> not ≥)
    max_area_km2   : strict upper bound on area in km²      (< not ≤)

    Returns
    -------
    mask : pd.Series[bool] — True = passes gate
    """
    elong_ok = df["elongation"] > min_elongation
    area_ok  = df["area_km2"]   < max_area_km2
    return elong_ok & area_ok


# ─── Primary filter function ──────────────────────────────────────────────────

def apply_bilge_filter(
    features_df: pd.DataFrame,
    prob_col:       str   = "prob_oil",
    is_night_col:   str   = "is_night",
    min_elongation: float = DEFAULT_MIN_ELONGATION,
    max_area_km2:   float = DEFAULT_MAX_AREA_KM2,
    night_boost:    float = DEFAULT_NIGHT_BOOST,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply bilge-dump post-processing filter to the RF-classified feature DataFrame.

    Pipeline (applied in order):
      1. Geometric gate  — reject patches with elongation ≤ 3 OR area ≥ 50 km²
      2. Night boost     — increase RF probability for night-time candidates
      3. Threshold       — final binary decision: prob_adjusted ≥ prob_threshold

    Parameters
    ----------
    features_df     : DataFrame containing at minimum:
                        - prob_col (e.g. "prob_oil" from LookalikeClassifier.predict_proba)
                        - "elongation"    (float, from features.py)
                        - "area_km2"      (float, from features.py)
                        - is_night_col    ("is_night", 0 or 1)
    prob_col        : Column name with RF oil probability (default "prob_oil")
    is_night_col    : Column name with night-time indicator (default "is_night")
    min_elongation  : Minimum elongation ratio (exclusive). Default 3.0.
    max_area_km2    : Maximum area in km² (exclusive). Default 50.0.
    night_boost     : Additive probability boost for night-time candidates.
                      Default 0.15 per Liao et al. (2023).
    prob_threshold  : Final classification threshold after boosting. Default 0.50.

    Returns
    -------
    result_df : pd.DataFrame — input df with the following ADDED columns:
        - "geom_pass"       : bool — True if passed elongation + area gate
        - "prob_adjusted"   : float — RF probability after night-time boost
        - "bilge_candidate" : bool — True = final bilge-dump detection
                              (geom_pass AND prob_adjusted ≥ threshold)
    Only rows that pass geom_pass are included in the output.
    Rows failing the geometric gate are logged and dropped.

    Raises
    ------
    KeyError    if required columns are missing from features_df
    ValueError  if features_df is empty
    """
    required = {prob_col, is_night_col, "elongation", "area_km2"}
    missing  = required - set(features_df.columns)
    if missing:
        raise KeyError(
            f"apply_bilge_filter: missing columns {missing}. "
            "Ensure predict_proba() and extract_scene_features() have been called first."
        )
    if len(features_df) == 0:
        raise ValueError("apply_bilge_filter: input DataFrame is empty.")

    df = features_df.copy()

    # ── 1. Geometric gate ─────────────────────────────────────────────────
    geom_pass = _apply_geometric_gate(df, min_elongation, max_area_km2)
    df["geom_pass"] = geom_pass

    n_total    = len(df)
    n_rejected = int((~geom_pass).sum())
    log.info(
        "Geometric gate: %d/%d rejected  (elongation≤%.1f OR area≥%.0f km²)",
        n_rejected, n_total, min_elongation, max_area_km2,
    )

    # ── 2. Night-time probability boost ───────────────────────────────────
    df["prob_adjusted"] = night_time_weight(
        base_prob  = df[prob_col].values,
        is_night   = df[is_night_col].values,
        night_boost= night_boost,
    )

    # ── 3. Final classification ────────────────────────────────────────────
    df["bilge_candidate"] = geom_pass & (df["prob_adjusted"] >= prob_threshold)

    n_candidates = int(df["bilge_candidate"].sum())
    log.info(
        "Bilge filter result: %d/%d final bilge-dump candidates  "
        "(threshold=%.2f, night_boost=%.2f)",
        n_candidates, n_total, prob_threshold, night_boost,
    )

    # Return only geometry-passing rows (rejections are dropped, not included)
    result = df[geom_pass].reset_index(drop=True)
    return result


def summarise_detections(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group bilge-dump candidates by scene_id and produce a detection summary.

    Parameters
    ----------
    result_df : Output of apply_bilge_filter() — must include "scene_id"
                and "bilge_candidate" columns.

    Returns
    -------
    summary_df : DataFrame with one row per scene:
        - "scene_id"
        - "n_candidates"    : total patches passing geometry
        - "n_bilge_dumps"   : patches also passing the RF + night threshold
        - "mean_prob_oil"   : mean adjusted probability over geometry-passing patches
        - "max_prob_oil"    : max adjusted probability
        - "any_night"       : bool — any night-time detection in this scene
    """
    if "scene_id" not in result_df.columns:
        raise KeyError("result_df must contain 'scene_id' column.")

    summary = (
        result_df.groupby("scene_id")
        .agg(
            n_candidates    = ("bilge_candidate", "count"),
            n_bilge_dumps   = ("bilge_candidate", "sum"),
            mean_prob_oil   = ("prob_adjusted", "mean"),
            max_prob_oil    = ("prob_adjusted", "max"),
            any_night       = ("is_night", lambda x: bool(x.any())),
        )
        .reset_index()
    )
    return summary


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(99)
    n   = 50

    df = pd.DataFrame({
        "scene_id":   [f"scene_{i % 5:02d}" for i in range(n)],
        "elongation": rng.uniform(0.5, 15.0, n),      # some below 3, some above
        "area_km2":   rng.uniform(1.0, 200.0, n),     # some above 50, some below
        "prob_oil":   rng.uniform(0.0, 1.0, n),
        "is_night":   rng.integers(0, 2, n),
    })

    result = apply_bilge_filter(df, prob_col="prob_oil")
    print(f"Input patches : {len(df)}")
    print(f"After geom gate: {len(result)} rows (all geom_pass=True)")
    print(f"Bilge candidates: {result['bilge_candidate'].sum()}")
    print(result[["scene_id", "elongation", "area_km2",
                  "prob_oil", "prob_adjusted", "bilge_candidate"]].to_string(index=False))

    # ── Assertions ──────────────────────────────────────────────────────────
    # All rows in result must have passed geometry
    assert result["geom_pass"].all(), "Result must only contain geom_pass=True rows"

    # All elongation > 3 AND area < 50
    assert (result["elongation"] > 3.0).all(), "elongation gate violated"
    assert (result["area_km2"]   < 50.0).all(), "area gate violated"

    # Night boost: if is_night=1, prob_adjusted = prob_oil + 0.15 (clipped)
    night_rows = result[result["is_night"] == 1]
    if len(night_rows) > 0:
        expected = np.clip(night_rows["prob_oil"].values + 0.15, 0, 1)
        assert np.allclose(night_rows["prob_adjusted"].values, expected, atol=1e-6), \
            "Night-time boost not applied correctly"

    # A patch with elongation=2 must never become a bilge candidate
    low_elong = df[df["elongation"] < 3.0].copy()
    low_elong["prob_oil"] = 0.99   # very high RF probability
    low_elong["is_night"] = 1
    result_low = apply_bilge_filter(low_elong, prob_col="prob_oil")
    assert len(result_low) == 0, "Low-elongation patches should all be rejected"

    summary = summarise_detections(result)
    print(f"\nDetection summary:\n{summary}")

    print("\nAll bilge_filter.py smoke tests passed.")
    sys.exit(0)
