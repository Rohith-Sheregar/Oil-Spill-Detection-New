"""
src/lookalike/feature_extraction.py — Compatibility Shim
=========================================================

This file originally contained a monolithic extract/filter/train scaffold.
It has been superseded by the Module 2 four-file architecture:

    src/lookalike/morphology.py    — 2-iter morphological closing + regionprops
    src/lookalike/features.py      — 12-feature extraction engine
    src/lookalike/classifier.py    — RandomForest + GroupKFold + save/load
    src/lookalike/bilge_filter.py  — elongation/area gate + night-time boost

This shim re-exports the key public symbols from the new modules so that any
existing code that does `from src.lookalike.feature_extraction import ...`
continues to work without modification.

New code should import directly from the sub-modules above.
"""
from __future__ import annotations

# ── Public re-exports (backward-compatible) ───────────────────────────────────

from src.lookalike.features import (          # noqa: F401
    FEATURE_NAMES,
    META_COLUMNS,
    extract_scene_features,
    build_feature_dataframe,
)
from src.lookalike.morphology import (        # noqa: F401
    apply_bilge_closing,
    extract_components,
    close_and_extract,
)
from src.lookalike.classifier import (        # noqa: F401
    LookalikeClassifier,
)
from src.lookalike.bilge_filter import (      # noqa: F401
    apply_bilge_filter,
    summarise_detections,
    night_time_weight,
)


# ── Legacy function shims ─────────────────────────────────────────────────────
# The original scaffold defined three standalone functions. Thin wrappers
# below maintain their call signatures for backward compatibility.

import numpy as np
import pandas as pd


def extract_patch_features(
    mask, vv, vh, H, alpha, wind_speed, hour_local, scene_id
) -> pd.DataFrame:
    """
    [DEPRECATED] Legacy wrapper around features.extract_scene_features().

    Use extract_scene_features() with RegionProperties from morphology.py directly.
    This shim exists only for backward compatibility.
    """
    import warnings
    warnings.warn(
        "extract_patch_features() is deprecated. Use "
        "src.lookalike.features.extract_scene_features() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.lookalike.morphology import close_and_extract
    _, regions = close_and_extract(mask)

    # Convert legacy (H, alpha_deg_raw) — alpha here is already in degrees
    return extract_scene_features(
        regions       = regions,
        vv_db         = vv,
        vh_db         = vh,
        scene_id      = scene_id,
        wind_speed_ms = wind_speed,
        hour_local    = hour_local,
    )


def apply_bilge_morphology_filter(
    features_df,
    min_elongation: float = 3.0,
    max_area_km2:   float = 50.0,
    px_area_km2:    float = 1e-4,
) -> pd.DataFrame:
    """
    [DEPRECATED] Legacy wrapper. Use bilge_filter.apply_bilge_filter() instead.

    Note: the original used px_area_km2 for conversion; the new pipeline
    computes area_km2 at extraction time. This shim calls the new filter
    assuming area_km2 is already in the DataFrame.
    """
    import warnings
    warnings.warn(
        "apply_bilge_morphology_filter() is deprecated. Use "
        "src.lookalike.bilge_filter.apply_bilge_filter() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # If old code passed area_px, convert it here
    df = features_df.copy()
    if "area_px" in df.columns and "area_km2" not in df.columns:
        df["area_km2"] = df["area_px"] * px_area_km2
    # Stub prob_oil column at 1.0 (pass-through: no RF, just geometric gate)
    df["prob_oil"] = 1.0
    df["is_night"] = df.get("is_night", 0)
    return apply_bilge_filter(
        df,
        prob_col        = "prob_oil",
        min_elongation  = min_elongation,
        max_area_km2    = max_area_km2,
        night_boost     = 0.0,   # no boost in legacy mode
        prob_threshold  = 0.5,
    )


def train_rf_with_scene_groups(
    features_df: pd.DataFrame,
    label_col: str = "label",
    n_splits:   int = 5,
):
    """
    [DEPRECATED] Legacy wrapper. Use LookalikeClassifier.fit() instead.
    """
    import warnings
    warnings.warn(
        "train_rf_with_scene_groups() is deprecated. Use "
        "src.lookalike.classifier.LookalikeClassifier.fit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    clf = LookalikeClassifier(n_estimators=200, n_folds=n_splits)
    clf.fit(features_df, label_col=label_col, group_col="scene_id")
    scores = clf.cv_scores_["balanced_accuracy"].values if clf.cv_scores_ is not None else np.array([])
    return clf._rf, scores
