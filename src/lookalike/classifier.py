"""
Module 2 — Random Forest Look-alike Classifier.

Wraps scikit-learn's RandomForestClassifier with:
  - GroupKFold cross-validation (scene-level grouping to prevent data leakage)
  - joblib save/load with metadata versioning
  - Feature importance extraction and ranked DataFrame output
  - Calibrated probability output (predict_proba)

RF hyperparameters (per synopsis)
-----------------------------------
    n_estimators=200          — ensemble size for stable feature importances
    class_weight='balanced_subsample'  — handles oil/lookalike class imbalance
                                         per-tree rather than globally
    max_features='sqrt'       — standard Breiman heuristic for classification
    min_samples_leaf=5        — prevents over-fitting on small patch sets
    random_state=42           — reproducibility

Why 'balanced_subsample' and not 'balanced'?
--------------------------------------------
'balanced' re-weights using the FULL dataset class ratio, which biases trees
towards the majority class when bootstrapping. 'balanced_subsample' re-weights
from the BOOTSTRAP SAMPLE — each tree sees balanced classes regardless of the
global imbalance. This is the correct choice when oil patches are rare.

References
----------
- Breiman (2001) Random Forests, Machine Learning 45(1): 5–32
- Synopsis Section 2.1: "Random Forest ensemble (n_estimators=200,
  class_weight='balanced_subsample')"
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize

from src.lookalike.features import FEATURE_NAMES

log = logging.getLogger(__name__)

# ─── Version stamp embedded in every saved model file ─────────────────────────
_MODEL_VERSION = "module2_rf_v1"


class LookalikeClassifier:
    """
    Random Forest ensemble for oil-spill look-alike discrimination.

    Designed to operate on the 12-feature DataFrame produced by
    `src.lookalike.features.build_feature_dataframe()`.

    Usage
    -----
    >>> clf = LookalikeClassifier()
    >>> clf.fit(features_df, label_col="label", group_col="scene_id")
    >>> proba_df = clf.predict_proba(test_df)
    >>> clf.save("results/module2/lookalike_rf.joblib")
    >>> clf2 = LookalikeClassifier.load("results/module2/lookalike_rf.joblib")
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_folds: int = 5,
        max_features: str = "sqrt",
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators     = n_estimators
        self.n_folds          = n_folds
        self.max_features     = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state

        self._rf: RandomForestClassifier | None = None
        self.cv_scores_: np.ndarray | None = None
        self.feature_names_in_: list[str] = list(FEATURE_NAMES)
        self._fit_time: str | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators    = self.n_estimators,
            class_weight    = "balanced_subsample",   # per-synopsis
            max_features    = self.max_features,
            min_samples_leaf= self.min_samples_leaf,
            n_jobs          = -1,                     # use all cores on Kaggle
            random_state    = self.random_state,
            oob_score       = True,                   # free validation estimate
        )

    def _validate_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the 12-feature matrix as a float32 ndarray, verifying columns.

        Raises
        ------
        KeyError if any FEATURE_NAMES column is missing from df.
        """
        missing = [f for f in self.feature_names_in_ if f not in df.columns]
        if missing:
            raise KeyError(
                f"LookalikeClassifier: missing feature columns: {missing}. "
                f"Run features.extract_scene_features() first."
            )
        return df[self.feature_names_in_].values.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        group_col: str = "scene_id",
    ) -> "LookalikeClassifier":
        """
        Fit the Random Forest using scene-grouped cross-validation.

        Cross-validation uses GroupKFold with n_folds splits, grouped by
        `group_col` (scene_id). This prevents patches from the same scene
        appearing in both train and test folds — the same anti-leakage
        strategy used in Module 1 (GroupShuffleSplit on scene_id).

        After CV, the model is re-fit on ALL data for inference deployment.

        Parameters
        ----------
        df         : DataFrame with FEATURE_NAMES columns plus label_col and group_col.
        label_col  : Column name for binary labels (1=oil, 0=lookalike).
        group_col  : Column name for scene grouping (default: "scene_id").

        Returns
        -------
        self — for method chaining.
        """
        X = self._validate_df(df)
        y = df[label_col].values.astype(int)
        groups = df[group_col].values

        n_unique_scenes = len(np.unique(groups))
        n_splits = min(self.n_folds, n_unique_scenes)
        if n_splits < 2:
            raise ValueError(
                f"Need at least 2 unique scenes for GroupKFold, got {n_unique_scenes}."
            )

        log.info(
            "LookalikeClassifier.fit: n=%d  n_unique_scenes=%d  n_folds=%d  "
            "n_estimators=%d",
            len(df), n_unique_scenes, n_splits, self.n_estimators,
        )

        # ── GroupKFold cross-validation ──────────────────────────────────────
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            rf_fold = self._make_rf()
            rf_fold.fit(X[train_idx], y[train_idx])

            # Compute fold metrics: accuracy, balanced accuracy, and AUC
            from sklearn.metrics import balanced_accuracy_score, roc_auc_score
            y_val  = y[val_idx]
            y_pred = rf_fold.predict(X[val_idx])
            y_prob = rf_fold.predict_proba(X[val_idx])[:, 1]

            acc  = float((y_pred == y_val).mean())
            bacc = float(balanced_accuracy_score(y_val, y_pred))
            try:
                auc = float(roc_auc_score(y_val, y_prob))
            except Exception:
                auc = float("nan")

            cv_scores.append({"fold": fold_idx, "accuracy": acc,
                               "balanced_accuracy": bacc, "auc": auc})
            log.info(
                "  Fold %d/%d — acc=%.4f  balanced_acc=%.4f  auc=%.4f",
                fold_idx + 1, n_splits, acc, bacc, auc,
            )

        self.cv_scores_ = pd.DataFrame(cv_scores)
        log.info(
            "CV summary: acc=%.4f±%.4f  balanced_acc=%.4f±%.4f  auc=%.4f±%.4f",
            self.cv_scores_["accuracy"].mean(),
            self.cv_scores_["accuracy"].std(),
            self.cv_scores_["balanced_accuracy"].mean(),
            self.cv_scores_["balanced_accuracy"].std(),
            self.cv_scores_["auc"].mean(),
            self.cv_scores_["auc"].std(),
        )

        # ── Final fit on ALL data ────────────────────────────────────────────
        log.info("Fitting final model on all %d samples...", len(X))
        self._rf = self._make_rf()
        self._rf.fit(X, y)
        self._fit_time = datetime.now().isoformat()
        log.info(
            "Final fit complete. OOB score = %.4f",
            self._rf.oob_score_ if hasattr(self._rf, "oob_score_") else float("nan"),
        )

        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return class probability estimates for all components in df.

        Parameters
        ----------
        df : DataFrame with FEATURE_NAMES columns (META_COLUMNS optional).

        Returns
        -------
        proba_df : DataFrame with columns:
            - "prob_lookalike" (class 0)
            - "prob_oil"       (class 1)
            - "pred_label"     (argmax class, int 0 or 1)
        Same row order as input df.
        """
        if self._rf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self._validate_df(df)
        proba = self._rf.predict_proba(X)
        return pd.DataFrame({
            "prob_lookalike": proba[:, 0],
            "prob_oil":       proba[:, 1],
            "pred_label":     self._rf.predict(X),
        }, index=df.index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 = lookalike, 1 = oil)."""
        if self._rf is None:
            raise RuntimeError("Call fit() before predict().")
        return self._rf.predict(self._validate_df(df))

    # ──────────────────────────────────────────────────────────────────────────
    # Feature importance
    # ──────────────────────────────────────────────────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Columns: "feature", "importance", "std" (across trees)

        Uses the RF's built-in mean decrease in impurity (MDI). Note that MDI
        over-rates high-cardinality continuous features; for publication use
        permutation importance from sklearn.inspection.permutation_importance.
        """
        if self._rf is None:
            raise RuntimeError("Call fit() before feature_importance_df().")
        importances = self._rf.feature_importances_
        stds = np.std(
            [tree.feature_importances_ for tree in self._rf.estimators_],
            axis=0,
        )
        df = pd.DataFrame({
            "feature":    self.feature_names_in_,
            "importance": importances,
            "std":        stds,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Persist the fitted classifier (RF + metadata) to a .joblib file.

        Saves the full object including cv_scores_, feature_names_in_,
        and hyperparameters. Compatible with joblib.load() directly.

        Parameters
        ----------
        path : str or Path — target file path (typically *.joblib)

        Returns
        -------
        saved_path : Path
        """
        if self._rf is None:
            raise RuntimeError("Cannot save unfitted classifier. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version":           _MODEL_VERSION,
            "rf":                self._rf,
            "cv_scores":         self.cv_scores_.to_dict() if self.cv_scores_ is not None else None,
            "feature_names_in":  self.feature_names_in_,
            "hyperparams": {
                "n_estimators":      self.n_estimators,
                "n_folds":           self.n_folds,
                "max_features":      self.max_features,
                "min_samples_leaf":  self.min_samples_leaf,
                "random_state":      self.random_state,
            },
            "fit_time":          self._fit_time,
        }
        joblib.dump(payload, path, compress=3)
        size_mb = path.stat().st_size / (1024 ** 2)
        log.info("LookalikeClassifier saved: %s  (%.1f MB)", path, size_mb)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LookalikeClassifier":
        """
        Load a previously saved LookalikeClassifier from a .joblib file.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        clf : LookalikeClassifier (fitted, ready for predict_proba)

        Raises
        ------
        FileNotFoundError  if path doesn't exist
        ValueError         if the file was saved by an incompatible version
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier file not found: {path}")

        payload = joblib.load(path)

        if payload.get("version") != _MODEL_VERSION:
            raise ValueError(
                f"Version mismatch: file has '{payload.get('version')}', "
                f"expected '{_MODEL_VERSION}'. Re-train the classifier."
            )

        hp = payload.get("hyperparams", {})
        obj = cls(
            n_estimators    = hp.get("n_estimators", 200),
            n_folds         = hp.get("n_folds", 5),
            max_features    = hp.get("max_features", "sqrt"),
            min_samples_leaf= hp.get("min_samples_leaf", 5),
            random_state    = hp.get("random_state", 42),
        )
        obj._rf                = payload["rf"]
        obj.feature_names_in_  = payload["feature_names_in"]
        obj._fit_time          = payload.get("fit_time")
        cv_raw = payload.get("cv_scores")
        obj.cv_scores_ = pd.DataFrame(cv_raw) if cv_raw is not None else None

        log.info("LookalikeClassifier loaded from %s (fit_time=%s)",
                 path, obj._fit_time)
        return obj


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile

    rng = np.random.default_rng(0)
    n   = 300

    # Synthetic feature DataFrame — 10 scenes
    scene_ids = rng.choice([f"scene_{i:03d}" for i in range(10)], n)
    df = pd.DataFrame({s: rng.standard_normal(n) for s in FEATURE_NAMES})
    df["anisotropy_A"] = 0.0   # degenerate, as in real data
    df["area_km2"]     = rng.uniform(0.01, 40.0, n)
    df["elongation"]   = rng.uniform(1.0, 12.0, n)
    df["compactness"]  = rng.uniform(0.01, 1.0, n)
    df["is_night"]     = rng.integers(0, 2, n)
    df["wind_speed_ms"]= rng.uniform(2.0, 15.0, n)
    df["proximity_shipping_lane_km"] = rng.uniform(0.0, 500.0, n)
    df["morphology_change_km2"] = rng.uniform(0.0, 5.0, n)
    df["scene_id"] = scene_ids
    df["label"]    = rng.integers(0, 2, n)

    print(f"Smoke test DataFrame: {df.shape}")

    clf = LookalikeClassifier(n_estimators=50, n_folds=5)
    clf.fit(df, label_col="label", group_col="scene_id")

    assert clf.cv_scores_ is not None, "cv_scores_ not set"
    assert clf.cv_scores_.shape[0] == 5, f"Expected 5 folds, got {clf.cv_scores_.shape[0]}"
    print(f"CV scores:\n{clf.cv_scores_}")

    proba = clf.predict_proba(df)
    assert len(proba) == n, "predict_proba length mismatch"
    assert set(proba.columns) == {"prob_lookalike", "prob_oil", "pred_label"}

    fi = clf.feature_importance_df()
    assert len(fi) == len(FEATURE_NAMES), "feature importance row count mismatch"
    print(f"\nTop 5 feature importances:\n{fi.head(5)}")

    # Save + reload
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test_rf.joblib"
        clf.save(p)
        clf2 = LookalikeClassifier.load(p)
        proba2 = clf2.predict_proba(df)
        assert np.allclose(proba["prob_oil"].values, proba2["prob_oil"].values), \
            "Reload mismatch in probabilities"

    print("All classifier.py smoke tests passed.")
    sys.exit(0)
