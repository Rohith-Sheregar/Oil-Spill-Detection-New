"""
Module 3 — Behavioral Anomaly Detection for AIS Vessel Candidates.

Extracts a compact 6-dimensional behavioral feature vector per trajectory
and scores each track for bilge-dump-consistent anomalous behavior using a
hybrid IsolationForest + RandomForestClassifier approach (Balsaraf et al. 2025).

Feature design rationale
--------------------------
Legitimate transit vessels maintain steady SOG with smooth, predictable course
changes. Bilge dumping vessels exhibit characteristic signatures:

- **SOG slowdown / stop**: Pumping takes time; vessels slow to 1–3 kn or heave-to.
- **High course deviation**: Manoeuvring to stay on a consistent bearing while
  pumping in open sea produces oscillating heading changes.
- **Minimum proximity**: Only vessels that actually passed close to the spill
  centroid are relevant.
- **Sudden SOG drop**: The transition from transit to pumping speed is abrupt.

Model architecture (Balsaraf et al. 2025 hybrid)
--------------------------------------------------
``IsolationForest`` is the unsupervised backbone — it generalises well to the
~5 confirmed incidents available, where a supervised classifier would overfit.
``RandomForestClassifier`` is added as an optional supervised layer: when
confirmed ground-truth labels exist (from validated incident reports), the RF
is trained on IsoForest scores + raw features to improve precision. With fewer
than ~20 labelled incidents, rely primarily on the IsoForest score.

Integration
-----------
- Input: output of :func:`trajectory_cleaning.apply_3d_dbscan` (clean GeoDataFrame)
- Output: ``candidate_features`` DataFrame → ``score_candidates()`` → Tier-1 flags
- Tier-1 threshold: 85th percentile anomaly score (synopsis §3.2)
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

# ─── Feature registry ─────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "sog_mean",              # Mean SOG (knots) — low = slow mover / pumping
    "sog_variance",          # SOG variance — high = erratic speed profile
    "course_deviation_std",  # Std-dev of ΔCOG (°) — high = erratic heading
    "n_stops",               # Count of pings where SOG < 0.5 kn
    "max_sudden_sog_drop",   # Most negative SOG delta between consecutive pings
    "min_proximity_km",      # Closest approach to spill centroid (km)
]

# Tier-1 anomaly score percentile threshold (synopsis §3.2)
_TIER1_PERCENTILE: float = 85.0

# IsolationForest contamination (fraction of expected anomalous vessels)
_DEFAULT_CONTAMINATION: float = 0.10

# Minimum pings per track to compute reliable features
_MIN_PINGS: int = 3


# ─── Task 2a: Feature extraction ─────────────────────────────────────────────

def extract_trajectory_features(
    clean_ais_gdf: "gpd.GeoDataFrame",
    spill_centroid_lon: float = 0.0,
    spill_centroid_lat: float = 0.0,
    time_col: str = "BaseDateTime",
    sog_col: str = "SOG",
    cog_col: str = "COG",
) -> pd.DataFrame:
    """
    Extract a 6-dimensional behavioral feature vector per trajectory.

    Processes the cleaned AIS GeoDataFrame (output of
    :func:`trajectory_cleaning.apply_3d_dbscan`) and returns one row of
    features per ``track_id``.

    Args:
        clean_ais_gdf:       GeoDataFrame with columns ``track_id``, ``LAT``,
                             ``LON``, ``SOG``, ``COG``, and ``BaseDateTime``.
                             Output of :func:`trajectory_cleaning.apply_3d_dbscan`.
        spill_centroid_lon:  Longitude of the verified spill centroid (°, WGS84).
                             Used to compute ``min_proximity_km``.
        spill_centroid_lat:  Latitude of the verified spill centroid (°, WGS84).
        time_col:            DateTime column name (default ``"BaseDateTime"``).
        sog_col:             Speed-over-ground column name (default ``"SOG"``).
        cog_col:             Course-over-ground column name (default ``"COG"``).

    Returns:
        DataFrame indexed by ``track_id`` with columns ``FEATURE_NAMES`` plus
        ``mmsi`` and ``n_pings``. Tracks with fewer than ``_MIN_PINGS`` pings
        are excluded.

    Raises:
        ValueError: If ``track_id`` column is missing.
    """
    required = {"track_id", "LAT", "LON", sog_col, cog_col}
    missing  = required - set(clean_ais_gdf.columns)
    if missing:
        raise ValueError(
            f"extract_trajectory_features: missing columns {missing}."
        )

    import geopandas as gpd  # local import avoids circular deps at module level

    # Import haversine from trajectory_cleaning to avoid code duplication
    from src.ais_attribution.trajectory_cleaning import _haversine_dist_km

    rows: list[dict] = []

    for track_id, g in clean_ais_gdf.groupby("track_id"):
        g = g.sort_values(time_col)
        n = len(g)

        if n < _MIN_PINGS:
            log.debug("track %s: only %d pings — skipping feature extraction.", track_id, n)
            continue

        sog = g[sog_col].to_numpy(dtype=np.float64)
        cog = g[cog_col].to_numpy(dtype=np.float64)
        lats = g["LAT"].to_numpy(dtype=np.float64)
        lons = g["LON"].to_numpy(dtype=np.float64)

        # ── SOG features ──────────────────────────────────────────────────────
        sog_mean     = float(np.nanmean(sog))
        sog_variance = float(np.nanvar(sog))
        n_stops      = int(np.sum(sog < 0.5))

        # Max sudden slowdown: most negative consecutive SOG delta
        sog_deltas          = np.diff(sog)
        max_sudden_sog_drop = float(np.nanmin(sog_deltas)) if len(sog_deltas) > 0 else 0.0

        # ── COG deviation ─────────────────────────────────────────────────────
        # Circular difference to handle 0°/360° wraparound correctly
        cog_deltas = np.diff(cog)
        cog_deltas = (cog_deltas + 180.0) % 360.0 - 180.0   # wrap to [−180, +180]
        course_deviation_std = float(np.nanstd(cog_deltas)) if len(cog_deltas) > 0 else 0.0

        # ── Minimum proximity to spill centroid ───────────────────────────────
        distances_km = _haversine_dist_km(lons, lats, spill_centroid_lon, spill_centroid_lat)
        min_proximity_km = float(np.nanmin(distances_km))

        # ── MMSI (for traceability) ───────────────────────────────────────────
        mmsi = g["MMSI"].iloc[0] if "MMSI" in g.columns else None

        rows.append({
            "track_id":            track_id,
            "mmsi":                mmsi,
            "n_pings":             n,
            "sog_mean":            sog_mean,
            "sog_variance":        sog_variance,
            "course_deviation_std": course_deviation_std,
            "n_stops":             n_stops,
            "max_sudden_sog_drop": max_sudden_sog_drop,
            "min_proximity_km":    min_proximity_km,
        })

    if not rows:
        log.warning("extract_trajectory_features: no valid tracks extracted.")
        return pd.DataFrame(columns=["track_id", "mmsi", "n_pings"] + FEATURE_NAMES)

    df = pd.DataFrame(rows).set_index("track_id")
    log.info("extract_trajectory_features: %d tracks → feature matrix %s.", len(df), df.shape)
    return df


# ─── Task 2b: AIS Anomaly Detector ───────────────────────────────────────────

class AISAnomalyDetector:
    """
    Hybrid IsolationForest + optional RandomForest anomaly scorer for AIS trajectories.

    Implements the two-layer approach from Balsaraf et al. (2025):

    1. **IsolationForest** (unsupervised): scores every trajectory on the 6
       behavioral features; generalises well with < 20 confirmed incidents.
    2. **RandomForestClassifier** (supervised, optional): once confirmed
       incident labels exist, trained on raw features + IsoForest score to
       improve precision. With ~5 confirmed incidents, treat the IsoForest
       score as the primary signal.

    The combined anomaly score ``S_AIS_anomaly ∈ [0, 1]`` is defined as::

        iso_score  = 1 − clip(isoforest.decision_function(X) + 0.5, 0, 1)
        rf_score   = rf.predict_proba(X)[:, 1]   # only if RF is fitted
        S_AIS      = 0.6 × iso_score + 0.4 × rf_score   # hybrid blend

    If no RF is fitted, ``S_AIS_anomaly = iso_score``.

    Example:
        >>> detector = AISAnomalyDetector()
        >>> detector.fit(historical_features_df)
        >>> result = detector.score_candidates(candidate_features_df)
        >>> tier1 = result[result["is_tier1_candidate"]]
    """

    # Weight of IsoForest vs RF in the hybrid score
    _ISO_WEIGHT: float = 0.6
    _RF_WEIGHT:  float = 0.4

    def __init__(
        self,
        contamination: float = _DEFAULT_CONTAMINATION,
        n_estimators_iso: int = 200,
        n_estimators_rf:  int = 100,
        tier1_percentile: float = _TIER1_PERCENTILE,
        random_state: int = 42,
    ) -> None:
        """
        Initialise the detector with hyperparameters.

        Args:
            contamination:     Expected fraction of anomalous vessels in the
                               training population. Default 0.10 (10%). Increase
                               to flag more candidates; decrease if precision is
                               more important than recall.
            n_estimators_iso:  Number of isolation trees (default 200).
            n_estimators_rf:   Number of RF trees (default 100).
            tier1_percentile:  Percentile threshold for Tier-1 candidate flag
                               (default 85th — synopsis §3.2).
            random_state:      Reproducibility seed.
        """
        self.contamination     = contamination
        self.n_estimators_iso  = n_estimators_iso
        self.n_estimators_rf   = n_estimators_rf
        self.tier1_percentile  = tier1_percentile
        self.random_state      = random_state

        self._iso:    IsolationForest | None       = None
        self._rf:     RandomForestClassifier | None = None
        self._scaler: RobustScaler                  = RobustScaler()
        self._is_fitted: bool = False
        self._rf_fitted: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_X(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Extract and validate the 6-feature matrix from a features DataFrame.

        Args:
            features_df: DataFrame with columns matching FEATURE_NAMES.

        Returns:
            float64 ndarray of shape (n_tracks, 6).

        Raises:
            KeyError: If any FEATURE_NAMES column is missing.
        """
        missing = [f for f in FEATURE_NAMES if f not in features_df.columns]
        if missing:
            raise KeyError(
                f"AISAnomalyDetector: missing feature columns {missing}. "
                "Run extract_trajectory_features() first."
            )
        return features_df[FEATURE_NAMES].to_numpy(dtype=np.float64)

    def _iso_to_score(self, raw_decision: np.ndarray) -> np.ndarray:
        """
        Map IsolationForest ``decision_function`` output → [0, 1] anomaly score.

        ``decision_function`` returns negative values for anomalies and positive
        for normal points. The typical range is roughly [–0.5, +0.5].

        Mapping: ``score = 1 − clip(raw + 0.5, 0, 1)``
        → anomalies → score ≈ 1; normals → score ≈ 0.

        Args:
            raw_decision: Raw output of ``isoforest.decision_function(X)``.

        Returns:
            Anomaly score array in [0, 1].
        """
        return 1.0 - np.clip(raw_decision + 0.5, 0.0, 1.0)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        historical_features: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "AISAnomalyDetector":
        """
        Train the anomaly detector on historical trajectory features.

        Args:
            historical_features: DataFrame indexed by ``track_id`` with
                                 columns matching FEATURE_NAMES (output of
                                 :func:`extract_trajectory_features`). Can be
                                 historical data from known non-incident voyages.
            labels:              Optional binary Series (1 = confirmed bilge dump,
                                 0 = clean voyage) aligned with
                                 ``historical_features``. If provided and contains
                                 at least 5 positive examples, the RF supervised
                                 layer is also fitted.

        Returns:
            ``self`` — for method chaining.
        """
        X = self._extract_X(historical_features)
        X_scaled = self._scaler.fit_transform(X)

        log.info("AISAnomalyDetector.fit: n=%d  features=%s", len(X), FEATURE_NAMES)

        # ── IsolationForest (always fitted) ───────────────────────────────────
        self._iso = IsolationForest(
            n_estimators = self.n_estimators_iso,
            contamination= self.contamination,
            n_jobs       = -1,
            random_state = self.random_state,
        )
        self._iso.fit(X_scaled)
        log.info("  IsolationForest fitted (n_estimators=%d).", self.n_estimators_iso)

        # ── RandomForest (optional supervised layer) ───────────────────────────
        if labels is not None:
            y = labels.to_numpy()
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos >= 5 and n_neg >= 5:
                iso_scores = self._iso_to_score(
                    self._iso.decision_function(X_scaled)
                ).reshape(-1, 1)
                X_rf = np.hstack([X_scaled, iso_scores])
                self._rf = RandomForestClassifier(
                    n_estimators = self.n_estimators_rf,
                    class_weight = "balanced",
                    random_state = self.random_state,
                    n_jobs       = -1,
                )
                self._rf.fit(X_rf, y)
                self._rf_fitted = True
                log.info("  RF layer fitted (n_pos=%d, n_neg=%d).", n_pos, n_neg)
            else:
                log.warning(
                    "  RF layer skipped: need ≥5 positives + ≥5 negatives, "
                    "got pos=%d neg=%d. Using IsoForest-only scoring.", n_pos, n_neg
                )

        self._is_fitted = True
        return self

    def score_candidates(
        self,
        candidate_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score vessel trajectories and flag Tier-1 bilge-dump candidates.

        Args:
            candidate_features: DataFrame indexed by ``track_id`` with columns
                                matching FEATURE_NAMES. Output of
                                :func:`extract_trajectory_features`.

        Returns:
            DataFrame (same index as ``candidate_features``) with columns:

            - All original feature columns.
            - ``S_AIS_anomaly`` (float, [0, 1]): Combined anomaly score.
              Higher = more anomalous = more likely to be a bilge dumper.
            - ``iso_score`` (float): Raw IsolationForest component.
            - ``rf_score`` (float | NaN): RF component (NaN if RF not fitted).
            - ``is_tier1_candidate`` (bool): True if ``S_AIS_anomaly`` exceeds
              the 85th-percentile threshold among all scored candidates.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Call AISAnomalyDetector.fit() before score_candidates().")

        if candidate_features.empty:
            log.warning("score_candidates: empty input — returning empty DataFrame.")
            return candidate_features.assign(
                S_AIS_anomaly       = pd.Series(dtype=float),
                iso_score           = pd.Series(dtype=float),
                rf_score            = pd.Series(dtype=float),
                is_tier1_candidate  = pd.Series(dtype=bool),
            )

        X = self._extract_X(candidate_features)
        X_scaled = self._scaler.transform(X)

        # ── IsoForest score ───────────────────────────────────────────────────
        iso_scores = self._iso_to_score(self._iso.decision_function(X_scaled))

        # ── RF score (optional) ───────────────────────────────────────────────
        if self._rf_fitted and self._rf is not None:
            iso_col = iso_scores.reshape(-1, 1)
            X_rf    = np.hstack([X_scaled, iso_col])
            rf_scores = self._rf.predict_proba(X_rf)[:, 1]
            combined  = (
                self._ISO_WEIGHT * iso_scores
                + self._RF_WEIGHT * rf_scores
            )
        else:
            rf_scores = np.full(len(X), np.nan)
            combined  = iso_scores

        # ── Tier-1 threshold ──────────────────────────────────────────────────
        threshold       = float(np.percentile(combined, self.tier1_percentile))
        is_tier1        = combined >= threshold

        result = candidate_features.copy()
        result["S_AIS_anomaly"]      = combined
        result["iso_score"]          = iso_scores
        result["rf_score"]           = rf_scores
        result["is_tier1_candidate"] = is_tier1

        n_tier1 = int(is_tier1.sum())
        log.info(
            "score_candidates: %d tracks scored, %d Tier-1 candidates "
            "(≥%.0f-th percentile, threshold=%.4f).",
            len(result), n_tier1, self.tier1_percentile, threshold,
        )
        return result

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Persist the fitted detector to a ``.joblib`` file.

        Args:
            path: Destination file path.

        Returns:
            Resolved ``Path`` of the saved file.

        Raises:
            RuntimeError: If the detector has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted detector.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        size_mb = path.stat().st_size / (1024 ** 2)
        log.info("AISAnomalyDetector saved: %s  (%.1f MB)", path, size_mb)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AISAnomalyDetector":
        """
        Load a previously saved detector from disk.

        Args:
            path: Path to the ``.joblib`` file.

        Returns:
            Fitted ``AISAnomalyDetector`` instance.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Detector file not found: {path}")
        obj = joblib.load(path)
        log.info("AISAnomalyDetector loaded from %s.", path)
        return obj


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(7)
    n   = 80

    # Synthetic feature DataFrame
    feat_df = pd.DataFrame({
        "track_id":             [f"mmsi_{i}_{j}" for i in range(8) for j in range(10)],
        "mmsi":                 np.repeat(range(100_000, 100_008), 10),
        "n_pings":              rng.integers(5, 50, n),
        "sog_mean":             rng.uniform(0.0, 14.0, n),
        "sog_variance":         rng.uniform(0.0, 25.0, n),
        "course_deviation_std": rng.uniform(0.0, 60.0, n),
        "n_stops":              rng.integers(0, 10, n),
        "max_sudden_sog_drop":  rng.uniform(-8.0, 0.0, n),
        "min_proximity_km":     rng.uniform(0.0, 50.0, n),
    }).set_index("track_id")

    # Inject 5 obvious anomalies
    feat_df.iloc[:5, feat_df.columns.get_loc("sog_mean")]     = 0.3
    feat_df.iloc[:5, feat_df.columns.get_loc("n_stops")]      = 20
    feat_df.iloc[:5, feat_df.columns.get_loc("min_proximity_km")] = 0.5

    # Train + score (unsupervised)
    detector = AISAnomalyDetector(contamination=0.10, tier1_percentile=85.0)
    detector.fit(feat_df)
    result = detector.score_candidates(feat_df)

    print(f"Scored {len(result)} tracks")
    print(f"Tier-1 candidates: {result['is_tier1_candidate'].sum()}")
    print(f"Score range: [{result['S_AIS_anomaly'].min():.4f}, {result['S_AIS_anomaly'].max():.4f}]")
    print(result[["S_AIS_anomaly", "is_tier1_candidate"]].head(10))

    assert "S_AIS_anomaly" in result.columns
    assert result["S_AIS_anomaly"].between(0, 1).all(), "Scores outside [0, 1]"
    assert result["is_tier1_candidate"].sum() > 0, "No Tier-1 candidates flagged"

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "detector.joblib"
        detector.save(p)
        d2 = AISAnomalyDetector.load(p)
        r2 = d2.score_candidates(feat_df)
        assert np.allclose(result["S_AIS_anomaly"], r2["S_AIS_anomaly"]), "Reload mismatch"

    print("All anomaly_detection smoke tests passed.")
    sys.exit(0)
