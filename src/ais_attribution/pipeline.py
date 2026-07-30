"""
Module 3 — End-to-End AIS Vessel Candidate Filtering Pipeline.

Chains the four Module 3 sub-tasks into a single executable class:

    Module 2 output (verified_spill_record)
        └─ Task 1: AIS fetch + 3D-DBSCAN trajectory cleaning
              └─ Task 2: Behavioral feature extraction + Anomaly scoring
                    └─ Task 3 (fallback): Dark-ship detection via FTM
                          └─ Combined output dict → Module 4 input

Pipeline decision logic
------------------------
1. Fetch AIS pings within ±50 km / ±6 h of the spill centroid.
2. Clean trajectories with per-MMSI 3D DBSCAN (Jeon 2023).
3. Extract 6-feature behavioral vectors per track.
4. Score with AISAnomalyDetector; flag Tier-1 candidates (≥85th-pct anomaly score).
5. **If zero Tier-1 AIS candidates found, OR if ``force_dark_ship=True``**:
   → Run FTM on the SAR VV array to detect non-broadcasting vessels.
   → Correlate SAR detections with AIS to isolate dark ships.
6. Return structured output dict for Module 4 (drift analysis).

Integration
-----------
Input ``verified_spill_record`` is a dict (or pandas Series row) from the
Module 2 bilge-filter output. Required keys:

    - ``centroid_lon``  (float): Spill centroid longitude (°, WGS84)
    - ``centroid_lat``  (float): Spill centroid latitude (°, WGS84)
    - ``timestamp``     (str | pd.Timestamp): SAR acquisition time (UTC)

Optional but recommended:
    - ``scene_transform``: rasterio.Affine for the SAR scene (used by dark-ship geocoding)
    - ``bbox``          : (min_lon, min_lat, max_lon, max_lat) for logging

References
----------
- Jeon (2023): 3D DBSCAN per-MMSI cleaning
- Balsaraf et al. (2025): IsoForest + RF hybrid anomaly detection
- Synopsis §3: AIS Vessel Candidate Filtering
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ais_attribution.trajectory_cleaning import (
    fetch_spill_candidates,
    apply_3d_dbscan,
)
from src.ais_attribution.anomaly_detection import (
    AISAnomalyDetector,
    extract_trajectory_features,
    FEATURE_NAMES,
)
from src.ais_attribution.dark_ship import (
    detect_ships_ftm,
    correlate_sar_to_ais,
)

log = logging.getLogger(__name__)

# ─── Pipeline defaults ────────────────────────────────────────────────────────
_DEFAULT_RADIUS_KM:      float = 50.0
_DEFAULT_WINDOW_HR:      float = 6.0
_DEFAULT_DBSCAN_EPS_KM:  float = 2.0
_DEFAULT_DBSCAN_EPS_HR:  float = 0.5
_DEFAULT_MIN_SAMPLES:    int   = 5
_DEFAULT_PIXEL_SPACING:  float = 10.0
_DEFAULT_DARK_TOL_M:     float = 500.0


class Module3Pipeline:
    """
    End-to-end Module 3 pipeline: AIS cleaning → anomaly scoring → dark-ship fallback.

    Designed as a stateful object so the ``AISAnomalyDetector`` (which requires
    a ``fit()`` call on historical data) can be initialised once and reused
    across multiple spill events in a batch run.

    Example:
        >>> pipeline = Module3Pipeline()
        >>> pipeline.fit_detector(historical_features_df)
        >>> result = pipeline.run(
        ...     verified_spill_record  = spill_row,
        ...     ais_csv_path           = "/data/AIS_2026_03_Zone15.csv",
        ...     sar_vv_array           = vv_linear_array,
        ... )
        >>> tier1 = result["tier1_candidates"]
        >>> dark  = result["dark_ship_flags"]
    """

    def __init__(
        self,
        radius_km:          float = _DEFAULT_RADIUS_KM,
        window_hours:       float = _DEFAULT_WINDOW_HR,
        dbscan_eps_km:      float = _DEFAULT_DBSCAN_EPS_KM,
        dbscan_eps_hr:      float = _DEFAULT_DBSCAN_EPS_HR,
        dbscan_min_samples: int   = _DEFAULT_MIN_SAMPLES,
        pixel_spacing_m:    float = _DEFAULT_PIXEL_SPACING,
        dark_tol_m:         float = _DEFAULT_DARK_TOL_M,
        detector:           AISAnomalyDetector | None = None,
        detector_path:      str | Path | None = None,
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            radius_km:          Spatial search radius around the spill (km).
            window_hours:       Temporal half-window around SAR acquisition (h).
            dbscan_eps_km:      DBSCAN spatial epsilon (km) for trajectory cleaning.
            dbscan_eps_hr:      DBSCAN temporal epsilon (h) for trajectory cleaning.
            dbscan_min_samples: DBSCAN min_samples.
            pixel_spacing_m:    SAR pixel spacing (m) for FTM vessel sizing.
            dark_tol_m:         SAR↔AIS match tolerance (m) for dark-ship flagging.
            detector:           Pre-fitted ``AISAnomalyDetector``. If ``None``,
                                a fresh detector is created.
            detector_path:      If provided, loads a previously saved detector
                                from this path (overrides ``detector``).
        """
        self.radius_km          = radius_km
        self.window_hours       = window_hours
        self.dbscan_eps_km      = dbscan_eps_km
        self.dbscan_eps_hr      = dbscan_eps_hr
        self.dbscan_min_samples = dbscan_min_samples
        self.pixel_spacing_m    = pixel_spacing_m
        self.dark_tol_m         = dark_tol_m

        if detector_path is not None:
            self.detector = AISAnomalyDetector.load(detector_path)
        elif detector is not None:
            self.detector = detector
        else:
            self.detector = AISAnomalyDetector()
            log.info("Module3Pipeline: created fresh AISAnomalyDetector (call fit_detector() before run()).")

    # ── Pre-training the detector ──────────────────────────────────────────────

    def fit_detector(
        self,
        historical_features: pd.DataFrame,
        labels: pd.Series | None = None,
    ) -> "Module3Pipeline":
        """
        Train the AIS anomaly detector on historical voyage features.

        Args:
            historical_features: DataFrame of behavioral features from known
                                 (non-incident) historical voyages. Columns must
                                 match ``anomaly_detection.FEATURE_NAMES``.
            labels:              Optional binary labels (1=bilge dump, 0=clean)
                                 for supervised RF layer training.

        Returns:
            ``self`` — for method chaining.
        """
        self.detector.fit(historical_features, labels=labels)
        log.info("Module3Pipeline: AISAnomalyDetector fitted.")
        return self

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(
        self,
        verified_spill_record: dict[str, Any] | pd.Series,
        ais_csv_path: str | None = None,
        sar_vv_array: np.ndarray | None = None,
        force_dark_ship: bool = False,
        scene_transform: Any | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full Module 3 pipeline for one verified spill event.

        Pipeline steps:

        1. **AIS fetch**: Load AIS pings within ±``radius_km`` / ±``window_hours``
           of the spill centroid using :func:`fetch_spill_candidates`.
        2. **Trajectory cleaning**: Per-MMSI 3D DBSCAN via
           :func:`apply_3d_dbscan`.
        3. **Feature extraction**: Compute 6 behavioral features per clean
           track via :func:`extract_trajectory_features`.
        4. **Anomaly scoring**: Score and flag Tier-1 candidates via
           :meth:`AISAnomalyDetector.score_candidates`.
        5. **Dark-ship fallback** (if zero Tier-1 candidates, or
           ``force_dark_ship=True``):
           a. :func:`detect_ships_ftm` on ``sar_vv_array``.
           b. :func:`correlate_sar_to_ais` to isolate non-broadcasting vessels.

        Args:
            verified_spill_record: Dict or Series from Module 2 output. Must
                                   contain ``centroid_lon``, ``centroid_lat``,
                                   and ``timestamp``. Optionally ``scene_transform``.
            ais_csv_path:          Path to the NOAA AIS monthly CSV file.
                                   Pass ``None`` to skip AIS steps and go directly
                                   to the dark-ship fallback.
            sar_vv_array:          2-D float32 array of **linear** VV intensity
                                   for the scene. Required for dark-ship detection;
                                   if ``None``, dark-ship step is skipped.
            force_dark_ship:       If ``True``, always run dark-ship detection
                                   regardless of Tier-1 AIS candidates found.
            scene_transform:       rasterio.Affine for the scene (used by FTM for
                                   geocoding). Overrides the ``scene_transform``
                                   key in ``verified_spill_record`` if provided.

        Returns:
            Dict with the following keys:

            - ``"tier1_candidates"`` (pd.DataFrame): Scored, Tier-1-flagged
              trajectory features. Ready as input for Module 4 drift analysis.
              Empty DataFrame if no AIS data available.
            - ``"dark_ship_flags"`` (list[dict]): JSON-serializable dark-ship
              evidence records. Empty list if FTM step was not triggered.
            - ``"clean_ais_gdf"`` (gpd.GeoDataFrame): Trajectory-cleaned AIS
              GeoDataFrame (after DBSCAN). Useful for visualisation / debugging.
            - ``"raw_ais_gdf"`` (gpd.GeoDataFrame): Unfiltered AIS pings from
              the spatiotemporal window. Empty if ``ais_csv_path`` is ``None``.
            - ``"pipeline_meta"`` (dict): Runtime statistics and config.

        Raises:
            KeyError: If ``verified_spill_record`` is missing required keys.
        """
        t_start = time.time()

        # ── Unpack spill record ───────────────────────────────────────────────
        rec = (
            verified_spill_record
            if isinstance(verified_spill_record, dict)
            else verified_spill_record.to_dict()
        )
        try:
            clon = float(rec["centroid_lon"])
            clat = float(rec["centroid_lat"])
            sar_time = pd.Timestamp(rec["timestamp"])
        except KeyError as exc:
            raise KeyError(
                f"Module3Pipeline.run: spill record missing required key {exc}. "
                "Expected: centroid_lon, centroid_lat, timestamp."
            ) from exc

        tx = scene_transform or rec.get("scene_transform", None)

        log.info(
            "Module3Pipeline.run: spill=(%.4f, %.4f) t=%s",
            clon, clat, sar_time.isoformat(),
        )

        meta: dict[str, Any] = {
            "spill_lon":   clon,
            "spill_lat":   clat,
            "sar_time":    sar_time.isoformat(),
            "radius_km":   self.radius_km,
            "window_hours":self.window_hours,
        }

        # ── Initialise empty outputs ──────────────────────────────────────────
        import geopandas as gpd
        raw_ais_gdf   = gpd.GeoDataFrame()
        clean_ais_gdf = gpd.GeoDataFrame()
        tier1_df      = pd.DataFrame(columns=FEATURE_NAMES + [
            "S_AIS_anomaly", "iso_score", "rf_score", "is_tier1_candidate",
        ])
        dark_flags: list[dict] = []

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1: AIS fetch
        # ══════════════════════════════════════════════════════════════════════
        if ais_csv_path is not None:
            log.info("Step 1/5: Fetching AIS candidates...")
            raw_ais_gdf = fetch_spill_candidates(
                spill_centroid_lon = clon,
                spill_centroid_lat = clat,
                spill_time         = sar_time,
                ais_csv_path       = ais_csv_path,
                radius_km          = self.radius_km,
                window_hours       = self.window_hours,
            )
            meta["n_raw_ais_pings"]  = len(raw_ais_gdf)
            meta["n_raw_mmsis"]      = raw_ais_gdf["MMSI"].nunique() if not raw_ais_gdf.empty else 0
            log.info(
                "  → %d pings, %d MMSIs",
                meta["n_raw_ais_pings"], meta["n_raw_mmsis"],
            )

            # ══════════════════════════════════════════════════════════════════
            # STEP 2: 3D DBSCAN trajectory cleaning
            # ══════════════════════════════════════════════════════════════════
            if not raw_ais_gdf.empty:
                log.info("Step 2/5: 3D DBSCAN trajectory cleaning...")
                clean_ais_gdf = apply_3d_dbscan(
                    ais_gdf         = raw_ais_gdf,
                    spatial_eps_km  = self.dbscan_eps_km,
                    temporal_eps_hr = self.dbscan_eps_hr,
                    min_samples     = self.dbscan_min_samples,
                )
                meta["n_clean_pings"]  = len(clean_ais_gdf)
                meta["n_clean_tracks"] = (
                    clean_ais_gdf["track_id"].nunique()
                    if not clean_ais_gdf.empty else 0
                )
                log.info(
                    "  → %d clean pings, %d tracks",
                    meta["n_clean_pings"], meta["n_clean_tracks"],
                )

            # ══════════════════════════════════════════════════════════════════
            # STEP 3: Feature extraction
            # ══════════════════════════════════════════════════════════════════
            if not clean_ais_gdf.empty:
                log.info("Step 3/5: Extracting behavioral features...")
                feat_df = extract_trajectory_features(
                    clean_ais_gdf      = clean_ais_gdf,
                    spill_centroid_lon = clon,
                    spill_centroid_lat = clat,
                )
                meta["n_feature_tracks"] = len(feat_df)
                log.info("  → %d tracks → feature matrix %s", len(feat_df), feat_df.shape)

                # ══════════════════════════════════════════════════════════════
                # STEP 4: Anomaly scoring
                # ══════════════════════════════════════════════════════════════
                if not feat_df.empty:
                    log.info("Step 4/5: Scoring anomalies...")
                    if not self.detector._is_fitted:
                        log.warning(
                            "  AISAnomalyDetector not fitted — fitting on "
                            "candidate features (unsupervised fallback)."
                        )
                        self.detector.fit(feat_df)

                    scored_df = self.detector.score_candidates(feat_df)
                    tier1_df  = scored_df[scored_df["is_tier1_candidate"]].copy()
                    meta["n_scored_tracks"]    = len(scored_df)
                    meta["n_tier1_candidates"] = len(tier1_df)
                    log.info(
                        "  → %d tracks scored, %d Tier-1 candidates",
                        meta["n_scored_tracks"], meta["n_tier1_candidates"],
                    )
                else:
                    meta["n_tier1_candidates"] = 0
            else:
                meta["n_tier1_candidates"] = 0
        else:
            meta["n_raw_ais_pings"]    = 0
            meta["n_raw_mmsis"]        = 0
            meta["n_tier1_candidates"] = 0
            log.info("Step 1–4: skipped (no ais_csv_path provided).")

        # ══════════════════════════════════════════════════════════════════════
        # STEP 5: Dark-ship fallback
        # ══════════════════════════════════════════════════════════════════════
        trigger_dark = (
            force_dark_ship
            or meta["n_tier1_candidates"] == 0
        )

        if trigger_dark and sar_vv_array is not None:
            reason = "force_dark_ship=True" if force_dark_ship else "zero Tier-1 AIS candidates"
            log.info("Step 5/5: Dark-ship FTM (reason: %s)...", reason)

            sar_detections = detect_ships_ftm(
                sar_intensity_array = sar_vv_array,
                pixel_spacing_m     = self.pixel_spacing_m,
                scene_transform     = tx,
                sar_timestamp       = sar_time,
            )
            meta["n_sar_detections"] = len(sar_detections)
            log.info("  → %d SAR vessel detections", len(sar_detections))

            if sar_detections:
                dark_flags = correlate_sar_to_ais(
                    sar_ship_centroids    = sar_detections,
                    ais_gdf               = raw_ais_gdf,
                    distance_tolerance_m  = self.dark_tol_m,
                )
            meta["n_dark_ships"] = len(dark_flags)
            log.info("  → %d dark ships flagged", len(dark_flags))
        elif trigger_dark and sar_vv_array is None:
            log.warning(
                "Step 5/5: Dark-ship fallback triggered but sar_vv_array is None — "
                "skipping FTM. Pass the linear VV array to enable dark-ship detection."
            )
            meta["n_dark_ships"] = 0
        else:
            log.info("Step 5/5: Dark-ship FTM not triggered (%d Tier-1 candidates found).",
                     meta["n_tier1_candidates"])
            meta["n_dark_ships"] = 0

        # ── Runtime ───────────────────────────────────────────────────────────
        meta["elapsed_s"] = round(time.time() - t_start, 2)
        log.info(
            "Module3Pipeline.run complete in %.1f s | "
            "tier1=%d  dark_ships=%d",
            meta["elapsed_s"],
            len(tier1_df),
            len(dark_flags),
        )

        return {
            "tier1_candidates": tier1_df,
            "dark_ship_flags":  dark_flags,
            "clean_ais_gdf":    clean_ais_gdf,
            "raw_ais_gdf":      raw_ais_gdf,
            "pipeline_meta":    meta,
        }


# ─── Convenience: load/save detector ─────────────────────────────────────────

def save_detector(detector: AISAnomalyDetector, path: str | Path) -> Path:
    """Save a fitted detector; thin wrapper around ``AISAnomalyDetector.save()``."""
    return detector.save(path)


def load_detector(path: str | Path) -> AISAnomalyDetector:
    """Load a previously saved detector."""
    return AISAnomalyDetector.load(path)


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(12)

    # Synthetic spill event
    spill = {
        "centroid_lon": -90.0,
        "centroid_lat":  28.5,
        "timestamp":    "2026-03-15T09:00:00Z",
    }

    # Synthetic SAR VV linear array (1024×1024) with 3 bright blobs
    H, W = 512, 512
    vv_arr = rng.exponential(scale=0.01, size=(H, W)).astype(np.float32)
    for r, c in [(100, 200), (300, 400), (450, 100)]:
        vv_arr[r-2:r+3, c-2:c+3] = rng.uniform(5.0, 10.0, (5, 5))

    # ── Test with no AIS (forces dark-ship path) ──────────────────────────────
    pipeline = Module3Pipeline()
    result   = pipeline.run(
        verified_spill_record = spill,
        ais_csv_path          = None,    # No AIS file available
        sar_vv_array          = vv_arr,
    )

    print(f"tier1_candidates : {len(result['tier1_candidates'])} rows")
    print(f"dark_ship_flags  : {len(result['dark_ship_flags'])}")
    print(f"pipeline_meta    : {result['pipeline_meta']}")

    assert isinstance(result["tier1_candidates"], pd.DataFrame)
    assert isinstance(result["dark_ship_flags"], list)
    assert isinstance(result["pipeline_meta"], dict)
    assert "elapsed_s" in result["pipeline_meta"]

    # Dark ships should be detected (no AIS → all SAR detections are dark)
    assert len(result["dark_ship_flags"]) >= 1, "Expected at least 1 dark ship"
    for ds in result["dark_ship_flags"]:
        assert ds["is_dark_ship"] is True
        assert "estimated_length_m" in ds
        assert "sar_timestamp" in ds

    # ── Test force_dark_ship=True ─────────────────────────────────────────────
    result2 = pipeline.run(spill, ais_csv_path=None, sar_vv_array=vv_arr, force_dark_ship=True)
    assert len(result2["dark_ship_flags"]) >= 1, "force_dark_ship=True must trigger FTM"

    print("\nAll pipeline.py smoke tests passed.")
    sys.exit(0)
