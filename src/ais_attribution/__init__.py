"""
src.ais_attribution — Module 3: AIS Vessel Candidate Filtering
==============================================================

Sub-modules
-----------
trajectory_cleaning  ±50 km / ±6 h AIS fetch + per-MMSI 3D DBSCAN de-spoofing
anomaly_detection    6-feature behavioral extractor + IsoForest/RF hybrid scorer
dark_ship            FTM vessel detection + SAR↔AIS dark-ship correlation
pipeline             Module3Pipeline — end-to-end orchestrator

Entry-points
------------
>>> from src.ais_attribution.pipeline import Module3Pipeline
>>> result = Module3Pipeline().run(spill_record, ais_csv, sar_vv)
"""
from src.ais_attribution.trajectory_cleaning import (   # noqa: F401
    fetch_spill_candidates,
    apply_3d_dbscan,
)
from src.ais_attribution.anomaly_detection import (     # noqa: F401
    extract_trajectory_features,
    AISAnomalyDetector,
    FEATURE_NAMES,
)
from src.ais_attribution.dark_ship import (             # noqa: F401
    detect_ships_ftm,
    correlate_sar_to_ais,
)
from src.ais_attribution.pipeline import Module3Pipeline  # noqa: F401
