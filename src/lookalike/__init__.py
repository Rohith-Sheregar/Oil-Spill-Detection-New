"""
src.lookalike — Module 2: Look-alike Discrimination & Bilge-Dump Detection
===========================================================================

Sub-modules
-----------
morphology      2-iteration morphological closing + connected-component extraction
features        12-feature extraction engine (polarimetric, geometric, contextual, temporal)
classifier      Random Forest ensemble with GroupKFold CV and joblib save/load
bilge_filter    Operational gating: elongation/area hard filter + night-time boost

Entry-points
------------
feature_extraction  Backward-compatible shim exposing legacy function signatures

CLI
---
src.training.train_module2  Full Kaggle training pipeline
"""

from src.lookalike.features    import FEATURE_NAMES, META_COLUMNS     # noqa: F401
from src.lookalike.morphology  import close_and_extract                # noqa: F401
from src.lookalike.classifier  import LookalikeClassifier              # noqa: F401
from src.lookalike.bilge_filter import apply_bilge_filter              # noqa: F401
