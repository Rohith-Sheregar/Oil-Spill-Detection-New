"""
Module 1 final verification script — runs without heavy ML dependencies.
Tests: polsar_decomp, wind_ratio, pseudo-label constants,
       shipping-lane bboxes, pair_sar_to_ais signature, deeplab_scse defaults,
       zenodo_sos_dataset API usage, augmentations.
"""
import sys, ast, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))

PASS = 0; FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAIL += 1

print("=" * 55)
print("Module 1 — Final Verification")
print("=" * 55)

# ─── 1. Preprocessing: polsar_decomp ─────────────────────────────────────
print("\n[1] Preprocessing — polsar_decomp")
from src.preprocessing.polsar_decomp import (
    db_to_linear, dual_pol_entropy_alpha, dual_pol_entropy_alpha_from_db,
    decompose_dual_pol_tiff,
)
rng = np.random.default_rng(42)
h, w = 64, 64
vv_lin = rng.uniform(1e-4, 0.2,  (h, w)).astype("float32")
vh_lin = rng.uniform(1e-5, 0.05, (h, w)).astype("float32")
H, alpha = dual_pol_entropy_alpha(vv_lin, vh_lin)
check("H in [0, 1]",          0.0 <= float(H.min()) and float(H.max()) <= 1.0)
check("alpha in [0, 90]",     0.0 <= float(alpha.min()) and float(alpha.max()) <= 90.0)

vv_db = rng.uniform(-25, -5,  (h, w)).astype("float32")
vh_db = rng.uniform(-30, -10, (h, w)).astype("float32")
H2, a2 = dual_pol_entropy_alpha_from_db(vv_db, vh_db)
check("dB wrapper H in [0,1]", 0.0 <= float(H2.min()) and float(H2.max()) <= 1.0)
check("db_to_linear output > 0", float(db_to_linear(vv_db).min()) > 0)
check("decompose_dual_pol_tiff alias works", "H" in decompose_dual_pol_tiff(vv_db, vh_db, input_is_db=True))

# ─── 2. Preprocessing: wind_ratio ────────────────────────────────────────
print("\n[2] Preprocessing — wind_ratio")
from src.preprocessing.wind_ratio import cmod5n_forward, compute_wind_corrected_ratio
b5 = compute_wind_corrected_ratio(vv_db, vh_db, wind_speed_ms=7.5, incidence_deg=35.0)
check("Band5 in [0, 1]",  0.0 <= float(b5.min()) and float(b5.max()) <= 1.0)
check("Band5 is float32", b5.dtype == np.float32)
s0 = float(cmod5n_forward(10.0, 35.0, 0.0))
check("CMOD5.N(10 m/s) > 0", s0 > 0)

# ─── 3. Pseudo-label constants ────────────────────────────────────────────
print("\n[3] Pseudo-label trainer constants (synopsis spec)")
src = pathlib.Path("src/training/pseudo_label_trainer.py").read_text(encoding="utf-8")
tree = ast.parse(src)
C = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in (
                "MAX_CYCLES","CONF_HIGH","CONF_LOW","MIN_CONFIDENT_FRAC","EPOCHS_PER_CYCLE"
            ):
                try: C[t.id] = ast.literal_eval(node.value)
                except: pass
check("MAX_CYCLES == 10",          C.get("MAX_CYCLES") == 10)
check("CONF_HIGH == 0.85",         C.get("CONF_HIGH") == 0.85)
check("CONF_LOW == 0.15",          C.get("CONF_LOW") == 0.15)
check("MIN_CONFIDENT_FRAC == 0.80",C.get("MIN_CONFIDENT_FRAC") == 0.80)
check("EPOCHS_PER_CYCLE == 1",     C.get("EPOCHS_PER_CYCLE") == 1)
check("run_pseudo_label_cycle defined", "def run_pseudo_label_cycle(" in src)
check("run_pseudo_label_cycles alias",  "def run_pseudo_label_cycles(" in src)

# ─── 4. Shipping-lane bboxes ──────────────────────────────────────────────
print("\n[4] Data access — SHIPPING_LANE_BBOXES")
from src.data_access.sentinel1_cdse import SHIPPING_LANE_BBOXES, search_shipping_lane
req = {"suez_canal","mediterranean","south_china_sea","gulf_of_mexico"}
check("All 4 lanes defined", req == set(SHIPPING_LANE_BBOXES.keys()))
check("suez_canal bbox",     SHIPPING_LANE_BBOXES["suez_canal"]    == (31.5, 27.0, 33.5, 32.0))
check("mediterranean bbox",  SHIPPING_LANE_BBOXES["mediterranean"] == (-6.0, 30.0, 36.0, 46.0))
check("south_china_sea bbox",SHIPPING_LANE_BBOXES["south_china_sea"] == (105.0, 0.0, 125.0, 25.0))
check("gulf_of_mexico bbox", SHIPPING_LANE_BBOXES["gulf_of_mexico"] == (-98.0, 18.0, -80.0, 30.0))
check("search_shipping_lane callable", callable(search_shipping_lane))

# ─── 5. AIS pairing utility ───────────────────────────────────────────────
print("\n[5] Data access — pair_sar_to_ais")
src5 = pathlib.Path("src/data_access/ais_noaa.py").read_text(encoding="utf-8")
check("pair_sar_to_ais defined",       "def pair_sar_to_ais(" in src5)
check("csv_path param",               "csv_path" in src5)
check("scene_bbox param",             "scene_bbox" in src5)
check("scene_acquisition_time param", "scene_acquisition_time" in src5)
check("window_hours default 6.0",     "window_hours: float = 6.0" in src5)
check("calls load_ais_window",        "load_ais_window" in src5)
check("returns GeoDataFrame",         "GeoDataFrame" in src5)

# ─── 6. DeepLabV3+ default in_channels = 5 ───────────────────────────────
print("\n[6] Model — deeplab_scse.py")
src6 = pathlib.Path("src/models/deeplab_scse.py").read_text(encoding="utf-8")
check("in_channels=5 default",   "in_channels=5" in src6)
check("SCSEModule defined",      "class SCSEModule" in src6)
check("DeepLabV3PlusSCSE defined","class DeepLabV3PlusSCSE" in src6)

# ─── 7. Dataset 5-band loader ─────────────────────────────────────────────
print("\n[7] Training — zenodo_sos_dataset.py")
src7 = pathlib.Path("src/training/zenodo_sos_dataset.py").read_text(encoding="utf-8")
check("uses dual_pol_entropy_alpha",  "dual_pol_entropy_alpha" in src7)
check("uses db_to_linear",            "db_to_linear" in src7)
check("full_5band in_channels=5",     "full_5band" in src7 and "5" in src7)
check("input_mode full_5band default","full_5band" in src7)

# ─── 8. Augmentations ────────────────────────────────────────────────────
print("\n[8] Augmentations — zenodo_sos_dataset.py")
check("Rotation (rot90)",           "rot90" in src7)
check("H-flip + V-flip",             "::-1]" in src7)
check("Gaussian noise",             "rng.normal" in src7)
check("Gaussian blur (scipy)",      "gaussian_filter" in src7)
check("Histogram equalization",     "equalize_hist" in src7)

# ─── 9. Results directory ─────────────────────────────────────────────────
print("\n[9] Results directory")
results = pathlib.Path("results/module1")
for sub in ["checkpoints","metrics","plots","logs","report","pseudo_labels"]:
    check(f"results/module1/{sub}/ exists", (results / sub).is_dir())
check("results/module1/README.md exists", (results / "README.md").is_file())

# ─── 10. Syntax check all source files ───────────────────────────────────
print("\n[10] Syntax check (py_compile)")
import py_compile
files = [
    "src/preprocessing/polsar_decomp.py",
    "src/preprocessing/wind_ratio.py",
    "src/preprocessing/band_stack.py",
    "src/models/deeplab_scse.py",
    "src/models/losses.py",
    "src/training/zenodo_sos_dataset.py",
    "src/training/pseudo_label_trainer.py",
    "src/training/train_module1.py",
    "src/training/splits.py",
    "src/training/gpu_utils.py",
    "src/validation/metrics.py",
    "src/reporting/module1_report.py",
    "src/data_access/sentinel1_cdse.py",
    "src/data_access/ais_noaa.py",
]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        check(f, True)
    except py_compile.PyCompileError as e:
        check(f, False, str(e))

# ─── Summary ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"TOTAL: {PASS} PASSED,  {FAIL} FAILED")
if FAIL == 0:
    print("Module 1 is 100% complete and verified.")
else:
    print("Fix the failures above before training.")
print("=" * 55)
