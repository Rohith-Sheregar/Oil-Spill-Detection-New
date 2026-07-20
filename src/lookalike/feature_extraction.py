"""
Random Forest look-alike rejection: feature engineering from segmentation
output + auxiliary bands, then scene-grouped training (same leakage risk
as Module 1 -- group by scene_id, never split patches/regions randomly).
"""
import numpy as np
import pandas as pd
from skimage import measure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold


def extract_patch_features(mask, vv, vh, H, alpha, wind_speed, hour_local, scene_id):
    """
    mask: binary segmentation output for one scene (oil/non-oil)
    vv, vh, H, alpha: same-shape arrays as the segmentation input bands
    wind_speed: scalar (ERA5, at scene center or per-patch centroid)
    hour_local: local hour 0-23, used for the night-time prior
    Returns one row per connected component (= one candidate dark patch).
    """
    rows = []
    for region in measure.regionprops(measure.label(mask)):
        ys, xs = region.coords[:, 0], region.coords[:, 1]
        vv_vals, vh_vals = vv[ys, xs], vh[ys, xs]
        rows.append({
            "scene_id": scene_id,
            "area_px": region.area,
            "elongation": region.major_axis_length / max(region.minor_axis_length, 1e-3),
            "compactness": region.perimeter ** 2 / max(region.area, 1e-3),
            "eccentricity": region.eccentricity,
            "mean_VV": vv_vals.mean(), "mean_VH": vh_vals.mean(),
            "ratio_VV_VH": vv_vals.mean() / max(vh_vals.mean(), 1e-6),
            "mean_H": H[ys, xs].mean(), "mean_alpha": alpha[ys, xs].mean(),
            "wind_speed": wind_speed,
            "is_night": int(hour_local < 6 or hour_local >= 20),
        })
    return pd.DataFrame(rows)


def apply_bilge_morphology_filter(features_df, min_elongation=3.0, max_area_km2=50.0, px_area_km2=1e-4):
    """
    Geometric signature filter from the synopsis: elongation > 3:1 AND
    area < 50 km^2. px_area_km2 depends on ground sampling distance -- for
    10m Sentinel-1 GRD this is ~1e-4 km^2/pixel; recompute if you've
    resampled to a different patch resolution.
    """
    area_km2 = features_df["area_px"] * px_area_km2
    keep = (features_df["elongation"] > min_elongation) & (area_km2 < max_area_km2)
    return features_df[keep].copy()


def train_rf_with_scene_groups(features_df, label_col="label", n_splits=5):
    X = features_df.drop(columns=[label_col, "scene_id"])
    y = features_df[label_col]
    groups = features_df["scene_id"]
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(clf.score(X.iloc[test_idx], y.iloc[test_idx]))
    clf.fit(X, y)  # final fit on everything once CV scores look sane
    return clf, np.array(scores)


if __name__ == "__main__":
    # Smoke test with synthetic data -- replace with real extracted features.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "scene_id": rng.integers(0, 10, n),
        "elongation": rng.uniform(1, 8, n),
        "compactness": rng.uniform(10, 200, n),
        "eccentricity": rng.uniform(0, 1, n),
        "mean_VV": rng.normal(-15, 3, n), "mean_VH": rng.normal(-20, 3, n),
        "ratio_VV_VH": rng.uniform(0.5, 3, n),
        "mean_H": rng.uniform(0, 1, n), "mean_alpha": rng.uniform(0, 90, n),
        "wind_speed": rng.uniform(1, 12, n), "is_night": rng.integers(0, 2, n),
        "label": rng.integers(0, 2, n),
    })
    clf, scores = train_rf_with_scene_groups(df)
    print("CV scores:", scores, "mean:", scores.mean())
