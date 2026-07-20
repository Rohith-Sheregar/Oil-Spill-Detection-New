"""
AIS trajectory cleaning (mixed-MMSI resolution) + behavioral anomaly
detection. Cleaning runs PER MMSI -- running DBSCAN across the whole
multi-vessel AIS table at once clusters by spatial proximity between
*different* vessels, not by identity continuity within one MMSI, which is
not the mixed-trajectory problem you're trying to solve.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


def clean_mixed_trajectories(ais_df, eps_km=2.0, min_samples=5, km_per_lat_deg=111.0):
    """
    Resolves the 'mixed MMSI' problem (multiple physical vessels briefly
    sharing/reusing an MMSI, or AIS spoofing artifacts) via 3D (lat, lon,
    time) DBSCAN run independently within each MMSI group. Adds a
    `track_id` column distinguishing sub-tracks within one MMSI.
    The time axis is scaled by an arbitrary factor (here x10) relative to
    the spatial axes so that eps_km has a comparable effect across both --
    tune this jointly with eps_km against a few known-good tracks, don't
    treat either as a fixed physical constant.
    """
    out = []
    for mmsi, g in ais_df.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").copy()
        t_hours = (g["BaseDateTime"] - g["BaseDateTime"].min()).dt.total_seconds() / 3600.0
        coords = np.column_stack([
            g["LAT"].to_numpy() * km_per_lat_deg,
            g["LON"].to_numpy() * km_per_lat_deg * np.cos(np.radians(g["LAT"].to_numpy())),
            t_hours.to_numpy() * 10.0,
        ])
        labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit(coords).labels_
        g["track_id"] = [f"{mmsi}_{l}" for l in labels]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def vessel_behavior_features(track_df):
    """One row of behavioral features per track_id, for anomaly detection."""
    rows = []
    for track_id, g in track_df.groupby("track_id"):
        g = g.sort_values("BaseDateTime")
        sog = g["SOG"].to_numpy()
        cog = g["COG"].to_numpy()
        rows.append({
            "track_id": track_id,
            "sog_mean": sog.mean(), "sog_std": sog.std() if len(sog) > 1 else 0.0,
            "course_deviation_std": np.std(np.diff(cog)) if len(cog) > 1 else 0.0,
            "n_stops": int((sog < 0.5).sum()),
            "max_slowdown": float(np.min(np.diff(sog))) if len(sog) > 1 else 0.0,
        })
    return pd.DataFrame(rows).set_index("track_id")


def flag_tier1_candidates(behavior_df, contamination=0.05):
    """
    Isolation Forest anomaly score gives you Tier-1 candidates without
    needing labeled anomalies. Layer a Random Forest in once you have any
    confirmed incident's vessel behavior to learn from -- but with ~5
    confirmed incidents total across the whole project, don't expect that
    RF half to generalize much beyond a hand-checkable sanity filter; treat
    Isolation Forest's unsupervised score as the real workhorse here.
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    behavior_df = behavior_df.copy()
    iso.fit(behavior_df)
    behavior_df["anomaly_score"] = iso.decision_function(behavior_df)
    behavior_df["is_tier1_candidate"] = iso.predict(behavior_df[["sog_mean", "sog_std",
                                                                   "course_deviation_std",
                                                                   "n_stops", "max_slowdown"]]) == -1
    return behavior_df


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "MMSI": rng.integers(100000000, 100000010, n),
        "BaseDateTime": pd.date_range("2026-03-15", periods=n, freq="5min"),
        "LAT": 28 + rng.normal(0, 0.1, n), "LON": -90 + rng.normal(0, 0.1, n),
        "SOG": rng.uniform(0, 15, n), "COG": rng.uniform(0, 360, n),
    })
    cleaned = clean_mixed_trajectories(df)
    behavior = vessel_behavior_features(cleaned)
    flagged = flag_tier1_candidates(behavior)
    print(flagged["is_tier1_candidate"].sum(), "Tier-1 candidates out of", len(flagged), "tracks")
