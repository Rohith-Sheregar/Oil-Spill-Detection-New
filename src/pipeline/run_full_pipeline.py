"""Unified end-to-end CLI for Sentinel-1 SAR and AIS bilge-dump attribution.

This script stitches Modules 1 through 4 into one forensic run:

1. Module 1: Sentinel-1 oil-slick segmentation.
2. Module 2: Look-alike discrimination and bilge-patch filtering.
3. Module 3: AIS vessel candidate extraction and anomaly scoring.
4. Module 4: Bidirectional drift attribution and composite confidence scoring.

Example:
    python -m src.pipeline.run_full_pipeline \
        --sar-tiff /path/to/scene.tiff \
        --ais-csv /path/to/noaa_ais.csv \
        --m1-weights results/module1/checkpoints/best_model.pt \
        --m2-weights results/module2/checkpoints/lookalike_rf.joblib \
        --metocean-nc /path/to/forcing.nc \
        --sar-time 2026-03-15T09:00:00Z \
        --output-dir results/forensic_reports
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ais_attribution.pipeline import Module3Pipeline
from src.drift.lagrangian_drift import (
    compute_drift_similarity,
    run_backward_simulation,
    run_forward_simulation,
)
from src.drift.scoring import (
    calculate_composite_score,
    compute_morphology_alignment,
    compute_temporal_weight,
)
from src.lookalike.bilge_filter import (
    DEFAULT_MAX_AREA_KM2,
    DEFAULT_MIN_ELONGATION,
    DEFAULT_NIGHT_BOOST,
    DEFAULT_PROB_THRESHOLD,
    apply_bilge_filter,
)
from src.lookalike.classifier import LookalikeClassifier
from src.lookalike.features import FEATURE_NAMES, extract_scene_features
from src.lookalike.morphology import close_and_extract
from src.preprocessing.polsar_decomp import db_to_linear

log = logging.getLogger(__name__)


@dataclass
class SarScene:
    """Loaded Sentinel-1 image arrays and optional geospatial metadata."""

    path: Path
    vv_db: np.ndarray
    vh_db: np.ndarray
    transform: Any | None
    crs: Any | None
    timestamp: pd.Timestamp | None


def _setup_logging(output_dir: Path) -> Path:
    """Configure console and file logging for a full pipeline run."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"full_pipeline_{stamp}.log"
    fmt = "%(asctime)s  %(levelname)-7s  %(name)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def _parse_timestamp(value: str | None) -> pd.Timestamp | None:
    """Parse an optional timestamp string as UTC."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = pd.Timestamp(text)
    except Exception as exc:
        raise ValueError(f"Could not parse timestamp {value!r}. Use ISO-8601, e.g. 2026-03-15T09:00:00Z.") from exc
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _infer_timestamp_from_text(text: str) -> pd.Timestamp | None:
    """Infer a Sentinel-style acquisition timestamp from tags or file names."""
    if not text:
        return None

    patterns = [
        (r"(20\d{6}T\d{6})", "%Y%m%dT%H%M%S"),
        (r"(20\d{6}_\d{6})", "%Y%m%d_%H%M%S"),
        (r"(20\d{2}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2})", "%Y:%m:%d %H:%M:%S"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return pd.Timestamp(datetime.strptime(match.group(1), fmt), tz="UTC")
            except Exception:
                continue

    iso_match = re.search(
        r"(20\d{2}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)",
        text,
    )
    if iso_match:
        try:
            return _parse_timestamp(iso_match.group(1))
        except Exception:
            return None
    return None


def _normalise_tiff_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vv_db, vh_db)`` arrays from common TIFF layouts."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return arr, arr.copy()
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF array shape {arr.shape}; expected 2D or 3D.")
    if arr.shape[0] <= 8 and arr.shape[1] > 8:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] < 2:
        return arr[..., 0], arr[..., 0].copy()
    return arr[..., 0], arr[..., 1]


def _load_sar_scene(path: Path, sar_time: str | None) -> SarScene:
    """Load a SAR TIFF with rasterio metadata when available."""
    if not path.exists():
        raise FileNotFoundError(f"SAR TIFF not found: {path}")

    timestamp = _parse_timestamp(sar_time)
    transform: Any | None = None
    crs: Any | None = None
    tags_text = ""

    try:
        import rasterio
        from affine import Affine

        with rasterio.open(path) as src:
            if src.count >= 2:
                vv_db = src.read(1).astype(np.float32)
                vh_db = src.read(2).astype(np.float32)
            elif src.count == 1:
                vv_db = src.read(1).astype(np.float32)
                vh_db = vv_db.copy()
            else:
                raise ValueError(f"Raster has no readable bands: {path}")
            if src.transform != Affine.identity():
                transform = src.transform
            crs = src.crs
            tags = src.tags()
            tags_text = " ".join(f"{key}={value}" for key, value in tags.items())
    except Exception as exc:
        log.info("rasterio load did not provide a usable raster; falling back to tifffile: %s", exc)
        import tifffile

        vv_db, vh_db = _normalise_tiff_array(tifffile.imread(str(path)))
        try:
            with tifffile.TiffFile(str(path)) as tif:
                tags_text = " ".join(str(tag.value) for tag in tif.pages[0].tags.values())
        except Exception:
            tags_text = ""

    if timestamp is None:
        timestamp = _infer_timestamp_from_text(tags_text) or _infer_timestamp_from_text(path.name)

    return SarScene(
        path=path,
        vv_db=np.asarray(vv_db, dtype=np.float32),
        vh_db=np.asarray(vh_db, dtype=np.float32),
        transform=transform,
        crs=crs,
        timestamp=timestamp,
    )


def _tile_starts(size: int, tile_size: int, stride: int) -> list[int]:
    """Compute deterministic tile starts that always cover an image dimension."""
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, stride))
    final = size - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _torch_load(path: Path, device: Any) -> Any:
    """Load a PyTorch checkpoint across torch versions."""
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Find a model state dict inside common checkpoint payloads."""
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Module 1 checkpoint does not contain a recognizable state dict.")


def _run_module1_inference(
    scene: SarScene,
    weights_path: Path,
    device_name: str,
    patch_size: int,
    stride: int,
    threshold: float,
    wind_speed_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled Module 1 segmentation and return mask plus probability map."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Module 1 weights not found: {weights_path}")

    import torch

    from src.models.deeplab_scse import DeepLabV3PlusSCSE
    from src.preprocessing.band_stack import build_5band_stack

    device = torch.device(device_name)
    stack = build_5band_stack(
        scene.vv_db,
        scene.vh_db,
        wind_speed_ms=wind_speed_ms,
        normalize=True,
    )
    channels, height, width = stack.shape
    model = DeepLabV3PlusSCSE(
        in_channels=channels,
        classes=1,
        input_size=patch_size,
        encoder_weights=None,
    )
    checkpoint = _torch_load(weights_path, device)
    state_dict = _extract_state_dict(checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        log.warning(
            "Module 1 checkpoint loaded with missing=%d unexpected=%d keys.",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
    model.to(device).eval()

    prob_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    row_starts = _tile_starts(height, patch_size, stride)
    col_starts = _tile_starts(width, patch_size, stride)
    total_tiles = len(row_starts) * len(col_starts)
    log.info("Module 1 inference: %d tiles over image %dx%d.", total_tiles, height, width)

    with torch.no_grad():
        for tile_index, row in enumerate(row_starts):
            for col in col_starts:
                row_end = min(row + patch_size, height)
                col_end = min(col + patch_size, width)
                tile = stack[:, row:row_end, col:col_end]
                pad_h = patch_size - tile.shape[1]
                pad_w = patch_size - tile.shape[2]
                if pad_h or pad_w:
                    tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                tensor = torch.from_numpy(tile[None, ...]).to(device)
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
                probs = probs[: row_end - row, : col_end - col]
                prob_sum[row:row_end, col:col_end] += probs
                weight_sum[row:row_end, col:col_end] += 1.0
            if (tile_index + 1) % 5 == 0 or tile_index == len(row_starts) - 1:
                log.info("Module 1 tiled rows complete: %d/%d.", tile_index + 1, len(row_starts))

    probabilities = prob_sum / np.maximum(weight_sum, 1.0)
    mask = (probabilities >= threshold).astype(np.uint8)
    return mask, probabilities


def _pixel_to_lonlat(
    row: float,
    col: float,
    transform: Any | None,
    crs: Any | None,
) -> tuple[float, float] | None:
    """Convert pixel coordinates to WGS84 lon/lat when georeferencing exists."""
    if transform is None:
        return None
    try:
        from rasterio.transform import xy as rio_xy

        x, y = rio_xy(transform, int(row), int(col))
        lon = float(x)
        lat = float(y)
        if crs is not None:
            epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
            if epsg != 4326:
                from pyproj import Transformer

                transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(lon, lat)
        if not np.isfinite(lon) or not np.isfinite(lat):
            return None
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            return None
        return float(lon), float(lat)
    except Exception as exc:
        log.debug("Pixel-to-lonlat transform failed: %s", exc)
        return None


def _region_axis_angle_deg(region: Any) -> float:
    """Approximate slick major-axis bearing in degrees from image regionprops."""
    orientation = float(getattr(region, "orientation", 0.0))
    return float((90.0 - np.degrees(orientation)) % 180.0)


def _component_metadata(scene: SarScene, regions: list[Any]) -> dict[int, dict[str, Any]]:
    """Build per-component metadata used by Modules 3 and 4."""
    metadata: dict[int, dict[str, Any]] = {}
    for region in regions:
        centroid_row, centroid_col = region.centroid
        lonlat = _pixel_to_lonlat(centroid_row, centroid_col, scene.transform, scene.crs)
        lon = lonlat[0] if lonlat is not None else np.nan
        lat = lonlat[1] if lonlat is not None else np.nan
        metadata[int(region.label)] = {
            "centroid_lon": lon,
            "centroid_lat": lat,
            "slick_major_axis_angle_deg": _region_axis_angle_deg(region),
            "centroid_row": float(centroid_row),
            "centroid_col": float(centroid_col),
            "bbox_min_row": int(region.bbox[0]),
            "bbox_min_col": int(region.bbox[1]),
            "bbox_max_row": int(region.bbox[2]),
            "bbox_max_col": int(region.bbox[3]),
        }
    return metadata


def _add_component_metadata(df: pd.DataFrame, metadata: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """Attach component lon/lat and orientation columns to a feature table."""
    if df.empty:
        return df
    result = df.copy()
    for column in (
        "centroid_lon",
        "centroid_lat",
        "slick_major_axis_angle_deg",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row",
        "bbox_max_col",
    ):
        result[column] = np.nan
    for idx, row in result.iterrows():
        meta = metadata.get(int(row["component_label"]), {})
        for key, value in meta.items():
            if key in result.columns:
                result.at[idx, key] = value
    return result


def _load_module2_model(path: Path) -> Any:
    """Load the Module 2 classifier, accepting wrapper or raw joblib payloads."""
    if not path.exists():
        raise FileNotFoundError(f"Module 2 weights not found: {path}")
    try:
        return LookalikeClassifier.load(path)
    except Exception as wrapper_exc:
        log.info("Module 2 wrapper load failed, trying raw joblib payload: %s", wrapper_exc)
        import joblib

        payload = joblib.load(path)
        if isinstance(payload, dict) and "rf" in payload:
            return payload["rf"]
        if hasattr(payload, "predict_proba"):
            return payload
        raise ValueError(f"Unsupported Module 2 model payload in {path}.") from wrapper_exc


def _predict_module2_proba(model: Any, features_df: pd.DataFrame) -> pd.DataFrame:
    """Return Module 2 probabilities for wrapper or sklearn-like models."""
    if isinstance(model, LookalikeClassifier):
        return model.predict_proba(features_df)
    missing = [feature for feature in FEATURE_NAMES if feature not in features_df.columns]
    if missing:
        raise KeyError(f"Module 2 feature table missing columns: {missing}")
    x = features_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    proba = model.predict_proba(x)
    if proba.ndim != 2:
        raise ValueError("Module 2 predict_proba returned a non-2D array.")
    if proba.shape[1] == 1:
        prob_oil = proba[:, 0]
        prob_lookalike = 1.0 - prob_oil
    else:
        prob_lookalike = proba[:, 0]
        prob_oil = proba[:, 1]
    pred_label = (prob_oil >= prob_lookalike).astype(int)
    return pd.DataFrame(
        {
            "prob_lookalike": prob_lookalike,
            "prob_oil": prob_oil,
            "pred_label": pred_label,
        },
        index=features_df.index,
    )


def _local_acquisition_time(
    sar_time_utc: pd.Timestamp,
    lon: float | None,
    offset_hours: float | None,
) -> pd.Timestamp:
    """Estimate local acquisition time using an explicit or longitude-derived offset."""
    ts = sar_time_utc.tz_convert("UTC") if sar_time_utc.tzinfo is not None else sar_time_utc.tz_localize("UTC")
    if offset_hours is not None:
        offset = float(offset_hours)
    elif lon is not None and np.isfinite(lon):
        offset = float(np.clip(round(lon / 15.0), -12.0, 14.0))
    else:
        offset = 0.0
    return ts + pd.Timedelta(hours=offset)


def _jsonable(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _is_wgs84_crs(crs: Any | None) -> bool:
    """Return True when a raster CRS is EPSG:4326."""
    if crs is None:
        return False
    try:
        return crs.to_epsg() == 4326
    except Exception:
        return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with robust conversion of numpy/pandas values."""
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _save_intermediate_arrays(
    output_dir: Path,
    mask: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, str]:
    """Persist Module 1 raster outputs."""
    artifacts: dict[str, str] = {}
    mask_npy = output_dir / "module1_mask.npy"
    prob_npy = output_dir / "module1_probability.npy"
    np.save(mask_npy, mask.astype(np.uint8))
    np.save(prob_npy, probabilities.astype(np.float32))
    artifacts["module1_mask_npy"] = str(mask_npy)
    artifacts["module1_probability_npy"] = str(prob_npy)
    try:
        import tifffile

        mask_tif = output_dir / "module1_mask.tif"
        tifffile.imwrite(str(mask_tif), mask.astype(np.uint8))
        artifacts["module1_mask_tif"] = str(mask_tif)
    except Exception as exc:
        log.warning("Could not write Module 1 mask TIFF: %s", exc)
    return artifacts


def _normalise_weights(text: str) -> tuple[float, float, float, float]:
    """Parse four comma-separated composite score weights."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated weights, e.g. 0.4,0.3,0.2,0.1")
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Composite weights must be numeric.") from exc


def _nearest_vessel_pose(
    tier1_row: pd.Series,
    clean_ais_gdf: Any,
    raw_ais_gdf: Any,
    discharge_time: pd.Timestamp,
) -> dict[str, Any] | None:
    """Find a candidate vessel position and COG nearest estimated discharge time."""
    track_id = str(tier1_row.name)
    groups: list[pd.DataFrame] = []
    if clean_ais_gdf is not None and len(clean_ais_gdf) and "track_id" in clean_ais_gdf.columns:
        track_group = clean_ais_gdf[clean_ais_gdf["track_id"].astype(str) == track_id]
        if len(track_group):
            groups.append(track_group)
    mmsi = tier1_row.get("mmsi", None)
    if raw_ais_gdf is not None and len(raw_ais_gdf) and mmsi is not None and "MMSI" in raw_ais_gdf.columns:
        mmsi_group = raw_ais_gdf[raw_ais_gdf["MMSI"].astype(str) == str(mmsi)]
        if len(mmsi_group):
            groups.append(mmsi_group)

    for group in groups:
        required = {"LON", "LAT", "BaseDateTime"}
        if not required.issubset(group.columns):
            continue
        g = group.copy()
        times = pd.to_datetime(g["BaseDateTime"], utc=True, errors="coerce")
        valid = times.notna()
        if not valid.any():
            continue
        g = g.loc[valid].copy()
        times = times.loc[valid]
        target = discharge_time.tz_convert("UTC") if discharge_time.tzinfo is not None else discharge_time.tz_localize("UTC")
        nearest_idx = (times - target).abs().idxmin()
        nearest = g.loc[nearest_idx]
        return {
            "track_id": track_id,
            "mmsi": nearest.get("MMSI", mmsi),
            "vessel_lon": float(nearest["LON"]),
            "vessel_lat": float(nearest["LAT"]),
            "vessel_cog_deg": float(nearest.get("COG", np.nan)),
            "vessel_sog": float(nearest.get("SOG", np.nan)),
            "vessel_time": pd.Timestamp(nearest["BaseDateTime"]).isoformat(),
        }
    return None


def _has_valid_lonlat(row: pd.Series) -> bool:
    """Return True if a spill row has finite lon/lat."""
    return bool(
        np.isfinite(float(row.get("centroid_lon", np.nan)))
        and np.isfinite(float(row.get("centroid_lat", np.nan)))
    )


def _run_module4_for_spill(
    spill_row: pd.Series,
    module3_result: dict[str, Any],
    sar_time: pd.Timestamp,
    wind_nc: str | None,
    current_nc: str | None,
    duration_hours: float,
    num_particles: int,
    decay_km: float,
    weights: tuple[float, float, float, float],
    local_time: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Run drift and composite scoring for all Tier-1 AIS candidates."""
    tier1_df = module3_result["tier1_candidates"]
    if tier1_df.empty:
        return []

    records: list[dict[str, Any]] = []
    discharge_time = sar_time - pd.Timedelta(hours=duration_hours)
    forcing_available = wind_nc is not None or current_nc is not None
    spill_lon = float(spill_row["centroid_lon"])
    spill_lat = float(spill_row["centroid_lat"])
    slick_axis = float(spill_row.get("slick_major_axis_angle_deg", np.nan))
    temporal_score = compute_temporal_weight(local_time.to_pydatetime())

    for _, candidate in tier1_df.iterrows():
        pose = _nearest_vessel_pose(
            candidate,
            module3_result.get("clean_ais_gdf"),
            module3_result.get("raw_ais_gdf"),
            discharge_time,
        )
        if pose is None:
            log.warning("Could not derive vessel pose for Tier-1 candidate %s; skipping Module 4 row.", candidate.name)
            continue

        s_ais = float(candidate.get("S_AIS_anomaly", 0.0))
        try:
            s_morphology = compute_morphology_alignment(slick_axis, pose["vessel_cog_deg"])
        except ValueError:
            s_morphology = 0.0

        s_drift = 0.0
        drift_status = "skipped_no_metocean"
        drift_forward_centroid: tuple[float, float] | None = None
        drift_backward_centroid: tuple[float, float] | None = None
        if forcing_available:
            try:
                forward_particles = run_forward_simulation(
                    pose["vessel_lon"],
                    pose["vessel_lat"],
                    discharge_time.to_pydatetime(),
                    wind_nc,
                    current_nc,
                    duration_hours=duration_hours,
                    num_particles=num_particles,
                )
                backward_particles = run_backward_simulation(
                    spill_lon,
                    spill_lat,
                    sar_time.to_pydatetime(),
                    wind_nc,
                    current_nc,
                    duration_hours=duration_hours,
                    num_particles=num_particles,
                )
                s_drift = compute_drift_similarity(
                    forward_particles,
                    backward_particles,
                    decay_km=decay_km,
                )
                drift_forward_centroid = (
                    float(np.mean(forward_particles[0])),
                    float(np.mean(forward_particles[1])),
                )
                drift_backward_centroid = (
                    float(np.mean(backward_particles[0])),
                    float(np.mean(backward_particles[1])),
                )
                drift_status = "ok"
            except Exception as exc:
                drift_status = f"failed: {exc}"
                log.warning("Module 4 drift failed for candidate %s: %s", candidate.name, exc)

        composite = calculate_composite_score(
            s_drift=s_drift,
            s_ais_anomaly=s_ais,
            s_morphology=s_morphology,
            s_temporal=temporal_score,
            weights=weights,
        )
        records.append(
            {
                "spill_component_label": int(spill_row["component_label"]),
                "spill_centroid_lon": spill_lon,
                "spill_centroid_lat": spill_lat,
                "slick_major_axis_angle_deg": slick_axis,
                "sar_time": sar_time.isoformat(),
                "estimated_discharge_time": discharge_time.isoformat(),
                "track_id": str(candidate.name),
                "mmsi": pose.get("mmsi"),
                "vessel_lon": pose["vessel_lon"],
                "vessel_lat": pose["vessel_lat"],
                "vessel_cog_deg": pose["vessel_cog_deg"],
                "vessel_sog": pose["vessel_sog"],
                "vessel_time": pose["vessel_time"],
                "S_drift": s_drift,
                "S_AIS_anomaly": s_ais,
                "S_morphology": s_morphology,
                "S_temporal": temporal_score,
                "composite_score": composite,
                "drift_status": drift_status,
                "drift_forward_centroid_lon": drift_forward_centroid[0] if drift_forward_centroid else np.nan,
                "drift_forward_centroid_lat": drift_forward_centroid[1] if drift_forward_centroid else np.nan,
                "drift_backward_centroid_lon": drift_backward_centroid[0] if drift_backward_centroid else np.nan,
                "drift_backward_centroid_lat": drift_backward_centroid[1] if drift_backward_centroid else np.nan,
            }
        )

    return records


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Modules 1-4 and write all output artifacts."""
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(output_dir)

    sar_path = Path(args.sar_tiff)
    ais_path = Path(args.ais_csv)
    m1_weights = Path(args.m1_weights)
    m2_weights = Path(args.m2_weights)
    wind_nc = args.wind_nc or args.metocean_nc
    current_nc = args.current_nc or args.metocean_nc

    if (args.spill_lon is None) != (args.spill_lat is None):
        raise ValueError("Pass both --spill-lon and --spill-lat, or neither.")
    if not ais_path.exists():
        raise FileNotFoundError(f"AIS CSV not found: {ais_path}")
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("--patch-size and --stride must both be positive.")

    log.info("Starting full pipeline.")
    log.info("SAR TIFF: %s", sar_path)
    log.info("AIS CSV: %s", ais_path)

    scene = _load_sar_scene(sar_path, args.sar_time)
    if scene.timestamp is None:
        raise ValueError(
            "SAR acquisition time could not be inferred. Provide --sar-time "
            "in ISO-8601 form, e.g. --sar-time 2026-03-15T09:00:00Z."
        )
    sar_time = scene.timestamp
    log.info("SAR acquisition time: %s", sar_time.isoformat())

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device
    log.info("Module 1 device: %s", device)

    mask, probabilities = _run_module1_inference(
        scene=scene,
        weights_path=m1_weights,
        device_name=device,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=args.m1_threshold,
        wind_speed_ms=args.wind_speed_ms,
    )
    artifacts = _save_intermediate_arrays(output_dir, mask, probabilities)
    log.info("Module 1 foreground pixels: %d.", int(mask.sum()))

    closed_mask, regions = close_and_extract(
        mask,
        iterations=args.morph_iterations,
        selem_size=args.morph_selem_size,
        min_area_px=args.min_component_px,
    )
    log.info("Module 2 morphology retained %d connected components.", len(regions))

    component_meta = _component_metadata(scene, regions)
    first_lon = next((meta["centroid_lon"] for meta in component_meta.values() if np.isfinite(meta["centroid_lon"])), np.nan)
    local_time = _local_acquisition_time(sar_time, float(first_lon) if np.isfinite(first_lon) else None, args.local_time_offset_hours)
    hour_local = int(local_time.hour)

    features_df = extract_scene_features(
        regions=regions,
        vv_db=scene.vv_db,
        vh_db=scene.vh_db,
        scene_id=sar_path.stem,
        gsd_m=args.gsd_m,
        wind_speed_ms=args.wind_speed_ms,
        hour_local=hour_local,
        scene_transform=scene.transform if _is_wgs84_crs(scene.crs) else None,
        scene_crs=str(scene.crs) if scene.crs is not None else None,
    )
    features_df = _add_component_metadata(features_df, component_meta)
    features_path = output_dir / "module2_features.csv"
    features_df.to_csv(features_path, index=False)
    artifacts["module2_features_csv"] = str(features_path)

    if features_df.empty:
        report = {
            "status": "no_module2_components",
            "message": "Module 1/2 found no connected slick components after morphology.",
            "artifacts": artifacts,
            "log_path": str(log_path),
        }
        _write_json(output_dir / "forensic_report.json", report)
        return report

    m2_model = _load_module2_model(m2_weights)
    proba_df = _predict_module2_proba(m2_model, features_df)
    module2_input = pd.concat([features_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    bilge_df = apply_bilge_filter(
        module2_input,
        prob_col="prob_oil",
        min_elongation=args.min_elongation,
        max_area_km2=args.max_area_km2,
        night_boost=args.night_boost,
        prob_threshold=args.prob_threshold,
    )
    bilge_path = output_dir / "module2_bilge_candidates.csv"
    bilge_df.to_csv(bilge_path, index=False)
    artifacts["module2_bilge_candidates_csv"] = str(bilge_path)

    verified_spills = bilge_df[bilge_df["bilge_candidate"]].copy()
    verified_spills = verified_spills.sort_values("prob_adjusted", ascending=False).head(args.top_k_spills)
    if args.spill_lon is not None and args.spill_lat is not None:
        verified_spills["centroid_lon"] = float(args.spill_lon)
        verified_spills["centroid_lat"] = float(args.spill_lat)

    if verified_spills.empty:
        report = {
            "status": "no_verified_bilge_spills",
            "message": "Module 2 produced no bilge_candidate=True patches.",
            "n_module2_geometry_passing": int(len(bilge_df)),
            "artifacts": artifacts,
            "log_path": str(log_path),
        }
        _write_json(output_dir / "forensic_report.json", report)
        return report

    invalid_geo = [int(row["component_label"]) for _, row in verified_spills.iterrows() if not _has_valid_lonlat(row)]
    if invalid_geo:
        raise ValueError(
            "Verified spill components lack WGS84 centroid coordinates: "
            f"{invalid_geo}. Provide a georeferenced TIFF or pass --spill-lon and --spill-lat."
        )

    sar_vv_linear = db_to_linear(scene.vv_db)
    module3 = Module3Pipeline(
        radius_km=args.ais_radius_km,
        window_hours=args.ais_window_hours,
        dbscan_eps_km=args.dbscan_eps_km,
        dbscan_eps_hr=args.dbscan_eps_hr,
        dbscan_min_samples=args.dbscan_min_samples,
        pixel_spacing_m=args.gsd_m,
        dark_tol_m=args.dark_tolerance_m,
    )

    all_tier1_frames: list[pd.DataFrame] = []
    dark_ship_records: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []

    for spill_index, spill_row in verified_spills.iterrows():
        local_time = _local_acquisition_time(
            sar_time,
            float(spill_row["centroid_lon"]),
            args.local_time_offset_hours,
        )
        spill_record = {
            "centroid_lon": float(spill_row["centroid_lon"]),
            "centroid_lat": float(spill_row["centroid_lat"]),
            "timestamp": sar_time,
            "component_label": int(spill_row["component_label"]),
        }
        log.info(
            "Module 3 for spill component %s at (%.5f, %.5f).",
            spill_record["component_label"],
            spill_record["centroid_lon"],
            spill_record["centroid_lat"],
        )
        module3_result = module3.run(
            verified_spill_record=spill_record,
            ais_csv_path=str(ais_path),
            sar_vv_array=sar_vv_linear,
            force_dark_ship=args.force_dark_ship,
            scene_transform=scene.transform if _is_wgs84_crs(scene.crs) else None,
        )

        tier1_df = module3_result["tier1_candidates"].copy()
        if not tier1_df.empty:
            tier1_df.insert(0, "spill_component_label", int(spill_row["component_label"]))
            tier1_reset = tier1_df.reset_index()
            first_col = tier1_reset.columns[0]
            if first_col != "track_id":
                tier1_reset = tier1_reset.rename(columns={first_col: "track_id"})
            all_tier1_frames.append(tier1_reset)

        for dark_record in module3_result["dark_ship_flags"]:
            dark_ship_records.append(
                {
                    **dark_record,
                    "spill_component_label": int(spill_row["component_label"]),
                }
            )

        attribution_records.extend(
            _run_module4_for_spill(
                spill_row=spill_row,
                module3_result=module3_result,
                sar_time=sar_time,
                wind_nc=wind_nc,
                current_nc=current_nc,
                duration_hours=args.drift_duration_hours,
                num_particles=args.num_particles,
                decay_km=args.drift_decay_km,
                weights=args.composite_weights,
                local_time=local_time,
            )
        )

    if all_tier1_frames:
        tier1_out = pd.concat(all_tier1_frames, ignore_index=True)
    else:
        tier1_out = pd.DataFrame()
    tier1_path = output_dir / "module3_tier1_candidates.csv"
    tier1_out.to_csv(tier1_path, index=False)
    artifacts["module3_tier1_candidates_csv"] = str(tier1_path)

    dark_path = output_dir / "module3_dark_ship_flags.json"
    _write_json(dark_path, {"dark_ship_flags": dark_ship_records})
    artifacts["module3_dark_ship_flags_json"] = str(dark_path)

    attribution_df = pd.DataFrame(attribution_records)
    if not attribution_df.empty:
        attribution_df = attribution_df.sort_values("composite_score", ascending=False)
    attribution_path = output_dir / "forensic_attribution.csv"
    attribution_df.to_csv(attribution_path, index=False)
    artifacts["forensic_attribution_csv"] = str(attribution_path)

    report = {
        "status": "complete",
        "elapsed_s": round(time.time() - start_time, 2),
        "sar_tiff": str(sar_path),
        "ais_csv": str(ais_path),
        "sar_time": sar_time.isoformat(),
        "local_time_reference": local_time.isoformat(),
        "n_module1_foreground_pixels": int(mask.sum()),
        "n_module2_components": int(len(features_df)),
        "n_module2_geometry_passing": int(len(bilge_df)),
        "n_verified_spills": int(len(verified_spills)),
        "n_tier1_candidates": int(len(tier1_out)),
        "n_dark_ship_flags": int(len(dark_ship_records)),
        "n_attribution_records": int(len(attribution_df)),
        "top_attribution": attribution_df.head(1).to_dict(orient="records") if not attribution_df.empty else [],
        "drift_forcing": {
            "wind_nc": wind_nc,
            "current_nc": current_nc,
            "metocean_nc": args.metocean_nc,
        },
        "artifacts": artifacts,
        "log_path": str(log_path),
    }
    _write_json(output_dir / "forensic_report.json", report)
    log.info("Full pipeline complete. Report: %s", output_dir / "forensic_report.json")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Modules 1-4 for illegal bilge-dump detection and vessel attribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sar-tiff", required=True, help="Raw Sentinel-1 TIFF/GeoTIFF scene.")
    parser.add_argument("--ais-csv", required=True, help="NOAA AIS CSV file.")
    parser.add_argument("--m1-weights", required=True, help="Module 1 segmentation checkpoint (.pt).")
    parser.add_argument("--m2-weights", required=True, help="Module 2 RF classifier checkpoint (.joblib).")
    parser.add_argument("--output-dir", required=True, help="Directory for forensic report outputs.")
    parser.add_argument("--sar-time", default=None, help="SAR acquisition time if not embedded in TIFF/filename.")
    parser.add_argument("--metocean-nc", default=None, help="Optional combined wind/current NetCDF forcing file.")
    parser.add_argument("--wind-nc", default=None, help="Optional ERA5 wind NetCDF. Overrides --metocean-nc for wind.")
    parser.add_argument("--current-nc", default=None, help="Optional CMEMS current NetCDF. Overrides --metocean-nc for currents.")
    parser.add_argument("--spill-lon", type=float, default=None, help="Manual spill centroid longitude override.")
    parser.add_argument("--spill-lat", type=float, default=None, help="Manual spill centroid latitude override.")

    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--patch-size", type=int, default=512, help="Module 1 tiled inference patch size.")
    parser.add_argument("--stride", type=int, default=384, help="Module 1 tiled inference stride.")
    parser.add_argument("--m1-threshold", type=float, default=0.5, help="Module 1 probability threshold.")
    parser.add_argument("--wind-speed-ms", type=float, default=7.0, help="Wind speed fallback used in Module 1/2 features.")
    parser.add_argument("--gsd-m", type=float, default=10.0, help="SAR ground sampling distance in metres.")
    parser.add_argument("--local-time-offset-hours", type=float, default=None, help="Local time offset from UTC; defaults to longitude-derived offset.")

    parser.add_argument("--morph-iterations", type=int, default=2, help="Binary closing iterations for Module 2.")
    parser.add_argument("--morph-selem-size", type=int, default=5, help="Square structuring element size for Module 2 closing.")
    parser.add_argument("--min-component-px", type=int, default=10, help="Minimum connected component area.")
    parser.add_argument("--min-elongation", type=float, default=DEFAULT_MIN_ELONGATION, help="Bilge geometry minimum elongation.")
    parser.add_argument("--max-area-km2", type=float, default=DEFAULT_MAX_AREA_KM2, help="Bilge geometry maximum area.")
    parser.add_argument("--night-boost", type=float, default=DEFAULT_NIGHT_BOOST, help="Module 2 night-time probability boost.")
    parser.add_argument("--prob-threshold", type=float, default=DEFAULT_PROB_THRESHOLD, help="Module 2 bilge probability threshold.")
    parser.add_argument("--top-k-spills", type=int, default=3, help="Maximum verified spill patches to send to Modules 3/4.")

    parser.add_argument("--ais-radius-km", type=float, default=50.0, help="AIS spatial search radius around spill.")
    parser.add_argument("--ais-window-hours", type=float, default=6.0, help="AIS temporal half-window around SAR time.")
    parser.add_argument("--dbscan-eps-km", type=float, default=2.0, help="Module 3 DBSCAN spatial epsilon.")
    parser.add_argument("--dbscan-eps-hr", type=float, default=0.5, help="Module 3 DBSCAN temporal epsilon.")
    parser.add_argument("--dbscan-min-samples", type=int, default=5, help="Module 3 DBSCAN min_samples.")
    parser.add_argument("--dark-tolerance-m", type=float, default=500.0, help="SAR-to-AIS dark ship match tolerance.")
    parser.add_argument("--force-dark-ship", action="store_true", help="Always run Module 3 dark-ship fallback.")

    parser.add_argument("--drift-duration-hours", type=float, default=6.0, help="Module 4 forward/backward drift duration.")
    parser.add_argument("--num-particles", type=int, default=500, help="Module 4 drift particle count.")
    parser.add_argument("--drift-decay-km", type=float, default=20.0, help="Exponential decay distance for S_drift.")
    parser.add_argument(
        "--composite-weights",
        type=_normalise_weights,
        default=(0.4, 0.3, 0.2, 0.1),
        help="Four comma-separated weights for drift,AIS,morphology,temporal.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    try:
        report = run_pipeline(_parse_args(argv))
        print(json.dumps(_jsonable({"status": report.get("status"), "report": report.get("artifacts", {})}), indent=2))
        return 0
    except Exception as exc:
        logging.exception("Full pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
