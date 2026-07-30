"""Composite confidence scoring for bilge-dump vessel attribution."""

from __future__ import annotations

from datetime import datetime

import numpy as np


def _bounded_unit(value: float, name: str) -> float:
    """Validate a finite score-like value and clip it to the unit interval."""
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"{name} must be finite; got {value}.")
    return float(np.clip(score, 0.0, 1.0))


def calculate_composite_score(
    s_drift: float,
    s_ais_anomaly: float,
    s_morphology: float,
    s_temporal: float,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> float:
    """Calculate the Section H.4.2 composite attribution confidence.

    The implemented formula is:

    ``C = w1*S_drift + w2*S_AIS_anomaly + w3*S_morphology + w4*S_temporal``

    Args:
        s_drift: Drift-consistency score in ``[0, 1]``.
        s_ais_anomaly: AIS behavioral anomaly score in ``[0, 1]``.
        s_morphology: Slick-to-course morphology alignment score in ``[0, 1]``.
        s_temporal: Night-time temporal prior score in ``[0, 1]``.
        weights: Four score weights, defaulting to ``(0.4, 0.3, 0.2, 0.1)``.

    Returns:
        Overall confidence that the candidate vessel caused the bilge dump,
        clipped to ``[0.0, 1.0]``.
    """
    if len(weights) != 4:
        raise ValueError(f"weights must contain exactly four values; got {weights}.")

    weight_arr = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weight_arr)):
        raise ValueError(f"weights must be finite; got {weights}.")
    if np.any(weight_arr < 0.0):
        raise ValueError(f"weights must be non-negative; got {weights}.")

    score_arr = np.asarray(
        [
            _bounded_unit(s_drift, "s_drift"),
            _bounded_unit(s_ais_anomaly, "s_ais_anomaly"),
            _bounded_unit(s_morphology, "s_morphology"),
            _bounded_unit(s_temporal, "s_temporal"),
        ],
        dtype=np.float64,
    )
    composite = float(np.dot(weight_arr, score_arr))
    return float(np.clip(composite, 0.0, 1.0))


def compute_morphology_alignment(
    slick_major_axis_angle_deg: float,
    vessel_course_over_ground_deg: float,
) -> float:
    """Score angular alignment between slick elongation and vessel heading.

    Slick major-axis orientation is an undirected axis: 0 degrees and 180
    degrees describe the same elongation. Vessel COG is directional, but for
    attribution the reverse heading is still aligned with the same slick axis.
    This function therefore uses the absolute cosine of the smallest angular
    difference.

    Args:
        slick_major_axis_angle_deg: Slick major-axis bearing in degrees.
        vessel_course_over_ground_deg: Vessel course-over-ground in degrees.

    Returns:
        Alignment score in ``[0, 1]`` where ``1`` is parallel or anti-parallel
        and ``0`` is perpendicular.
    """
    slick_angle = float(slick_major_axis_angle_deg)
    cog_angle = float(vessel_course_over_ground_deg)
    if not np.isfinite(slick_angle) or not np.isfinite(cog_angle):
        raise ValueError(
            "Angles must be finite: "
            f"slick_major_axis_angle_deg={slick_major_axis_angle_deg}, "
            f"vessel_course_over_ground_deg={vessel_course_over_ground_deg}."
        )

    diff = (slick_angle - cog_angle + 180.0) % 360.0 - 180.0
    score = abs(np.cos(np.deg2rad(diff)))
    return float(np.clip(score, 0.0, 1.0))


def compute_temporal_weight(acquisition_time_local: str | datetime | np.datetime64) -> float:
    """Return the bilge-dump night-time prior for a local acquisition time.

    Args:
        acquisition_time_local: Local SAR acquisition time. Strings are parsed
            with ``datetime.fromisoformat`` after translating a trailing ``Z``
            to ``+00:00``; ``numpy.datetime64`` values are supported as well.

    Returns:
        ``1.0`` when the local hour is in the night-time interval
        ``20:00 <= hour <= 23:59`` or ``00:00 <= hour < 06:00``; otherwise
        ``0.5``.
    """
    if isinstance(acquisition_time_local, datetime):
        dt = acquisition_time_local
    elif isinstance(acquisition_time_local, np.datetime64):
        seconds = acquisition_time_local.astype("datetime64[s]").astype(int)
        dt = datetime.utcfromtimestamp(int(seconds))
    else:
        text = str(acquisition_time_local).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Could not parse acquisition_time_local={acquisition_time_local!r}.") from exc

    hour = int(dt.hour)
    return 1.0 if hour >= 20 or hour < 6 else 0.5


__all__ = [
    "calculate_composite_score",
    "compute_morphology_alignment",
    "compute_temporal_weight",
]
