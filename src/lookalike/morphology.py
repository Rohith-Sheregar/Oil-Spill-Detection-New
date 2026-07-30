"""
Module 2 — Morphological pre-processing for bilge-dump discrimination.

Applies 2-iteration binary closing to Module 1 segmentation masks to fill
fragmentation artifacts, then extracts connected components as scikit-image
RegionProperties objects for downstream feature extraction.

Physics rationale (Chang et al. 2024)
--------------------------------------
Bilge dumping produces a continuous narrow streak. The deep-learning boundary
detector may fragment it into disconnected blobs when the streak crosses a
calm-water specular reflection region. Two iterations of binary closing with a
5×5 square structuring element bridge gaps up to ~25m (2.5 pixels × 10m GSD)
without merging genuinely separate slicks (which are typically >100m apart).

References
----------
- Chang et al. (2024): morphological closing for bilge-dump SAR post-processing
- Skimage docs: skimage.measure.regionprops
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_closing, label as nd_label
from skimage.measure import regionprops, label as sk_label

log = logging.getLogger(__name__)

# Default structuring element: 5×5 square (all True).
# A disk might round corners more naturally, but the square is exactly what
# Chang et al. (2024) describe for bilge-dump post-processing.
_DEFAULT_SELEM = np.ones((5, 5), dtype=bool)


# ─── Core morphological operations ────────────────────────────────────────────

def apply_bilge_closing(
    binary_mask: np.ndarray,
    iterations: int = 2,
    selem_size: int = 5,
) -> np.ndarray:
    """
    Apply binary morphological closing to a Module 1 segmentation mask.

    The closing fills small gaps/holes without significantly altering the
    overall slick geometry. Two iterations are used per Chang et al. (2024)
    as the minimum needed to bridge typical fragmentation artifacts.

    Parameters
    ----------
    binary_mask : (H, W) array-like, dtype convertible to bool
        Raw binary prediction from Module 1 (1 = oil candidate, 0 = background).
        May contain dtype uint8, int32, float32, etc.
    iterations  : int
        Number of closing iterations. Default 2 per Chang et al. (2024).
        Set to 0 to skip closing (pass-through, useful for ablation studies).
    selem_size  : int
        Side length of the square structuring element in pixels.
        Default 5 → covers 50m at 10m GSD. Must be odd and ≥ 1.

    Returns
    -------
    closed : (H, W) bool ndarray
        Morphologically closed binary mask.

    Raises
    ------
    ValueError
        If selem_size is even or < 1.
    """
    if selem_size < 1 or selem_size % 2 == 0:
        raise ValueError(
            f"selem_size must be a positive odd integer; got {selem_size}."
        )

    mask = np.asarray(binary_mask, dtype=bool)

    if iterations == 0:
        log.debug("apply_bilge_closing: iterations=0, returning input unchanged.")
        return mask

    selem = np.ones((selem_size, selem_size), dtype=bool)
    closed = mask.copy()
    for _ in range(iterations):
        closed = binary_closing(closed, structure=selem)

    n_added = int(closed.sum()) - int(mask.sum())
    log.debug(
        "apply_bilge_closing: %d iter, selem=%dx%d → +%d pixels filled.",
        iterations, selem_size, selem_size, n_added,
    )
    return closed.astype(bool)


def extract_components(
    closed_mask: np.ndarray,
    min_area_px: int = 10,
    connectivity: int = 2,
) -> list:
    """
    Label connected components in a binary mask and return their RegionProperties.

    Parameters
    ----------
    closed_mask  : (H, W) bool array — output of apply_bilge_closing()
    min_area_px  : int
        Minimum component area in pixels to retain. Components smaller than
        this are treated as noise. Default 10 pixels (= 1000 m² at 10m GSD).
        Prevents the RF from operating on single-pixel detections.
    connectivity : int
        1 = 4-connectivity, 2 = 8-connectivity (default).
        8-connectivity is standard for elongated streak detection.

    Returns
    -------
    regions : list[skimage.measure._regionprops.RegionProperties]
        One entry per connected component passing the area threshold.
        Each entry exposes: .label, .area, .bbox, .centroid,
        .major_axis_length, .minor_axis_length, .perimeter, .eccentricity,
        .coords (pixel coordinates), .image (sub-image bool mask).

    Notes
    -----
    If closed_mask is all-zero (no detections), returns an empty list.
    """
    mask_bool = np.asarray(closed_mask, dtype=bool)
    if not mask_bool.any():
        log.debug("extract_components: mask is all-zero, no components.")
        return []

    labelled = sk_label(mask_bool, connectivity=connectivity)
    all_regions = regionprops(labelled)

    kept = [r for r in all_regions if r.area >= min_area_px]
    log.debug(
        "extract_components: %d total components, %d kept (min_area_px=%d).",
        len(all_regions), len(kept), min_area_px,
    )
    return kept


def close_and_extract(
    binary_mask: np.ndarray,
    iterations: int = 2,
    selem_size: int = 5,
    min_area_px: int = 10,
) -> tuple[np.ndarray, list]:
    """
    Convenience wrapper: closing + extraction in one call.

    Parameters
    ----------
    binary_mask : (H, W) — raw Module 1 output
    iterations  : closing iterations (default 2)
    selem_size  : structuring element side length (default 5)
    min_area_px : minimum component area (default 10 px)

    Returns
    -------
    (closed_mask, regions)
        closed_mask : (H, W) bool ndarray
        regions     : list[RegionProperties]
    """
    closed = apply_bilge_closing(binary_mask, iterations=iterations, selem_size=selem_size)
    regions = extract_components(closed, min_area_px=min_area_px)
    return closed, regions


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(42)
    h, w = 512, 512

    # Synthetic mask: two elongated streaks + some speckle noise
    mask = np.zeros((h, w), dtype=bool)
    mask[100:110, 50:300]  = True   # Horizontal streak (250px long, 10px wide)
    mask[300:315, 150:400] = True   # Another streak
    mask[200, 200]         = True   # Single-pixel noise

    # Add a small fragmentation gap in streak 1
    mask[104:107, 160:170] = False  # 10px gap in the middle

    print(f"Input mask: {mask.sum()} foreground pixels, "
          f"{int(nd_label(mask)[1])} raw components")

    closed, regions = close_and_extract(mask, iterations=2, selem_size=5, min_area_px=10)

    print(f"Closed mask: {closed.sum()} foreground pixels")
    print(f"Kept components: {len(regions)}")
    for r in regions:
        maj = getattr(r, 'axis_major_length', None) or r.major_axis_length
        minn = getattr(r, 'axis_minor_length', None) or r.minor_axis_length
        print(f"  label={r.label}  area={r.area}px  "
              f"major={maj:.1f}  minor={minn:.1f}  "
              f"elongation={maj / max(minn, 1e-3):.2f}")

    # Assertions
    assert len(regions) == 2, f"Expected 2 components, got {len(regions)}"
    assert closed.sum() >= mask.sum(), "Closing must not remove foreground pixels"
    _get_axes = lambda r: (
        getattr(r, 'axis_major_length', None) or r.major_axis_length,
        getattr(r, 'axis_minor_length', None) or r.minor_axis_length,
    )
    elongs = [_get_axes(r)[0] / max(_get_axes(r)[1], 1e-3) for r in regions]
    assert all(e > 3.0 for e in elongs), "Both streaks should be highly elongated"

    # Test pass-through mode (iterations=0)
    closed0, _ = close_and_extract(mask, iterations=0)
    assert np.array_equal(closed0, mask), "iterations=0 must return input unchanged"

    print("All morphology smoke tests passed.")
    sys.exit(0)
