"""
Scene/region-level train-val splits. Never split SAR patches randomly --
adjacent patches from one pass share speckle realization and weather
conditions, so a random split leaks information across the train/val
boundary (the data-leakage gap flagged against Liao et al. 2023 in your own
lit review). Group by scene_id, and for the external-geography validation
tier, hold out an entire region.
"""
from sklearn.model_selection import GroupKFold


def scene_level_split(patch_df, scene_id_col="scene_id", n_splits=5, fold=0):
    """
    patch_df: one row per patch, with scene_id_col identifying the parent
              SAR scene each patch was cut from.
    Returns (train_idx, val_idx) positional indices for the requested fold.
    Patches from the same scene never appear on both sides of a fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(patch_df, groups=patch_df[scene_id_col]))
    return splits[fold]


def external_region_holdout(patch_df, region_col="region", holdout_region=None):
    """
    For the external-geography validation tier: hold out an entire region
    (e.g. South China Sea) that contributed ZERO patches to training,
    rather than a random subset of patches from a region the model has
    already seen elsewhere. holdout_region=None picks whichever region has
    the fewest patches (cheapest to hold out without starving training).
    Returns (train_index, holdout_index) as label-based indices into patch_df.
    """
    if holdout_region is None:
        holdout_region = patch_df[region_col].value_counts().idxmin()
    is_holdout = patch_df[region_col] == holdout_region
    return patch_df.index[~is_holdout].to_numpy(), patch_df.index[is_holdout].to_numpy()
