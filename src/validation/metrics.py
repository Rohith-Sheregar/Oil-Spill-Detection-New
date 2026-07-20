"""
mIoU + pixel F1 for binary SAR segmentation, the composite attribution
score, and a rank-based validation check sized to how much data you
actually have.

On weight-fitting: with ~5 confirmed incidents, fitting w1-w4 by regression
is overfitting by construction -- 4 free parameters against ~5 data points,
with no held-out set possible. Use equal weights or a coarse manual grid
search instead, and validate by rank, not magnitude.

On "rank correlation" specifically: a true Spearman/Kendall correlation
needs enough candidates *per incident* to be meaningful, and needs several
incidents to aggregate over -- shaky with n~5. What you actually care about
operationally is simpler and more honest given the sample size: for each
confirmed incident, does the correct vessel get the HIGHEST composite score
among the candidates considered? `rank_correlation_check` below computes
that top-1 hit rate rather than a formal correlation coefficient.
"""
import numpy as np
from sklearn.metrics import f1_score


def compute_miou(pred_mask, gt_mask, n_classes=2):
    """pred_mask, gt_mask: integer arrays, same shape, values in [0, n_classes)."""
    ious = []
    for c in range(n_classes):
        pred_c, gt_c = (pred_mask == c), (gt_mask == c)
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue  # class absent from both pred and gt in this scene -- skip, don't score as 0 or 1
        ious.append(np.logical_and(pred_c, gt_c).sum() / union)
    return float(np.mean(ious)) if ious else float("nan")


def compute_pixel_f1(pred_mask, gt_mask):
    """Binary pixel-level F1 -- flatten and treat every pixel as one sample."""
    return f1_score(gt_mask.flatten(), pred_mask.flatten(), average="binary", zero_division=0)


def composite_attribution_score(s_drift, s_ais_anomaly, s_morphology, s_temporal, weights=None):
    """weights default to equal (0.25 each) -- see module docstring for why
    NOT to fit these by regression on a ~5-incident validation set."""
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)
    w1, w2, w3, w4 = weights
    return w1 * s_drift + w2 * s_ais_anomaly + w3 * s_morphology + w4 * s_temporal


def rank_correlation_check(candidates_by_incident: dict):
    """
    candidates_by_incident: {incident_id: [(vessel_id, composite_score, is_correct_vessel), ...]}
    Returns the fraction of incidents where the confirmed vessel received
    the highest composite score among all candidates considered for that
    incident (top-1 hit rate).
    """
    hits = 0
    for incident_id, candidates in candidates_by_incident.items():
        ranked = sorted(candidates, key=lambda c: c[1], reverse=True)
        if ranked and ranked[0][2]:
            hits += 1
    return hits / len(candidates_by_incident) if candidates_by_incident else float("nan")


if __name__ == "__main__":
    pred = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    gt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    print("mIoU:", compute_miou(pred, gt))
    print("F1:", compute_pixel_f1(pred, gt))

    demo = {
        "incident_1": [("vessel_A", 0.81, True), ("vessel_B", 0.62, False)],
        "incident_2": [("vessel_C", 0.55, False), ("vessel_D", 0.71, True)],
    }
    print("Top-1 hit rate:", rank_correlation_check(demo))
