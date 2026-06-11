"""
Distribution-free conformal prediction for classification, implemented from scratch
(no MAPIE dependency, so it is robust to library version churn).

Two conformity scores:
  - LAC  (Least Ambiguous set-valued Classifier, a.k.a. "score"/THR):
         score_i = 1 - p_model(y_i | x_i)
  - APS  (Adaptive Prediction Sets, Romano et al. 2020), non-randomized variant.

Two coverage regimes:
  - marginal: one global quantile -> guarantees P(y in C(x)) >= 1 - alpha overall.
  - Mondrian (group-conditional): a separate quantile per subgroup -> guarantees
         the coverage holds *within each group*. This is the fairness fix.

All functions take class-probability matrices `probs` of shape [n_samples, n_classes]
whose columns are aligned to integer labels 0..K-1 (label-encode y first!).
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Conformal quantile (finite-sample corrected)
# --------------------------------------------------------------------------- #
def conformal_qhat(cal_scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile of calibration scores."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    n = cal_scores.shape[0]
    if n == 0:
        return np.inf
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(cal_scores, level, method="higher"))


# --------------------------------------------------------------------------- #
# LAC score
# --------------------------------------------------------------------------- #
def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    idx = np.arange(len(labels))
    return 1.0 - probs[idx, labels]


def lac_sets(test_probs: np.ndarray, qhat: float) -> np.ndarray:
    """Include class k iff (1 - p_k) <= qhat  <=>  p_k >= 1 - qhat."""
    sets = test_probs >= (1.0 - qhat)
    # never return an empty set: keep the argmax as a fallback
    empty = ~sets.any(axis=1)
    if empty.any():
        sets[empty, test_probs[empty].argmax(axis=1)] = True
    return sets


# --------------------------------------------------------------------------- #
# APS score (non-randomized)
# --------------------------------------------------------------------------- #
def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    order = np.argsort(-probs, axis=1)                 # classes, high -> low prob
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)                  # cumulative incl. current
    ranks = (order == labels[:, None]).argmax(axis=1)  # position of true label
    return cum[np.arange(n), ranks]


def aps_sets(test_probs: np.ndarray, qhat: float) -> np.ndarray:
    order = np.argsort(-test_probs, axis=1)
    sorted_p = np.take_along_axis(test_probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    prefix = cum - sorted_p                            # cumulative BEFORE this class
    include_sorted = prefix < qhat                     # includes the crossing class
    include_sorted[:, 0] = True                        # always keep the top class
    sets = np.zeros_like(test_probs, dtype=bool)
    np.put_along_axis(sets, order, include_sorted, axis=1)
    return sets


# --------------------------------------------------------------------------- #
# Unified marginal + Mondrian interface
# --------------------------------------------------------------------------- #
_SCORE_FNS = {"lac": (lac_scores, lac_sets), "aps": (aps_scores, aps_sets)}


def marginal_conformal(cal_probs, cal_labels, test_probs, alpha, score="lac"):
    score_fn, set_fn = _SCORE_FNS[score]
    s = score_fn(cal_probs, cal_labels)
    qhat = conformal_qhat(s, alpha)
    return set_fn(test_probs, qhat)


def mondrian_conformal(cal_probs, cal_labels, cal_groups,
                       test_probs, test_groups, alpha, score="lac",
                       min_group_cal=20):
    """Run conformal separately within each group. Groups too small in the
    calibration set fall back to the global quantile so they are never undefined."""
    score_fn, set_fn = _SCORE_FNS[score]
    cal_groups = np.asarray(cal_groups)
    test_groups = np.asarray(test_groups)
    sets = np.zeros_like(test_probs, dtype=bool)

    global_qhat = conformal_qhat(score_fn(cal_probs, cal_labels), alpha)
    for g in np.unique(test_groups):
        tmask = test_groups == g
        cmask = cal_groups == g
        if cmask.sum() >= min_group_cal:
            qhat = conformal_qhat(score_fn(cal_probs[cmask], cal_labels[cmask]), alpha)
        else:
            qhat = global_qhat
        sets[tmask] = set_fn(test_probs[tmask], qhat)
    return sets


def covered(pred_sets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample indicator: was the true label inside the prediction set?"""
    return pred_sets[np.arange(len(labels)), labels]


def set_sizes(pred_sets: np.ndarray) -> np.ndarray:
    return pred_sets.sum(axis=1)
