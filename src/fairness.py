"""Subgroup-fairness metrics computed on conformal prediction sets."""
from __future__ import annotations
import numpy as np
from .conformal import covered, set_sizes


def per_group_coverage(pred_sets, labels, groups):
    cov = covered(pred_sets, labels)
    groups = np.asarray(groups)
    return {str(g): float(cov[groups == g].mean()) for g in np.unique(groups)}


def per_group_set_size(pred_sets, groups):
    sz = set_sizes(pred_sets)
    groups = np.asarray(groups)
    return {str(g): float(sz[groups == g].mean()) for g in np.unique(groups)}


def coverage_gap(pred_sets, labels, groups):
    """Returns (gap = max-min coverage, worst-group coverage)."""
    c = list(per_group_coverage(pred_sets, labels, groups).values())
    return float(max(c) - min(c)), float(min(c))


def set_size_disparity(pred_sets, groups):
    """max - min of mean set size across groups (an uncertainty 'tax')."""
    s = list(per_group_set_size(pred_sets, groups).values())
    return float(max(s) - min(s))


def fairness_summary(pred_sets, labels, groups):
    gap, worst = coverage_gap(pred_sets, labels, groups)
    return {
        "marginal_coverage": float(covered(pred_sets, labels).mean()),
        "mean_set_size": float(set_sizes(pred_sets).mean()),
        "coverage_gap": gap,
        "worst_group_coverage": worst,
        "set_size_disparity": set_size_disparity(pred_sets, groups),
        "per_group_coverage": per_group_coverage(pred_sets, labels, groups),
        "per_group_set_size": per_group_set_size(pred_sets, groups),
    }
