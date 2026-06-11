"""Statistical rigor helpers: bootstrap CIs and across-dataset significance tests."""
from __future__ import annotations
import numpy as np
from scipy import stats


def bootstrap_coverage_ci(indicator, n_boot=1000, ci=0.95, seed=0):
    """95% bootstrap CI for a coverage (or any Bernoulli) rate."""
    rng = np.random.default_rng(seed)
    indicator = np.asarray(indicator, dtype=float)
    n = len(indicator)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    boots = np.array([rng.choice(indicator, n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(indicator.mean()), float(lo), float(hi)


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test across datasets (a, b are per-dataset means)."""
    a, b = np.asarray(a), np.asarray(b)
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def friedman_nemenyi(score_matrix, method_names):
    """
    score_matrix: [n_datasets, n_methods] of a metric (lower-is-better recommended,
                  e.g. mean set size or |coverage - target|).
    Returns (friedman_p, avg_ranks, nemenyi_pvalue_df or None).
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    # average ranks (1 = best/lowest)
    ranks = np.apply_along_axis(stats.rankdata, 1, score_matrix)
    avg_ranks = dict(zip(method_names, ranks.mean(axis=0)))
    try:
        _, fried_p = stats.friedmanchisquare(*score_matrix.T)
    except ValueError:
        fried_p = np.nan
    nemenyi = None
    try:
        import scikit_posthocs as sp
        nemenyi = sp.posthoc_nemenyi_friedman(score_matrix)
        nemenyi.index = method_names
        nemenyi.columns = method_names
    except Exception:
        pass  # scikit-posthocs optional
    return float(fried_p), avg_ranks, nemenyi
