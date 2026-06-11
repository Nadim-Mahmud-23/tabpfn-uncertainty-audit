"""Calibration metrics for probabilistic classifiers."""
from __future__ import annotations
import numpy as np


def _confidence_and_correct(probs, labels):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    return conf, correct


def expected_calibration_error(probs, labels, n_bins=15, adaptive=True):
    """ECE. `adaptive=True` uses equal-mass bins (more robust than equal-width)."""
    conf, correct = _confidence_and_correct(probs, labels)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def maximum_calibration_error(probs, labels, n_bins=15):
    conf, correct = _confidence_and_correct(probs, labels)
    edges = np.linspace(0, 1, n_bins + 1)
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            gaps.append(abs(correct[m].mean() - conf[m].mean()))
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs, labels, n_classes):
    onehot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs, labels, eps=1e-12):
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))


def reliability_curve(probs, labels, n_bins=15):
    """Returns (bin_confidence, bin_accuracy, bin_count) for plotting."""
    conf, correct = _confidence_and_correct(probs, labels)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, cs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            cs.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(cs)


def all_calibration_metrics(probs, labels, n_classes):
    return {
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "ece_adaptive": expected_calibration_error(probs, labels, adaptive=True),
        "ece_equalwidth": expected_calibration_error(probs, labels, adaptive=False),
        "mce": maximum_calibration_error(probs, labels),
        "brier": brier_score(probs, labels, n_classes),
        "nll": negative_log_likelihood(probs, labels),
    }
