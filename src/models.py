"""
Model factory. Every model exposes the sklearn API (.fit / .predict_proba)
and is label-aligned because we label-encode y to 0..K-1 upstream.

Families covered:
  - tabpfn            : the foundation model under audit
  - tabpfn_temp       : TabPFN + post-hoc temperature scaling (a cheap calibration fix)
  - xgboost, lightgbm : gradient-boosted decision trees (the entrenched champions)
  - mlp               : a small neural net (sklearn, no torch needed)
"""
from __future__ import annotations
import numpy as np


def _make_base(name, seed):
    if name in ("tabpfn", "tabpfn_temp"):
        from tabpfn import TabPFNClassifier
        # default device (MPS/CPU auto on Apple Silicon); fast for <=10k rows
        return TabPFNClassifier()
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, verbose=-1)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                          early_stopping=True, random_state=seed))
    raise ValueError(f"unknown model {name}")


class TemperatureScaler:
    """Wraps a fitted probabilistic model and rescales its logits by 1/T,
    fitting T on a held-out set by minimizing NLL. Standard Guo et al. (2017)."""

    def __init__(self, base):
        self.base = base
        self.T = 1.0

    def fit_temperature(self, X_val, y_val):
        from scipy.optimize import minimize_scalar
        p = np.clip(self.base.predict_proba(X_val), 1e-12, 1.0)
        logits = np.log(p)
        n = len(y_val)

        def nll(T):
            z = logits / T
            z = z - z.max(axis=1, keepdims=True)
            sm = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(sm[np.arange(n), y_val], 1e-12, 1.0)))

        self.T = float(minimize_scalar(nll, bounds=(0.05, 10.0),
                                       method="bounded").x)
        return self

    def predict_proba(self, X):
        p = np.clip(self.base.predict_proba(X), 1e-12, 1.0)
        z = np.log(p) / self.T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def build_model(name, seed):
    """Returns an object with .fit(X,y) and .predict_proba(X)."""
    return _make_base(name, seed)


MODEL_NAMES = ["tabpfn", "tabpfn_temp", "xgboost", "lightgbm", "mlp"]
