"""
Orchestration for a single (dataset, model, seed) cell.

Produces two kinds of long-format metric rows:
  calibration rows  : one per (dataset, model, seed)         -> results/calibration.csv
  conformal rows    : one per (dataset, model, seed, alpha,  -> results/conformal_fairness.csv
                                score, axis, method)
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

from .models import build_model, TemperatureScaler
from .calibration import all_calibration_metrics
from . import conformal as cf
from .fairness import fairness_summary


def three_way_split(X, y, sensitive, seed,
                    cal_frac=0.25, test_frac=0.30):
    idx = np.arange(len(y))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=cal_frac + test_frac, random_state=seed, stratify=y)
    rel_test = test_frac / (cal_frac + test_frac)
    cal_idx, te_idx = train_test_split(
        tmp_idx, test_size=rel_test, random_state=seed, stratify=y[tmp_idx])

    def take(i):
        return X[i], y[i], {k: v[i] for k, v in sensitive.items()}
    return take(tr_idx), take(cal_idx), take(te_idx)


def run_cell(dataset_name, loader_fn, loader_kwargs, model_name, seed,
             alphas=(0.05, 0.10, 0.20), scores=("lac", "aps")):
    data = loader_fn(seed=seed, **loader_kwargs)
    X, y, sens, K = data["X"], data["y"], data["sensitive"], data["n_classes"]

    (Xtr, ytr, _), (Xcal, ycal, scal), (Xte, yte, ste) = \
        three_way_split(X, y, sens, seed)

    # ---- fit base model -------------------------------------------------- #
    model = build_model(model_name, seed)
    model.fit(Xtr, ytr)
    if model_name == "tabpfn_temp":
        model = TemperatureScaler(model).fit_temperature(Xcal, ycal)

    cal_probs = model.predict_proba(Xcal)
    test_probs = model.predict_proba(Xte)

    # ---- calibration (RQ1) ---------------------------------------------- #
    calib = all_calibration_metrics(test_probs, yte, K)
    calib_row = dict(dataset=dataset_name, model=model_name, seed=seed,
                     n_classes=K, n_test=len(yte), **calib)

    # ---- conformal + fairness (RQ2/RQ3) --------------------------------- #
    # axes: every dataset gets a class-conditional axis ("predclass");
    #       datasets with sensitive attributes get those too.
    axes = {"predclass": test_probs.argmax(1)}
    cal_axes = {"predclass": cal_probs.argmax(1)}
    for ax, vals in ste.items():
        axes[ax] = vals
        cal_axes[ax] = scal[ax]

    conf_rows = []
    for alpha in alphas:
        for score in scores:
            # marginal once per (alpha, score); fairness measured on each axis
            marg_sets = cf.marginal_conformal(cal_probs, ycal, test_probs,
                                              alpha, score)
            for ax, gvals in axes.items():
                fs = fairness_summary(marg_sets, yte, gvals)
                conf_rows.append(_row(dataset_name, model_name, seed, alpha,
                                      score, ax, "marginal", fs))

            # Mondrian: separate quantile per group of THIS axis
            for ax, gvals in axes.items():
                mon_sets = cf.mondrian_conformal(
                    cal_probs, ycal, cal_axes[ax],
                    test_probs, gvals, alpha, score)
                fs = fairness_summary(mon_sets, yte, gvals)
                conf_rows.append(_row(dataset_name, model_name, seed, alpha,
                                      score, ax, "mondrian", fs))

    return calib_row, conf_rows


def _row(dataset, model, seed, alpha, score, axis, method, fs):
    return dict(
        dataset=dataset, model=model, seed=seed, alpha=alpha, score=score,
        axis=axis, method=method,
        target_coverage=1 - alpha,
        marginal_coverage=fs["marginal_coverage"],
        mean_set_size=fs["mean_set_size"],
        coverage_gap=fs["coverage_gap"],
        worst_group_coverage=fs["worst_group_coverage"],
        set_size_disparity=fs["set_size_disparity"],
    )
