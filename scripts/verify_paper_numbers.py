#!/usr/bin/env python3
"""
Verify every number in the FINAL (v2/IJAR-revision) manuscript against the
committed CSVs and the prediction cache. Reviewer-response artifact.

Protocol (paper Sec. 5.8): LAC for all conformal/fairness numbers; conformal
analyses pooled over the FOUR base models (tabpfn, xgboost, lightgbm, mlp),
excluding tabpfn_temp; RQ3 marginal-vs-Mondrian contrasts paired at the
model-averaged (dataset x seed) level, n=15, Holm-adjusted across axes.

Exit 0 iff all checks pass.
Run from repo root:  python scripts/verify_paper_numbers.py
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
CAL = pd.read_csv(ROOT / "results" / "calibration.csv")
CONF = pd.read_csv(ROOT / "results" / "conformal_fairness.csv")
BASE = ["tabpfn", "xgboost", "lightgbm", "mlp"]
LAC = CONF[(CONF.score == "lac") & (CONF.model.isin(BASE))]
R = []


def chk(label, claimed, got, tol=1.5e-3):
    ok = got is not None and abs(claimed - got) <= tol
    R.append((label, claimed, got, ok))


def chk_p(label, claimed, got, rel=0.25, thresh=None):
    if thresh is not None:
        ok = got is not None and got <= thresh
    else:
        ok = got is not None and abs(got - claimed) <= rel * max(claimed, 1e-12)
    R.append((label, claimed, got, ok))


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0; m = len(p)
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def pooled(axis, method, alpha, col):
    s = LAC[(LAC.axis == axis) & (LAC.method == method) & np.isclose(LAC.alpha, alpha)]
    return float(s.groupby(["dataset", "seed"])[col].mean().mean())


# --- Table rq3 @90% (4-model) --------------------------------------------- #
T3 = {  # axis: (gap_m, gap_d, wgc_m, wgc_d, size_m, size_d)
    "sex": (0.012, 0.013, 0.895, 0.895, 1.331, 1.334),
    "age": (0.081, 0.051, 0.862, 0.881, 1.331, 1.353),
    "race": (0.044, 0.057, 0.875, 0.873, 1.331, 1.341),
    "predclass": (0.044, 0.013, 0.881, 0.894, 1.331, 1.340),
}
for ax, (gm, gd, wm, wd, sm, sd) in T3.items():
    chk(f"rq3 {ax} gap marg", gm, pooled(ax, "marginal", 0.1, "coverage_gap"))
    chk(f"rq3 {ax} gap mond", gd, pooled(ax, "mondrian", 0.1, "coverage_gap"))
    chk(f"rq3 {ax} wgc marg", wm, pooled(ax, "marginal", 0.1, "worst_group_coverage"))
    chk(f"rq3 {ax} wgc mond", wd, pooled(ax, "mondrian", 0.1, "worst_group_coverage"))
    chk(f"rq3 {ax} size marg", sm, pooled(ax, "marginal", 0.1, "mean_set_size"))
    chk(f"rq3 {ax} size mond", sd, pooled(ax, "mondrian", 0.1, "mean_set_size"))

# --- Table tax (set-size disparity @90%) ---------------------------------- #
TAX = {"sex": (0.059, 0.086), "age": (0.240, 0.373), "race": (0.118, 0.154), "predclass": (0.115, 0.160)}
for ax, (m, d) in TAX.items():
    chk(f"tax {ax} marg", m, pooled(ax, "marginal", 0.1, "set_size_disparity"))
    chk(f"tax {ax} mond", d, pooled(ax, "mondrian", 0.1, "set_size_disparity"))

# --- Table rq3targets ------------------------------------------------------ #
TGT = {0.2: {"sex": (0.029, 0.020), "age": (0.151, 0.105), "race": (0.057, 0.058)},
       0.1: {"sex": (0.012, 0.013), "age": (0.081, 0.051), "race": (0.044, 0.057)},
       0.05: {"sex": (0.012, 0.010), "age": (0.045, 0.038), "race": (0.033, 0.038)}}
for a, axes in TGT.items():
    for ax, (m, d) in axes.items():
        chk(f"targets {ax}@{a} marg", m, pooled(ax, "marginal", a, "coverage_gap"))
        chk(f"targets {ax}@{a} mond", d, pooled(ax, "mondrian", a, "coverage_gap"))

# --- per-dataset race @90% ------------------------------------------------- #
for ds, (m, d) in {"ACSEmployment-CA": (0.034, 0.051), "ACSIncome-CA": (0.048, 0.058),
                   "ACSPublicCoverage-CA": (0.052, 0.062)}.items():
    s = LAC[(LAC.axis == "race") & np.isclose(LAC.alpha, 0.1) & (LAC.dataset == ds)]
    chk(f"race {ds} marg", m, float(s[s.method == "marginal"].groupby("seed")["coverage_gap"].mean().mean()))
    chk(f"race {ds} mond", d, float(s[s.method == "mondrian"].groupby("seed")["coverage_gap"].mean().mean()))

# --- RQ3 Holm p-values @90% (n=15) ----------------------------------------- #
cell = LAC[np.isclose(LAC.alpha, 0.1)].groupby(
    ["dataset", "seed", "axis", "method"], as_index=False)[["coverage_gap", "worst_group_coverage"]].mean()
praw_gap, praw_wgc, axes_o = [], [], []
for ax in ["age", "predclass", "race", "sex"]:
    piv = cell[cell.axis == ax].pivot_table(index=["dataset", "seed"], columns="method",
                                             values=["coverage_gap", "worst_group_coverage"])
    g = wilcoxon(piv[("coverage_gap", "marginal")], piv[("coverage_gap", "mondrian")]).pvalue
    w = wilcoxon(piv[("worst_group_coverage", "marginal")], piv[("worst_group_coverage", "mondrian")]).pvalue
    praw_gap.append(g); praw_wgc.append(w); axes_o.append(ax)
hg = dict(zip(axes_o, holm(praw_gap))); hw = dict(zip(axes_o, holm(praw_wgc)))
chk_p("age gap Holm p<1e-3", 3.7e-4, hg["age"], thresh=1e-3)
chk_p("race gap Holm p~4e-3", 4.0e-3, hg["race"], rel=0.4)
chk_p("age wgc Holm p<1e-3", 4.9e-4, hw["age"], thresh=1e-3)
chk_p("sex gap Holm p~0.6", 0.60, hg["sex"], rel=0.4)

# --- MC null table reproduces (ground rule 2: nulls unchanged) ------------- #
from scripts.mc_noise_floor import simulate_gap, AXES  # noqa: E402
def null_avg(axis_prefix, alpha, mond):
    vals = [simulate_gap(ngs, alpha, reps=60_000, mondrian=mond).mean()
            for name, ngs in AXES.items() if name.startswith(axis_prefix)]
    return float(np.mean(vals))
for axp, m90, d90 in [("Age", 0.032, 0.047), ("Race", 0.036, 0.052), ("Sex", 0.010, 0.014)]:
    chk(f"mcnull {axp} marg@90", m90, null_avg(axp, 0.10, False), tol=3e-3)
    chk(f"mcnull {axp} mond@90", d90, null_avg(axp, 0.10, True), tol=3e-3)

# --- cache-derived: GBDT+T ECE, TabPFN Brier/NLL advantage, APS, empty ----- #
CACHE = ROOT / "cache"
if CACHE.exists() and len(list(CACHE.glob("*.npz"))) == 60:
    from src.calibration import all_calibration_metrics
    from src import conformal as cf
    ece = {"xgboost_T": [], "lightgbm_T": []}
    brier = {"tabpfn": [], "xgboost_T": [], "lightgbm_T": []}
    nll = {"tabpfn": [], "xgboost_T": [], "lightgbm_T": []}
    aps_cov, aps_size, empty80 = [], [], []
    for ds in ["ACSEmployment-CA", "ACSIncome-CA", "ACSPublicCoverage-CA"]:
        for seed in range(5):
            for model in ["xgboost", "lightgbm"]:
                d = np.load(CACHE / f"{ds}__{model}__seed{seed}.npz", allow_pickle=True)
                m = all_calibration_metrics(d["test_probs_T"], d["yte"], int(d["n_classes"]))
                ece[f"{model}_T"].append(m["ece_adaptive"]); brier[f"{model}_T"].append(m["brier"]); nll[f"{model}_T"].append(m["nll"])
            dt = np.load(CACHE / f"{ds}__tabpfn__seed{seed}.npz", allow_pickle=True)
            mt = all_calibration_metrics(dt["test_probs"], dt["yte"], int(dt["n_classes"]))
            brier["tabpfn"].append(mt["brier"]); nll["tabpfn"].append(mt["nll"])
            calp, tep, ycal, yte = dt["cal_probs"], dt["test_probs"], dt["ycal"], dt["yte"]
            rng = np.random.default_rng(7000 + seed)
            cs = cf.aps_scores_rand(calp, ycal, rng); qh = cf.conformal_qhat(cs, 0.1)
            sets = cf.aps_sets_rand(tep, qh, rng)
            aps_cov.append(cf.covered(sets, yte).mean()); aps_size.append(cf.set_sizes(sets).mean())
            for mdl in BASE:
                dm = np.load(CACHE / f"{ds}__{mdl}__seed{seed}.npz", allow_pickle=True)
                s = cf.lac_scores(dm["cal_probs"], dm["ycal"]); q80 = cf.conformal_qhat(s, 0.2)
                empty80.append((~(dm["test_probs"] >= 1 - q80).any(axis=1)).mean())
    chk("GBDT+T xgb ECE", 0.026, float(np.mean(ece["xgboost_T"])))
    chk("GBDT+T lgbm ECE", 0.026, float(np.mean(ece["lightgbm_T"])))
    chk_p("TabPFN<xgb+T brier p", 6.1e-5, wilcoxon(brier["tabpfn"], brier["xgboost_T"]).pvalue, thresh=1e-4)
    chk_p("TabPFN<lgbm+T nll p", 6.1e-5, wilcoxon(nll["tabpfn"], nll["lightgbm_T"]).pvalue, thresh=1e-4)
    chk("randAPS cov tabpfn@90", 0.90, float(np.mean(aps_cov)), tol=0.01)
    chk("empty-set raw@80 ~0.79%", 0.0079, float(np.mean(empty80)), tol=4e-3)
else:
    print("(cache absent -> skipping cache-derived checks; run build_cache.py)")

fails = [r for r in R if not r[3]]
for label, c, g, ok in R:
    if not ok:
        print(f"  FAIL {label:32s} paper={c:.6g} recomputed={'None' if g is None else f'{g:.6g}'}")
print("=" * 70)
print(f"{len(R) - len(fails)}/{len(R)} checks PASS, {len(fails)} FAIL")
sys.exit(1 if fails else 0)
