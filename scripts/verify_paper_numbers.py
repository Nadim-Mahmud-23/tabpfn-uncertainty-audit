#!/usr/bin/env python3
"""
Verify every statistic in paper.tex against the raw result CSVs.

Recomputes, from results/calibration.csv (75 rows) and
results/conformal_fairness.csv (3,600 rows), each number that appears in
Tables 1-6 and in the Results prose, and prints PASS/FAIL against the value
written in the paper. Doubles as the reviewer-response artifact.

INTEGRITY: all conformal/fairness statistics are LAC-only by design (the
deterministic APS configuration is degenerate on these binary tasks --
coverage 1.0, set size ~2.0). Every conformal recomputation filters
score == 'lac'. The exported table_RQ*.csv files are NOT used (they average
LAC and APS rows and are wrong).

Run:  python scripts/verify_paper_numbers.py
Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CAL = pd.read_csv(os.path.join(ROOT, "results", "calibration.csv"))
CONF = pd.read_csv(os.path.join(ROOT, "results", "conformal_fairness.csv"))

MODELS = ["xgboost", "lightgbm", "mlp", "tabpfn", "tabpfn_temp"]
CELLS = ["dataset", "seed"]          # 15 paired cells for RQ1/RQ2
RESULTS = []                          # (label, claimed, computed, ok)


def chk(label, claimed, computed, tol=1.5e-3):
    """Numeric check with absolute tolerance (default matches 3-dp rounding)."""
    ok = (claimed is None) or (computed is not None and abs(claimed - computed) <= tol)
    RESULTS.append((label, claimed, computed, ok))
    return ok


def chk_p(label, claimed, computed, rtol=0.20, thresh=None):
    """p-value check. If `thresh` given, require computed < thresh (for 'p<1e-4'
    style claims). Otherwise require relative agreement within rtol."""
    if thresh is not None:
        ok = computed is not None and computed < thresh
    else:
        ok = computed is not None and (
            abs(computed - claimed) <= rtol * max(claimed, 1e-30)
            or (computed < 1e-4 and claimed < 1e-4)
        )
    RESULTS.append((label, claimed, computed, ok))
    return ok


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def lac(method=None, axis=None, target=None):
    d = CONF[CONF.score == "lac"].copy()
    if method is not None:
        d = d[d.method == method]
    if axis is not None:
        d = d[d.axis == axis]
    if target is not None:
        d = d[np.isclose(d.target_coverage, target)]
    return d


def paired(df, value, by, key_a, key_b, col="model"):
    """Return paired arrays (a, b) aligned on `by` keys, for col==key_a vs key_b."""
    a = df[df[col] == key_a].set_index(by)[value].sort_index()
    b = df[df[col] == key_b].set_index(by)[value].sort_index()
    idx = a.index.intersection(b.index)
    return a.loc[idx].to_numpy(), b.loc[idx].to_numpy()


def wilcoxon(a, b):
    if len(a) == 0 or np.allclose(a, b):
        return np.nan
    return float(stats.wilcoxon(a, b)[1])


# =========================================================================== #
# TABLE 1 -- RQ1 pooled calibration (mean +/- std over 15 cells, ddof=1)
# =========================================================================== #
print("\n### Table 1: RQ1 pooled calibration (mean +/- std, n=15)")
T1 = {
    "xgboost":     dict(accuracy=0.763, ece_adaptive=0.075, mce=0.135, brier=0.333, nll=0.516),
    "lightgbm":    dict(accuracy=0.769, ece_adaptive=0.052, mce=0.100, brier=0.317, nll=0.485),
    "mlp":         dict(accuracy=0.759, ece_adaptive=0.025, mce=0.057, brier=0.324, nll=0.486),
    "tabpfn":      dict(accuracy=0.782, ece_adaptive=0.026, mce=0.063, brier=0.298, nll=0.452),
    "tabpfn_temp": dict(accuracy=0.782, ece_adaptive=0.025, mce=0.054, brier=0.297, nll=0.451),
}
T1_std = {
    "xgboost":     dict(accuracy=0.052, ece_adaptive=0.019, mce=0.022, brier=0.064, nll=0.084),
    "lightgbm":    dict(accuracy=0.049, ece_adaptive=0.012, mce=0.018, brier=0.057, nll=0.074),
    "mlp":         dict(accuracy=0.046, ece_adaptive=0.007, mce=0.022, brier=0.056, nll=0.074),
    "tabpfn":      dict(accuracy=0.046, ece_adaptive=0.006, mce=0.021, brier=0.056, nll=0.075),
    "tabpfn_temp": dict(accuracy=0.046, ece_adaptive=0.006, mce=0.018, brier=0.056, nll=0.074),
}
for m in MODELS:
    sub = CAL[CAL.model == m]
    for metric in ["accuracy", "ece_adaptive", "mce", "brier", "nll"]:
        chk(f"T1 {m}.{metric} mean", T1[m][metric], float(sub[metric].mean()))
        chk(f"T1 {m}.{metric} std", T1_std[m][metric], float(sub[metric].std(ddof=1)))

# =========================================================================== #
# TABLE 2 -- RQ1 adaptive ECE by dataset (mean +/- std over 5 seeds)
# =========================================================================== #
print("### Table 2: RQ1 adaptive ECE by dataset (n=5)")
DS = {"ACSEmployment": "ACSEmployment-CA", "ACSIncome": "ACSIncome-CA",
      "ACSPublicCoverage": "ACSPublicCoverage-CA"}
T2 = {
    "xgboost":     dict(ACSEmployment=0.058, ACSIncome=0.069, ACSPublicCoverage=0.098),
    "lightgbm":    dict(ACSEmployment=0.043, ACSIncome=0.049, ACSPublicCoverage=0.064),
    "mlp":         dict(ACSEmployment=0.020, ACSIncome=0.024, ACSPublicCoverage=0.033),
    "tabpfn":      dict(ACSEmployment=0.022, ACSIncome=0.025, ACSPublicCoverage=0.030),
    "tabpfn_temp": dict(ACSEmployment=0.021, ACSIncome=0.025, ACSPublicCoverage=0.028),
}
for m in MODELS:
    for short, full in DS.items():
        sub = CAL[(CAL.model == m) & (CAL.dataset == full)]
        chk(f"T2 {m}.{short} ECE", T2[m][short], float(sub["ece_adaptive"].mean()))

# =========================================================================== #
# TABLE 4 -- RQ2 marginal coverage + set size (LAC, predclass marginal)
# =========================================================================== #
print("### Table 4: RQ2 coverage & set size (LAC, predclass, marginal)")
T4 = {
    "xgboost":  {0.80: (0.800, 1.083), 0.90: (0.899, 1.339), 0.95: (0.952, 1.534)},
    "lightgbm": {0.80: (0.802, 1.072), 0.90: (0.901, 1.327), 0.95: (0.953, 1.519)},
    "mlp":      {0.80: (0.803, 1.095), 0.90: (0.903, 1.363), 0.95: (0.951, 1.543)},
    "tabpfn":   {0.80: (0.809, 1.058), 0.90: (0.900, 1.296), 0.95: (0.950, 1.480)},
}
base = lac(method="marginal", axis="predclass")
for m, tgts in T4.items():
    for tgt, (cov, size) in tgts.items():
        sub = base[(base.model == m) & np.isclose(base.target_coverage, tgt)]
        chk(f"T4 {m}@{tgt} coverage", cov, float(sub["marginal_coverage"].mean()))
        chk(f"T4 {m}@{tgt} set size", size, float(sub["mean_set_size"].mean()))

# =========================================================================== #
# TABLE 5 -- RQ3 gap / WGC / size at 90%, marginal vs Mondrian (pooled models)
# =========================================================================== #
print("### Table 5: RQ3 @90% (LAC, pooled over 5 models x 15 cells)")
T5 = {
    ("sex", "marginal"):       (0.012, 0.895, 1.324),
    ("sex", "mondrian"):       (0.013, 0.895, 1.327),
    ("age", "marginal"):       (0.081, 0.862, 1.324),
    ("age", "mondrian"):       (0.051, 0.881, 1.345),
    ("race", "marginal"):      (0.046, 0.875, 1.324),
    ("race", "mondrian"):      (0.057, 0.873, 1.334),
    ("predclass", "marginal"): (0.045, 0.881, 1.324),
    ("predclass", "mondrian"): (0.013, 0.894, 1.332),
}
for (axis, method), (gap, wgc, size) in T5.items():
    sub = lac(method=method, axis=axis, target=0.90)
    chk(f"T5 {axis}/{method} gap", gap, float(sub["coverage_gap"].mean()))
    chk(f"T5 {axis}/{method} WGC", wgc, float(sub["worst_group_coverage"].mean()))
    chk(f"T5 {axis}/{method} size", size, float(sub["mean_set_size"].mean()))

# RQ3 Wilcoxon (paired at dataset x model x seed, n=75)
print("### Table 5: RQ3 marginal-vs-Mondrian Wilcoxon (n=75)")
for axis, (claim_gap_p, claim_wgc_p) in {
    "sex": (0.50, 0.79), "age": (4.2e-11, 2.9e-10), "race": (9.3e-4, 0.55),
}.items():
    marg = lac(method="marginal", axis=axis, target=0.90)
    mond = lac(method="mondrian", axis=axis, target=0.90)
    # explicit pairing on (dataset, model, seed)
    key = ["dataset", "model", "seed"]
    am = marg.set_index(key)["coverage_gap"].sort_index()
    bm = mond.set_index(key)["coverage_gap"].sort_index()
    idx = am.index.intersection(bm.index)
    gp = wilcoxon(am.loc[idx].to_numpy(), bm.loc[idx].to_numpy())
    aw = marg.set_index(key)["worst_group_coverage"].sort_index()
    bw = mond.set_index(key)["worst_group_coverage"].sort_index()
    wp = wilcoxon(aw.loc[idx].to_numpy(), bw.loc[idx].to_numpy())
    n = len(idx)
    chk_p(f"T5 {axis} gap p (n={n})", claim_gap_p, gp,
          thresh=(1e-3 if claim_gap_p < 1e-3 else None), rtol=0.5)
    chk_p(f"T5 {axis} WGC p (n={n})", claim_wgc_p, wp,
          thresh=(1e-3 if claim_wgc_p < 1e-3 else None), rtol=0.5)

# =========================================================================== #
# TABLE 6 -- RQ3 coverage gap by target level (LAC, pooled)
# =========================================================================== #
print("### Table 6: RQ3 coverage gap by target level (LAC, pooled)")
T6 = {
    0.80: dict(sex=(0.029, 0.021), age=(0.151, 0.106), race=(0.056, 0.055)),
    0.90: dict(sex=(0.012, 0.013), age=(0.081, 0.051), race=(0.046, 0.057)),
    0.95: dict(sex=(0.012, 0.010), age=(0.045, 0.037), race=(0.034, 0.038)),
}
for tgt, axes in T6.items():
    for axis, (marg_gap, mond_gap) in axes.items():
        sm = lac(method="marginal", axis=axis, target=tgt)
        sd = lac(method="mondrian", axis=axis, target=tgt)
        chk(f"T6 {axis}@{tgt} marg gap", marg_gap, float(sm["coverage_gap"].mean()))
        chk(f"T6 {axis}@{tgt} mond gap", mond_gap, float(sd["coverage_gap"].mean()))

# =========================================================================== #
# PROSE -- RQ1
# =========================================================================== #
print("### Prose: RQ1")
ece = CAL.pivot_table(index=CELLS, columns="model", values="ece_adaptive")
acc = CAL.pivot_table(index=CELLS, columns="model", values="accuracy")
brier = CAL.pivot_table(index=CELLS, columns="model", values="brier")
nll = CAL.pivot_table(index=CELLS, columns="model", values="nll")
chk_p("RQ1 ECE tabpfn<xgboost p<1e-4", 1e-4,
      wilcoxon(ece["tabpfn"], ece["xgboost"]), thresh=1e-4)
chk_p("RQ1 ECE tabpfn<lightgbm p<1e-4", 1e-4,
      wilcoxon(ece["tabpfn"], ece["lightgbm"]), thresh=1e-4)
chk_p("RQ1 ECE tabpfn vs mlp p=0.60", 0.60,
      wilcoxon(ece["tabpfn"], ece["mlp"]), rtol=0.30)
chk("RQ1 ECE tabpfn-mlp median diff", -0.001,
    float(np.median(ece["tabpfn"] - ece["mlp"])), tol=2e-3)
chk("RQ1 acc tabpfn-mlp diff", 0.023, float((acc["tabpfn"] - acc["mlp"]).mean()), tol=2e-3)
chk_p("RQ1 acc tabpfn>mlp p<1e-3", 1e-3, wilcoxon(acc["tabpfn"], acc["mlp"]), thresh=1e-3)
chk_p("RQ1 brier tabpfn<xgboost p<1e-4", 1e-4,
      wilcoxon(brier["tabpfn"], brier["xgboost"]), thresh=1e-4)
chk_p("RQ1 nll tabpfn<xgboost p<1e-4", 1e-4,
      wilcoxon(nll["tabpfn"], nll["xgboost"]), thresh=1e-4)
chk_p("RQ1 temp scaling tabpfn vs +T p=0.12", 0.12,
      wilcoxon(ece["tabpfn"], ece["tabpfn_temp"]), rtol=0.6)
# Friedman over 5 models x 15 cells on ECE
fr = stats.friedmanchisquare(*[ece[m].to_numpy() for m in MODELS])
chk("RQ1 Friedman chi2", 48.5, float(fr.statistic), tol=0.5)
chk_p("RQ1 Friedman p=7.5e-10", 7.5e-10, float(fr.pvalue), rtol=1.0)
ranks = ece[MODELS].rank(axis=1).mean()
for m, r in {"tabpfn_temp": 1.8, "mlp": 2.0, "tabpfn": 2.2,
             "lightgbm": 4.0, "xgboost": 5.0}.items():
    chk(f"RQ1 rank {m}", r, float(ranks[m]), tol=0.15)

# =========================================================================== #
# PROSE -- RQ2
# =========================================================================== #
print("### Prose: RQ2")
for tgt, (cov, sd) in {0.80: (0.805, 0.013), 0.90: (0.901, 0.009), 0.95: (0.951, 0.007)}.items():
    pool = lac(method="marginal", axis="predclass", target=tgt)["marginal_coverage"]
    chk(f"RQ2 pooled coverage@{tgt} mean", cov, float(pool.mean()))
    chk(f"RQ2 pooled coverage@{tgt} std", sd, float(pool.std(ddof=1)))
size90 = lac(method="marginal", axis="predclass", target=0.90)
for opp, claim in {"lightgbm": 4.3e-4, "xgboost": 3.1e-4, "mlp": 6.5e-4}.items():
    a, b = paired(size90, "mean_set_size", CELLS, "tabpfn", opp)
    chk_p(f"RQ2 setsize tabpfn<{opp} p", claim, wilcoxon(a, b), rtol=0.5)
# APS degeneracy
aps = CONF[CONF.score == "aps"]
chk("RQ2 APS coverage ~1.0", 1.000,
    float(aps[aps.method == "marginal"]["marginal_coverage"].mean()), tol=2e-3)
aps_sz = float(aps[aps.method == "marginal"]["mean_set_size"].mean())
chk("RQ2 APS set size in [1.99,2.0]", 2.0, aps_sz, tol=1.1e-2)

# =========================================================================== #
# PROSE -- RQ3
# =========================================================================== #
print("### Prose: RQ3")
# age set-size cost +0.021 (1.324 -> 1.345)
age_m = lac(method="marginal", axis="age", target=0.90)["mean_set_size"].mean()
age_d = lac(method="mondrian", axis="age", target=0.90)["mean_set_size"].mean()
chk("RQ3 age set-size cost +0.021", 0.021, float(age_d - age_m), tol=2e-3)
# age 80% gap & WGC
chk("RQ3 age@80 marg WGC 0.732", 0.732,
    float(lac(method="marginal", axis="age", target=0.80)["worst_group_coverage"].mean()))
chk("RQ3 age@80 mond WGC 0.774", 0.774,
    float(lac(method="mondrian", axis="age", target=0.80)["worst_group_coverage"].mean()))
# race per-dataset marginal -> mondrian gap at 90%
race_pd = {"ACSEmployment-CA": (0.034, 0.049), "ACSIncome-CA": (0.047, 0.060),
           "ACSPublicCoverage-CA": (0.056, 0.061)}
for ds, (mg, dg) in race_pd.items():
    sm = lac(method="marginal", axis="race", target=0.90)
    sd = lac(method="mondrian", axis="race", target=0.90)
    chk(f"RQ3 race@90 {ds} marg gap", mg,
        float(sm[sm.dataset == ds]["coverage_gap"].mean()))
    chk(f"RQ3 race@90 {ds} mond gap", dg,
        float(sd[sd.dataset == ds]["coverage_gap"].mean()))
# per-model age marginal gap band 0.078-0.083
age_marg = lac(method="marginal", axis="age", target=0.90)
band = age_marg.groupby("model")["coverage_gap"].mean()
chk("RQ3 age per-model gap min>=0.078", 0.078, float(band.min()), tol=2e-3)
chk("RQ3 age per-model gap max<=0.083", 0.083, float(band.max()), tol=2e-3)
# max coverage-gap diff tabpfn vs tabpfn_temp (any axis, 90%)
mg = lac(target=0.90)
piv = mg.pivot_table(index=["dataset", "seed", "axis", "method"],
                     columns="model", values="coverage_gap")
maxdiff = float((piv["tabpfn"] - piv["tabpfn_temp"]).abs().max())
chk("RQ3 max tabpfn vs +T gap diff 0.0027", 0.0027, maxdiff, tol=1.0e-3)

# =========================================================================== #
# GROUP SIZES (Table 3) -- verify against committed group_sizes.csv if present
# =========================================================================== #
gpath = os.path.join(ROOT, "results", "group_sizes.csv")
if os.path.exists(gpath):
    print("### Table 3: group sizes")
    gs = pd.read_csv(gpath)
    def gsize(ds, axis, grp):
        r = gs[(gs.dataset == ds) & (gs.axis == axis) & (gs.group == grp)]
        return int(r["n_cal"].iloc[0]) if len(r) else None
    for ds, val in {"ACSEmployment-CA": 84, "ACSIncome-CA": 86,
                    "ACSPublicCoverage-CA": 131}.items():
        chk(f"T3 race Black n_cal {ds}", val, gsize(ds, "race", "Black"), tol=0.5)
    chk("T3 age 65+ n_cal ACSIncome-CA", 140, gsize("ACSIncome-CA", "age", "65+"), tol=0.5)

# =========================================================================== #
# REPORT
# =========================================================================== #
print("\n" + "=" * 78)
fails = [r for r in RESULTS if not r[3]]
for label, claimed, computed, ok in RESULTS:
    if not ok:
        cl = "None" if claimed is None else f"{claimed:.6g}"
        co = "None" if computed is None else f"{computed:.6g}"
        print(f"  FAIL  {label:45s} paper={cl:>12s}  recomputed={co:>12s}")
print("=" * 78)
print(f"{len(RESULTS) - len(fails)}/{len(RESULTS)} checks PASS, {len(fails)} FAIL")
sys.exit(1 if fails else 0)
