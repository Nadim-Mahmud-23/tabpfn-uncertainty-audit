#!/usr/bin/env python3
"""
Derive the revision quantities that need model predictions, from the cached
probabilities in cache/ (built by build_cache.py). No model refitting.

Computes, pooled over the four base models (tabpfn, xgboost, lightgbm, mlp):
  (A) per-group coverage, marginal vs Mondrian, LAC @90%, with bootstrap 95% CIs;
  (B) LAC empty-set / fallback-trigger rate per target level;
  (C) randomized-APS marginal coverage, mean set size, and per-axis gaps
      (marginal vs Mondrian) vs LAC, with n=15 paired Wilcoxon on the gaps;
  (D) GBDT+T calibration metrics (XGB+T, LGBM+T) and TabPFN-vs-+T paired tests.

Run from repo root:  python scripts/derive_from_cache.py
"""
from __future__ import annotations
import sys
import pathlib
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src import conformal as cf                      # noqa: E402
from src.fairness import per_group_coverage          # noqa: E402
from src.calibration import all_calibration_metrics  # noqa: E402

CACHE = ROOT / "cache"
BASE = ["tabpfn", "xgboost", "lightgbm", "mlp"]
DATASETS = ["ACSEmployment-CA", "ACSIncome-CA", "ACSPublicCoverage-CA"]
SEEDS = range(5)
AXES = ["sex", "race", "age"]
ALPHAS = {0.20: "80%", 0.10: "90%", 0.05: "95%"}


def load(ds, model, seed):
    return np.load(CACHE / f"{ds}__{model}__seed{seed}.npz", allow_pickle=True)


def boot_ci(vals, reps=5000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(reps)])
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# =========================================================================== #
# (A) per-group coverage @90% LAC, marginal vs Mondrian
# =========================================================================== #
def per_group():
    # group -> {(dataset,seed): [coverage over models]}
    marg = defaultdict(lambda: defaultdict(list))
    mond = defaultdict(lambda: defaultdict(list))
    AXIS_OF = {}
    for ds in DATASETS:
        for seed in SEEDS:
            for model in BASE:
                d = load(ds, model, seed)
                calp, tep, ycal, yte = d["cal_probs"], d["test_probs"], d["ycal"], d["yte"]
                for ax in AXES:
                    cg, tg = d[f"cal_{ax}"], d[f"te_{ax}"]
                    ms = cf.marginal_conformal(calp, ycal, tep, 0.10, "lac")
                    ds_ = cf.mondrian_conformal(calp, ycal, cg, tep, tg, 0.10, "lac")
                    for grp, cov in per_group_coverage(ms, yte, tg).items():
                        marg[(ax, grp)][(ds, seed)].append(cov); AXIS_OF[(ax, grp)] = ax
                    for grp, cov in per_group_coverage(ds_, yte, tg).items():
                        mond[(ax, grp)][(ds, seed)].append(cov)
    print("\n=== (A) Per-group coverage @90% LAC (mean [95% CI] over cells) ===")
    out = {}
    for key in sorted(marg, key=lambda k: (AXES.index(k[0]), k[1])):
        mvals = [np.mean(v) for v in marg[key].values()]   # model-avg per cell
        dvals = [np.mean(v) for v in mond[key].values()]
        mm, mlo, mhi = boot_ci(mvals); dm, dlo, dhi = boot_ci(dvals)
        out[key] = (mm, mlo, mhi, dm, dlo, dhi)
        print(f"  {key[0]:5s} {str(key[1]):8s}  marg {mm:.3f} [{mlo:.3f},{mhi:.3f}]"
              f"   mond {dm:.3f} [{dlo:.3f},{dhi:.3f}]")
    return out


# =========================================================================== #
# (B) LAC empty-set / fallback-trigger rate per target
# =========================================================================== #
def empty_rates():
    print("\n=== (B) LAC raw-empty (fallback-trigger) rate by target (pooled base models) ===")
    for alpha, tgt in ALPHAS.items():
        rates = []
        for ds in DATASETS:
            for seed in SEEDS:
                for model in BASE:
                    d = load(ds, model, seed)
                    calp, tep, ycal = d["cal_probs"], d["test_probs"], d["ycal"]
                    s = cf.lac_scores(calp, ycal)
                    qhat = cf.conformal_qhat(s, alpha)
                    raw = tep >= (1.0 - qhat)              # before the argmax fallback
                    rates.append((~raw.any(axis=1)).mean())
        print(f"  {tgt}: raw-empty rate = {np.mean(rates)*100:.2f}% "
              f"(realized empty rate after argmax fallback = 0.00%)")


# =========================================================================== #
# (C) randomized APS
# =========================================================================== #
def rand_aps():
    print("\n=== (C) Randomized APS (pooled base models) ===")
    cov_by_t = defaultdict(list); size_by_t = defaultdict(list)
    # per-axis gaps at 90%: axis -> {(ds,seed): [gap over models]} for marg/mond
    gmarg = {ax: defaultdict(list) for ax in AXES}
    gmond = {ax: defaultdict(list) for ax in AXES}
    for ds in DATASETS:
        for seed in SEEDS:
            for model in BASE:
                d = load(ds, model, seed)
                calp, tep, ycal, yte = d["cal_probs"], d["test_probs"], d["ycal"], d["yte"]
                rng = np.random.default_rng(7000 + seed)
                for alpha, tgt in ALPHAS.items():
                    cs = cf.aps_scores_rand(calp, ycal, rng)
                    qhat = cf.conformal_qhat(cs, alpha)
                    sets = cf.aps_sets_rand(tep, qhat, rng)
                    cov_by_t[tgt].append(cf.covered(sets, yte).mean())
                    size_by_t[tgt].append(cf.set_sizes(sets).mean())
                    if alpha == 0.10:
                        for ax in AXES:
                            tg, cg = d[f"te_{ax}"], d[f"cal_{ax}"]
                            # marginal gap (global qhat already computed)
                            pgc = per_group_coverage(sets, yte, tg)
                            gmarg[ax][(ds, seed)].append(max(pgc.values()) - min(pgc.values()))
                            # mondrian gap (per-group qhat with randomized score)
                            msets = np.zeros_like(tep, bool)
                            gqh = cf.conformal_qhat(cf.aps_scores_rand(calp, ycal, rng), alpha)
                            for g in np.unique(tg):
                                cm, tm = (cg == g), (tg == g)
                                qh = (cf.conformal_qhat(cf.aps_scores_rand(calp[cm], ycal[cm], rng), alpha)
                                      if cm.sum() >= 20 else gqh)
                                msets[tm] = cf.aps_sets_rand(tep[tm], qh, rng)
                            pgm = per_group_coverage(msets, yte, tg)
                            gmond[ax][(ds, seed)].append(max(pgm.values()) - min(pgm.values()))
    for tgt in ALPHAS.values():
        print(f"  {tgt}: coverage {np.mean(cov_by_t[tgt]):.3f}, mean set size {np.mean(size_by_t[tgt]):.3f}")
    print("  per-axis gap @90% (marg -> mond), n=15 paired Wilcoxon:")
    for ax in AXES:
        mc = np.array([np.mean(gmarg[ax][k]) for k in sorted(gmarg[ax])])
        dc = np.array([np.mean(gmond[ax][k]) for k in sorted(gmond[ax])])
        p = wilcoxon(mc, dc).pvalue if not np.allclose(mc, dc) else 1.0
        print(f"    {ax:5s} {mc.mean():.3f} -> {dc.mean():.3f}  (p={p:.3g}, n={len(mc)})")


# =========================================================================== #
# (D) GBDT+T calibration metrics
# =========================================================================== #
def gbdt_temp():
    print("\n=== (D) GBDT+T RQ1 metrics (mean +/- sd over 15 cells) ===")
    rows = {"xgboost_T": defaultdict(list), "lightgbm_T": defaultdict(list),
            "tabpfn": defaultdict(list)}
    cellkey = []
    for ds in DATASETS:
        for seed in SEEDS:
            cellkey.append((ds, seed))
            for model in ["xgboost", "lightgbm"]:
                d = load(ds, model, seed)
                m = all_calibration_metrics(d["test_probs_T"], d["yte"], int(d["n_classes"]))
                for k, v in m.items():
                    rows[f"{model}_T"][k].append(v)
            dt = load(ds, "tabpfn", seed)
            mt = all_calibration_metrics(dt["test_probs"], dt["yte"], int(dt["n_classes"]))
            for k, v in mt.items():
                rows["tabpfn"][k].append(v)
    for name in ["xgboost_T", "lightgbm_T"]:
        r = rows[name]
        print(f"  {name}: acc {np.mean(r['accuracy']):.3f}+/-{np.std(r['accuracy'],ddof=1):.3f}  "
              f"ECE {np.mean(r['ece_adaptive']):.3f}+/-{np.std(r['ece_adaptive'],ddof=1):.3f}  "
              f"MCE {np.mean(r['mce']):.3f}+/-{np.std(r['mce'],ddof=1):.3f}  "
              f"Brier {np.mean(r['brier']):.3f}+/-{np.std(r['brier'],ddof=1):.3f}  "
              f"NLL {np.mean(r['nll']):.3f}+/-{np.std(r['nll'],ddof=1):.3f}")
    print("  TabPFN vs +T paired Wilcoxon (n=15):")
    for name in ["xgboost_T", "lightgbm_T"]:
        for metric in ["ece_adaptive", "brier", "nll"]:
            a = np.array(rows["tabpfn"][metric]); b = np.array(rows[name][metric])
            p = wilcoxon(a, b).pvalue
            print(f"    TabPFN vs {name} {metric}: {np.mean(a):.3f} vs {np.mean(b):.3f}  p={p:.3g}")


if __name__ == "__main__":
    per_group()
    empty_rates()
    rand_aps()
    gbdt_temp()
