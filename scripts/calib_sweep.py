#!/usr/bin/env python3
"""
Calibration-size sweep (paper Sec. 6.4, Fig. gap_vs_support).

Holds each fitted model fixed and subsamples its 2,000-point calibration set to
n_cal in {500, 1000, 1500, 2000}, recomputing the LAC marginal and Mondrian
race/age gaps on the full test set. This isolates the effect of calibration
support (and hence min_g n_g) on the marginal-vs-Mondrian crossover. The exact
zero-bias Mondrian null (mc_noise_floor.simulate_gap), evaluated on the
subsampled per-group sizes, is overlaid.

Output: paper_final/figures/gap_vs_support.pdf
Run from repo root:  python scripts/calib_sweep.py
"""
from __future__ import annotations
import sys
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src import conformal as cf                       # noqa: E402
from src.fairness import per_group_coverage           # noqa: E402
from scripts.mc_noise_floor import simulate_gap       # noqa: E402

CACHE = ROOT / "cache"
BASE = ["tabpfn", "xgboost", "lightgbm", "mlp"]
DATASETS = ["ACSEmployment-CA", "ACSIncome-CA", "ACSPublicCoverage-CA"]
NCALS = [500, 1000, 1500, 2000]
ALPHA = 0.10
AXES = ["race", "age"]
COLOR = {"race": "#EE6677", "age": "#228833"}


def gap(sets, yte, groups):
    pgc = per_group_coverage(sets, yte, groups)
    return max(pgc.values()) - min(pgc.values())


def main():
    # observed: ncal -> axis -> {'marg':[per-cell],'mond':[...]}; and min_n: ncal->axis->[min stratum]
    obs = {n: {ax: {"marg": defaultdict(list), "mond": defaultdict(list)} for ax in AXES} for n in NCALS}
    minn = {n: {ax: [] for ax in AXES} for n in NCALS}
    null = {n: {ax: [] for ax in AXES} for n in NCALS}     # simulated Mondrian null per (n,axis)
    for ds in DATASETS:
        for seed in range(5):
            for model in BASE:
                d = np.load(CACHE / f"{ds}__{model}__seed{seed}.npz", allow_pickle=True)
                calp, tep, ycal, yte = d["cal_probs"], d["test_probs"], d["ycal"], d["yte"]
                ncal_full = len(ycal)
                rng = np.random.default_rng(900 + seed)
                for n in NCALS:
                    idx = rng.choice(ncal_full, size=min(n, ncal_full), replace=False)
                    cp, yc = calp[idx], ycal[idx]
                    ms = cf.marginal_conformal(cp, yc, tep, ALPHA, "lac")
                    for ax in AXES:
                        cg, tg = d[f"cal_{ax}"][idx], d[f"te_{ax}"]
                        ds_ = cf.mondrian_conformal(cp, yc, cg, tep, tg, ALPHA, "lac")
                        obs[n][ax]["marg"][(ds, seed)].append(gap(ms, yte, tg))
                        obs[n][ax]["mond"][(ds, seed)].append(gap(ds_, yte, tg))
                        if model == "tabpfn":   # group sizes are model-independent
                            sizes = [int((cg == g).sum()) for g in np.unique(cg)]
                            minn[n][ax].append(min(sizes))
                            null[n][ax].append(simulate_gap(sizes, ALPHA, reps=20_000, mondrian=True).mean())

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for axis in AXES:
        xs, mg, dg, nl = [], [], [], []
        for n in NCALS:
            xs.append(np.mean(minn[n][axis]))
            mg.append(np.mean([np.mean(v) for v in obs[n][axis]["marg"].values()]))
            dg.append(np.mean([np.mean(v) for v in obs[n][axis]["mond"].values()]))
            nl.append(np.mean(null[n][axis]))
        c = COLOR[axis]
        ax.plot(xs, mg, "o-", color=c, lw=1.8, label=f"{axis}: marginal (obs.)")
        ax.plot(xs, dg, "s--", color=c, lw=1.8, mfc="white", label=f"{axis}: Mondrian (obs.)")
        ax.plot(xs, nl, ":", color=c, lw=1.2, alpha=0.7, label=f"{axis}: Mondrian null")
    ax.set_xscale("log")
    ax.set_xlabel(r"smallest calibration stratum $\min_g n_g$")
    ax.set_ylabel(r"coverage gap at $90\%$ target")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "paper_final" / "figures" / "gap_vs_support.pdf"
    fig.savefig(out)
    print(f"wrote {out}")
    # print the crossover summary
    for axis in AXES:
        for n in NCALS:
            mg = np.mean([np.mean(v) for v in obs[n][axis]["marg"].values()])
            dg = np.mean([np.mean(v) for v in obs[n][axis]["mond"].values()])
            print(f"  {axis} n_cal={n:5d} min_n={np.mean(minn[n][axis]):5.0f}  "
                  f"marg {mg:.3f}  mond {dg:.3f}  {'MOND>MARG' if dg>mg else ''}")


if __name__ == "__main__":
    main()
