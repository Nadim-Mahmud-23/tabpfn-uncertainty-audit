#!/usr/bin/env python3
"""
Regenerate the revised figures into paper_final/figures/.

coverage_gap_{age,sex,race}.pdf : per-(base model) marginal-vs-Mondrian bars at
  the 90% target (LAC), error bars = sd over the 15 (dataset x seed) cells,
  SHARED y-range across the three axes, clean labels, no in-figure title.
reliability_example.pdf : the four base models on ACSPublicCoverage, reliability
  curve as the across-seed mean with a +/- sd band, axes cropped to [0.5, 1],
  no in-figure title.

Run from repo root:  python scripts/make_figures_final.py
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.calibration import reliability_curve   # noqa: E402

FIG = ROOT / "paper_final" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BASE = ["tabpfn", "xgboost", "lightgbm", "mlp"]
LABEL = {"tabpfn": "TabPFN", "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}
BLUE, ORANGE = "#1f77b4", "#ff7f0e"


def coverage_gap_figs():
    conf = pd.read_csv(ROOT / "results" / "conformal_fairness.csv")
    lac = conf[(conf.score == "lac") & (conf.model.isin(BASE))
               & np.isclose(conf.target_coverage, 0.90)]
    # shared y-range: max (mean + sd) over all axes/models/methods
    ymax = 0.0
    stats = {}
    for axis in ("age", "sex", "race"):
        s = lac[lac.axis == axis]
        d = {}
        for method in ("marginal", "mondrian"):
            sm = s[s.method == method]
            means = sm.groupby("model")["coverage_gap"].mean()
            sds = sm.groupby("model")["coverage_gap"].std(ddof=1)
            d[method] = (means, sds)
            ymax = max(ymax, float((means + sds).max()))
        stats[axis] = d
    ylim = (0, np.ceil(ymax * 110) / 100)   # round up a touch

    for axis in ("age", "sex", "race"):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        x = np.arange(len(BASE)); w = 0.38
        for i, method in enumerate(("marginal", "mondrian")):
            means, sds = stats[axis][method]
            ax.bar(x + (i - 0.5) * w, [means[m] for m in BASE], w,
                   yerr=[sds[m] for m in BASE], capsize=3,
                   color=BLUE if method == "marginal" else ORANGE,
                   label=method)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in BASE])
        ax.set_ylabel("coverage gap (max$-$min)")
        ax.set_ylim(*ylim)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / f"coverage_gap_{axis}.pdf")
        plt.close(fig)
        print(f"wrote coverage_gap_{axis}.pdf  (shared ylim {ylim})")


def reliability_fig():
    cache = ROOT / "cache"
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect")
    colors = {"tabpfn": BLUE, "xgboost": ORANGE, "lightgbm": "#2ca02c", "mlp": "#9467bd"}
    grid = np.linspace(0.5, 1.0, 11)
    centers = 0.5 * (grid[:-1] + grid[1:])
    for model in BASE:
        curves = []
        for seed in range(5):
            d = np.load(cache / f"ACSPublicCoverage-CA__{model}__seed{seed}.npz", allow_pickle=True)
            x, y, _ = reliability_curve(d["test_probs"], d["yte"], n_bins=15)
            # resample onto a common grid in [0.5,1] by interpolation
            keep = (x >= 0.5)
            if keep.sum() >= 2:
                curves.append(np.interp(centers, x[keep], y[keep], left=np.nan, right=np.nan))
        C = np.array(curves)
        mean = np.nanmean(C, axis=0); sd = np.nanstd(C, axis=0)
        ax.plot(centers, mean, "o-", color=colors[model], ms=3, lw=1.4, label=LABEL[model])
        ax.fill_between(centers, mean - sd, mean + sd, color=colors[model], alpha=0.15)
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "reliability_example.pdf")
    plt.close(fig)
    print("wrote reliability_example.pdf")


if __name__ == "__main__":
    coverage_gap_figs()
    reliability_fig()
