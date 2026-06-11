#!/usr/bin/env python3
"""
Regenerate the paper figures at publication resolution (dpi=300, + vector PDF).

INTEGRITY: the RQ3 coverage-gap figures filter score == 'lac'. The deterministic
APS rows are degenerate (gap == 0) on these binary tasks, so averaging them in
would halve every bar -- exactly the artifact in the original PNGs.

Outputs (PNG + PDF) into figures/:
  coverage_gap_age.png/.pdf, coverage_gap_sex.png/.pdf, coverage_gap_race.png/.pdf
  reliability_example.png/.pdf

Run:  python scripts/make_figures.py
The reliability panel refits TabPFN + XGBoost on ACSPublicCoverage (seed 0), so it
needs TABPFN_TOKEN (from .env / environment) and ~1 minute; pass --no-reliability
to skip it and regenerate only the (fast) coverage-gap figures.
"""
from __future__ import annotations
import os
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# --- secrets / SSL bootstrap (mirrors the notebooks) ----------------------- #
import certifi  # noqa: E402
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)
CONF = pd.read_csv(ROOT / "results" / "conformal_fairness.csv")
TARGET = 0.90


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{stem}.png and .pdf")


def coverage_gap_figure(axis):
    sub = CONF[(CONF.score == "lac") & (CONF.axis == axis)
               & np.isclose(CONF.target_coverage, TARGET)]
    piv = (sub.groupby(["model", "method"])["coverage_gap"].mean()
              .unstack()[["marginal", "mondrian"]])
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.plot(kind="bar", ax=ax, color={"marginal": "#1f77b4", "mondrian": "#ff7f0e"})
    ax.set_ylabel("coverage gap (max$-$min across groups)")
    ax.set_xlabel("model")
    ax.set_title(f"Subgroup coverage gap by {axis} @ 90% target (LAC)")
    ax.legend(title="method")
    ax.tick_params(axis="x", rotation=0)
    save(fig, f"coverage_gap_{axis}")


def reliability_figure():
    from src.data_loaders import load_folktables
    from src.models import build_model
    from src.pipeline import three_way_split
    from src.calibration import reliability_curve

    d = load_folktables(task="ACSPublicCoverage", state="CA", seed=0)
    (Xtr, ytr, _), _, (Xte, yte, _) = three_way_split(d["X"], d["y"], d["sensitive"], 0)
    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, color in [("tabpfn", "#1f77b4"), ("xgboost", "#ff7f0e")]:
        mdl = build_model(name, 0)
        mdl.fit(Xtr, ytr)
        x, y_, _ = reliability_curve(mdl.predict_proba(Xte), yte)
        plt.plot(x, y_, "o-", color=color, label=name)
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Reliability --- ACSPublicCoverage")
    save(fig, "reliability_example")


def main():
    for axis in ("age", "sex", "race"):
        coverage_gap_figure(axis)
    if "--no-reliability" not in sys.argv:
        reliability_figure()


if __name__ == "__main__":
    main()
