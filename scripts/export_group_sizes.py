#!/usr/bin/env python3
"""
Export per-group calibration/test sample sizes (n_g) for each Tier-A dataset
and protected axis, using the *exact* loaders and split that the experiments
use (src.data_loaders + src.pipeline.three_way_split), at seed 0.

This substantiates the RQ3 small-group Mondrian-failure explanation: the race
axis contains strata whose calibration support n_g is far smaller than the age
axis, so per-group conformal quantiles are estimated from very few points.

Output: results/group_sizes.csv with columns
    dataset, axis, group, n_cal, n_test, frac_cal

Run:  python scripts/export_group_sizes.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_folktables          # noqa: E402
from src.pipeline import three_way_split               # noqa: E402

SEED = 0
TASKS = ["ACSEmployment", "ACSIncome", "ACSPublicCoverage"]
AXES = ["sex", "race", "age"]


def main() -> None:
    rows = []
    for task in TASKS:
        d = load_folktables(task=task, state="CA", seed=SEED)
        # identical split to the experiment runner (same default fracs + seed)
        (_, _, _), (_, ycal, scal), (_, yte, ste) = three_way_split(
            d["X"], d["y"], d["sensitive"], SEED
        )
        n_cal_total = len(ycal)
        for axis in AXES:
            if axis not in scal:
                continue
            gcal = np.asarray(scal[axis])
            gte = np.asarray(ste[axis])
            for g in sorted(np.unique(gcal)):
                n_cal = int((gcal == g).sum())
                n_test = int((gte == g).sum())
                rows.append(
                    dict(
                        dataset=f"{task}-CA",
                        axis=axis,
                        group=str(g),
                        n_cal=n_cal,
                        n_test=n_test,
                        frac_cal=round(n_cal / n_cal_total, 4),
                    )
                )

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "group_sizes.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} rows, seed={SEED})\n")

    # console summary: smallest calibration stratum per axis (pooled view)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(out.to_string(index=False))
    print("\nSmallest calibration stratum per axis (min n_cal over datasets/groups):")
    print(out.groupby("axis")["n_cal"].min().to_string())


if __name__ == "__main__":
    main()
