"""Re-paired RQ3 statistics at the (dataset x seed) level (paper Sec. 5.8).

Fixes the unit-of-analysis error in the original draft: marginal-vs-Mondrian
contrasts were paired at (dataset x model x seed), n=75, but the four base
models share identical splits and produce nearly identical conformal fairness
outcomes, so model-level replicates are not independent. This script:

  1. loads results/conformal_fairness.csv,
  2. keeps LAC rows and the four BASE models (drops tabpfn_temp),
  3. AVERAGES each fairness metric over models within a (dataset, seed) cell,
  4. runs paired two-sided Wilcoxon tests (marginal vs mondrian) at n = 15
     per (alpha, axis), reports direction counts,
  5. Holm-adjusts p-values within each metric family across axes.

ADJUST THE COLUMN MAP BELOW to your CSV schema before running, then paste the
printed values into the \\fillme{} slots of paper.tex (Table rq3 and the RQ3
prose) and into the abstract if you choose to quote p-values there.

Run:  python scripts/repaired_stats.py
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# EDIT ME: map the script's canonical names to your CSV's column names.
# ---------------------------------------------------------------------------
COL = {
    "dataset": "dataset",
    "model": "model",
    "seed": "seed",
    "alpha": "alpha",          # miscoverage level, e.g. 0.1
    "score": "score",          # 'lac' / 'aps'
    "axis": "axis",            # 'sex' / 'race' / 'age' / 'predclass' (a.k.a. taxonomy)
    "method": "method",        # 'marginal' / 'mondrian'
    "gap": "coverage_gap",     # max-min per-group coverage
    "wgc": "worst_group_coverage",
    "size": "mean_set_size",
}
BASE_MODELS = ["tabpfn", "xgboost", "lightgbm", "mlp"]   # excludes tabpfn_temp
CSV = "results/conformal_fairness.csv"
METRICS = ["gap", "wgc", "size"]


def holm(pvals):
    """Holm step-down adjustment; returns adjusted p-values in input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    df = pd.read_csv(CSV)
    missing = [v for v in COL.values() if v not in df.columns]
    if missing:
        sys.exit(f"Columns not found in {CSV}: {missing}\n"
                 f"Available: {list(df.columns)}\nEdit COL at the top of this script.")

    c = {k: COL[k] for k in COL}
    df = df[df[c["score"]].str.lower() == "lac"]
    df = df[df[c["model"]].str.lower().isin(BASE_MODELS)]

    # Average over models within each (dataset, seed) cell.
    keys = [c["dataset"], c["seed"], c["alpha"], c["axis"], c["method"]]
    cell = df.groupby(keys, as_index=False)[[c["gap"], c["wgc"], c["size"]]].mean()

    for alpha, sub_a in cell.groupby(c["alpha"]):
        print(f"\n=== alpha = {alpha} (target {100 * (1 - alpha):.0f}%) ===")
        rows = []
        for axis, sub in sub_a.groupby(c["axis"]):
            piv = sub.pivot_table(index=[c["dataset"], c["seed"]],
                                  columns=c["method"],
                                  values=[c["gap"], c["wgc"], c["size"]])
            res = {"axis": axis, "n": len(piv)}
            for metric in METRICS:
                m_col, d_col = (c[metric], "marginal"), (c[metric], "mondrian")
                marg, mond = piv[m_col].values, piv[d_col].values
                diff = mond - marg
                if np.allclose(diff, 0):
                    stat_p = 1.0
                else:
                    stat_p = wilcoxon(marg, mond, zero_method="wilcox",
                                      alternative="two-sided").pvalue
                res[f"{metric}_marg"] = marg.mean()
                res[f"{metric}_mond"] = mond.mean()
                res[f"{metric}_p_raw"] = stat_p
                res[f"{metric}_dir"] = int((diff > 0).sum())  # cells where Mondrian larger
            rows.append(res)

        out = pd.DataFrame(rows)
        for metric in METRICS:
            out[f"{metric}_p_holm"] = holm(out[f"{metric}_p_raw"])
        with pd.option_context("display.float_format", "{:0.4g}".format,
                               "display.width", 200):
            cols = ["axis", "n"] + [f"{m}_{s}" for m in METRICS
                                    for s in ("marg", "mond", "dir", "p_holm")]
            print(out[cols].to_string(index=False))
        print("(dir = number of the n cells in which Mondrian > marginal)")


if __name__ == "__main__":
    main()
