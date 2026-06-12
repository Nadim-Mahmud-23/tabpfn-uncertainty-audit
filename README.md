# Auditing the Uncertainty of Tabular Foundation Models
### A Conformal Prediction & Fairness Analysis of TabPFN

Companion code for the project. Everything runs on a MacBook (Apple Silicon) with no GPU.
The experiments compare **TabPFN** against gradient-boosted trees and a small neural net on:

- **Calibration** (ECE, Brier, NLL) — RQ1
- **Conformal validity & efficiency** (coverage, prediction-set size) — RQ2
- **Subgroup fairness** (per-group coverage gap; marginal vs **Mondrian** conformal) — RQ3

Conformal prediction is implemented from scratch in NumPy (`src/conformal.py`), so there is
**no dependency on a specific MAPIE version**.

---

## Paper

The manuscript (submitted to the *International Journal of Approximate Reasoning*) lives in
[`paper/`](paper/):

- `paper/paper.tex` — Elsevier `elsarticle` source; `paper/refs.bib` — verified references;
  `paper/highlights.tex` — submission highlights; `paper/paper.pdf` — compiled (43 pp).
- Build it with [Tectonic](https://tectonic-typesetting.github.io) — `cd paper && tectonic -X compile paper.tex`
  — or with `pdflatex paper.tex; bibtex paper; pdflatex paper.tex; pdflatex paper.tex`.
  For Overleaf, upload `paper.tex`, `refs.bib`, `highlights.tex`, and the `figures/` PDFs.

**Every number in the paper is reproducible from the committed result CSVs:**

- `python scripts/export_group_sizes.py` → `results/group_sizes.csv` (per-group calibration sizes).
- `python scripts/verify_paper_numbers.py` → recomputes every final-manuscript table/prose
  statistic from `results/*.csv` and the prediction cache, prints PASS/FAIL (72/72 pass).
  Conformal/fairness numbers are **LAC-only** and pooled over the **four base models**
  (tabpfn, xgboost, lightgbm, mlp); RQ3 contrasts are paired at the model-averaged
  (dataset × seed) level, n=15, Holm-adjusted.
- `python scripts/repaired_stats.py` → re-paired RQ3 Wilcoxon p-values (n=15, the corrected unit).
- `python scripts/mc_noise_floor.py` → exact Beta–binomial null model (Section 4; no data needed).
- `python scripts/build_cache.py` → caches per-cell probabilities (re-runs the matrix once),
  then `python scripts/derive_from_cache.py` → per-group coverage, empty-set rates, randomized
  APS, and temperature-scaled-GBDT metrics.
- `python scripts/make_figures.py` (or `make_figures_final.py` for the revised, error-bar figures).

`paper/CHANGES.md` is the full preparation log (every value's source, with `file:line` evidence).

---

## 1. Setup (5 minutes)

```bash
# from the project root
conda env create -f environment.yml      # or: python -m venv .venv && source .venv/bin/activate
conda activate tabpfn-audit
pip install -r requirements.txt          # if you used venv instead of the env file

jupyter lab                              # opens the notebooks
```

> Apple Silicon note: TabPFN uses the MPS/CPU backend automatically. For datasets ≤10k rows
> (everything here) inference is seconds. 16 GB RAM is enough.

---

## 2. Data — **you do not download anything manually**

| Tier | Source | How it arrives | Caches to |
|------|--------|----------------|-----------|
| **A** (fairness headline) | US Census ACS via **folktables** | `folktables` auto-downloads when first used | `data/folktables/` |
| **B** (breadth) | **OpenML** classification datasets | `fetch_openml` auto-downloads + caches | `data/openml/` |

- folktables pulls the ACS PUMS files straight from the Census Bureau
  (`https://www2.census.gov/programs-surveys/acs/...`). See https://github.com/socialfoundations/folktables.
- OpenML datasets are fetched by stable `data_id` from https://www.openml.org.

The first run of notebook **02** populates `data/` automatically. Just run the cells.

*(Optional Tier C — MC-MED clinical data — is **not** in this repo; it needs PhysioNet
credentialing. Add it later by writing one more loader that returns the same dict shape.)*

---

## 3. How to run (in order)

1. **`notebooks/01_smoke_test.ipynb`** — Run All. Confirms TabPFN downloads, predicts, and that
   the conformal/calibration code works. (~1–3 min first time.)

2. **`notebooks/02_run_experiments.ipynb`** — Run All.
   - Starts in **`QUICK=True`** mode (Tier A, 2 models, 2 seeds) so you see results in minutes.
   - When happy, set **`QUICK=False`** for the full paper run: 3 ACS datasets, 5 model
     configurations (TabPFN, TabPFN+T, XGBoost, LightGBM, MLP; +T GBDT variants for RQ1),
     **5 seeds**, 3 confidence levels, and the LAC / deterministic-APS / randomized-APS scores.
     This is about **one hour** of wall-clock time on a laptop, dominated by TabPFN CPU inference.
   - Writes `results/calibration.csv` and `results/conformal_fairness.csv` (checkpointed after
     every dataset, so it is safe to interrupt and resume).

3. **`notebooks/03_analysis_figures.ipynb`** — Run All. Produces the paper tables and figures
   into `figures/`:
   - calibration table (RQ1)
   - coverage/efficiency table (RQ2)
   - **coverage-gap bar chart: marginal vs Mondrian** (RQ3, the headline)
   - reliability diagram
   - Friedman/Nemenyi ranking across Tier-B datasets

---

## 4. Project layout

```
tabpfn-uncertainty-audit/
├── README.md
├── requirements.txt / environment.yml
├── data/        (auto-filled cache; gitignored)
├── results/     (metrics CSVs)
├── figures/     (plots)
├── src/
│   ├── data_loaders.py   # Tier A folktables + Tier B OpenML -> uniform dict
│   ├── models.py         # TabPFN, TabPFN+temp-scaling, XGBoost, LightGBM, MLP
│   ├── conformal.py      # LAC + APS scores; marginal + Mondrian (from scratch)
│   ├── calibration.py    # ECE / MCE / Brier / NLL / reliability
│   ├── fairness.py       # per-group coverage, coverage gap, set-size disparity
│   ├── stats_utils.py    # bootstrap CI, Wilcoxon, Friedman/Nemenyi
│   └── pipeline.py       # run_cell(dataset, model, seed) -> metric rows
└── notebooks/
    ├── 01_smoke_test.ipynb
    ├── 02_run_experiments.ipynb
    └── 03_analysis_figures.ipynb
```

---

## 5. Tuning the experiment

Edit the config cell in notebook 02:
- `DATASETS` — add/remove from `src/data_loaders.dataset_registry()`
- `MODELS` — subset of `tabpfn, tabpfn_temp, xgboost, lightgbm, mlp`
- `SEEDS`, `ALPHAS` (= 1 − confidence level), `SCORES` (`lac`, `aps`)
- Change the fairness axis plotted in notebook 03 via `AXIS = 'sex' | 'race' | 'age' | 'predclass'`
- Dataset size is capped via the loaders' `n=` argument (default 8000) to stay in TabPFN's sweet spot.

---

## 6. Reproducibility

All randomness is seeded (`SEEDS`). Pin your versions with `pip freeze > pip-freeze.txt`
before submitting, and commit `results/` so reviewers can re-make every figure from the CSVs.
