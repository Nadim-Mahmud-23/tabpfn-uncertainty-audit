# Cover note — paper.tex

Every number in the draft was **recomputed from the raw `calibration.csv` (75 rows) and
`conformal_fairness.csv` (3,600 rows)**, not taken from your exported `table_RQ*.csv` files.
Reason below (item 1 — read it before anything else).

---

## ⚠️ 1. Your exported RQ2/RQ3 tables and gap figures contain an averaging artifact

Decomposing RQ2 by conformity score reveals that **deterministic APS is degenerate on these
binary tasks**: coverage = 1.000 and mean set size ≈ 2.0 for every model at every target.
(Mechanism: without randomization, the APS score of the lower-ranked true label is 1, so the
calibration quantile saturates.)

Consequences:
- `table_RQ2_coverage_efficiency.csv/.tex` averages LAC and APS rows → the reported
  "coverage 0.901 at 0.80 target" is the mean of LAC ≈ 0.805 (correct) and APS = 1.0
  (vacuous). The paper's Table 4 uses **LAC only** and is correct as written.
- `table_RQ3_coverage_gap_{sex,age,race}.csv` and the three `coverage_gap_*.png` figures
  also average over both scores. Since APS gaps are identically 0, **all your exported gap
  values are exactly half the real (LAC) values** (e.g., age marginal: exported 0.0407,
  true LAC value 0.0809).
- **Action:** in notebook 03, add `& (conf.score=='lac')` to the RQ2/RQ3 filters and
  regenerate the three gap PNGs. The figure captions in paper.tex are written for the
  corrected figures and contain [NOTE TO AUTHORS] reminders.
- Silver lining: the APS degeneracy is itself a publishable negative result and is written
  up as a paragraph in §5.2.

## ⚠️ 2. Security: revoke your API tokens

Notebooks 01–03 contain a live Hugging Face token and a TabPFN API token in plain text.
Treat both as compromised: revoke and rotate them now, and strip them from the notebooks
(use environment variables or a gitignored `.env`) before releasing the repo.

## 3. The honest headline (already written into the paper)

- **RQ1:** TabPFN ECE 0.026 vs XGBoost 0.075 / LightGBM 0.052 (Wilcoxon p<1e-4, n=15
  paired cells); ties the MLP (p=0.60) with higher accuracy; temperature scaling not
  significant (p=0.12). Friedman χ²=48.5, p=7.5e-10.
- **RQ2 (LAC):** all models within 1pt of nominal coverage; TabPFN smallest sets at every
  level (1.296 vs 1.327–1.363 @90%, all p<1e-3).
- **RQ3 (LAC @90%, n=75 pairs):** age — Mondrian gap 0.081→0.051, WGC 0.862→0.881
  (p<1e-9), set-size cost +0.021; sex — no effect (p=0.50); **race — Mondrian increases
  the gap 0.046→0.057 (p=9.3e-4)**, WGC unchanged. The race result is the most novel
  finding; do not bury it.
- TabPFN and TabPFN+T produce (near-)identical conformal sets (max gap diff 0.0027) —
  monotone transform; explained in §4.3.

## 4. [MISSING] placeholders you must fill (search paper.tex for "MISSING")

Protocol (from your src/ code):
1. Number of bins in adaptive ECE (src/calibration.py)
2. Whether APS uses randomization (src/conformal.py) — confirm deterministic
3. three_way_split proportions and absolute train/cal sizes (n_test=2400 is known)
4. folktables survey year/horizon; preprocessing details; whether protected attributes
   are model features
5. Race categories retained / merged; age bin edges; **per-group calibration counts n_g**
   (needed to substantiate the race-axis explanation — export a small table)
6. Model hyperparameters / tuning budget for XGBoost, LightGBM, MLP (src/models.py)
7. TabPFN package version and checkpoint
8. Hardware + total runtime; repo URL + license; pinned dependency versions
9. Author names/affiliations, CRediT, competing interests
10. (Optional) Holm-adjusted p-values if a reviewer asks

References: all 23 entries are [CIT: ...] stubs — fill with verified BibTeX; none were
invented.

## 5. Scope changes vs. your original outline

- Tier-B OpenML and the Friedman–Nemenyi across-dataset ranking are **not in the CSVs**;
  the paper scopes them to future work and says so in Limitations. Same for shift
  experiments (RQ4/RQ5). If you run Tier-B before submission, §4.7 and §7 need one-line
  updates and a new results subsection.
- The Friedman/rank analysis was instead run on RQ1 ECE over the 15 Tier-A cells
  (legitimate, and reported descriptively).

## 6. Word count

~7,400 words excluding references — inside your 7–9k target. The race-axis finite-sample
discussion (§5.3, §6) is where reviewers will push; the per-group n_g table (item 4.5) is
your strongest defense.
