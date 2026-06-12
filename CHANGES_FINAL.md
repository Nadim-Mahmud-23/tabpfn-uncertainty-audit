# CHANGES_FINAL.md — IJAR revision finalization log

Integrity: every value below was produced by a script run in this session (named per item)
or read from the committed CSVs. Nothing invented.

## Phase 0 — setup
- `paper_final/` = copy of `overleaf-submission-revised/`. 24 real `\fillme{}` placeholders
  (+3 in comments). COL map in `repaired_stats.py` already matches CSV schema — no edit needed.
- **MC null check (ground rule 2):** ran `paper_final/scripts/mc_noise_floor.py`; all "Null"
  columns of `tab:mcnull` reproduce exactly (dataset-averaged): Sex 0.013/0.010/0.007 (marg),
  0.019/0.014/0.010 (mond); Race 0.048/0.036/0.026, 0.070/0.052/0.037; Age 0.043/0.032/0.023,
  0.063/0.047/0.034. Floor prose (0.011 sex, 0.027–0.045 age, 0.044–0.055 race) confirmed.
  **Nulls untouched.**

## 4-model recomputation (ground rule 3) — source: repaired_stats.py + recompute_4model.py
Switching pooling from 5 models (incl. tabpfn_temp) to the 4 base models shifts (3 dp):
- Marginal mean set size **1.324 → 1.331** (everywhere); axis Mondrian sizes: sex 1.327→1.334,
  age 1.345→1.353, race 1.334→1.341, predclass 1.332→1.340.
- Race marginal gap @90% **0.046 → 0.044**; predclass marginal gap @90% 0.045 → 0.044.
- Per-dataset race gap @90% marg→mond: Employment 0.034→0.051 (was →0.049), Income 0.048→0.058
  (was 0.047→0.060), PublicCoverage 0.052→0.062 (was 0.056→0.061).
- Table rq3targets: 80% sex mond 0.021→0.020, age mond 0.106→0.105, race 0.056/0.055→0.057/0.058;
  90% race marg 0.046→0.044; 95% age mond 0.037→0.038, race marg 0.034→0.033.
- Table tax (set-size disparity): sex 0.058/0.086→0.059/0.086, age 0.236/0.369→0.240/0.373,
  race 0.123/0.154→0.118/0.154, predclass 0.115/0.157→0.115/0.160. "56%"→"55%".
- mcnull Observed cols: 80% Sex-Mond 0.021→0.020, Age-Mond 0.106→0.105, Race-Marg 0.056→0.057,
  Race-Mond 0.055→0.058; 90% Race-Marg 0.046→0.044; 95% Race-Marg 0.034→0.033.

## RQ3 Holm-adjusted Wilcoxon, n=15 (alpha=0.1) — source: repaired_stats.py
- Age: gap p=3.7e-4 (dir 14/15 smaller), WGC p=4.9e-4 (14/15 higher), size p=2.4e-4.
- Race: gap p=4.0e-3 (dir 12/15 larger), WGC p=1.0 (n.s.), size p=2.0e-3.
- Sex: gap p=0.60, WGC p=1.0. Predclass: gap p=2.4e-4.
- Age set-size cost +0.021 (1.331→1.353); WGC age@80 0.731→0.774, @95 0.927→0.935.

## Phase 2 — cache-derived results (source: build_cache.py → derive_from_cache.py, 60 cells)
- **GBDT+T (M6):** XGBoost+T acc 0.763, ECE 0.026, MCE 0.054, Brier 0.319, NLL 0.479;
  LightGBM+T acc 0.769, ECE 0.026, MCE 0.055, Brier 0.312, NLL 0.470. Temperature scaling
  drives both GBDTs' ECE to **0.026 = TabPFN's** (paired p=0.98 / 1.0), but TabPFN keeps a
  significant Brier/NLL edge (p=6.1e-5). Filled Table rq1 + the "Fourth" observation + intro
  bullet. **This confirms the paper's predicted pattern — no claim rewrite needed.**
- **Per-group coverage Table 7 (M8):** filled all three axes (e.g. age <25 0.935 over-covered,
  55-64 0.884 worst, Black 0.891 under-covered under marginal; Mondrian column too). CI
  half-widths 0.005–0.024 noted in caption. Added the named-group prose sentence in §6.3.
- **Empty-set (min-2):** realized LAC empty rate 0 at every level; raw fallback-trigger rate
  0.79% at 80%, 0% at 90%/95%. Filled Table rq2 caption + §6.2 inline.
- **Randomized APS (M5):** valid (cov 0.798/0.897/0.953), sets larger than LAC
  (1.152/1.409/1.618). Marginal age gap 0.038 (vs LAC 0.081, near the null 0.032) — APS removes
  most structural age bias without conditioning. Mondrian still nudges race up
  (0.041→0.054, p=0.055). Wrote the full §6.2 Randomized-APS paragraph + intro/§5.6/abstract.
  Added randomized APS functions to src/conformal.py (aps_scores_rand / aps_sets_rand).
- **Group-size sd (min-5):** vary by ≤72 points over seeds (per-group sd ≤33); filled caption.
- **Intersectional sex×race (M9 iii):** RAN it (from cache): ≤8 strata, smallest ≈34;
  marginal gap 0.082, Mondrian *increases* to 0.099 (n=15, p=0.013) — sharper small-group
  failure, as predicted. Filled the Limitations sentence (second-state replication deferred).

## Phase 3 — figures (make_figures_final.py)
- coverage_gap_{age,sex,race}.pdf: per-base-model marginal/Mondrian bars, error bars = sd over
  15 cells, shared y-range (0,0.14), clean labels, no titles. reliability_example.pdf: 4 base
  models, across-seed mean ± sd band, cropped to [0.5,1], no title. Removed all 4 caption
  `\fillme` notes and updated caption text.

## Phase 4 — repo hygiene
- verify_paper_numbers.py rewritten to the 4-model rule + cache-derived checks: **72/72 PASS**
  (includes MC-null reproduction = ground rule 2). Added mc_noise_floor.py, repaired_stats.py,
  build_cache.py, derive_from_cache.py, recompute_4model.py, make_figures_final.py to scripts/.
  README reconciled: verification 72/72, 5 seeds (not 10), ~1 hour (not overnight), randomized
  APS, new scripts listed.

## Phase 5 — manuscript + packaging
- All 24 `\fillme` resolved; `\fillme` macro + revision-notes comment block deleted. Only the
  Zenodo line remains, as a plain non-red bracket. Compiles with tectonic: **0 undefined refs,
  0 errors, 59 pp**. Package `overleaf-submission-final.zip` = paper.tex, refs.bib,
  highlights.tex, figures/ (5 referenced PDFs) — nothing else; compiled `paper_final.pdf` saved.

## Calibration-size sweep (M3b) — DONE (scripts/calib_sweep.py)
- Subsample each model's calibration set to n_cal in {500,1000,1500,2000}; recompute LAC
  race/age marginal & Mondrian gaps vs min_g n_g, with the exact Mondrian null overlaid.
  Result (Fig. gap_vs_support): race Mondrian rises 0.057 (min_n 97) -> 0.080 (min_n 23),
  above marginal at every support level; age Mondrian repair weakens 0.051 (258) -> 0.072 (66).
  Observed Mondrian gaps track their nulls. Added Fig. + paragraph to Sec 6.4. (Used n_cal up to
  2000, the committed calibration size; n_cal=4000 would need a larger split, not pursued.)

## Items NOT done (with reason)
- **Second-state (TX) replication (M9 i):** deferred to future work (stated in Limitations) —
  needs a fresh data download + cache build.
- **Zenodo DOI:** author's manual deposit step (left as a plain bracketed note in Reproducibility).

## Numbers that contradicted no claim (checked)
- Recalibrated GBDTs reach ECE 0.026 = TabPFN, but TabPFN wins Brier/NLL — matches the paper's
  contingent claim. Randomized APS does not beat LAC on fairness (it shifts where gaps start but
  Mondrian still worsens race). No claim required softening.
