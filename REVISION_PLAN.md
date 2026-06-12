# REVISION_PLAN — referee report → manuscript v2

Status legend: ✅ done in this revision · 🔴 **you must run** (placeholder `\fillme{}` in red in paper.tex) · 📝 do before submission

**Do not submit while any red `\fillme{}` remains in the compiled PDF.** Search the .tex for `\fillme` — there are 24 occurrences. When all are resolved, delete the macro definition so any stragglers break the build.

---

## Major comments

**M1 — n=75 pairing error.** ✅ Statistical protocol rewritten (Sec. 5.8): RQ3 contrasts now paired at the model-averaged (dataset × seed) level, n=15, Holm-adjusted; the n=75 pairing is disowned explicitly. Abstract no longer quotes p-values. 🔴 Run `scripts/repaired_stats.py` (edit the COL map to your CSV schema first) and fill the p-values + direction counts in Table 6 and the RQ3 prose.

**M2 — bias/noise conflation.** ✅ Sec. 4.1 now decomposes the marginal gap into structural bias + calibration noise + **test-side measurement noise** (the previously ignored term), and Sec. 6.4 quantifies it with the zero-bias marginal null: race marginal gap 0.046 vs null 0.036 → mostly noise; age 0.081 vs 0.032 → real bias. The "essentially independent of calibration size" claim is gone. Real numbers — no action needed.

**M3a — Monte Carlo validation.** ✅ DONE WITH REAL NUMBERS. `scripts/mc_noise_floor.py` (committed, 200k replicates, exact Beta–binomial law, seconds on CPU) produced Table 1 (tab:mcnull) and Fig. 1 (figures/mc_noise_floor.pdf). Headline: observed Mondrian gaps match their simulated nulls within 0.005 at the 90% target on every axis; the closed-form floor is accurate for one-dominant-stratum taxonomies (race) but *underestimates* the null when several strata are comparably small (age) — honestly characterized in Sec. 4.3, which now recommends the simulation as the operational diagnostic and demotes Eq. (gapfloor) to a scaling mnemonic. Commit the script to the repo and re-run once to confirm identical output on your machine.

**M3b — calibration-size sweep.** 🔴 Not runnable without your pipeline. Recommended (cheap, overnight-at-most): vary n_cal ∈ {500, 1000, 2000, 4000} or subsample strata directly; plot observed marginal/Mondrian gaps vs min n_g with the simulated nulls overlaid — the crossover figure. Add as Sec. 6.4 figure or appendix. The paper as revised stands without it, but this is the single most persuasive addition for a Q1 referee.

**M4 — manufactured small-group regime.** ✅ New Sec. 5.2 "The small-group regime is a designed stress test" states it plainly; framing throughout changed from "discovery about ACS" to "controlled demonstration of a regime that arises in genuinely small-data domains"; Limitations updated.

**M5 — vacuous APS half / randomized APS missing.** ✅ Deterministic-APS finding demoted from contribution list to a "practical caution" paragraph (Sec. 6.2); abstract reworded. 🔴 Implement randomized APS (one-line change in src/conformal.py per your README), regenerate the matrix, and fill the "Randomized APS" paragraph: coverage, set size, and — critically — its per-group gaps vs LAC's (APS exists for conditional coverage; this is your RQ3 in disguise).

**M6 — recalibrated GBDT baseline.** ✅ XGBoost+T and LightGBM+T added to the design (Sec. 5.4) with the rationale; RQ1 conclusion reframed as "matches the recalibrated recipe without the recalibration step", explicitly contingent. 🔴 Fit temperature on the calibration split for both GBDTs (you already have the machinery from TabPFN+T), fill the two rows of Table 2 and the "Fourth" observation in Sec. 6.1.

**M7 — novelty claims & missing citations.** ✅ Added (all bibliographically verified): Vovk 2012 (PMLR 25:475–490), Barber et al. 2021 (Inf. & Inference 10(2):455–482), Lu et al. 2022 (AAAI 36(11):12008–12016), Löfström et al. 2015 (IDA 19(6):1355–1375). "The intersection is empty" deleted; novelty re-scoped in Sec. 2 to (a) the exact pre-deployment null model and (b) the marginal-vs-Mondrian crossover demonstration + the foundation-model pipeline; "first calibration audit" softened against Hollmann et al. 2025's own calibration results. ✅ **Bibliography expanded to 71 entries** (round 2): the related-work, background, methods, and statistics sections now also cite the CP foundations and extensions (Shafer–Vovk tutorial, Fontana review, Lei 2014, RAPS, Cauchois–Gupta–Duchi, CQR, jackknife+, RCPS, Bian–Barber, Guan, Stutz et al.), the algorithmic-fairness canon (Dwork, Feldman, Hardt, Chouldechova, Kleinberg, Pleiss, multicalibration, Cherian–Candès auditing, Mehrabi survey, Barocas–Hardt–Narayanan, Obermeyer), the calibration lineage (Brier, Gneiting–Raftery, Platt, Zadrozny–Elkan, Naeini, Kull Dirichlet, Vaicenavicius, Ovadia), the tabular canon (Breiman, Friedman, XGBoost, LightGBM, CatBoost, Borisov survey, Shwartz-Ziv, Kadra, van Breugel position), and methodology (Wilcoxon, Holm, Friedman 1937, Demšar, scikit-learn). Every entry's venue/volume/pages was verified against publisher or index records; every entry is cited in the text (checked programmatically: 71 in bib, 71 cited, zero orphans). 📝 Run a fresh literature search at submission time for 2025–26 conformal-fairness work.

**M8 — no per-group coverage shown.** ✅ New Table 7 (tab:pergroup) skeleton with named groups + bootstrap-CI columns; Sec. 3.5 now designates worst-group coverage as the primary cross-axis statistic and mandates reading gaps against the null, not zero. 🔴 Export per-group coverage from your pipeline (fairness.py computes it; add an export if only summaries are committed) and fill the table. Also state in prose *which* group is worst-covered on age (likely <25 or 65+ — check) and the *direction* of the Black stratum's deviation under marginal calibration.

**M9 — thin external validity.** ✅ Limitations now states the three-tasks-one-population point explicitly; intersectional (sex × race) axis flagged. 🔴 (i) Run at least one additional state (folktables parameter; overnight per your README) and report whether the null-model reading replicates; (ii) reconcile seeds: README promises a 10-seed full run, paper commits 5 — either run 10 or fix the README; (iii) run sex × race or explicitly defer with justification in Limitations.

**M10 — TabPFN+T conformal rows invalid.** ✅ TabPFN+T (and the new GBDT+T) excluded from all conformal analyses with the exchangeability rationale (Sec. 5.4); conformal analyses now use the 4 base models; tables/captions updated ("pooled over 4 base models"). Note: pooled RQ3 numbers in the current draft were computed over 5 models including tabpfn_temp — since its conformal rows coincide with TabPFN's to ≤0.0027, the 4-model means will shift negligibly, but 🔴 recompute them (repaired_stats.py already filters to BASE_MODELS) and update Table 6/8 values if any third decimal moves.

---

## Minor comments

1. Theory promoted out of Background into its own Section 4. ✅
2. Empty LAC sets acknowledged (Sec. 3.3); rate slot in Table 5 caption and Sec. 6.2. 🔴 compute empty-set rate per level (likely nonzero at 80%).
3. Holm adjustment adopted, α=0.05 on adjusted p (Sec. 5.8). ✅
4. n_g ≳ 10/α restated as a corollary of the null-model rule (Discussion). ✅
5. Table 4 caption asks for mean ± sd of group sizes over seeds. 🔴 one-liner from export_group_sizes.py.
6. Race-coarsening sensitivity acknowledged (Sec. 5.1 + Limitations). 📝 optional: 5-way split sensitivity paragraph.
7. Debiased/smooth ECE appendix. 📝 optional.
8. Abstract rewritten: 247 words, no p-values. ✅
9. Title shortened to the question form. ✅
10–11. Figure regeneration specs embedded as red notes in each caption: error bars over seeds, shared y-range, clean labels ("TabPFN+T" not "tabpfn_temp"), no in-figure titles, reliability diagram cropped to [0.5, 1] with all models as small multiples. 🔴 update scripts/make_figures.py accordingly.
12. Repo/paper inconsistencies. 📝 fix before submission: 180/180 vs 197/197 verification counts; "overnight" vs "~1 hour"; 5 vs 10 seeds (see M9).
13. gibbs2023 was already correct (JRSS-B 87(4):1100–1126, 2025) — verified, kept. ✅
14. Eq. (5) presentation fixed: randomized score defined, deterministic variant stated in prose. ✅
15. TabPFN-vs-MLP "indistinguishable" now flagged as failure-to-reject (Sec. 6.1). 📝 optional TOST.
16. "honestly" header → "scoped"; first-person hedge removed; long sentences split. ✅
17. Highlights rewritten (5 bullets; longest = 84 chars — verify after any edit). ✅
18. Zenodo DOI slot added to Reproducibility statement. 📝 deposit results/ + scripts before submission.

---

## Order of operations (suggested)

1. `python scripts/mc_noise_floor.py` — confirm Table 1 numbers reproduce (no data needed).
2. Implement randomized APS; kick off full matrix regeneration overnight (covers M5; optionally fold in second state + 10 seeds, M9).
3. While that runs: GBDT+T rows (M6, minutes), per-group coverage export (M8), empty-set rates (min-2), group-size sd (min-5), figure regeneration (min-10/11).
4. `python scripts/repaired_stats.py` on the fresh CSVs → fill all p-value slots (M1, M10).
5. (Strongly recommended) calibration-size sweep (M3b) → one figure.
6. Update verify_paper_numbers.py to cover the new tables; reconcile README; Zenodo deposit.
7. Grep for `\fillme` → must return nothing → delete the macro → final compile → submit.
