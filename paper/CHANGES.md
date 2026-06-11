# CHANGES.md — running log for IJAR submission preparation

Integrity rule: every value below is traceable to a file:line or to a computation against
`results/*.csv`. Nothing invented. Items the author must supply are tagged `[STILL MISSING]`.

---

## PHASE 1 — placeholders resolved from the codebase

Each entry: paper location → value inserted → evidence (file:line).

| # | Paper placeholder | Value inserted | Evidence |
|---|---|---|---|
| 1 | Adaptive-ECE bin count (`§3.1`) | **B = 15** equal-mass bins | `src/calibration.py:13` (`n_bins=15`), used via `all_calibration_metrics` `:69` |
| 2 | APS randomization (`§3.3`) | **Deterministic, no randomization** — score accumulates sorted probs incl. the true label's own mass (= u≡1) | `src/conformal.py:57-63` (`aps_scores` returns `cum[ranks]`, no `u`) |
| 3 | Split proportions + sizes (`§4.4`) | **45/25/30%** = 3,600 / 2,000 / 2,400 of 8,000, label-stratified | `src/pipeline.py:19-26` (`cal_frac=0.25, test_frac=0.30`); n_test=2400 matches `calibration.csv` |
| 4 | folktables survey year/horizon (`§4.1`) | **2018, 1-Year, person survey, CA** | `src/data_loaders.py:73,85` |
| 5 | Race categories (`§4.1`) | **4 groups**: White(1), Black(2), Asian(6), Other(rest merged) | `src/data_loaders.py:47-50` |
| 6 | Age bins (`§4.1`) | **5 bands**, edges 25/40/55/65: <25,25-39,40-54,55-64,65+ (PublicCoverage has 4: no 65+) | `src/data_loaders.py:41-44`; `results/group_sizes.csv` (no 65+ row for PublicCoverage) |
| 7 | Per-group n_g table (`§4.1`, new Table 3) | Inserted full table; **Black n_cal = 84/86/131**, smallest age band 65+ = **140** | `scripts/export_group_sizes.py` → `results/group_sizes.csv` (seed 0) |
| 8 | Preprocessing (`§4.2`) | Median-impute numerics; ordinal-encode categoricals (unknown→−1); float32; MLP-only StandardScaler; subsample cap **n=8,000** | `src/data_loaders.py:53-67,31-38,73-74`; `src/models.py:36-39` |
| 9 | Protected attrs as features? (`§4.2`) | **Included as model inputs** (part of folktables standard feature sets; not withheld) | `src/data_loaders.py:88,99` (`df_to_pandas` returns SEX/RAC1P/AGEP; `_preprocess_frame` uses all cols) |
| 10 | TabPFN version + checkpoint (`§4.3`) | **tabpfn 8.0.7**, checkpoint **tabpfn-v3-classifier-v3_default.ckpt** (repo Prior-Labs/tabpfn_3) | `importlib.metadata`; cache `~/Library/Caches/tabpfn/`; `tabpfn/model_loading.py:173,180,504-508` |
| 11 | XGBoost hyperparams (`§4.3`) | 300 trees, depth 6, lr 0.1, subsample/colsample 0.9, hist | `src/models.py:22-25` |
| 12 | LightGBM hyperparams (`§4.3`) | 300 trees, 31 leaves, lr 0.05, subsample/colsample 0.9 | `src/models.py:28-31` |
| 13 | MLP architecture (`§4.3`) | (128,64), standardized inputs, early stopping 10% val, ≤300 epochs | `src/models.py:36-39` |
| 14 | Tuning budget (`§4.3,§5.4,§6`) | **None** — fixed defaults, no per-dataset search, for all models | `src/models.py:15-40` (no CV/search anywhere) |
| 15 | Training rows/cell (`§5.4`) | **3,600** | derived from split (item 3) |
| 16 | Race n_g in mechanism para (`§5.3`) | Black ≈ **84–131** vs White 1,100–1,300 | `results/group_sizes.csv` |
| 17 | n_g threshold, deployment (`§5.3`) | Rule 10/α ≈ 100 at α=0.1; age-min 140 clears, race-min (Black) 84 fails | `results/group_sizes.csv` |
| 18 | Dependency versions (Reproducibility) | tabpfn 8.0.7, xgboost 3.2.0, lightgbm 4.6.0, sklearn 1.8.0, folktables 0.0.12, numpy 2.3.5, scipy 1.17.0, torch 2.9.1 | `importlib.metadata` |
| 19 | Hardware (Reproducibility) | Apple Silicon MacBook, no GPU (CPU/MPS) | `README.md:4,27-28` |

---

## Integrity corrections made in Phase 1 (beyond filling blanks)

These are factual fixes; **no results numbers changed**.

1. **"tuned GBDTs" → "untuned / fixed-default GBDTs"** throughout (abstract, §1 RQ1, §1
   contributions, §1 setup, §7). The code does **no** hyperparameter tuning (`src/models.py:15-40`),
   so the original "tuned" wording was inaccurate. (Lines describing the *external literature's*
   tuned-GBDT benchmarks — §1 ¶1 and §2 — were left unchanged, as they are accurate there.)
   *Impact:* slightly softens the RQ1 framing — TabPFN beats **out-of-the-box** GBDTs, not tuned ones.
2. **TabPFN v2 → v3.** The audited artifact is the v3 default checkpoint (package 8.0.7), not the
   v2 model of the Nature 2025 paper. §4.3 now states the exact checkpoint. **See open question for
   author below — there is no dedicated v3 paper to cite.**
3. **Protected attributes are model features** — now stated explicitly in §4.2 (was an open
   question in the draft). Affects fairness interpretation (disparities arise even with the
   attribute visible to the model).
4. **Holm-correction claim removed** from §4.6. The draft asserted all significant effects survive
   Holm correction; this was never computed. Removed rather than left as an unverified claim.
   (Can be added back if Phase 2 computes it / a reviewer requests it.)
5. **Reliability-diagram caption** claims "equal-mass bins" but `reliability_curve`
   (`src/calibration.py:52-55`) uses **equal-width** `np.linspace` bins. **To be fixed in Phase 3**
   alongside figure regeneration (caption → "15 equal-width confidence bins").

---

## [STILL MISSING] — author must supply

- Author names, affiliations, country (`frontmatter`).
- Declaration of competing interest; CRediT authorship statement.
- Repository URL and license (Reproducibility statement).
- Total wall-clock compute time for the full matrix (not logged during the run; hardware is known).

## Open question for the author (framing)

- **TabPFN v3 vs v2 citation.** The model under audit is TabPFN **v3** (checkpoint
  `tabpfn-v3-classifier-v3_default.ckpt`, package 8.0.7). The Nature 2025 / ICLR 2023 papers
  describe v1–v2. There is no separate peer-reviewed v3 paper. Current draft cites v2 Nature as
  lineage and states the exact checkpoint in §4.3. Confirm this is how you want it framed, or
  provide a v3 reference if one exists.

## [CIT: ...] reference stubs — still to resolve (Phase 4)

All 23 `\bibitem` entries remain `[CIT: ...]` stubs (none invented). Phase 4 will fill verified
BibTeX into `paper/refs.bib` **only if** records can be web-verified; otherwise stubs remain and
are listed here.

---

## Author / repo / licensing (resolved from user instruction)

- Author: **Nadim Mahmud Dipu**, Brac University, Dhaka, Bangladesh. Single-author CRediT
  (all roles); competing interest: none declared. Data-availability statement added (ACS PUMS
  2018 1-Year via folktables).
- Repository: **https://github.com/Nadim-Mahmud-23/tabpfn-uncertainty-audit** (public),
  **MIT license** (`LICENSE`, © 2026 Nadim Mahmud Dipu).
- Compute time: ~1 hour wall-clock (≈55 min for the 75-cell matrix, TabPFN CPU inference
  dominant) on an Apple Silicon MacBook (observed during the run).

## PHASE 2 — number verification — DONE

- `scripts/verify_paper_numbers.py` recomputes Tables 1–6 + all Results prose from the raw
  CSVs (LAC-only for conformal/fairness). **180/180 checks PASS, 0 FAIL.** No tex number
  needed correction — the draft's values were already exact. Script is committed (reviewer
  artifact). Group-size Table 3 also verified against `results/group_sizes.csv`.

## PHASE 3 — figures (LAC-only, dpi=300 + vector PDF) — DONE

- `scripts/make_figures.py` regenerates `coverage_gap_{age,sex,race}` and
  `reliability_example` as **PNG (dpi=300) + PDF**, with the `score=='lac'` filter (the old
  PNGs averaged in degenerate APS rows and were half-height). Verified age bars now read
  ≈0.08 marginal / ≈0.05 Mondrian, matching Table 5/6.
- Removed all three `[NOTE TO AUTHORS]` caption reminders. Fixed the reliability caption
  ("equal-mass" → "15 equal-width confidence bins", per `src/calibration.py:52-55`). Added
  "(LAC score)" to the sex/race captions. Figures now referenced via `\graphicspath{{../figures/}}`
  with vector-PDF includes (paper.tex lives in `paper/`, figures in repo-root `figures/`).

## PHASE 4 — IJAR reframe — DONE (references pending verification)

- `\journal{}` → International Journal of Approximate Reasoning; header comment updated.
- New title: *"Subgroup coverage in split conformal classification: marginal validity,
  Mondrian repair, and small-group failure, with an audit of TabPFN."*
  (variants considered, see below).
- Abstract and Introduction rewritten to **lead with the conformal/subgroup-coverage
  contribution**; TabPFN demoted to "model under audit" (RQ1/RQ2 retained as secondary).
  Contribution bullets reordered (coverage-fairness first). Related Work reordered
  (conformal + group-conditional first). Conclusion first paragraph reframed. Keywords
  reordered (conformal first).
- Finite-sample variance argument made **precise**: new Eq. (sd ≈ sqrt(alpha(1-alpha)/n_g))
  in §3.5, with the gap scaling as 1/sqrt(min_g n_g); used consistently in abstract/intro/§5.3.
- `paper/highlights.tex` created — 5 bullets, each ≤85 chars (verified 71–79).
- elsarticle (review,12pt) + line numbers + elsarticle-num retained.

### Title variants considered
1. (chosen) "Subgroup coverage in split conformal classification: marginal validity,
   Mondrian repair, and small-group failure, with an audit of TabPFN."
2. "When does group-conditional conformal prediction help? Marginal validity, Mondrian
   repair, and small-stratum failure, audited on a tabular foundation model."
3. "Equitable coverage is procedural, not model-borne: split conformal validity and the
   small-group limits of Mondrian calibration, with TabPFN under audit."

## PHASE 5 — security / build / word count — PARTIAL

- **Security (DONE):** removed hardcoded HF_TOKEN/TABPFN_TOKEN from all three notebooks →
  dependency-free `.env` loader; added `.env.example`, `.env` in `.gitignore`. Repo-wide grep
  confirms no token strings outside the gitignored `.env`; the pushed remote tree (45 files)
  contains no `.env` and no `data/` cache. (User to revoke the leaked tokens.)
- **Word count:** `wc -w paper.tex` = **7,566** (the loose measure the cover note's "7,400"
  used — counts LaTeX markup, table cells, bibliography). True **main-text prose ≈ 2,600–2,800
  words** excluding references/tables/equations; abstract 300. **This is under the 7,000–9,000
  target by the honest measure** — flagged for the author; not padded. Expanding would mean
  real new content (e.g., the Tier-B suite, a longer Related Work/Discussion).
- **Build (DONE):** installed **Tectonic** (via Homebrew, no sudo; auto-fetches `elsarticle.cls`
  and CTAN deps). `paper.pdf` compiles to **35 pages** (review/double-spaced 12pt). Log is clean:
  **no undefined citations or references**, no overfull hbox > 15 pt. Remaining 14 small overfull /
  11 underfull boxes are cosmetic URL line-breaks in the bibliography. Rebuild with
  `tectonic -X compile paper.tex` (or `pdflatex; bibtex; pdflatex x2`).

## Post-submission expansion (a) + reference finalization (b)

**(b) All 5 `UNVERIFIED` reference sub-fields resolved** (web-verified, flags removed):
nixon2019 (5-author proceedings list, pp. 38-41, DBLP); papadopoulos2002 (Harris Papadopoulos,
Kostas Proedrou — DBLP/Springer); jung2023 (OpenReview); and **gibbs2023 is now formally
published** — JRSS-B 2025, vol 87(4), pp. 1100-1126, DOI 10.1093/jrsssb/qkaf008 (Oxford Academic),
entry upgraded from arXiv. No `UNVERIFIED` flags remain in refs.bib.

**(a) Paper expanded with genuine, verified content** (no padding; every new number checked):
- New Background §3.6 **"When does Mondrian help? A finite-sample prediction"** — derives the
  noise floor `gap_Mond >~ sqrt(alpha(1-alpha)/min n_g) * sqrt(2 ln|G|)` and shows it predicts the
  sign of the marginal->Mondrian change on all three axes (age floor 0.046 < bias 0.081 -> helps;
  race floor 0.054 > bias 0.046 -> hurts; sex ~0.012 ~ bias 0.012 -> neutral).
- New Results §5.4 **"set-size tax"** (Table 7): Mondrian lowers the coverage gap but *raises*
  set-size disparity on every axis (age 0.236->0.369) — coverage equity bought with efficiency
  inequity. + an explicit validation-of-theory paragraph.
- Added: per-dataset RQ1 consistency; RQ2 efficiency-vs-calibration link + per-target widening;
  RQ3 per-target/WGC walk-through; Discussion paragraphs (theory as an a priori diagnostic; the
  coverage-vs-efficiency fairness conflict); deeper Related Work (impossibility of exact conditional
  coverage); Methodology split rationale; Background exchangeability note; expanded
  Limitations (theoretical-scope + model-specificity caveats) and Conclusion (4 future directions
  + a methodological-lesson paragraph).
- `verify_paper_numbers.py` extended to **197/197 checks** (added Table 7, theory floors, per-dataset
  accuracy, age@95 WGC). Fixed a citation-spacing bug in `apply_citations.py` (7 glued `\cite`).
- Compiles to **43 pp**; `wc -w` = **9,075** (within the 7,000-9,000 target by that measure; true
  narrative prose is ~4,000-4,500 — for substantially more *real* depth, run the Tier-B suite).

## References — DONE (verified BibTeX)

- All 23 references web-verified (DBLP / official proceedings / arXiv / Nature / JASA / Royal
  Holloway) into `paper/refs.bib`; `scripts/apply_citations.py` converted the 72 inline `[CIT:]`
  stubs → 32 `\cite{}` commands and switched the manual bibliography to `\bibliography{refs}`
  (`\bibliographystyle{elsarticle-num}`). 0 stubs remain; all 23 cited keys resolve.
- **Five entries carry `% UNVERIFIED` sub-field flags** (author first names only, except one):
  `nixon2019` (first names of Dusenberry/Zhang/Jerfel), `papadopoulos2002` (first names
  Harris/Kostas), `jung2023` (first names Christopher/Georgy/Ramya), and `gibbs2023`
  (cited as arXiv:2305.12616; reportedly accepted to JRSS-B 2025 but vol/pages/DOI unconfirmed).
  Surnames, titles, venues, and years for all five are verified. Please glance at these before
  submission.
- IJAR guide (sub-agent, partly from Elsevier house style as the live page blocks automated
  fetch): Highlights 3-5 bullets ≤85 chars (✓), elsarticle-num correct (✓), abstract ~250 words
  (trimmed to **246**), single-column elsarticle for submission. Double-check the exact IJAR
  abstract cap on the live Guide-for-Authors page.
