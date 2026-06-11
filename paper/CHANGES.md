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

## PHASE 2 — number verification — *pending*
## PHASE 3 — figure regeneration (LAC-only, dpi=300) — *pending*
## PHASE 4 — IJAR reframe — *pending*
## PHASE 5 — security, build, word count — *pending*
