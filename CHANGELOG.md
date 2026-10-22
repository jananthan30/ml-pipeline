# Changelog

## 0.5.0 — big data, real approvals, one step at a time

### Fixed
- **Overrides unlocked training by accident.** Any `Override:` line mentioning "model" or "training"
  (a red-flag override, a final-test override) unlocked Gate B, and a final-test override unlocked
  Gate C. Overrides now unlock only what they name: `Override: Gate B`, `Override: Gate C`,
  `Override: red flag <name>`, `Override: final test`.
- **Test-set names were easy to miss.** `X_holdout`, `X_te`, `df_hold_out`, `XTest` and more are now
  recognized; `ML_PIPELINE_TEST_NAMES` adds your own. `test_size=` is no longer mistaken for a test set.
- **Code could run without being scanned.** Shell commands are now judged by the local scripts,
  notebooks, shell scripts and imported local modules they run (`python train.txt`,
  `exec(open(...).read())`, `jupyter nbconvert --execute`, `bash run.sh`).
- **Hidden fits** — `getattr(m, "fit")`, `f = m.fit`, `partial_fit` — are detected.
- **State could be tampered with.** `.guard_state.json` and the new `.approvals.json` can't be
  written, moved or deleted by the agent, and recorded PIPELINE.md lines can't be deleted or
  rewritten. `final_test` also counts its own ledger lines, so resetting the state file doesn't
  reset the count.
- **EDA was blocked.** PCA, t-SNE, UMAP, clustering, outlier scores and distribution fits are allowed
  before Gate A.
- **Codex plugin installs had no guard library.** The library moved into `skills/ml-pipeline/lib/`,
  so every install (Claude Code, Codex plugin, `install-other-tools.sh`) ships it.
- Leakage checks no longer flag every category on a highly imbalanced target; unique integer amounts
  are no longer mistaken for ids; stratified splits work on frames with a non-default index.

### Added
- **User-only gate approvals.** A `UserPromptSubmit` hook records approvals and overrides from the
  user's own messages; gate and override lines without that record don't count.
- **Step-by-step enforcement.** `- Step N done: <result>` lines are accepted only in order, after the
  phase gate, and with each step's evidence on disk. Complex models need the baseline (step 9) first;
  hyperparameter search needs the candidates (step 10) first. Gate A needs a written prediction contract.
- **Big data.** `guard.profile`, `guard.eda_figures`, `guard.split_to_files` and `guard.final_test`
  stream Parquet, CSV/TSV, JSON lines, Feather, Arrow and polars data in batches with bounded memory
  (2M rows × 44 columns profiles in ~13 s at ~1 GB peak; doubling the rows adds ~0.1 GB).
  Every-row statistics are exact (distinct counts via a sketch above 4,096 values); distribution
  statistics and figures use a uniform sample. `data_profile_summary.md` gives the agent a short,
  ranked view instead of a huge JSON. The PostToolUse hook warns when an output floods the context.
- **Complete EDA.** Every column is covered: a one-figure overview of all columns, paged
  distributions and feature-vs-target plots ranked by target signal, categorical-vs-target plots,
  an outlier figure, target drift over time, and the full correlation matrix as CSV.
- **Leakage-safe splits.** Dates stored as text or epoch numbers are detected and force a time split;
  repeating entity columns force a group split (each with a user-agreed escape hatch); time splits
  never cut through one timestamp and accept a `gap`; group splits stay near the requested sizes;
  rare classes reach validation and test when they can.
- **Stronger leakage detection:** non-linear (correlation ratio), label conflicts, duplicates that
  differ only by id, ids ordered by the outcome, and columns named after the target.
- `guard.cv_splits` (stratified / group / time-ordered folds with a gap, repeatable),
  `guard.log_run` / `guard.runs` (reproducible run log), `guard.manifest` (image and text data),
  `guard.parse_datetimes`, `guard.status`, `guard.summary`.
- An independent review found 21 issues in the first v0.5 draft; all are fixed with regression tests:
  questions mentioning a gate no longer count as approvals; a bare "yes" counts only after an explicit
  approval question; fits are classified by what the receiver was built from; `cross_val_score` and
  friends count as training; `guard.final_test` needs Gate C (hook and library); `python -m`, `cd dir &&`,
  `from pkg import mod` are followed; indirect state writes and `git checkout -- ml_pipeline` are
  blocked; read-only commands (`cat | jq`, `grep`) pass; `lab_test` / `test_index` / `'test'` strings
  are not the test split; "not" in a reason no longer cancels a line; mixed UTC offsets parse; missing
  targets are refused instead of silently dropped; CSV types come from the whole file; `inf` values and
  late-typed Parquet columns no longer crash.
- CI runs the full test suite (and the hook tests on a bare Python); three new eval cases:
  `self-approval`, `wide-data`, `text-dates`.
