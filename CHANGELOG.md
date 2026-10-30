# Changelog

## 0.5.0 — big data, real approvals, one step at a time, rigorous data work

This release combines two independently built 0.5 lines: big data and hardening (user-only approvals,
step order, tamper protection) and rigorous data work before training (per-column records, full-data
proof, the pilot rule). Where both solved the same problem the stricter behaviour was kept.

### Added — rigorous data work
- **Full-data proof.** `guard.load()` refuses `nrows`, `usecols`, `skiprows`, chunks, `columns=` and
  `filters=`, and records each file's SHA-256, rows and columns in `source.json`; a file profiled by path
  is recorded the same way; `guard.derive()` registers joined frames. Gate A needs a profile that covers
  the full data, unless the user said "override partial data".
- **Profile schema 2.** Every column gets the describe/info statistics (dtype, non-null, missing,
  memory, percentiles, mean/std/skew/kurtosis, zeros, negatives, outliers, top levels with counts, rare
  levels, date span, text lengths) and the target its exact class counts. On big data every count and
  moment is exact over every row; only percentiles come from the uniform sample.
- **Per-column records the gates check** (new stdlib module `hooks/gate_records.py`):
  `data_dictionary.csv` (meaning, unit, allowed, source, available at prediction, role, and a figure note
  per column), `missingness.csv` (cause and evidence at Gate A, decision and reason at Gate B) with a
  missing-pattern figure, `label_audit.md` (computed class counts and label conflicts plus five written
  sections), every column on `02_variables_pNN.png` pages with `pages.json`, `cleaning_log.csv`,
  `categorical_plan.csv`, `associations.json` and two `07_*_train.png` figures from the training split,
  `training_runs.jsonl` receipts (`guard.fit`, `guard.record_training`), and `importance.json` with
  `12_permutation_importance.png` from the validation split plus one `12_attribution_*.png` at Gate C.
- **The real-explanation rule.** Every figure explanation has at least 40 characters and a number;
  every column note cites one of that column's own figures; no two columns share a note. `guard.fig()`
  and the hook apply the same rule (shared test vectors).
- **Pilot rule.** After Gate B, a slice of the rows inside a training call (`X_train[:5000]`, `.iloc`,
  `.head()`, `.sample()`, `nrows=`) is denied unless the user said "override pilot"; Gate C still needs a
  full-data training run.
- **Red flag `target-stats-outside-train`.** Target-based feature selection must use the training split.
- **Runs.** A `## Run: <name> (date)` heading starts a new experiment: only its lines count, and the
  user's chat approvals are tagged with the run they were given in, so nothing carries over.
- Records saved by Excel or Numbers (byte-order mark, cp1252, `;` or tab) parse; a record the guard can't
  read is never overwritten.
- `tests/probe_hook_audit.py` feeds 23 crafted tool calls to the real hook (0 of 23 wrong); an end-to-end
  test walks one project through big data and every record via the real hooks; the canary's honest final
  test score is checked against its ceiling.

### Changed — where the two lines met
- **Ledger lines are strict and run-scoped.** Only `- Gate X: approved ...` and `- Override: <what> - ...`
  count, the last line per gate decides, and only in the newest run; "to be approved", "needs an approved
  rationale" and "never approved" no longer count even when the user approved in chat.
- **Overrides need the same records as approvals**, and the user's word in chat.
- **No target statistics before the split.** The profile no longer publishes a ranked `feature_signal`
  (the target is still used to look for leaks); step-2 figures leave the target out (no
  feature-vs-target pages, no target drift; correlations between features only; the overview plots
  missingness against distinctness); feature-target strength comes from `guard.train_associations` on
  the training split.
- **Every column is drawn.** The paged numeric distributions with a page cap are replaced by variable
  pages for every column of every kind.
- **The test set stays used.** A used test set can't be re-split; every further touch needs one more
  user-approved `final test` override, and one in the newest run.
- **Wider detection:** any `*Trainer(...)`, a backward pass plus an optimizer step under any names,
  `permutation_importance`; `.txt` files are scanned when written as well as when run; `sh -c '...'`,
  `bash -lc "..."` and `.venv/bin/python` are followed; `.ts`/`.js` and other languages are never scanned
  for Python fits; `jupyter nbconvert` without `--execute` is not a run.
- **The agent can't switch enforcement off.** Assigning `ML_PIPELINE_ENFORCE`, `ML_PIPELINE_FILE` or
  `ML_PIPELINE_REQUIRE_USER_APPROVAL` (shell, Python, or a settings file) is denied; messages say "ask
  the user" instead of naming the switch.
- A crash while checking a gate's records denies that ledger write instead of failing open.

### From the big-data and hardening line

#### Fixed
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

#### Added
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
