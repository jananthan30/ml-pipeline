---
name: ml-pipeline
description: MANDATORY whenever a task involves training, fine-tuning, tuning, or evaluating a machine-learning model on data (tabular, time series, text, images — any modality). Enforces a strict 16-step pipeline that starts with inspecting the raw data, gates each phase behind the user's explicit permission, and produces marimo notebooks with matplotlib visuals so the user can see and understand every step. Never jump straight to model training. Not for auditing an existing project, saved model, or claimed score ("can I trust this model?") - that is the ml-audit skill, and an audit does not start this pipeline.
---

# ML Pipeline Discipline

**Hard rule: no model is trained until every earlier step in the pipeline is done and the
user has explicitly approved the phase gates before it.** "Train a model on this data" is a
request to *start the pipeline at step 1*, not at step 10.

## The pipeline (strict order — never reorder, never skip silently)

```
RAW DATA
  → 1. Data inspection
  → 2. Exploratory data analysis (EDA)
  → 3. Define the prediction problem
  ──────────── GATE A: user approval ────────────
  → 4. Data cleaning
  → 5. Data engineering
  → 6. Train / validation / test split
  → 7. Feature engineering
  → 8. Preprocessing
  ──────────── GATE B: user approval ────────────
  → 9. Baseline model
  → 10. Model training
  → 11. Hyperparameter tuning
  → 12. Model evaluation
  ──────────── GATE C: user approval ────────────
  → 13. Error analysis
  → 14. Final test (test set touched ONCE)
  → 15. Deployment (only if user asks)
  → 16. Monitoring + retraining plan
  ──────────── GATE D: wrap-up report ───────────
```

## Phase gates — explicit permission, every time

At each gate, STOP and give the user, in plain non-jargon language:

1. **What was done** — the steps completed, 1–2 sentences each.
2. **What was found** — key findings, with the visuals that show them.
3. **Decisions made and why** — e.g. "dropped 312 duplicate rows", "chose time-based split
   because the data has dates".
4. **What comes next** — the next phase's steps, in one short list. At Gate B this section
   includes the `Model rationale:` block (format under Progress tracking).
5. **The question** — ask for explicit permission to continue. Wait for a clear yes.
   Silence, ambiguity, or "hmm" is not a yes. If the user redirects, incorporate it.

**Only the user can approve a gate.** In Claude Code a hook records approvals from the user's own
messages ("approve gate B", "gate B approved", or a plain "yes" when your last question was
"Do you approve Gate B?"; questions and passing mentions of a gate never count)
in `ml_pipeline/.approvals.json`. A gate line you write without that record does not count, and the
hook refuses to record it. Never write, edit, move, or delete `.approvals.json` or `.guard_state.json`.

If the user says "skip ahead" or "just train it": explain in 2–3 sentences which steps are
missing and the concrete risk (usually leakage or garbage-in), then ask once for explicit
override confirmation ("Do you want to override Gate B?"). If they confirm, the override is
recorded from their reply; then write the override line in PIPELINE.md. An override records their
choice; it does not skip the data work — the gate's records are still required.

## Progress tracking (survives across sessions)

On first use in a project, create `ml_pipeline/PIPELINE.md`. It is an **append-only ledger**: add
lines, never delete or rewrite recorded ones (the hook refuses). On any new session, read it first
and resume from the first unrecorded step — never restart, never skip ahead of it.

**One step at a time.** When a step is finished, record it on its own line — the hook parses it:

```
- Step 4 done: dropped 312 exact duplicates; capped age at 110; 3 rules, before/after counts above
- Step 5 skipped: single table, nothing to join
```

A step line is accepted only when every earlier step is recorded, the gate before its phase is
approved by the user, and its own artifacts exist: step 1 needs `data_profile.json`; step 2 the
`01_*`/`02_*` figures; step 3 the prediction contract; step 6 a frozen split; step 9 a `baseline`
run in `runs.jsonl`; step 10 a `candidate` run; step 12 a `12_*` figure; step 13 a `13_*` figure;
step 14 the line `guard.final_test()` writes. Steps 1, 2, 3, 6, 9, 12 and 14 can't be skipped; any
other step may be `skipped` with a reason.

Record each gate approval, after the user approved it in chat, on its own line in exactly this shape:

```
- Gate A: approved 2026-09-16
- Gate B: approved 2026-09-17
- Override: Gate B - user approved skipping to training 2026-09-17 - reason: <why>
```

A gate line is accepted only when its steps are recorded, its records are complete (listed under
Enforcement below), and the user approved it in chat. Only these line shapes count: a line starting
`- Gate X:` whose first word is `approved`, or a line starting `- Override: Gate X -`. Headings, table
rows, bold prose, and checkboxes never count, and anything after `approved` is a note. For each gate
the **last** such line decides, so `- Gate B: hold - <why>` locks a gate again.

**Each new experiment is a new run.** Start it with a heading of this shape, and record its steps and
gates below it:

```
## Run: <short name> (2026-09-23)
```

Only lines after the newest `## Run:` heading count, and the user's chat approvals belong to the run
they were given in — approvals from earlier runs never carry over, so a new run starts with every gate
locked. A ledger with no `## Run:` heading is one run. A new run never resets the test set: it stays
used until the user agrees a new test strategy.

At step 3 write the contract in exactly this shape (the hook checks the five bullets exist):

```
Prediction contract:
- target: churned = 1 if no purchase within 30 days after the snapshot date
- unit: one row per customer per monthly snapshot; active customers only
- horizon: predict on the snapshot date for the next 30 days
- available: only data timestamped on or before the snapshot date
- metric: PR-AUC primary; recall at 80% precision secondary
```

After recording a gate approval in a git repository, checkpoint it:
`git add -A && git commit -m "ml-pipeline: Gate X approved" && git tag -f gate-X`. Every approved
phase is then a reproducible point to return to.

Before recording `Gate B: approved`, PIPELINE.md must contain a model rationale in exactly this shape
(the hook checks the five bullets exist; the user judges their content):

```
Model rationale:
- traits: binary, 1,428 rows (small), minority 11.4%, has_datetime, no groups
- baseline: majority-class dummy + logistic regression (class_weight=balanced)
- candidates: gradient boosting with small trees; regularized logistic
- ruled out: neural nets (small data); k-NN (mixed scales, weak on tabular); random split (temporal)
- metric: PR-AUC primary, recall at fixed precision secondary
```

## Tools: marimo notebooks + matplotlib visuals

- **The workbench is a marimo notebook**, not loose scripts. Keep notebooks in
  `ml_pipeline/`: `01_eda.py` (steps 1–3), `02_prep.py` (steps 4–8), `03_model.py`
  (steps 9–12), `04_eval.py` (steps 13–16). In Claude Code, drive them live with the
  `marimo-pair` skill so the user watches the work happen. In other harnesses, write the
  notebook files and tell the user to open them with `marimo edit <file>`.
- **Every step that looks at data produces matplotlib figures** (seaborn on top is fine).
  Also save each figure to `ml_pipeline/figures/<step>_<name>.png` so gates can reference
  them even without a live notebook.
- **Explain every figure in 1–2 plain sentences**: what it shows and why it matters for
  the next decision. A figure without an explanation is not done.

## What each step must produce

1. **Data inspection** — copy the plugin's guard library with `cp` to `ml_pipeline/guard.py` (its path
   is given at session start; in Codex/Kimi it is `lib/mlpipeline_guard.py` next to this file) and run
   `guard.profile(source, target=..., time_col=..., group_col=...)` on **all** the data. `source` may be
   a **file path** (Parquet, CSV/TSV, JSON lines, Feather) — read in batches, so data larger than memory
   is fine, and the file's hash and row count go into `source.json` — or a DataFrame loaded whole with
   `df = guard.load(path)` (never `nrows`, `usecols`, or a sample; a frame joined from several files is
   registered with `guard.derive(df, name, parents=[...], how=...)`). Do not `pd.read_csv` a large file
   yourself. Counts and moments (missingness, distinct counts, duplicates, label conflicts, min/max,
   mean/std/skew, zeros, outliers, level and class counts) cover every row; percentiles and figures use
   a uniform 200k-row sample on larger data. The profile's `source.full_data` must be true. Then **read
   `ml_pipeline/data_profile_summary.md`** (short) — not the full JSON, which can be huge on wide data;
   look up single columns in the JSON when needed. Its leakage suspects, date columns (including dates
   stored as text or epoch numbers), repeated entity columns, and traits decide the split, the metric,
   and the model family. Create the per-column records: `guard.data_dictionary()`,
   `guard.missingness(source)`, `guard.label_audit(source, target=...)`. Report rows × columns, column
   kinds, missing values, duplicates, label conflicts, and every leakage suspect. For images or text,
   run `guard.manifest(folder, kind="image"|"text", group_pattern=...)` first and profile the manifest
   (`group_col="group"` keeps a source and exact duplicate files together). No modification yet.
2. **EDA** — `guard.eda_figures(source, target=..., time_col=...)` writes the figures with
   explanations computed from the data: `01_missingness`, `02_target_balance`, `02_feature_overview`
   (every column as one dot: missingness vs. distinctness), `02_outliers`, `02_correlations` between
   features only (full matrix in `correlations.csv`), `02_temporal_coverage` when a date column exists,
   and **every column on grouped pages** `02_variables_pNN` (`figures/pages.json` lists which; call
   `guard.variable_figures(source, target=...)` again after adding columns). Nothing here relates a
   feature to the target: that is a training-split statistic (step 7). Then fill every word field: in
   `data_dictionary.csv` the meaning, unit, allowed values, source, available-at-prediction (yes/no), role
   (feature/target/id/leak/drop), and a `figure_note` for every column; in `missingness.csv` the likely
   cause (random, depends-on-other-data, depends-on-itself, structural, unknown) and evidence; every
   section of `label_audit.md`, naming each leakage suspect kept as a feature. A note is at least 40
   characters and cites a figure of that column's own from the profile — a percentile, mean, level
   count, top count, span, or its missing count ("median 54, 2% missing") — not a count every column
   has; no two columns share a note. Add other figures with `guard.fig(step, name, figure,
   explanation)`; every explanation is at least 40 characters with a number from the data. Exploratory
   fits (PCA, t-SNE, UMAP, clustering, distribution fits) are allowed here. If matplotlib is missing,
   install it — text descriptions are not figures and the gate will not accept them. The Gate A report
   shows the row chain (file rows → profiled rows), the dictionary, the missingness table, and the
   label audit.
3. **Define the prediction problem** — write a short contract: target (exact definition,
   units), prediction unit and population, prediction time/horizon, information actually
   available at prediction time, objective, evaluation metric, constraints. **The user must
   approve this contract at Gate A** — it controls everything after. Consult
   `references/model-selection.md` with the profile's traits when choosing the metric and the
   candidate model families.
4. **Data cleaning** — rows with no target value (the split refuses them — drop or label them here and
   record the rule), missing values, duplicates, invalid/impossible values, inconsistent
   categories, unit/format issues. Apply every rule as
   `cleaned = guard.log_cleaning(rule, before, after, reason=...)`, which records before/after counts in
   `cleaning_log.csv`; if nothing needs cleaning, record `- Cleaning: none needed - reason: <why>`. Fill
   `decision` and `reason` for every row of `missingness.csv`. Prefer preserving data over deleting.
   Never use information from the future or from the test rows to decide a cleaning rule.
5. **Data engineering** — joins/integration, aggregation, time alignment to an index date,
   business rules, one-row-per-prediction-unit feature table, data-quality checks
   (row counts, uniqueness, ranges).
6. **Split before any fitting** — with the guard library:
   `train, val, test = guard.split(df, target=..., time_col=<col> if the data is temporal, group_col=<col> if the same entity appears in multiple rows, gap=<embargo>)`.
   It drops exact duplicates, refuses a random split when any date column exists (native, text, or
   epoch numbers) and a row-level split when an entity column repeats, never lets one timestamp
   straddle a boundary, keeps groups together at sensible sizes, checks that no row lands in two
   splits, and freezes a fingerprint of the test set. If the user agrees the dates or entities don't
   matter, pass `not_temporal_reason=` / `not_grouped_reason=` with their reason. For data that does
   not fit in memory use `guard.split_to_files(path, target=..., ...)`, which streams train/val/test
   Parquet files (identical rows always share a split). The test set is touched exactly once, at
   step 14, through `guard.final_test()`.
7. **Feature engineering** — run `guard.categorical_plan()` and fill an encoding and a reason for every
   categorical feature. Run `guard.train_associations(train, target=...)` on the training split only (the
   frame `split` returned or the train file `split_to_files` wrote) for the correlation matrix and each
   feature's strength with the target — never on the full data. Design features on the training set's
   statistics only, then apply the same transformations to validation/test.
8. **Preprocessing** — scalers, encoders, imputers fit on train only, wrapped in a pipeline
   object so train and inference can never diverge.
9. **Baseline first** — a dummy predictor (majority class / mean) AND one simple model
   (logistic/linear regression or small tree). Train through `guard.fit(model, X_train, y_train)` (custom
   loops: `guard.record_training(train_data, name=...)`; after `fit_resample(X_train, y_train)` pass
   `rows_from=X_train`) so each run's row count is recorded: Gate C needs one run on the whole training
   split, and a pilot on a slice needs the user's "override pilot" first. Log both with
   `guard.log_run(name, kind="baseline", metrics={...}, model=...)` — it records the split
   fingerprints, seed, package versions and git commit so the run is reproducible. Every later model
   is judged against this line; a complex model that can't beat it gets reported as such. The hook
   denies other model families until `- Step 9 done` is recorded.
10. **Model training** — train candidate models on train, compare on validation, and log every run
    with `guard.log_run(kind="candidate")`. On small data use `guard.cv_splits(train, ...)` (repeated
    stratified, group, or time-ordered folds with a gap) and report the spread, not one number.
11. **Hyperparameter tuning** — on validation or `guard.cv_splits` folds only (`cv=folds`), logged
    with `kind="tuned"`. The test set is never part of tuning. The hook denies a search until
    `- Step 10 done` is recorded.
12. **Model evaluation** — the contract's metric plus supporting views: confusion matrix and
    ROC/PR curves for classification, residual plots for regression, always compared to the
    baseline. Figures required. Run `guard.importance(model.predict, val, target=..., metric_fn=...)` for
    permutation importance on the validation split, and save one attribution figure (SHAP, partial
    dependence, or coefficients) with `guard.fig(12, 'attribution_<name>', ...)`.
13. **Error analysis** — worst predictions, performance by slice/subgroup, calibration,
    where the model fails and a hypothesis for why.
14. **Final test** — `score = guard.final_test(model.predict, test, target=..., metric_fn=...)`, where
    `test` is the frame `split` returned or the test file `split_to_files` wrote; it predicts in
    batches (`batch_rows=`). It verifies the data is the frozen test set, refuses a second call, and
    logs the result to PIPELINE.md. Report the number honestly, even if it is worse than validation.
    No going back to tune on it — if the result forces changes, agree a new test strategy with the
    user; their "override final test" allows exactly one more evaluation, and you record
    `- Override: final test - user approved a second evaluation YYYY-MM-DD - reason: <why>`.
15. **Deployment** — only when the user asks. Save the full pipeline artifact
    (preprocessing + model together), verify a reloaded artifact reproduces predictions,
    document the inference input contract.
16. **Monitoring + retraining** — write down: what drift to watch (input and prediction
    distributions), what metric threshold triggers retraining, and how retraining reuses
    this same pipeline from step 1.

## Big data, wide data, and long runs

- **Keep the data on disk.** Pass file paths to `guard.profile`, `guard.eda_figures`,
  `guard.split_to_files` and `guard.final_test`; they stream in batches. Convert CSV to Parquet once
  if you will read it more than twice. Never print whole frames: print shapes, `.head(10)`, and
  summaries, and write big tables to files under `ml_pipeline/`. The PostToolUse hook warns when an
  output floods the context.
- **Wide data.** `02_feature_overview` shows every column at once and the variable pages draw every
  column, about 12 per page; every column still gets its own dictionary note. Rank features by target
  strength only after the split (`guard.train_associations` on train). Report coverage honestly.
- **Training on large data.** Tune on a sample of train (state its size), then refit the chosen
  configuration on all of train. Prefer models that stream or subsample (histogram gradient boosting,
  SGD, `partial_fit`). Use early stopping on validation.
- **Long runs.** A run over ~5 minutes goes in the background (the Bash tool's `run_in_background`),
  writes progress to a log file under `ml_pipeline/logs/`, checkpoints so it can resume, and logs its
  result with `guard.log_run`. Check on it instead of waiting in the foreground, and tell the user what
  is running and roughly how long it will take.

## Enforcement in Claude Code (hook)

Installed as a Claude Code plugin, `hooks/guard_training.py` runs before every Bash, Write,
Edit, and notebook tool call. It scans the code — and, for shell commands, every local script,
notebook, or shell script the command runs plus the local modules they import — and **denies**:

- **model training** — `.fit`, `cross_val_score` and its relatives, a backward pass plus optimizer step,
  any `*Trainer(...)` — until the user approved Gate B (or asked to override it) and the current run says
  so. With no `ml_pipeline/PIPELINE.md` at all, training is denied with a pointer to step 1.
- **any fitting** — scalers, encoders, imputers included — until Gate A. Exploratory fits for EDA
  (PCA, t-SNE, UMAP, clustering, outlier scores, distribution fits) are allowed before it.
- **complex models before the baseline** (`- Step 9 done`) and **hyperparameter search before the
  candidates** (`- Step 10 done`).
- **fitting on the test split** — always, at every gate — and **evaluating on it** before Gate C.
  Test-set names it knows: `X_test`, `df_test`, `test_*`, `*_holdout`, `X_te`, `XTest`, … plus any in
  `ML_PIPELINE_TEST_NAMES` (comma-separated).
- **hidden fits** — `getattr(model, "fit")`, `f = model.fit`, `partial_fit`.
- **recording a step or gate before its prerequisites** — the denial lists exactly what is missing:
  - `Gate A`: steps 1–3; the `Prediction contract:` block; `source.json` and a full-data profile (or the
    user's `- Override: partial data - ...`); the schema-2 profile; `data_dictionary.csv` with every column
    and every word field; every column on a `02_variables_pNN.png` page; `missingness.csv` with a cause
    for every column with gaps; a complete `label_audit.md`; one `01_*.png` and two `02_*.png`.
  - `Gate B`: steps 4–8; the frozen split; a cleaning log (or `- Cleaning: none needed - reason: ...`); a
    decision for every column with gaps; `categorical_plan.csv` for every categorical feature;
    `associations.json` from the training split with its two `07_*_train.png` figures; a `04_*.png`; the
    `Model rationale:` block.
  - `Gate C`: steps 9–12; a baseline run in `runs.jsonl`; a full-data training run in
    `training_runs.jsonl`; `importance.json` from the validation split; `12_permutation_importance.png`
    and one `12_attribution_*.png`.
  - `Gate D`: steps 13–14, a `13_*.png` and the `Step 14 final test:` line.
  - Every PNG is larger than 1 KB and has a `- Figure <file>: <explanation>` line of at least 40
    characters with a number. An override line (`- Override: Gate X - ...`) needs the same records.
- **training on a slice** after Gate B — `X_train[:5000]`, `.sample(...)`, `.head(...)` in the fit call,
  or `nrows=` in a reader — unless the user said "override pilot" and the current run has
  `- Override: pilot - <size> - user approved YYYY-MM-DD - reason: ...`.
- **changing the enforcement settings** — only the user switches enforcement or the approval rule off,
  or points the hook at another ledger.
- **tampering** — writing, moving, or deleting `.approvals.json` / `.guard_state.json`, and deleting
  or rewriting recorded PIPELINE.md lines (gates, overrides, steps, final tests).
- **red flags the profile can prove** — `neural-net-small-data`, `random-split-temporal`,
  `group-split`, `accuracy-imbalanced`, `resample-before-split`, `target-stats-outside-train`, plus `baseline-first` and
  `tune-before-train` (see `references/model-selection.md`). Override one only after the user said
  "override red flag <name>": `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>`.

A PostToolUse hook also warns (never blocks) when a result looks too good to be honest
(accuracy/AUC/F1 ≥ 0.98) or when `train_test_split()` is called without `stratify=` or on data that
mentions dates. To prove a pipeline does not leak, run it on `examples/canary/canary.csv`: honest
test accuracy cannot exceed 0.80, and `mlpipeline_canary.verdict(score, n_test)` says whether a
number is above the ceiling.

A denial is not an obstacle to route around: do the missing steps, get the user's explicit
approval, record the gate line, and retry. The hook fails open on its own errors (except while
checking a gate's records: then that ledger write is denied). If the project is not an ML pipeline at
all, ask the user to switch enforcement off — never change that setting yourself. Codex and Kimi
have no hook support, so there the same rules apply as instructions only — and there, record a gate
only after the user's explicit yes in the conversation.

## Non-negotiables

- Only the user approves a gate or an override, in their own words. Never approve one yourself.
- Selection leakage is the one that matters most: the test set is never used to pick a model, a
  seed, a feature, or a threshold. Validation only.
- Every gate is recorded with its figures on disk and their explanations in PIPELINE.md. A figure
  that was described but not drawn does not exist.
- Test set is used exactly once. No tuning, no peeking, no "just checking".
- All fitting (cleaning statistics, features, preprocessing, models) uses training data only, and so
  does every statistic that relates a feature to the target.
- Every column is profiled, drawn, and explained from the full data; a sample or a slice is a pilot.
- Temporal data gets temporal splits; repeated entities get group splits.
- Baseline before any complex model; every result is reported relative to it.
- Failures and disappointing numbers are reported plainly — never hidden or reframed.
- Chat explanations stay beginner-friendly; the code stays production-grade.
