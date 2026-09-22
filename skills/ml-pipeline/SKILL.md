---
name: ml-pipeline
description: MANDATORY whenever a task involves training, fine-tuning, tuning, or evaluating a machine-learning model on data (tabular, time series, text, images — any modality). Enforces a strict 16-step pipeline that starts with inspecting the raw data, gates each phase behind the user's explicit permission, and produces marimo notebooks with matplotlib visuals so the user can see and understand every step. Never jump straight to model training.
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

If the user says "skip ahead" or "just train it": explain in 2–3 sentences which steps are
missing and the concrete risk (usually leakage or garbage-in), then ask once for explicit
override confirmation. If they confirm, proceed and record the override in PIPELINE.md. An override
records their choice; it does not skip the data work — the gate's records are still required.

## Progress tracking (survives across sessions)

On first use in a project, create `ml_pipeline/PIPELINE.md` — a checklist of the 16 steps
with status (`todo / in progress / done / approved-gate / overridden`), one line of results
per finished step, and dated gate approvals. Update it after every step. On any new session,
read it first and resume from the first unfinished step — never restart, never skip ahead
of it.

Record each gate approval on its own line in exactly this shape — the enforcement hook
parses it:

```
- Gate A: approved 2026-09-16
- Gate B: approved 2026-09-17
- Override: Gate B - user approved skipping to training 2026-09-17 - reason: <why>
```

Only these line shapes count: a line starting `- Gate X:` whose first word is `approved`, or a line
starting `- Override: Gate X -`. Headings, table rows, bold prose, and checkboxes never count, and
anything after `approved` is a note. For each gate the **last** such line decides, so
`- Gate B: hold - <why>` locks a gate again.

**Each new experiment is a new run.** Start it with a heading of this shape, and record its gates
below it:

```
## Run: <short name> (2026-09-23)
```

Only lines after the newest `## Run:` heading count — approvals from earlier runs never carry
over, so a new run starts with every gate locked. A ledger with no `## Run:` heading is one run.

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

1. **Data inspection** — copy the plugin's guard library to `ml_pipeline/guard.py` with `cp` (its path is
   given at session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py` next to this
   file). Load every raw file whole with `df = guard.load(path)` — never `nrows`, `usecols`, or a sample —
   which records the file's hash and row count; register a frame joined from several files with
   `guard.derive(df, name, parents=[...], how=...)`. Run `guard.profile(df, target=..., time_col=...,
   group_col=...)` on the whole frame and read `ml_pipeline/data_profile.json` before anything else:
   `source.full_data` must be true, and its `leakage_suspects` and `traits` decide the split, the metric,
   and the model family. Create the per-column records: `guard.data_dictionary(df)`,
   `guard.missingness(df)`, `guard.label_audit(df, target=...)`. For images or text, build a manifest table
   first (path, label, size or length) and profile that. No modification yet.
2. **EDA** — `guard.eda_figures(df, target=..., time_col=...)` writes the required figures
   (`01_missingness`, `02_target_balance`, `02_distributions`, `02_correlations` between features only,
   `02_temporal_coverage` when a date column exists) and `guard.variable_figures(df, target=...)` draws every
   column on grouped pages. Then fill every word field: in `data_dictionary.csv` the meaning, unit, allowed
   values, source, available-at-prediction (yes/no), role (feature/target/id/leak/drop), and a `figure_note`
   for every column; in `missingness.csv` the likely cause and evidence; every section of `label_audit.md`,
   naming each leakage suspect kept as a feature. A note is at least 40 characters and cites a figure of that
   column's own from the profile — a percentile, mean, level count, top count, span, or its missing count
   ("median 54, 2% missing") — not a count every column has; no two columns share a note. Add other figures
   with `guard.fig(step, name, figure, explanation)`; every explanation has a number. If matplotlib is
   missing, install it — text descriptions are not figures. The Gate A report shows the row chain (file rows
   → profiled rows), the dictionary, the missingness table, and the label audit.
3. **Define the prediction problem** — write a short contract: target (exact definition,
   units), prediction unit and population, prediction time/horizon, information actually
   available at prediction time, objective, evaluation metric, constraints. **The user must
   approve this contract at Gate A** — it controls everything after. Consult
   `references/model-selection.md` with the profile's traits when choosing the metric and the
   candidate model families.
4. **Data cleaning** — missing values, duplicates, invalid/impossible values, inconsistent
   categories, unit/format issues. Apply every rule as
   `cleaned = guard.log_cleaning(rule, before, after, reason=...)`, which records before/after counts in
   `cleaning_log.csv`; if nothing needs cleaning, record `- Cleaning: none needed - reason: <why>`. Fill
   `decision` and `reason` for every row of `missingness.csv`. Prefer preserving data over deleting. Never use information from the
   future or from the test rows to decide a cleaning rule.
5. **Data engineering** — joins/integration, aggregation, time alignment to an index date,
   business rules, one-row-per-prediction-unit feature table, data-quality checks
   (row counts, uniqueness, ranges).
6. **Split before any fitting** — copy the plugin's guard library to `ml_pipeline/guard.py`
   (its path is given at session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py`
   next to this file) and split with it:
   `train, val, test = guard.split(df, target=..., time_col=<col> if the data is temporal, group_col=<col> if the same entity appears in multiple rows)`.
   It drops exact duplicates, refuses a random split when a datetime column exists, keeps groups
   together, checks that no row lands in two splits, and freezes a fingerprint of the test set.
   The test set is touched exactly once, at step 14, through `guard.final_test()`.
7. **Feature engineering** — run `guard.categorical_plan(df)` and fill an encoding and a reason for every
   categorical feature. Run `guard.train_associations(train, target=...)` on the training split only for the
   correlation matrix and each feature's strength with the target — never on the full frame. Design features
   on the training set's statistics only, then apply the same transformations to validation/test.
8. **Preprocessing** — scalers, encoders, imputers fit on train only, wrapped in a pipeline
   object so train and inference can never diverge.
9. **Baseline first** — a dummy predictor (majority class / mean) AND one simple model
   (logistic/linear regression or small tree). Record their metrics. Every later model is
   judged against this line; a complex model that can't beat it gets reported as such. Train through
   `guard.fit(model, X_train, y_train)` (custom loops: `guard.record_training(train_data, name=...)`) so each
   run's row count is recorded (after `fit_resample(X_train, y_train)`, pass `rows_from=X_train`); Gate C needs one run on the whole training split, and a pilot on a slice
   needs the user's `- Override: pilot - ...` first.
10. **Model training** — train candidate models on train, compare on validation. Log every
    run's config and score.
11. **Hyperparameter tuning** — on validation/cross-validation only. The test set is never
    part of tuning.
12. **Model evaluation** — the contract's metric plus supporting views: confusion matrix and
    ROC/PR curves for classification, residual plots for regression, always compared to the
    baseline. Figures required. Run `guard.importance(model.predict, val, target=..., metric_fn=...)` for
    permutation importance on the validation split, and save one attribution figure (SHAP, partial
    dependence, or coefficients) with `guard.fig(12, 'attribution_<name>', ...)`.
13. **Error analysis** — worst predictions, performance by slice/subgroup, calibration,
    where the model fails and a hypothesis for why.
14. **Final test** — `score = guard.final_test(model.predict, test, target=..., metric_fn=...)`.
    It verifies the frame is the frozen test set, refuses a second call, and logs the result to
    PIPELINE.md. Report the number honestly, even if it is worse than validation. No going back to
    tune on it — if the result forces changes, agree a new test strategy with the user and record
    `- Override: final test - user approved a second evaluation YYYY-MM-DD - reason: <why>`.
15. **Deployment** — only when the user asks. Save the full pipeline artifact
    (preprocessing + model together), verify a reloaded artifact reproduces predictions,
    document the inference input contract.
16. **Monitoring + retraining** — write down: what drift to watch (input and prediction
    distributions), what metric threshold triggers retraining, and how retraining reuses
    this same pipeline from step 1.

## Enforcement in Claude Code (hook)

Installed as a Claude Code plugin, `hooks/guard_training.py` runs before every Bash, Write,
Edit, and notebook tool call, scans the Python code for fitting and training calls — including the
file, module, or notebook a command runs (`python train.py`, `python -m pkg`, `uv run`, `marimo`,
`jupyter nbconvert --execute`, `papermill`) — and **denies**:

- **model training** — `.fit`, `cross_val_score` and its relatives, a backward pass plus optimizer
  step, any `*Trainer(...)` — until the current run has `- Gate B: approved …` or
  `- Override: Gate B - …`. With no `ml_pipeline/PIPELINE.md` at all, training is denied with a
  pointer to step 1.
- **any fitting** — scalers, encoders, imputers included — until `- Gate A: approved …`.
- **fitting on the test split** (`X_test`, `df_test`, `test_*`) — always, at every gate.
- **evaluating on the test split** — `.predict/.score` or a metric function whose arguments name
  `X_test`, `df_test`, `test_*` — until `- Gate C: approved …` (or `- Override: Gate C - …`).
- **changing the enforcement setting** — only the user switches enforcement off.
- **recording a gate before its records are complete** — the denial lists exactly what is missing:
  - `Gate A`: `source.json` and a full-data profile (or `- Override: partial data - ...`); the schema-2
    profile; `data_dictionary.csv` with every column and every word field; every column on a
    `02_variables_pNN.png` page; `missingness.csv` with a cause for every column with gaps; a complete
    `label_audit.md`; one `01_*.png` and two `02_*.png`.
  - `Gate B`: a cleaning log (or `- Cleaning: none needed - reason: ...`); a decision for every column with
    gaps; `categorical_plan.csv` for every categorical feature; `associations.json` from the training split
    with its two `07_*_train.png` figures; a `04_*.png`; the `Model rationale:` block.
  - `Gate C`: a full-data training run in `training_runs.jsonl`; `importance.json` from the validation
    split; `12_permutation_importance.png` and one `12_attribution_*.png`.
  - `Gate D`: a `13_*.png` and the `Step 14 final test:` line.
  - Every PNG is larger than 1 KB and has a `- Figure <file>: <explanation>` line of at least 40
    characters with a number.
- **training on a slice** after Gate B — `X_train[:5000]`, `.sample(...)`, `.head(...)` in the fit call, or
  `nrows=` — unless the current run has `- Override: pilot - <size> - user approved YYYY-MM-DD - reason: ...`.
- **red flags the profile can prove** — `neural-net-small-data`, `random-split-temporal`,
  `group-split`, `accuracy-imbalanced`, `resample-before-split`, `target-stats-outside-train` (see `references/model-selection.md`).
  Override one only after the user agreed: `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>`.

A PostToolUse hook also warns (never blocks) when a result looks too good to be honest
(accuracy/AUC/F1 ≥ 0.98) or when `train_test_split()` is called without `stratify=` or on data that
mentions dates. To prove a pipeline does not leak, run it on `examples/canary/canary.csv`: honest
test accuracy cannot exceed 0.80, and `mlpipeline_canary.verdict(score, n_test)` says whether a
number is above the ceiling.

A denial is not an obstacle to route around: do the missing steps, get the user's explicit
approval, record the gate line, and retry. The hook fails open on its own errors. If the project
is not an ML pipeline at all, ask the user to switch enforcement off — never change that setting
yourself. Codex and Kimi have no hook support, so there the same rules apply as instructions only.

## Non-negotiables

- Selection leakage is the one that matters most: the test set is never used to pick a model, a
  seed, a feature, or a threshold. Validation only.
- Every gate is recorded with its figures on disk and their explanations in PIPELINE.md. A figure
  that was described but not drawn does not exist.
- Test set is used exactly once. No tuning, no peeking, no "just checking".
- All fitting (cleaning statistics, features, preprocessing, models) uses training data only.
- Temporal data gets temporal splits; repeated entities get group splits.
- Baseline before any complex model; every result is reported relative to it.
- Failures and disappointing numbers are reported plainly — never hidden or reframed.
- Chat explanations stay beginner-friendly; the code stays production-grade.
