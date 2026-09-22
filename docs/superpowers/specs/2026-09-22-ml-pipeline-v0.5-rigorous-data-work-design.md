# ml-pipeline v0.5.0 — Rigorous data work before training (design)

**Status:** design approved in conversation 2026-09-22 (five sections, approach A). Awaiting the
owner's review of this written spec before an implementation plan is written. Supersedes the
requirements draft `2026-09-22-ml-pipeline-v0.5-rigorous-data-work-DRAFT.md`. Builds on the hook
fixes in commit `246f1a7` (strict, run-scoped ledger; probe 0 of 23).

## Goal

Make the data work before training complete and provable. In v0.4.0, steps 4–8 are instructions
only, the Phase 2 checkpoint needs one `04_*.png` and the model rationale, and nothing proves that
every variable was inspected or that the full dataset was used. Agents profile a sample, run a
small pilot, and move on.

**Principle.** The harness guarantees **completeness**: every column, every missing value, every
label, and every figure has a recorded, explained decision, machine-checked against the data's
real column list and row count. The owner judges **quality** at each gate; the harness makes sure
there is something complete to judge. Guard-library helpers compute every number; agents write
only words (meanings, reasons, decisions, notes).

## Decisions (owner, 2026-09-22)

1. **Approach A:** the guard library pre-fills one small file per artifact with computed numbers;
   the agent fills the word fields; the hook checks every artifact against the profile's column
   list when a gate is recorded. (Rejected: all records as PIPELINE.md lines — unreadable on wide
   data, fragile prose parsing; a checker command writing a receipt — two moving parts, and the
   receipt can be hand-written.)
2. **Data dictionary: completeness only.** No separate sign-off line; the dictionary is shown in
   the Gate A report and the owner's Gate A approval is the sign-off.
3. **Pilot runs: only with an override.** Training on a slice is denied unless the current run
   records `- Override: pilot - ...`; Gate C needs proof of at least one full-data training run.
4. **Wide data: grouped figure pages** (about 12 columns per page), with a written note for every
   column. The per-column note lives in the data dictionary's `figure_note` column, not in
   PIPELINE.md; each page PNG keeps its normal `- Figure` ledger line.
5. **Feature importance and attribution belong to Phase 3** (after the baseline, checked at
   Gate C), never before training.
6. **Target statistics are computed on the training split only.** The step-2 correlation heatmap
   drops the target; feature–target association moves to Phase 2 on the training split. The
   pre-split leakage-suspect scan stays: its job is to catch leaks, not to select features.

## Scope

**In:** §1 full-data proof and the pilot rule; §2 Phase 1 records (Gate A); §3 Phase 2 records
(Gate B); §4 Phase 3 records (Gate C) and the real-explanation rule; §5 hook integration; §6 fixes
riding along; §7 skill, docs, eval; §8 tests.

**Out (later):** approvals written only from the owner's own message (UserPromptSubmit ledger);
profiling non-tabular data beyond a manifest table; configurable thresholds (constants for now);
any content grading beyond the machine checks below.

## File layout

All records live in `ml_pipeline/` next to `PIPELINE.md`:

| File | Written by | Checked at |
|---|---|---|
| `source.json` | `guard.load()`, `guard.derive()` | Gate A |
| `data_profile.json` (schema 2) | `guard.profile()` | Gate A |
| `data_dictionary.csv` | `guard.data_dictionary()` + agent | Gate A |
| `figures/02_variables_pNN.png`, `figures/pages.json` | `guard.variable_figures()` | Gate A |
| `missingness.csv`, `figures/01_missing_pattern.png` | `guard.missingness()` + agent | Gate A (cause), Gate B (decision) |
| `label_audit.md` | `guard.label_audit()` + agent | Gate A |
| `cleaning_log.csv` | `guard.log_cleaning()` | Gate B |
| `categorical_plan.csv` | `guard.categorical_plan()` + agent | Gate B |
| `associations.json`, `figures/07_*_train.png` | `guard.train_associations()` | Gate B |
| `training_runs.jsonl` | `guard.fit()`, `guard.record_training()` | Gate C |
| `importance.json`, `figures/12_permutation_importance.png` | `guard.importance()` | Gate C |
| `.guard_state.json` (extended) | `guard.split()` | Gates B, C |

Every guard helper that writes a file the agent also edits **merges**: it refreshes computed
fields, keeps the agent's words for rows that still exist, adds rows for new columns with blank
word fields, and removes rows for columns no longer in the frame, warning with their names
(`UserWarning`). CSVs are read as `utf-8-sig`, so a file saved from Excel or Numbers (byte-order
mark, CRLF line endings) still parses; they are written as plain UTF-8.

## 1. Full-data proof and the pilot rule

### 1.1 Loading

```python
guard.load(path, *, state_dir="ml_pipeline", **read_kwargs) -> pd.DataFrame
guard.derive(df, name, *, parents, how, state_dir="ml_pipeline") -> pd.DataFrame
```

`load` picks the pandas reader by suffix (`.csv`, `.tsv`, `.txt` → `read_csv`, tab-separated for
`.tsv`; `.parquet`, `.pq`; `.xlsx`, `.xls`; `.json`; `.jsonl`, `.ndjson` with `lines=True`;
`.feather`), streams a SHA-256 of the file bytes (1 MiB chunks), and records the source. It raises
`PartialData` (a `LeakageError`) when `read_kwargs` contains `nrows`, `skiprows`, `skipfooter`,
`chunksize`, `iterator`, or `usecols`: a slice of rows or columns is a pilot, and a column dropped
before inspection is a column never inspected (drop it later with role `drop` in the dictionary).
Unsupported suffixes raise `ValueError` naming the supported ones.

`derive` registers a frame built from loaded sources at step 5 (joins, aggregations). Every name in
`parents` must already be in `source.json` (else `KeyError`); `how` is a required non-empty
sentence. It returns `df` unchanged.

`source.json`:

```json
{"schema_version": "1",
 "sources": {
   "data/raw.csv":  {"kind": "file", "sha256": "…", "bytes": 48213344, "n_rows": 1204331, "n_cols": 31,
                     "columns": ["…"], "loaded": "2026-09-23"},
   "patients_joined": {"kind": "derived", "parents": ["data/raw.csv", "data/visits.parquet"],
                     "how": "left join on patient_id, one row per admission", "n_rows": 1204331,
                     "n_cols": 44, "columns": ["…"], "loaded": "2026-09-23"}}}
```

File sources are keyed by the path as given; derived sources by `name`.

### 1.2 Profile schema 2

```python
guard.profile(df, *, target, time_col=None, group_col=None, source=None, state_dir="ml_pipeline") -> dict
```

- **No sampling.** `SAMPLE_ROWS` is removed; every statistic and the leakage-suspect scan use
  every row. `shape.sampled` / `shape.sample_rows` are removed.
- **Source match.** With `source=None`, the profile matches the first source (in `source.json`
  order) whose `n_rows == len(df)` and whose column set equals `df`'s; with `source="name"` it
  checks only that source. It
  writes `"source": {"name", "n_rows", "full_data": true}` on a match, else
  `{"name": null or the given name, "full_data": false, "reason": "<one sentence>"}` — no exception:
  a partial profile is recordable only with the owner's override (§5).
- **Per-column statistics** (the describe + info equivalent), a superset of schema 1 (existing keys
  keep their meaning):
  - every kind: `kind`, `dtype`, `non_null`, `missing`, `missing_frac`, `n_unique`, `unique_frac`,
    `memory_bytes`
  - `numeric`: `mean`, `std`, `min`, `p01`, `p05`, `p25`, `median`, `p75`, `p95`, `p99`, `max`,
    `skew`, `kurtosis`, `zeros`, `negatives`, `outliers` (1.5×IQR count), `outlier_frac`
  - `categorical`, `bool`: `n_levels`, `top` (up to 10 levels → fraction), `top_counts` (same
    levels → count), `rare_levels` (levels under `RARE_LEVEL_FRAC` = 1% of non-null rows)
  - `datetime`: `start`, `end`, `span_days`
  - `text`: `length_min`, `length_median`, `length_max`
  - `id`: the common keys only
- **Target:** schema 1 keys plus `class_counts` (label → count) for classification and
  `mean`, `std`, `min`, `max` for regression.
- `schema_version` becomes `"2"`. `traits` is unchanged, so red flags and warnings keep working.

### 1.3 Training receipts

```python
guard.fit(model, X, y=None, *, name=None, state_dir="ml_pipeline", **fit_kwargs) -> model
guard.record_training(data, *, name, state_dir="ml_pipeline") -> None
```

`fit` requires split state (§6.2; else `LeakageError("call guard.split() at step 6 first")`), calls
`model.fit(X, y, **fit_kwargs)` (`model.fit(X, **fit_kwargs)` when `y` is None), and appends one
JSON line to `training_runs.jsonl`:

```json
{"model": "HistGradientBoostingClassifier", "n_rows": 963, "n_train": 963, "full": true, "date": "2026-09-24"}
```

`full` is `len(X) == n_train`. `record_training` serves custom loops (PyTorch, Keras): `data` must
support `len()` — a frame, array, or dataset — so the row count is measured, never typed; it
writes the same line. The model name defaults to `type(model).__name__`.

### 1.4 Pilot rule (hook)

A **slice inside a model-training call** is a pilot: in the argument list of a model `.fit(` (any
receiver, including `guard.fit(`), a `[:N]` / `[a:N]` slice, `.iloc[` or `.loc[` with a slice,
`.head(`, or `.sample(`; or `nrows=` anywhere in code that also trains. After Gate B it is denied
unless the current run has a strict line `- Override: pilot - ...`:

> ml-pipeline: this trains on a slice of the data ('…'). Train on the full training split. If the
> user explicitly agreed to a quick pilot, record `- Override: pilot - <size> - user approved
> YYYY-MM-DD - reason: ...` first; Gate C still needs a full-data training run.

A slice taken in an earlier statement (`small = X_train.sample(...)`) is not seen by the text
scan; it is caught at Gate C, which requires a `full: true` receipt (§4.1).

## 2. Phase 1 records (Gate A)

### 2.1 Data dictionary

```python
guard.data_dictionary(df, *, state_dir="ml_pipeline") -> Path
```

`data_dictionary.csv`, header exactly:

```
column,kind,dtype,observed,meaning,unit,allowed,source,available_at_prediction,role,figure_note
```

- Computed: `column`, `kind`, `dtype`, `observed` (numeric `min..max, median m`; categorical
  `k levels: a, b, c…`; datetime `start..end`; text `length a..b`; id `n unique`). `role` is
  pre-filled `target` for the profile's target, blank otherwise.
- Agent: `meaning`, `unit` (`none` when unitless), `allowed` (range or levels), `source`,
  `available_at_prediction` (`yes` / `no`), `role` (`feature` / `target` / `id` / `leak` / `drop`),
  `figure_note` (§4.3).
- Requires a schema-2 profile in `state_dir` (else `LeakageError("run guard.profile() first")`).

### 2.2 Variable figure pages

```python
guard.variable_figures(df, *, target, per_page=12, state_dir="ml_pipeline") -> list[Path]
```

Draws every column, target included, on pages `figures/02_variables_p01.png`, `…_p02.png`, …:
numeric → histogram with a median line; categorical and bool → top-10 level bar chart;
datetime → rows per month; text → length histogram; id → a text panel with the unique count.
Writes `figures/pages.json` (`{"02_variables_p01.png": ["age", "zip", …], …}`) and one ledger line
per page with a computed explanation, e.g. `- Figure 02_variables_p01.png: page 1 of 3, 12 columns
(age … zip); 4 have missing values, 2 are strongly skewed (income, tenure). Per-column notes are
in data_dictionary.csv.` Distributions only — no target relationship before the split (decision 6).

### 2.3 Missingness

```python
guard.missingness(df, *, state_dir="ml_pipeline") -> Path
```

`missingness.csv`, header exactly `column,n_missing,pct_missing,mechanism,evidence,decision,reason`,
one row per column with at least one missing value (`n_missing`, `pct_missing` computed; merge
rules apply). It also draws `figures/01_missing_pattern.png` with `fig()`: with two or more gappy
columns, a heatmap of the correlation between their missing-indicators; with one, a bar of its
count. Explanation computed, e.g. `3 columns have gaps; income and employer are missing together
in 94% of income's gap rows.` With no gaps it writes the header only and no pattern figure.

- `mechanism` (Gate A): `random`, `depends-on-other-data`, `depends-on-itself`, `structural`,
  `unknown` (aliases accepted: `MCAR`, `MAR`, `MNAR`). `evidence` (Gate A): non-empty.
- `decision` (Gate B, §3.2) and `reason` (Gate B): see §3.2.

### 2.4 Label audit

```python
guard.label_audit(df, *, target, state_dir="ml_pipeline") -> Path
```

Writes `label_audit.md` if absent; on later calls it refreshes only the computed block:

```markdown
# Label audit: readmitted_30d

<!-- computed by guard.label_audit - do not edit -->
- rows: 1,428
- missing labels: 0
- classes: 0 = 1,265 (88.6%); 1 = 163 (11.4%)
- identical feature rows with different labels: 12 rows in 5 groups
<!-- end computed -->

## Definition
<!-- the exact rule that produces the label, with units and time window -->

## Source
<!-- who or what produced the label, and how -->

## When labeled
<!-- when the label becomes known, relative to the prediction time -->

## Leakage check
<!-- which columns could encode the label; every leakage suspect kept as a feature is named here -->

## Noise and consistency
<!-- how reliable the label is; what the conflicting-label count above means -->
```

Regression replaces the `classes:` line with `- target: min a, max b, mean m, std s`.
"Identical feature rows" groups rows by every non-target column except columns of kind `id`.

## 3. Phase 2 records (Gate B)

### 3.1 Cleaning log

```python
guard.log_cleaning(rule, before, after, *, reason, state_dir="ml_pipeline") -> pd.DataFrame
```

Appends to `cleaning_log.csv` (header `rule,reason,rows_before,rows_after,cells_changed,
columns_changed,date`): row counts; `cells_changed` over the shared index labels and shared
columns, treating NaN == NaN; `columns_changed` as `;`-separated names. Empty `rule` or `reason`
raises `ValueError`. Returns `after`. A project that needs no cleaning records instead the strict
PIPELINE.md line `- Cleaning: none needed - reason: <why>`.

### 3.2 Missingness decisions

The same `missingness.csv`. `decision` is one of `drop-column`, `drop-rows`, `impute-median`,
`impute-mean`, `impute-mode`, `impute-constant`, `impute-model`, `indicator`, `keep-native`, or a
`+`-joined combination (`impute-median+indicator`); `reason` is non-empty.

### 3.3 Categorical plan

```python
guard.categorical_plan(df, *, state_dir="ml_pipeline") -> Path
```

`categorical_plan.csv`, header exactly `column,n_levels,rare_levels,high_cardinality,encoding,reason`:
one row per `categorical` or `bool` column except the target (`n_levels`, `rare_levels`,
`high_cardinality` = `yes` when `n_levels > HIGH_CARD_LEVELS` computed). `encoding` is one of
`one-hot`, `ordinal`, `target-encoding-cv`, `frequency`, `rare-grouping`, `hashing`, `native`,
`drop`, or a `+`-joined combination; `reason` is non-empty. Rows for columns whose dictionary role
is not `feature` may stay blank; the hook ignores them.

### 3.4 Train-only associations

```python
guard.train_associations(train, *, target, state_dir="ml_pipeline") -> dict
```

Raises `NotTrainingSplit` (a `LeakageError`) unless split state exists and `len(train) ==
n_train` (row count, because step-7 features change the columns). Computes, with pandas and numpy
only, dropping missing values pairwise:

- **Feature–feature:** Spearman correlation among numeric and bool features; `collinear_pairs`
  lists pairs with |ρ| > `COLLINEAR_RHO` = 0.9.
- **Feature–target**, one value in [0, 1] per feature: classification target → correlation ratio η
  for a numeric feature, Cramér's V for a categorical or bool feature; regression target → |Spearman
  ρ| for a numeric feature, η of the target across levels for a categorical feature. Features with
  more than `HIGH_CARD_LEVELS` levels are marked `"high_cardinality": true` (V overstates them).

Writes `associations.json`:

```json
{"schema_version": "1", "computed_on": {"split": "train", "n_rows": 963, "date": "2026-09-23"},
 "target": "readmitted_30d", "task": "binary",
 "feature_target": [{"column": "prior_visits", "measure": "eta", "value": 0.41}],
 "collinear_pairs": [{"a": "weight_kg", "b": "bmi", "rho": 0.93}]}
```

and two figures through `fig()`: `07_correlations_train.png` (heatmap of up to 40 numeric
features; explanation names the count of pairs above 0.9 and the strongest one) and
`07_feature_target_train.png` (top 25 features by strength; explanation names the strongest
feature and value and how many features score below 0.05).

### 3.5 Red flag `target-stats-outside-train` (hook)

Evaluated like the other red flags (only when a profile exists; lifted by
`- Override: red flag target-stats-outside-train - ...` in the current run). Fires when a call to
`mutual_info_classif`, `mutual_info_regression`, `f_classif`, `f_regression`, `chi2`, or
`r_regression` has arguments that do not contain `train`, or when a `.fit(` / `.fit_transform(`
on a selector has arguments that do not contain `train`. A selector is `SelectKBest`,
`SelectPercentile`, `GenericUnivariateSelect`, `SelectFpr`, `SelectFdr`, or `SelectFwe`, called
directly (`SelectKBest(k=5).fit(...)`) or through a name assigned from one of those constructors in
the same code (`sel = SelectKBest(k=5)` ... `sel.fit(...)`). Message: `target-based feature selection must use the training split only (…) -
select features inside the training data or a CV fold`.

## 4. Phase 3 records (Gate C) and the real-explanation rule

### 4.1 Full training run

Gate C requires `training_runs.jsonl` with at least one line where `full` is true and `n_rows`
equals `n_train` in split state.

### 4.2 Importance and attribution

```python
guard.importance(predict_fn, val, *, target, metric_fn, greater_is_better=True, n_repeats=5,
                 seed=0, state_dir="ml_pipeline") -> dict
```

Raises `NotValidationSplit` (a `LeakageError`) unless `len(val) == n_val`. Scores the unshuffled
validation frame, then shuffles one feature at a time (`n_repeats` permutations each, seeded) and
records the mean and standard deviation of the score drop (sign flipped when lower is better).
Writes `importance.json` (`computed_on` as in §3.4 with `"split": "validation"`, `metric`,
`baseline_score`, `features` sorted by mean drop) and `figures/12_permutation_importance.png`
(top 25 with error bars; explanation names the top feature and its drop, and how many features
cost less than 0.001).

Attribution: at least one `figures/12_attribution_*.png` drawn by the agent (SHAP summary, partial
dependence, or coefficients) and saved with `guard.fig(12, "attribution_<name>", …)`.

### 4.3 The real-explanation rule

Implemented identically in the guard library (raise early) and the hook (deny at gate time):

- **Every figure explanation** (the text of a `- Figure <file>.png:` line): at least
  `MIN_NOTE_CHARS` = 40 characters and at least one digit. `guard.fig()` raises `ValueError` when an
  explanation fails; the hook lists every failing figure at each gate. v0.4's no-gaps text becomes
  `0 of N columns have missing values.`
- **Every column note** (`figure_note` in the dictionary): at least 40 characters; at least one
  number that **matches a statistic of that column** in the profile; and no two columns share a
  note (compared case- and whitespace-insensitively).
- **Matching a number.** Numbers are read with `-?\d[\d,]*(?:\.\d+)?%?` (commas removed). A number
  written with `d` decimals matches a statistic `f` when `round(f, d)` equals it, or, when it
  carries `%`, when `round(100·f, d)` equals it. The statistics are every number in the column's
  profile entry, recursively (so `top_counts` values count). The values 0 and 1 never count as a
  match: they occur in almost every column and prove nothing.

```
✗  age: shows the distribution
✗  age: looks fine, 12 things to check           (12 is not a statistic of age)
✓  age: right-skewed, median 54, 2% missing; cap at the 99th percentile (91) before scaling
```

## 5. Hook integration

The artifact checks live in a new stdlib module `hooks/gate_records.py`, imported by
`guard_training.py` the way `post_tool_warn.py` imports it. `gate_problems()` gains the checks
below; the message format is unchanged (`ml-pipeline: Gate X can't be recorded: p1; p2; …`), and
each problem names at most `LIST_CAP` = 10 columns followed by `and N more`.

| Gate | v0.4 (kept) | v0.5 adds |
|---|---|---|
| A | profile; ≥ 1 `01_*.png`; ≥ 2 `02_*.png` | `source.json` present and `source.full_data` true, or `- Override: partial data - …` in the current run; profile schema 2 with the common keys for every column and the kind-specific keys for every column with at least one value; dictionary: column set equals the profile's (missing, extra, duplicate named), no blank word field, valid `available_at_prediction` and `role`, exactly one `target` row naming the profile target, no `feature` with `available_at_prediction = no`, every figure note passes §4.3; every profile column on some page in `pages.json` and every listed page on disk; missingness: every gappy column listed once with `n_missing` equal to the profile's, valid `mechanism`, non-empty `evidence`, `01_missing_pattern.png` when any column has gaps; label audit: file present, computed block present with class counts equal to the profile's `class_counts` (or the regression line), each of the five sections has ≥ 40 characters outside HTML comments, and the Leakage check section contains the name (case-insensitive) of every leakage suspect whose role is `feature` or `id` |
| B | ≥ 1 `04_*.png`; model rationale | `cleaning_log.csv` with ≥ 1 row, or `- Cleaning: none needed - reason: …` in the current run; every missingness row has a valid `decision` and a `reason`; categorical plan: every categorical or bool column whose role is `feature` has one row with a valid `encoding` and a `reason`; `associations.json` with `computed_on.split = train` and `n_rows = n_train`; `07_correlations_train.png` and `07_feature_target_train.png` present |
| C | ≥ 1 `12_*.png` | a full training run (§4.1); `importance.json` with `computed_on.split = validation` and `n_rows = n_val`; `12_permutation_importance.png`; ≥ 1 `12_attribution_*.png` |
| D | ≥ 1 `13_*.png`; final-test line | — |
| all | every PNG > 1 KB with a ledger line | every ledger explanation passes §4.3 |

New override shapes, strict and current-run-scoped like the gate lines: `- Override: partial data -
…`, `- Override: pilot - …`, `- Override: red flag target-stats-outside-train - …`. The
`- Cleaning: none needed - reason: …` line is parsed the same way.

**Projects in progress.** Checks run only when a gate is recorded, so gates already recorded stay
valid; the next gate recorded needs the new records.

## 6. Fixes riding along

1. **Final-test override parsing (guard).** `final_test()`'s second-touch check currently accepts
   any line mentioning "override" and "final test". It adopts the hook's rules: only a strict
   `- Override: final test - …` line in the current run (after the newest `## Run:` heading).
2. **Split state records train and validation.** `split()` also writes `n_train`, `n_val`,
   `train_fingerprint`, `val_fingerprint`. Helpers that need them and find an older state file
   raise `LeakageError("split state is from an older guard: re-run guard.split()")`.
3. **A used test set cannot be reset.** `split()` today overwrites the state file, so calling it
   again resets `test_touches` to 0. It now raises `TestSetAlreadyUsed` when the existing state has
   `test_touches >= 1`, unless the current run has `- Override: final test - …`.

## 7. Skill, docs, eval

- **SKILL.md:** step 1 → `guard.load` (or `guard.derive` after joins), `guard.profile`,
  `guard.data_dictionary`, `guard.missingness`, `guard.label_audit`; step 2 →
  `guard.eda_figures`, `guard.variable_figures`, then fill every word field and figure note;
  the Gate A report shows the row chain (file rows → profiled rows), the dictionary, the
  missingness table, and the label audit. Steps 4–8 → `guard.log_cleaning`, missingness
  decisions, `guard.categorical_plan`, `guard.train_associations` after the split. Steps 9–12 →
  `guard.fit` / `guard.record_training`, `guard.importance`, an attribution figure. The
  enforcement section gains the Gate A–C table, the pilot rule, the new overrides, and §4.3.
- **SessionStart context:** names the new helpers in one clause.
- **README:** "Enforced, not just instructed" gains a paragraph on the full-data proof,
  per-column records, and the pilot rule.
- **Eval `understand-first`:** new graders `file_exists ml_pipeline/source.json` and
  `file_exists ml_pipeline/data_dictionary.csv`.

## 8. Tests (test-first)

- **Guard (`.venv`, pandas + numpy + matplotlib):** `load` (hash and rows recorded; each refused
  keyword raises; unsupported suffix); `derive` (unknown parent raises); `profile` schema 2 (full
  data true and false; no sampling on a frame above 200,000 rows; kind-specific keys;
  `class_counts`); `data_dictionary`, `missingness`, `categorical_plan` (skeleton; merge keeps
  words; stale rows removed and returned); `variable_figures` (every column on a page;
  `pages.json`; ledger lines pass §4.3); `label_audit` (computed block; conflict count on planted
  duplicates; sections never overwritten); `log_cleaning` (counts; empty reason raises);
  `train_associations` (wrong row count raises; planted signal ranks first; planted collinear pair
  flagged; figures written); `fit` / `record_training` (receipt lines; `full` flag; no split state
  raises); `importance` (planted feature ranks first; wrong split raises); final-test override
  parsing; `split` refuses to reset a used test set.
- **Hook (stdlib):** fixture builders write profile, CSV, JSON, and Markdown files without pandas;
  one test per Gate A/B/C problem in §5 (missing → named in the denial → complete → allowed);
  pilot detection and its override; `target-stats-outside-train` fires and is lifted; §4.3 cases.
- **Shared vectors:** one table of note cases (pass/fail with reason) run against both the guard's
  and the hook's implementation of §4.3.
- **End to end (`tests/test_v05_end_to_end.py`, `.venv`):** a synthetic messy frame (numeric with
  gaps, a high-cardinality categorical, bool, datetime, id, conflicting labels) written to CSV; the
  honest path through every helper, with word fields filled programmatically, records Gates A, B,
  and C through the real hook; shortcut variants are each denied — sampled profile, pilot fit,
  missing column note, copy-pasted notes, associations on the full frame.
- **Probe:** stays at 0 of 23. **CI:** the guard-library job also runs the end-to-end test.

## Constants

`RARE_LEVEL_FRAC = 0.01`, `COLLINEAR_RHO = 0.9`, `MIN_NOTE_CHARS = 40`, `LIST_CAP = 10`,
`PAGE_COLUMNS = 12`; unchanged: `SMALL_DATA_ROWS = 5000`, `IMBALANCE_MINORITY = 0.10`,
`HIGH_CARD_LEVELS = 50`, `LEAK_CORR = 0.95`, `LEAK_PURITY = 0.98`, `MIN_PNG_BYTES = 1024`.
Removed: `SAMPLE_ROWS`.

## Global constraints

- Hooks: Python ≥ 3.9 (CI runs 3.9 and 3.14), stdlib only, no `subprocess` / `eval` / `exec`, fail
  open, 10-second timeout.
- Guard: pandas + numpy required, matplotlib imported lazily with the `Agg` backend; no new
  dependencies. Canary stays stdlib.
- Versions → `0.5.0` in `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Amendments after the final review (2026-09-22)

A fresh whole-branch review found ten Important issues; these amendments record the resulting
behaviour and supersede the text above where they differ.

1. **The hook never fails open on unreadable input.** Each shell segment is read under its own error
   guard (a `~user` path that cannot be resolved is skipped, not fatal). A crash while checking a gate's
   records denies that ledger write only (`could not check the Gate X records (...)`); no tool is locked.
2. **Records saved by Excel or Numbers parse:** UTF-8 with or without a byte-order mark, cp1252, and `,`,
   `;`, or tab delimiters (the delimiter whose header has the record's key). The guard never overwrites
   a record it cannot read; it raises and leaves the file unchanged.
3. **Only gate lines a call adds are checked.** A recording line counts as new when it occurs more often in
   the new text than in what the call replaces (the `old_string` of an Edit, otherwise the existing
   ledger). Rewriting the ledger or anchoring an edit on a recorded gate does not re-check that gate; a new
   run's gate lines are checked.
4. **Overrides need the same records as approvals** (owner-facing ruling; the gate messages and the skill
   say so). An override records the user's choice; it does not skip the data work.
5. **Pilot rule, narrowed:** a row slice inside a model-training call is `[:N]`, `[a:N]`, `[-N:]`,
   `.iloc/.loc[...:N]`, `.head(`, or `.sample(` without `frac=1`; slices nested inside another subscript
   or taken of `.columns` are column selection, and negative stops are shifts. `nrows=` counts only inside
   a pandas `read_*(` call and never as `nrows=0`.
6. **Following execution, widened:** `torchrun`, `accelerate`, `deepspeed`, `horovodrun`, and
   `python -m torch.distributed.run|launch|accelerate.commands|deepspeed` run the `.py` files in their
   arguments; `sh|bash|zsh <file>` and `-c '<code>'` are read one level deep.
7. **Training receipts** count rows by `shape[0]` when the input has a shape (sparse matrices), else
   `len()`. `guard.fit(..., rows_from=X_train)` and `guard.record_training(..., rows_from=...)` record
   training on a resampled frame; the run is full when `rows_from` is the whole training split, and
   Gate C accepts a receipt whose `n_rows` or `rows_from` equals `n_train`.
8. **Column notes cite the column's own figures** (§4.3): the statistics a note may cite are its kind's
   figures (numeric percentiles, mean, std, min, max, skew, kurtosis, zeros, negatives, outliers;
   categorical and bool levels, top fractions and counts, rare levels; datetime span; text lengths) plus
   its missing count and fraction when it has gaps. Columns without such figures (ids, all-missing
   columns) may cite unique and missing counts. Row, non-null, distinct-value, and memory counts no
   longer count, so one template cannot pass every column.
9. **`load()` also refuses `columns=` and `filters=`** (parquet and feather subsets), and leakage suspects
   are stored by column-name string.
