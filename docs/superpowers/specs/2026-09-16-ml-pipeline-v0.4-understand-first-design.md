# ml-pipeline v0.4.0 — Understand First (design)

**Status:** design approved in conversation 2026-09-16 (three sections, approach A). Awaiting the
user's review of this written spec before an implementation plan is written.

## Goal

Make "understand the data before modelling" mechanical. Today the skill *asks* for figures, a data
description, and a sensible model choice; the v0.3 eval showed an agent producing "text figures"
and moving on. v0.4 makes three things impossible to skip: a **data profile** and **figures** before
Gate A, a written **model rationale** before Gate B, and a short list of **provably wrong model
applications** that the hook denies outright.

Decisions taken during design: gates **block** when artifacts are missing (no warn mode);
model selection is enforced as **rationale + red flags**; implementation extends the existing
guard library and hook (approach A) rather than adding a gatekeeper command.

## Scope

**In**
1. `guard.profile()` → `ml_pipeline/data_profile.json` with per-column facts, leakage suspects,
   and a `traits` block every later check reads.
2. `guard.eda_figures()` (required set, factual explanations) and `guard.fig()` (any figure);
   every PNG gets a ledger line in `PIPELINE.md`.
3. Hook: a **gate-recording check** — a `Gate X: approved` line is denied until that gate's
   artifacts exist; exact missing items in the message.
4. `skills/ml-pipeline/references/model-selection.md` playbook keyed on traits, and a required
   `Model rationale:` block before Gate B.
5. Hook: five **red flags** denied from profile traits + code patterns, each with an override.
6. PostToolUse: three softer smells as warnings.
7. Skill, README, SessionStart hook, installer updates; eval case `understand-first`; release 0.4.0.

**Out (later):** the persistent process-wisdom store (L3); the evaluator agent; profiling of
non-tabular data beyond a manifest table; configurable thresholds (constants for now).

## 1. Data profile

```python
guard.profile(df, *, target, time_col=None, group_col=None, state_dir="ml_pipeline") -> dict
```

Writes `ml_pipeline/data_profile.json` and returns the same dict.

```json
{
  "schema_version": "1", "created": "2026-09-16",
  "shape": {"n_rows": 1428, "n_cols": 31, "memory_mb": 0.4, "sampled": false, "sample_rows": null},
  "target": {"name": "y", "task": "binary", "n_classes": 2,
             "class_balance": {"0": 0.886, "1": 0.114}, "minority_frac": 0.114},
  "columns": {
    "age":        {"kind": "numeric", "missing_frac": 0.02, "n_unique": 71, "unique_frac": 0.05,
                   "min": 18, "median": 54, "max": 97, "skew": 0.3, "outlier_frac": 0.01},
    "zip":        {"kind": "categorical", "missing_frac": 0.0, "n_unique": 412, "unique_frac": 0.29,
                   "n_levels": 412, "top": {"94110": 0.02, "94103": 0.02}},
    "admitted_at":{"kind": "datetime", "missing_frac": 0.0, "n_unique": 1400, "unique_frac": 0.98,
                   "start": "2023-01-02", "end": "2024-12-30"},
    "notes":      {"kind": "text", "missing_frac": 0.4, "n_unique": 850, "unique_frac": 0.6, "median_length": 62},
    "patient_id": {"kind": "id", "missing_frac": 0.0, "n_unique": 1428, "unique_frac": 1.0}
  },
  "duplicates": {"exact_row_frac": 0.0},
  "constant_columns": [],
  "high_cardinality_categoricals": ["zip"],
  "datetime_columns": ["admitted_at"],
  "time_col": "admitted_at", "group_col": null,
  "entity_candidates": ["patient_id"],
  "leakage_suspects": [
    {"column": "outcome_recorded_at", "reason": "datetime later than time_col in 97% of rows"},
    {"column": "risk_score_final",    "reason": "|corr| with target = 0.97"},
    {"column": "discharge_outcome",   "reason": "name matches *_outcome"}
  ],
  "traits": {"n_rows": 1428, "n_features": 30, "task": "binary", "minority_frac": 0.114,
             "has_datetime": true, "has_groups": false, "small_data": true, "imbalanced": false,
             "high_card_categoricals": ["zip"]}
}
```

**Column kind:** bool dtype → `bool`; datetime64 → `datetime`; numeric dtype → `numeric`, or `id`
when `unique_frac > 0.95` and values are integer-like; object/string/category → `id` when
`unique_frac > 0.95`, else `text` when median string length > 30, else `categorical`.
**Target task:** regression when the target is numeric with `n_unique > 20`; binary when
`n_unique == 2`; multiclass otherwise. **Outliers:** 1.5×IQR rule. **High cardinality:** categorical
with `n_levels > 50`. **Entity candidates:** columns whose name matches `(^|_)id$` (case-insensitive)
or `unique_frac` in (0.01, 0.95) with an integer-like or string dtype and name ending in `id`.

**Leakage suspects** (any of): numeric column with `|corr(col, target)| > 0.95` (target label-encoded
if needed); categorical with ≤ 50 levels where predicting the per-level majority class gives accuracy
`> 0.98`; datetime column whose values are later than `time_col` in > 50% of rows; column name
matching `(outcome|result|after|label|target|final|actual)` or ending in `_y`, excluding the target.

**Traits:** `small_data = n_rows < 5000`; `imbalanced = task != regression and minority_frac < 0.10`;
`has_datetime = any datetime column or time_col given`; `has_groups = group_col given` (entity
candidates are suggestions, not a trait — they must not trigger red flags).

**Sampling:** above 200,000 rows, distributions and correlations use a 200,000-row sample
(`sampled: true, sample_rows: 200000`); counts, missingness, duplicates use the full table.

## 2. Figures and the ledger

Directory `ml_pipeline/figures/`; file names `NN_<slug>.png` where `NN` is the two-digit step.
Every saved figure appends one ledger line to `PIPELINE.md` (created if absent):

```
- Figure 02_target_balance.png: y=1 is 11.4% of 1,428 rows, so a majority-class dummy scores 88.6%.
```

```python
guard.eda_figures(df, *, target, time_col=None, state_dir="ml_pipeline") -> list[Path]
guard.fig(step: int, name: str, figure, explanation: str, state_dir="ml_pipeline") -> Path
```

`eda_figures` always writes `01_missingness.png` (missing fraction per column), `02_target_balance.png`
(class fractions, or a histogram for regression), `02_distributions.png` (histograms of up to 12
numeric columns), `02_correlations.png` (numeric correlation heatmap incl. target, up to 30 columns),
and `02_temporal_coverage.png` (rows per period over `time_col`) when a datetime column exists.
Explanations are sentences computed from the data. `fig` saves at 120 dpi, tight bounding box,
and requires `explanation`. matplotlib is imported lazily inside these two functions with the `Agg`
backend, so `split()` / `final_test()` keep working without it. If matplotlib is missing the
functions raise `RuntimeError("matplotlib is required for figures: pip install matplotlib")` —
there is no text-figure fallback.

## 3. Gate-recording check (hook)

**Trigger:** any string in `tool_input` (Write `content`, Edit `new_string`, Bash command including
heredocs) that contains a non-negated `Gate X: approved …` line. This check runs **before** the
existing `.md` skip in `main()`.

**Project root:** the parent of `ml_pipeline/` if the edited `file_path` ends in `PIPELINE.md`;
otherwise the existing upward search from `cwd`.

**Requirements** (all PNGs in `figures/` must be > 1,024 bytes and have a ledger line; the ledger
text checked is the existing `PIPELINE.md` plus the text being added):

| Gate | Required |
|---|---|
| A | `data_profile.json`; ≥ 1 `01_*.png`; ≥ 2 `02_*.png` |
| B | ≥ 1 `04_*.png`; `Model rationale:` block (§4) |
| C | ≥ 1 `12_*.png` |
| D | ≥ 1 `13_*.png`; a `- Step 14 final test:` line |

**Denial message:** `ml-pipeline: Gate A can't be recorded: data_profile.json missing; figures/01_*.png 0 found (need 1); 02_correlations.png has no '- Figure 02_correlations.png:' line.` One item per missing requirement, nothing else.

## 4. Playbook and the rationale block

`skills/ml-pipeline/references/model-selection.md` (about two pages), organised by trait: tabular
small (< 5,000 rows) / medium / large; imbalanced target; high-cardinality categoricals; temporal
data; repeated entities; text; images; interpretability required. Each section lists: baseline
(always), strong default, worth trying, **avoid and why**, metric, split. It ends with a
wrong-application table (the five red flags plus advisory smells). The skill reads it at step 3
and before the Gate B report. The SessionStart hook names its path; `install-other-tools.sh`
copies `references/` for Codex and Kimi.

Required in `PIPELINE.md` before `Gate B: approved` is accepted:

```
Model rationale:
- traits: binary, 1,428 rows (small), minority 11.4%, has_datetime, no groups
- baseline: majority-class dummy + logistic regression (class_weight=balanced)
- candidates: gradient boosting with small trees; regularized logistic
- ruled out: neural nets (small data); k-NN (mixed scales, weak on tabular); random split (temporal)
- metric: PR-AUC primary, recall at fixed precision secondary
```

**Check:** a line matching `^\s*Model rationale:` followed, within the next 12 lines, by bullets
beginning `- traits:`, `- baseline:`, `- candidates:`, `- ruled out:`, `- metric:` (case-insensitive).
Missing bullets are named in the denial. Content is not graded by the hook.

## 5. Red flags (hook, PreToolUse)

Evaluated only when `ml_pipeline/data_profile.json` exists (found next to `PIPELINE.md`). Each
denial names its override line: `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: …`
(non-negated) in `PIPELINE.md` lifts that one flag.

| name | fires when | code pattern |
|---|---|---|
| `neural-net-small-data` | `traits.small_data` and a fit/train call is present | `\b(?:MLPClassifier\|MLPRegressor\|Sequential\|keras\|torch\.nn\|nn\.Module\|tensorflow\|tf\.keras)\b` |
| `random-split-temporal` | `traits.has_datetime` | `\b(?:train_test_split\|KFold\|StratifiedKFold\|ShuffleSplit\|RepeatedKFold)\s*\(` |
| `group-split` | `traits.has_groups` | any of the above plus `TimeSeriesSplit\s*\(` — i.e. a split that is not `Group*` |
| `accuracy-imbalanced` | `traits.imbalanced` | `scoring\s*=\s*["']accuracy["']` |
| `resample-before-split` | always | `fit_resample\s*\(([^)]*)\)` whose arguments do not contain `train` |

Denial message: `ml-pipeline: red flag <name>: <one-sentence why> - <what to do instead>. Override with '- Override: red flag <name> - ...' in ml_pipeline/PIPELINE.md.`

## 6. PostToolUse warnings (never block)

With a profile present: `KNeighbors*`, `SVC`, `SVR` when `traits.n_rows > 50000`; `OneHotEncoder` or
`get_dummies` when `traits.high_card_categoricals` is non-empty; `accuracy_score(` when
`traits.imbalanced`. Existing warnings unchanged.

## 7. Skill, README, hooks, installer

- **SKILL.md** step 1: call `guard.profile(...)` first and read `leakage_suspects` and `traits` before
  anything else. Step 2: `guard.eda_figures(...)` produces the required figures with factual
  explanations; add more with `guard.fig`; if matplotlib is missing, install it — no text figures.
  Step 3: consult `references/model-selection.md` with the traits. Gate B report: include the
  `Model rationale:` block. Enforcement section: the gate table (§3), red flags and override names
  (§5). Non-negotiables: "Every gate is recorded with its figures on disk and their explanations in
  PIPELINE.md."
- **README:** the "Enforced, not just instructed" section gains a paragraph on profile, figures,
  rationale and red flags.
- **SessionStart hook:** also names `references/model-selection.md`.
- **install-other-tools.sh:** copies `references/` next to `lib/`.

## 8. Eval

New case `evals/understand-first/` (tags `[gates, needs-deps]`, `runs: 1`, `max_turns: 40`): the
scaffold writes `data.csv` and runs `python3 -m pip install --quiet --user pandas matplotlib`
(exit 0 either way). Prompt: the same "train a classifier" request, noting pandas and matplotlib
are installed. Graders: `tool_used Skill`; `tool_used Bash|Write` with `\.fit\(` min 0 max 0;
`file_exists ml_pipeline/data_profile.json`; `file_exists ml_pipeline/figures/02_*.png`; `regex`
over `{source: file, path: ml_pipeline/PIPELINE.md}` for `- Figure 02_target_balance\.png:`; the
existing `stops-at-gate-a` llm rubric. If the sandbox blocks network, the case fails on the
file graders; CI runs `--tag gates` without `needs-deps` until that is resolved. `gate-stop`
stays stdlib-only and unchanged.

## 9. Tests

- Guard (venv adds matplotlib): profile kinds, missing/duplicate fractions, each leakage-suspect
  rule on planted columns, traits, sampling flag; `eda_figures` writes every required PNG > 1 KB with
  a ledger line; `fig` writes PNG + ledger; `RuntimeError` when matplotlib is absent (simulated by
  hiding the module).
- Hook (stdlib): gate check for A (no profile → no `01_*` → PNG without ledger line → complete),
  B (rationale missing → missing bullets named → complete), C, D; Bash `echo >>` route; each red
  flag fires with a fixture profile, is lifted by its override, and is silent without a profile.
- PostToolUse: the three new warnings.

## 10. Edge cases

No matplotlib → clear error, gate stays blocked. Eval sandbox without pandas → `needs-deps` tag.
Large tables → sampled statistics, flagged. Images/text → the skill's step 1 builds a manifest table
(path, label, size or length) and profiles that. Hook stays fail-open; the gate check runs only when
a gate line is present in the edited text, red flags only when a profile exists and a pattern
matches (~30 ms).

## Global constraints

- Hooks: Python ≥ 3.10, stdlib only, no `subprocess`/`eval`/`exec`, fail open.
- Guard: pandas + numpy required; matplotlib imported lazily, `Agg` backend. Canary stays stdlib.
- Constants: `SMALL_DATA_ROWS = 5000`, `IMBALANCE_MINORITY = 0.10`, `HIGH_CARD_LEVELS = 50`,
  `LEAK_CORR = 0.95`, `LEAK_PURITY = 0.98`, `MIN_PNG_BYTES = 1024`, `SAMPLE_ROWS = 200_000`.
- Gate lines `- Gate X: approved YYYY-MM-DD`; override lines `- Override: red flag <name> - …`.
- HOL scanner ≥ 98/100 with zero critical/high. Versions → `0.4.0` in both manifests.
- Branch `v0.4-understand-first`; commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
