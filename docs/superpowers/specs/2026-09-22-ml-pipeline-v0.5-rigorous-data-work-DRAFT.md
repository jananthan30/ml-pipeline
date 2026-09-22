# ml-pipeline v0.5 — rigorous data work before training (DRAFT)

Status: draft requirements from the owner, 2026-09-22. Confirm scope before implementing.
Pairs with the hook audit in `tests/probe_hook_audit.py` (20 of 23 cases wrong on v0.4.0).

## Problem

Agents skip or shortcut the data work: they profile a sample, run a small pilot, and move to
training. In v0.4.0, steps 4–8 (cleaning, engineering, split, features, preprocessing) are
instructions only. The Phase 2 checkpoint needs just one `04_*.png` figure and the model
rationale block. Nothing proves that every variable was inspected or that the full dataset was
used.

## Principle

The harness guarantees **completeness**: every column, every missing value, every label, and
every figure has a recorded, explained decision. Each check is machine-verifiable against the
data's actual column list and row count. **Quality** is judged by the user at each checkpoint;
the harness makes sure there is something complete to judge.

## Required artifacts

### Phase 1 — understand (checked before the first checkpoint can be recorded)

| Artifact | Machine check |
|---|---|
| `data_dictionary.csv`: one row per column with meaning, unit, type, allowed range or levels, source, available-at-prediction-time (yes/no), role (feature / target / id / leak / drop) | Every column in the data appears exactly once; no empty required fields |
| `variables/`: per-variable report (describe + info equivalent) for every column | One entry per column; the numbers are computed by the guard library, not typed |
| Per-variable figures (distribution; relationship to target for key features) | A figure per column (grouped pages allowed for wide data), each with a ledger explanation |
| Missingness table and missingness-pattern figure | Every column with missing values is listed with count, percentage, and suspected mechanism |
| `label_audit.md`: exact label definition, source, class counts, time of labeling, leakage check, noise/consistency check | Present; class counts match the profile |
| Full-data proof: profile row count and source-file hash | Profile rows equal the raw source rows; a sampled profile needs a recorded user override |

### Phase 2 — prepare (checked before the modeling checkpoint can be recorded)

| Artifact | Machine check |
|---|---|
| Cleaning log: every rule with before/after row and value counts | At least one entry, or an explicit "no cleaning needed" statement with reason |
| Missingness plan: per column, the decision (drop / impute method / missing-indicator / keep) and reason | Every column with missing values has a decision |
| Categorical plan: per categorical column, the encoding (one-hot / ordinal / target encoding inside CV / rare-level grouping) and reason | Every categorical feature has a decision; high-cardinality columns never get plain one-hot without a reason |
| Correlation matrix and multicollinearity check, computed on the training split | Figure plus ledger explanation; flagged pairs listed |
| Feature–target association (mutual information or similar), training split only | Present; computed after the split |
| Imputers, scalers, and encoders fit on training data only | Already enforced by the hook |

### Phase 3 — after the baseline (evaluation checkpoint)

| Artifact | Why here, not earlier |
|---|---|
| Feature importance (permutation, on validation) and attribution (e.g. SHAP) with explanations | Both need a trained model, so they belong after the baseline |

## Anti-shortcut rules (hook)

- Deny training code that subsamples (`head`, `sample` with a fraction, `nrows`, fixed small
  slices) unless PIPELINE.md records a user-approved pilot, and require a full-data run before
  the evaluation checkpoint.
- Every figure's explanation must be a real explanation: minimum length and at least one number
  from the data (reject "shows the distribution").
- Target-related statistics used for feature selection must come from the training split only.

## Open questions for the owner

1. ~~Wide data: one figure per column, or grouped pages?~~ **Decided 2026-09-22: grouped pages**
   (several variables per figure page), with a written note for every column in the ledger. Every
   column must appear on some page; the check is per column, not per figure.
2. Should the data dictionary need the user's sign-off, or only completeness?
3. Pilot runs: allowed with an override, or never before the preparation checkpoint?
