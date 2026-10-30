# ml-pipeline

**Stop your coding agent from jumping straight to model training.**

Coding agents love to call `.fit()` five minutes into an ML task — before looking at the
data, before there's a test set, before anyone agreed on what is being predicted. This
plugin forces a strict, data-first 16-step pipeline with explicit user-permission gates —
and, in Claude Code, a hook that makes skipping them impossible:

![The ml-pipeline flow: 16 steps across 4 phases — Understand, Prepare, Model, Prove & Ship — each ending in a permission gate](assets/pipeline.svg)

## Install

**Claude Code**

```
/plugin marketplace add jananthan30/ml-pipeline
/plugin install ml-pipeline@ml-pipeline
```

The skill activates automatically on any ML training/tuning/evaluation task, or invoke it
directly with `/ml-pipeline`. Auditing existing work is a separate skill, `/ml-audit` (below).

**Codex CLI**

```
codex plugin marketplace add jananthan30/ml-pipeline
codex plugin add ml-pipeline@ml-pipeline
```

**Kimi Code CLI** (or Codex without plugins)

```bash
git clone https://github.com/jananthan30/ml-pipeline && cd ml-pipeline && ./install-other-tools.sh
```

Copies both skills (pipeline and audit) into `~/.codex/skills/` and `~/.kimi-code/skills/` and
appends an enforcement section and an audit section to each tool's global `AGENTS.md`.

## How it works

- **Strict order, no skipping** — the split happens *before* feature engineering and
  preprocessing; all fitting uses training data only; the test set is touched exactly once.
- **You stay in control** — at each gate the agent explains, in plain language, what it
  did, what it found (with figures), and what comes next, then waits for your explicit OK.
  Only *your* message approves a gate — the agent can't approve its own. "Just train it" gets
  the risk explained and needs you to say "override gate B".
- **One step at a time** — each finished step is recorded as `- Step N done: …`, and a step is
  accepted only after the steps before it and with its own evidence on disk (profile, figures,
  frozen split, logged baseline run, …).
- **Big and wide data** — the guard library reads Parquet/CSV/JSON-lines/Feather in batches, so
  profiling, EDA, splitting and the final test work on files larger than memory. EDA draws every
  column, and the agent reads a short summary instead of a huge JSON.
- **Every column accounted for** — before Gate A each column has a data-dictionary row and a written
  note that cites its own numbers, each gap has a likely cause, and the label is audited. You judge
  the quality; the hooks check that nothing is missing.
- **See everything** — [marimo](https://marimo.io) notebooks as the workbench, matplotlib
  figures saved to `ml_pipeline/figures/`, each explained in 1–2 plain sentences.
- **Resumable** — `ml_pipeline/PIPELINE.md` tracks every step and approval, so a new
  session continues where the last one stopped.

## Enforced, not just instructed

In Claude Code the plugin ships hooks. Before any command, file write, or notebook edit runs, a
`PreToolUse` hook scans the code — including the scripts, notebooks, and local modules a shell
command runs — and **denies**: any non-exploratory fitting before Gate A; model training before
Gate B; complex models before the baseline; tuning before the candidates; evaluating on the test
split before Gate C; fitting on the test split, ever; training on a slice of the data unless you agreed
to a pilot. It also refuses edits to its own state files, deletion of recorded ledger lines, and any
attempt by the agent to switch enforcement off. A `UserPromptSubmit` hook records gate approvals from your
messages only, so a gate line the agent writes on its own doesn't count. A denied call tells the
agent exactly what is missing. A `PostToolUse` hook warns when a metric looks too good to be honest,
a split ignores class balance or time, or an output is so large it floods the agent's context.

At step 6 the agent switches to `guard.split()` / `guard.split_to_files()` / `guard.final_test()`, a
small runtime library that drops duplicates, refuses random splits when dates exist (even dates
stored as text), refuses row splits when an entity (patient, user, device) repeats, keeps a gap
between time periods, and lets the frozen test set be scored exactly once. `guard.cv_splits()` gives
leakage-safe folds, and `guard.log_run()` keeps a reproducible run log.

It also refuses to let understanding be skipped. `guard.profile()` writes a data profile — column
kinds, missingness, duplicates and label conflicts, class balance, dates (even when stored as text),
repeating entities, and **leakage suspects** (linear and non-linear) — plus a one-page summary for the
agent to read. `guard.eda_figures()` draws the figures — every column, on pages — with explanations
computed from the data. Gate A can't be recorded without them and a written prediction contract; Gate B
can't be recorded without a written
model rationale (traits → baseline → candidates → what was ruled out and why → metric). And seven
provably wrong applications are denied outright: a neural net on a small table, a random split on
temporal data, a non-group split on grouped data, accuracy as the selection metric on an imbalanced
target, resampling before the split, target-based feature selection outside the training split, a
complex model before the baseline, and tuning before the candidates.

The data work is complete and provable. A file passed to `guard.profile()` (or loaded with
`guard.load()`) has its hash and row count recorded, and Gate A needs a profile of the whole file — a
sample or an `nrows=` slice is stopped. Counts and moments cover every row, even on files larger than
memory. Every column gets a data-dictionary row, a place on a figure page, a written note that cites a
real number from that column, and — if it has gaps — a likely cause and a decision; the label gets an
audit. Correlations and feature–target strength are computed on the training split only, training
runs are recorded so Gate C can prove one used the whole training split, and permutation importance
comes from the validation split after the baseline. Approvals never carry over between experiments:
each starts with a `## Run: <name> (date)` heading, and your approvals belong to the run you gave them in.

Prove it on the **canary**: `examples/canary/canary.csv` has a known honest ceiling of 0.80
accuracy. Beat it and you leaked (the test suite runs the honest path on it and checks the score
stays below). You (never the agent) set `ML_PIPELINE_ENFORCE=0` to switch enforcement off for non-ML
projects; `ML_PIPELINE_TEST_NAMES=X_lockbox,y_lockbox` teaches the hook your own test-set names.
Codex and Kimi get the same rules as instructions (no hook support there).

The hooks stop an agent from skipping steps by accident or by habit, and from the common shortcuts
(running a hidden script, renaming the test set, editing its own approvals). They are pattern checks,
not a sandbox: code built specifically to evade them can still run. For a hard boundary, run the agent
without direct write access to `ml_pipeline/.approvals.json` and `.guard_state.json`.

## Audit an existing project

The gates guard new work. For work that already exists — a saved model, a data folder, a claimed
score, built before the gates or by someone else — ask the agent to **audit** it ("can I trust this
model?", or `/ml-audit`). The audit skill runs one read-only tool:

```
python3 <plugin>/skills/ml-audit/lib/mlpipeline_audit.py <project folder>
```

It inventories the folder (data and its train/val/test roles, predictions, models, code, claims;
environment, dependency and version-control folders are skipped), runs every check it has evidence
for, and writes one new dated folder, `<project>/ml_audit/<date_time>/`, with `report.md`,
`findings.json` and `inventory.json`. It changes nothing else: no ledger lines, no gates, no approvals,
no test-touch count. Repeated audits sit side by side.

- **One plain verdict, no score:** *trustworthy*, *not trustworthy*, or *not enough evidence*. One proven
  critical finding (a claimed score the predictions don't give, rows shared between train and test, a
  feature that gives the label away) means *not trustworthy*. *Trustworthy* needs the three required
  checks — score recomputation, train/test overlap, feature–target leakage — to have run, and it covers
  only what the report lists as checked; "could not check" is never a pass.
- **Every finding** has a severity, its evidence (proven by the tool, suspected, or agent review), counts,
  the files and columns involved, and a command that reproduces it. The report holds column names and
  counts, never raw row values.
- **Checks:** claimed scores recomputed from saved predictions (accuracy, F1, AUC, RMSE, MAE, no
  scikit-learn) within the claim's own rounding, and compared with a trivial baseline; data problems
  (missing values, duplicates, constant columns, conflicting and invalid labels — every row streamed);
  leakage (identical rows or entities across splits, near-perfect single-feature association, training
  data later than test data); weak subgroups with intervals and calibration; and signs of test-set reuse
  in the run log, the ledger, and tuning code.
- **Prose counts too:** the agent passes scores it finds in a README as `--claim
  'metric=accuracy,value=0.93,split=test,source=README.md:12'`, and adds its own reading of the code as
  `--review` findings, labelled as such. Review findings never make the verdict on their own, and the agent
  never states a number the tool did not compute.
- **Model files are never loaded by default.** Pickle, joblib and zip-format torch files are scanned
  statically — the pickle instructions are read, never run — and reported as *no risky imports found*,
  *risky imports found* (named), or *unverified*. A clean scan lowers the risk; it does not prove the file
  is safe.
- **To trust a model,** tell the agent "trust models in `<project folder name>`". Your message is recorded
  by the hook; the agent then reruns with `--trust <name>`, and the tool loads each model in a child
  process with the project's own interpreter (code-free modes first, such as torch `weights_only`),
  predicts the test file, and checks the claims against it. Files that import dangerous names are not
  loaded even then. Because this views the test labels, its numbers must never be used to choose a model.
- **Retrain checks need your approval** ("approve audit retrain for `<name>`"). The shuffled-label check
  gives every near-identical row one random label, trains a nearest-neighbour model on a copy of the
  training data inside the audit folder, and scores the test copy: above chance means the test rows copy
  training rows. It never runs project code, and the copies are deleted afterwards.

The guard lets exactly one command through for this: the plugin's own tool, run as one plain command
with its documented flags. A script with the same name, a training script that mentions "audit", the tool
chained to another command, an environment prefix, or a `--trust` the user never gave are all denied.
In Codex and Kimi (no hooks) the read-only, one-command, trust and retrain rules are instructions only.

Not covered yet: production drift and monitoring, fairness review, backdoor or poisoning scans of
weights, and fixing anything. The audit does not send project data anywhere.

## Development

```bash
pip install -r tests/requirements-dev.txt
python -m pytest tests
```

## License

Apache-2.0
