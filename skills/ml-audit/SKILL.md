---
name: ml-audit
description: Use when the user asks to AUDIT, check, or verify an EXISTING machine-learning project, saved model, or claimed score ("can I trust this model / this accuracy?", "audit this folder", "check this project for leakage"). Runs the plugin's read-only audit tool on the folder and reports a verdict (trustworthy, not trustworthy, not enough evidence) with proven findings. Never trains, fixes, or edits the project. Not for building or training a new model - that is the ml-pipeline skill.
---

# ML Audit

**Hard rule: an audit reads; it never changes the project.** You do not train, retrain, tune, fix, re-split, or edit
any project file. The only writes are the audit tool's own dated folder under `<project>/ml_audit/`. An audit is not a
pipeline: do not start the 16-step ml-pipeline for an audit request.

## 1. Run the tool (one plain command)

The session context names the tool's absolute path. Run exactly:

```
python3 <plugin>/skills/ml-audit/lib/mlpipeline_audit.py <project folder>
```

- One command, nothing before or after it: no `cd ... &&`, no pipes, no `> file`, no `$VAR`, no `VAR=... python3`.
  The training guard allows this exact shape only (it is the one place test data may be scored before Gate C).
  Use the project's own interpreter (for example `<project>/.venv/bin/python`) if the system `python3` lacks pandas.
- Never copy the tool into the project or wrap it in a script; both are denied.
- Flags take data only: `--claim`, `--review`, `--check <family>`, `--sample-rows N`, `--timeout S`, and the two
  approval flags below. The tool takes no code, expressions, or import paths.

It prints `audit folder: <path>` and `verdict: ...`. Read `<audit folder>/report.md` and `findings.json`.

## 2. Show the inventory before trusting the result

Tell the user what the tool found: data files and their roles (train / val / test), predictions, models, code,
claims, the target column, and any time or entity column. Ask them to confirm. If something is wrong (a test file
labelled as validation, the wrong target), the USER corrects it: they copy the audit's `inventory.json` to
`<project>/ml_audit/inventory.json` and edit it, or ask you to write that one file - it is the only project file you
may write, and only when they ask. Then rerun.

## 3. Claims in prose

The tool parses machine-readable claims (metrics files, `ml_pipeline/runs.jsonl`, the ledger's final-test line).
Read the README, notebooks and reports for scores stated in prose and pass each one in with its source:

```
--claim 'metric=accuracy,value=0.93,split=test,source=README.md:12'
```

Metrics: accuracy, f1, auc, rmse, mae. Write the value exactly as the prose states it (its rounding sets the tolerance).

## 4. Your review layer (labelled, never proven)

Read the project's code and prose and add what the tool cannot prove, as review findings:

```
--review 'family=leakage,severity=high,text=StandardScaler is fit on the full data before the split,source=train.py:41'
```

Families: honest-scores, data-problems, leakage, weak-spots, test-misuse, model-safety. Severities: critical, high,
medium, info. Typical review findings: preprocessing fit before the split, test data used in tuning, a metric unsuited
to the class balance, a score too good for the domain, which subgroups matter for this domain, labels that look wrong.

Review findings are labelled "agent review" in the report. They never count as proven and never make the verdict
"not trustworthy" on their own; a review critical keeps the verdict at "not enough evidence" until resolved.

## 5. Model files: scanned, not loaded

Model files are scanned statically (pickle instructions read without running them) and reported as no risky imports
found, risky imports found, or unverified. A clean scan lowers risk; it does not prove safety - say so. Models are
loaded only if the user trusts the project: ask them; if they reply "trust models in <project folder name>", the
approval is recorded and you rerun with `--trust <project folder name>`. Never add `--trust` on your own; the guard
denies it without the user's message.

## 6. Retrain checks need approval

Checks that need a retrain (the shuffled-label check) are listed as "needs approval". Explain what it does (it trains
a small model on copies inside the audit folder, never on project files) and ask. If the user replies
"approve audit retrain for <project folder name>", rerun with `--retrain <project folder name>`.

## 7. Report in plain words

- Lead with the verdict exactly as the tool gave it, then the reason.
- Then the list of what was and was not checked: "trustworthy" covers only what was examined.
- Then each finding: severity, proven / suspected / review, what it means, and its reproduction command.
- **Never type a number the tool did not compute.** Quote scores, counts and intervals from `findings.json`; if a
  number is not there, say it was not computed.
- If the audit viewed test labels (trusted scoring), say that its numbers must not be used to choose a model.
- Offer next steps as suggestions only. Fixing is the user's decision and is outside the audit.

## Codex and Kimi

Without hooks, the read-only rule, the one-command rule, and the trust and retrain approvals are instructions only:
follow them exactly. Approvals there are the user's explicit words in chat; never write `ml_audit/.audit_approvals.json`.
