# ml-pipeline audit: check existing models and data for issues

Status: ready to build. Builds on the combined v0.5 line, not on v0.4.

## Problem Statement

The plugin's gates stop bad work before and during training. They say nothing about work that already exists. A person
with a saved model, a data folder and a claimed score has no way to ask: can I believe this number? Models built before
the gates existed, inherited from someone else, or built with the gates switched off are all outside the plugin's reach.

## Solution

`/ml-pipeline:audit <folder>` inspects a project folder and reports what it found, what it could not check, and whether
the project's results can be trusted. It works with whatever the folder holds and does not need a pipeline ledger. It
changes nothing in the project: its only writes are a new audit folder.

The result is one of three verdicts, backed by findings. Each finding has a severity, its evidence and a command that
reproduces it. There is no numeric score.

## Decisions

### Owner's decisions (accepted recommendations, 2026-10-30)

1. **Input:** a project folder. Everything found there is inventoried, the user can correct the inventory, and anything
   missing becomes "could not check" rather than an error.
2. **Model safety:** formats that can run code on load are never loaded by default. The audit scans them statically and
   loads a model only after the user says they trust it, per project.
3. **Version 1 checks:** data problems, leakage, honest scores, hidden weak spots, test-set misuse.
4. **Report only:** no fixing, no retraining. Checks that need a retrain are listed as needing approval and run only if
   the user approves.
5. **Report shape:** severity, evidence and reproduction per finding, and a verdict of trustworthy, not trustworthy, or
   not enough evidence. No numeric score. One serious proven leak means not trustworthy.
6. **Build order:** after the v0.5 combination, on a separate branch, tested with planted problems.

### Proposals made while writing this spec (the owner delegated these)

**Two layers of findings.** The tool computes every number and makes every proven finding. The agent reads code and prose
and adds review findings. Review findings are labelled as such and can never produce a "proven" result, which keeps to the
existing rule that agents never type numbers.

**Verdict rule.**

| Verdict | When |
|---|---|
| Not trustworthy | At least one proven critical finding |
| Not enough evidence | No proven critical, but a required check could not run, or a suspected critical is unresolved |
| Trustworthy | Every required check ran and found no proven critical or high finding, and no suspected critical remains |

Required checks are score recomputation, train/test overlap, and feature-target leakage. The verdict always appears next
to the list of what was and was not checked, because "trustworthy" only covers what was examined.

**Finding shape.** Identifier, check family, severity (critical, high, medium, info), evidence class (proven, suspected,
review), evidence counts, affected files and columns, and a reproduction command.

**Audit output.** A dated subfolder under an audit folder in the audited project, holding a human report, a machine
findings file and the inventory. The report holds column names and counts, never raw row values, so it is safe to commit.

**Exception in the training guard.** The guard denies scoring on test data, and the audit must do exactly that. The guard
allows only an exact invocation of the plugin's own audit tool, resolved inside the plugin's install folder. The exception
must not become a bypass:
- The tool never calls a fit, except in the approved retrain check.
- It takes no code, expressions or import paths as arguments.
- Training code that merely mentions "audit" is still denied.
- A script in the project with the same name as the tool is still denied.

The audit neither reads nor writes the guard's approval and state files, and it adds no ledger lines.

**Model loading.** The static scan reads pickle opcodes without executing them, covering `.pkl`, `.pickle`, `.joblib`, and
the pickle inside zip-format `.pt` files. It reports imports outside a safe list. A file it cannot parse is "unverified",
never "safe". The scan lowers risk and does not prove safety, and the report says so: a clean scan never loads the file
without approval. Trust is granted only through an explicit flag naming the project. Loaded models run in a child process
using the project's own interpreter. Where a format has a code-free mode, it is used first.

**Claimed scores.** Machine-readable claims (metrics files, the run log, the final-test ledger line) are parsed by the
tool. Claims in prose are found by the agent and passed in with a source location, and are then checked the same way. A
claim with no predictions behind it is "could not check", not "wrong".

**Tolerances are named constants.** Score mismatch uses a tolerance that accounts for rounding at the claim's stated
precision. Subgroup findings require a minimum group size and report an interval. Near-perfect single-feature association
is the leakage trigger. Each threshold is a documented constant with a test on both sides of it.

**Scale.** Large files go through the guard library's streaming paths. Any check run on a sample says so, with the sample
size, in the finding. Environment, dependency and version-control folders are skipped.

**Dependencies.** Same as the guard library. No scikit-learn: metrics are implemented directly and tested against known
values, so the audit runs in projects that use any framework.

**Skill boundaries.** The audit is its own skill, and its description is narrow enough not to compete with the pipeline
skill. The pipeline skill's "any evaluation" trigger gains an exception for audits, so an audit does not start a 16-step
pipeline. Codex and Kimi installs receive the skill too. They have no hooks, so there the read-only and trust rules are
instructions only, and the docs say so.

**Testing seam: one.** Run the audit tool on a fixture folder and assert on its findings file and verdict. Accepted without
a separate question under the owner's delegation.

## The checks

| Family | Proven by the tool | Suspected or review |
|---|---|---|
| Data problems | Missing values per column, duplicate rows, constant columns, identical features with conflicting labels, invalid labels | Labels that look wrong for the domain |
| Leakage | Identical rows or entities across splits, near-perfect single-feature association with the target, training data later than test data when a time column exists | Preprocessing fit before the split, seen in code |
| Honest scores | Recomputed score differs from the claim, score no better than a trivial baseline, claim labelled test but computed on other rows | Too-good-to-be-true scores, metric unsuited to class balance |
| Hidden weak spots | Subgroup performance gap with an interval, calibration error | Which subgroups matter for this domain |
| Test-set misuse | Run log shows repeated test scoring, ledger overrides for extra test runs | Test data referenced in tuning code, selection among many runs by test score |

## User Stories

1. As a project owner, I want to audit a folder with one command, so that I do not have to list every file.
2. As a project owner, I want to see what the audit found before trusting its result, so that I can spot a misidentified file.
3. As a project owner, I want to correct the inventory, so that a mislabelled test file does not distort the findings.
4. As a project owner, I want a partial project to still get a report, so that missing files do not stop the audit.
5. As a project owner, I want "could not check" to be visibly different from "passed", so that silence never looks like safety.
6. As a project owner, I want a single plain verdict, so that I know whether to rely on the result.
7. As a project owner, I want no numeric score, so that a broken model cannot look almost fine.
8. As a project owner, I want one proven serious leak to decide the verdict, so that it cannot be averaged away.
9. As a project owner, I want every finding to carry a reproduction command, so that I can verify it myself.
10. As a project owner, I want proven findings separated from the agent's opinions, so that I know how much weight each deserves.
11. As a project owner, I want the audit to leave my files untouched, so that I can run it on anything without worry.
12. As a project owner, I want repeated audits kept side by side, so that I can see what changed.
13. As a project owner, I want the report free of raw data rows, so that I can commit or share it.
14. As a project owner, I want claimed scores recomputed from saved predictions, so that I learn whether a number is real.
15. As a project owner, I want scores compared with a trivial baseline, so that I see when a model adds nothing.
16. As a project owner, I want a claim in a README checked like any other claim, so that prose is not exempt.
17. As a project owner, I want rows shared between train and test detected, so that inflated scores are caught.
18. As a project owner, I want features that give away the answer flagged, so that I do not ship a model that cheats.
19. As a project owner, I want time-order violations detected, so that forecasts are not built on the future.
20. As a project owner, I want label conflicts and invalid labels reported, so that I know how far the targets can be trusted.
21. As a project owner, I want results for weak subgroups with their sample sizes, so that small-group noise is not mistaken for a problem.
22. As a project owner, I want to know whether predicted probabilities can be believed, so that I can act on them.
23. As a project owner, I want signs of repeated test-set use shown, so that I know the final number may be optimistic.
24. As a project owner, I want saved models left unopened by default, so that a malicious file cannot run on my machine.
25. As a project owner, I want a static risk scan of model files, so that I can decide which to trust.
26. As a project owner, I want a clean scan not to be described as a guarantee, so that I am not falsely reassured.
27. As a project owner, I want the audit to work without a pipeline ledger, so that older projects can be audited.
28. As a project owner, I want large datasets handled without exhausting memory, so that big projects can be audited.
29. As a project owner, I want to be told when a check used a sample, so that I know how strong the evidence is.
30. As a project owner, I want retrain-based checks to run only after I approve them, so that nothing trains without my say.
31. As a plugin maintainer, I want the guard exception to be narrow, so that the audit cannot become a route around the gates.
32. As a plugin maintainer, I want an audit not to change gates, approvals or the test-touch count, so that auditing a live project does not disturb it.
33. As a Codex or Kimi user, I want the audit available too, so that it is not limited to Claude Code.
34. As a plugin maintainer, I want every check proven against planted problems and clean controls, so that I know both its catch rate and its false alarms.

## Testing Decisions

A good test runs the audit on a folder and asserts on the findings and the verdict. It does not test private functions,
except where a metric or threshold needs both sides of its boundary covered.

- **Planted-problem fixtures, one per proven check.** Each folder has exactly one defect and must produce that finding and
  the expected verdict. Generated by code, like the existing fixtures, not checked-in data.
- **Clean controls.** A clean fixture must produce no critical or high finding. False alarms count as failures. Include a
  borderline clean case for each threshold.
- **Read-only proof.** Hash every file before and after the run. Only the audit folder may differ.
- **No execution proof.** A pickle whose load would create a marker file must leave no marker after an audit without trust.
- **Scanner fixtures.** Malicious-import pickles are flagged, ordinary pickles are not, and a corrupt file is "unverified".
- **Guard tests.** Extend the existing hook tests and the attack harness with: the exact tool invocation is allowed; a
  look-alike script, a training script that mentions audit, and a tool invocation chained to another command are denied.
- **Agent behaviour.** An eval case using the existing eval format: given a folder with a planted leak, the agent reports
  "not trustworthy", does not train, and does not edit project files.
- **Regression.** All existing tests and the attack harness still pass. Hook tests still run on a bare standard-library
  Python.

## Out of Scope

- Production drift, live monitoring, and fairness review (planned for version 2).
- Fixing anything, including suggested patches applied automatically.
- A numeric score or ranking.
- Scanning model weights for backdoors or poisoning.
- Sending project data to any outside service.

## Further Notes

- The build runs in a cloud session, which cannot see local projects. The first test on a real project is a separate step
  for the owner, and only then can real-world false alarms be measured.
- The installed plugin may be disabled on a given machine. The audit needs the release installed, but it does not change
  that setting.
