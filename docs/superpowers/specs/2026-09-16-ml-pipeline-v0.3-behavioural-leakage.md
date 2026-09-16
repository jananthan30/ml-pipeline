# ml-pipeline v0.3.0 — Behavioural leakage enforcement + proof

**Status:** approved by the user 2026-09-16 ("put it in a plan md todo file and work on it one by one").

## Why (research basis)

A 2026 landscape study across 2,047 datasets (arXiv 2604.04199) ranks leakage by how much it
inflates reported results:

| Class | Example | Effect |
|---|---|---|
| Selection | peeking at the test set, seed cherry-picking | worst — ~90% noise exploitation |
| Memorization | duplicate rows across splits | substantial, grows with model capacity |
| Boundary | random split on temporal data | invisible under random CV |
| Estimation | fitting a scaler on all rows | negligible — \|ΔAUC\| ≤ 0.005 |

v0.2.0's hook polices `.fit()` (the negligible class). v0.3.0 re-aims enforcement at the
behavioural classes, adds a runtime guard that can see what static checks cannot (row overlap,
temporal order, how many times the test set was touched), and ships a **canary dataset** with a
known accuracy ceiling (CapCode, arXiv 2606.07379): a result above the ceiling proves leakage
without judgement. A `claude plugin eval` case proves the gate discipline holds.

## Scope

**In:** (1) hook denies evaluating on the test split before Gate C; (2) `lib/mlpipeline_guard.py`
with `split()` (dedupe, chronological/group/stratified, overlap check, fingerprint freeze) and
`final_test()` (single touch, tamper check, PIPELINE.md log); (3) `lib/mlpipeline_canary.py`
(stdlib) + `examples/canary/`; (4) SessionStart hook that tells the agent where the helpers are;
(5) PostToolUse hook that warns on too-good metrics and unsafe `train_test_split` calls;
(6) skill/README updates incl. `git tag gate-X` checkpoints; (7) one eval case `gate-stop`;
(8) release 0.3.0.

**Out (backlog, separate plans):** independent evaluator agent + hacker-fixer hardening;
trying `mlw` (PyPI 1.1.2) as a runtime boundary; bounded search module for steps 10–11;
per-phase compute budgets; a `canary-honesty` eval case (needs pandas inside the eval sandbox).

## Global constraints

- Hooks: Python ≥ 3.10, **stdlib only**, no `subprocess`/`eval`/`exec`, fail open, ~30 ms.
- Guard library: pandas + numpy only (it is copied into the user's project). Canary: stdlib only.
- Gate lines stay `- Gate X: approved YYYY-MM-DD`; override lines start `- Override: <what> - `.
- Codex/Kimi keep working as instructions; `install-other-tools.sh` also copies `lib/`.
- HOL scanner stays ≥ 98/100 with zero critical/high on the PR.
- Versions bumped in both `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`.
- Dev env: `uv venv .venv && uv pip install --python .venv/bin/python pandas numpy` — `.venv/`
  and `evals/results/` are gitignored. Hook tests: `python3 -m unittest discover tests`.
  Guard tests: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`.
- Commits: conventional style, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
