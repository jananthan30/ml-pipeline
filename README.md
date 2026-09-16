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
directly with `/ml-pipeline`.

**Codex CLI**

```
codex plugin marketplace add jananthan30/ml-pipeline
codex plugin add ml-pipeline@ml-pipeline
```

**Kimi Code CLI** (or Codex without plugins)

```bash
git clone https://github.com/jananthan30/ml-pipeline && cd ml-pipeline && ./install-other-tools.sh
```

Copies the skill into `~/.codex/skills/` and `~/.kimi-code/skills/` and appends an
enforcement section to each tool's global `AGENTS.md`.

## How it works

- **Strict order, no skipping** — the split happens *before* feature engineering and
  preprocessing; all fitting uses training data only; the test set is touched exactly once.
- **You stay in control** — at each gate the agent explains, in plain language, what it
  did, what it found (with figures), and what comes next, then waits for your explicit OK.
  "Just train it" gets the risk explained and requires a logged override.
- **See everything** — [marimo](https://marimo.io) notebooks as the workbench, matplotlib
  figures saved to `ml_pipeline/figures/`, each explained in 1–2 plain sentences.
- **Resumable** — `ml_pipeline/PIPELINE.md` tracks every step and approval, so a new
  session continues where the last one stopped.

## Enforced, not just instructed

In Claude Code the plugin ships a `PreToolUse` hook. Before any command, file write, or
notebook edit runs, it scans the code for training calls and **denies** them unless
`ml_pipeline/PIPELINE.md` records the gate that unlocks them: nothing is fit before Gate A,
no model is trained before Gate B, and the test split is never fit on. A denied call tells
the agent exactly which steps are missing — so "just train it" fails closed until you have
approved the gate. `ML_PIPELINE_ENFORCE=0` switches it off for non-ML projects. Codex and
Kimi get the same rules as instructions (no hook support there).

## License

Apache-2.0
