# ml-pipeline

**Stop your coding agent from jumping straight to model training.**

Coding agents (Claude Code, Codex, Kimi, …) love to call `.fit()` five minutes into an ML
task — before they've looked at the data, before there's a test set, before anyone agreed
on what is actually being predicted. The result is leakage, garbage features, and scores
that don't survive contact with reality.

This plugin forces a strict, data-first 16-step pipeline with explicit user-permission
gates:

```
RAW DATA
  → 1 Data inspection → 2 EDA → 3 Define prediction problem   ── GATE A 🛑
  → 4 Cleaning → 5 Engineering → 6 Train/val/test split
  → 7 Feature engineering → 8 Preprocessing                    ── GATE B 🛑
  → 9 Baseline → 10 Training → 11 Tuning → 12 Evaluation       ── GATE C 🛑
  → 13 Error analysis → 14 Final test (touched ONCE)
  → 15 Deployment → 16 Monitoring + retraining                 ── GATE D 🛑
```

At every gate the agent stops, explains **in plain language** what it did, what it found
(with figures), and what comes next — and waits for your explicit permission.

## What you get

- **Strict ordering** — the split happens *before* feature engineering and preprocessing;
  all fitting uses training data only; the test set is touched exactly once.
- **Visuals at every data-facing step** — [marimo](https://marimo.io) notebooks as the
  workbench, matplotlib figures saved to `ml_pipeline/figures/`, each explained in 1–2
  plain sentences.
- **Resumable progress** — a `ml_pipeline/PIPELINE.md` checklist in your project tracks
  every step and gate approval, so a new session resumes instead of restarting.
- **Override with consent** — "just train it" gets a short explanation of the risk and a
  single explicit-override confirmation, recorded in the checklist.

## Install (Claude Code)

```
/plugin marketplace add jananthan30/ml-pipeline
/plugin install ml-pipeline@ml-pipeline
```

The skill then activates automatically on any ML training/tuning/evaluation task, or
invoke it directly with `/ml-pipeline`.

## Install (Codex / Kimi CLI)

Both tools read `SKILL.md`-style skills and global `AGENTS.md` instructions:

```bash
./install-other-tools.sh
```

This copies the skill into `~/.codex/skills/` and `~/.kimi-code/skills/` and appends an
enforcement section to each tool's global `AGENTS.md`.

## License

MIT
