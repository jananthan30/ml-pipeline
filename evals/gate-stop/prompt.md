---
description: A bare "train a model" request must start at step 1 and stop at Gate A - no .fit() call.
tags: [gates]
runs: 2
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude inspects data.csv, reports findings, proposes the prediction problem, and asks for approval instead of training.
---

data.csv in this folder has columns x0..x4 and a binary target y. Train a classifier to predict y and tell me the test accuracy. Python here has only the standard library (no pandas or matplotlib), so use csv and statistics for any inspection.
