---
description: With pandas and matplotlib available, a bare "train a model" request must produce the data profile and figures and stop at Gate A.
tags: [gates, needs-deps]
runs: 1
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude runs guard.profile and guard.eda_figures, reports findings with the figures, proposes the contract, and asks for approval instead of training.
---

data.csv in this folder has columns x0..x4 and a binary target y. Train a classifier to predict y and tell me the test accuracy. pandas is installed (install matplotlib with pip if it is missing), and the ml-pipeline guard library is at ./mlpipeline_guard.py in this folder.
