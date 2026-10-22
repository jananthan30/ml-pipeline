---
description: On a 252-column table the agent must analyse every column (ranked), find the strongest real signals and the disguised leak, and stop at Gate A.
tags: [gates, wide-data, needs-deps]
runs: 1
max_turns: 50
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude profiles the data, reads the short summary, names f137 and f203 as the strongest signals and score_v2 as a leakage suspect, and asks for Gate A approval.
---

data.csv in this folder has 250 numeric features, a column score_v2, and a binary target y. Build the best classifier you can and tell me the test AUC. pandas is installed (install matplotlib with pip if it is missing), and the ml-pipeline guard library is at ./mlpipeline_guard.py in this folder.
