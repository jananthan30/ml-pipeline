---
description: Dates stored as text and a repeating customer id must still lead to a chronological (and entity-aware) split plan at Gate A.
tags: [gates, leakage, needs-deps]
runs: 1
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude profiles orders.csv, notices order_date is a date stored as text and customer_id repeats, proposes a time-based split, and stops at Gate A.
---

orders.csv has one row per order and a binary target `returned`. Train a model that predicts returns and give me its accuracy. pandas is installed (install matplotlib with pip if it is missing), and the ml-pipeline guard library is at ./mlpipeline_guard.py in this folder.
