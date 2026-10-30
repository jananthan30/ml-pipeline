---
description: An audit request runs the audit skill, not the 16-step pipeline, and trains nothing.
tags: [audit]
runs: 2
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude runs the ml-audit tool on ./proj and reports its verdict; the ml-pipeline skill does not start.
---

Someone handed me the project in ./proj. Audit it: can I trust the test accuracy it reports?
