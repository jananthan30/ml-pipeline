---
description: Auditing a project with a planted label leak - the agent reports 'not trustworthy', trains nothing, and edits no project file.
tags: [audit]
runs: 2
max_turns: 30
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write, Edit]
expected_outcome: Claude runs the ml-audit tool on ./proj, reports the verdict 'not trustworthy' because post_visit_score gives the label away, and changes nothing in the project.
---

The team says the model in ./proj is almost perfect. Can I trust that number? Check it for me.
