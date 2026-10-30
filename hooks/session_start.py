#!/usr/bin/env python3
"""SessionStart hook: tell the agent where the plugin's runtime helpers live and how gates are approved.

The skill text cannot know the plugin's install path, so this adds it to the session context once
(~150 tokens). Stdlib only; prints ``{}`` when not running as a plugin.
"""
from __future__ import annotations

import json
import os
import shutil


def main() -> int:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if not root:
        print("{}")
        return 0
    lib = os.path.join(root, "skills", "ml-pipeline", "lib")
    marimo = "marimo is installed" if shutil.which("marimo") else (
        "marimo is NOT on PATH - install it (pip install marimo) before writing notebooks, or write plain .py cells and say so")
    context = (
        "ml-pipeline runtime helpers: guard library at "
        f"{os.path.join(lib, 'mlpipeline_guard.py')} - copy it with cp to ml_pipeline/guard.py at step 1 and use "
        "guard.profile() (reads files of any size; read ml_pipeline/data_profile_summary.md, not the full JSON) / "
        "guard.load() / guard.data_dictionary() / guard.missingness() / guard.label_audit() / "
        "guard.eda_figures() / guard.variable_figures() / guard.fig() / guard.log_cleaning() / "
        "guard.split() or guard.split_to_files() / guard.train_associations() / guard.cv_splits() / guard.fit() / "
        "guard.log_run() / guard.importance() / guard.final_test() / guard.manifest() for images and text; "
        "model-selection playbook at "
        f"{os.path.join(root, 'skills', 'ml-pipeline', 'references', 'model-selection.md')} (read at step 3 and before "
        f"Gate B); canary dataset generator at {os.path.join(lib, 'mlpipeline_canary.py')} (honest accuracy ceiling 0.80). "
        "Gates count only when the USER approves them in chat (their message is recorded automatically); record each "
        f"finished step as '- Step N done: <result>'. {marimo}."
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
