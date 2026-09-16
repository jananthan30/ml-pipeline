#!/usr/bin/env python3
"""SessionStart hook: tell the agent where the plugin's runtime helpers live.

The skill text cannot know the plugin's install path, so this adds it to the session context
once (~60 tokens). Stdlib only; prints ``{}`` when not running as a plugin.
"""
from __future__ import annotations

import json
import os


def main() -> int:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if not root:
        print("{}")
        return 0
    lib = os.path.join(root, "lib")
    context = (
        "ml-pipeline runtime helpers: guard library at "
        f"{os.path.join(lib, 'mlpipeline_guard.py')} - copy it to ml_pipeline/guard.py at step 6, not before, and use "
        "guard.split() / guard.final_test(); canary dataset generator at "
        f"{os.path.join(lib, 'mlpipeline_canary.py')} (honest accuracy ceiling 0.80; above it means leakage)."
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
