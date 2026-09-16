"""Tests for hooks/session_start.py: python3 -m unittest tests/test_session_start.py"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import session_start  # noqa: E402


def run(env: dict[str, str]) -> dict:
    saved = dict(os.environ)
    os.environ.pop("CLAUDE_PLUGIN_ROOT", None)
    os.environ.update(env)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            session_start.main()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return json.loads(out.getvalue())


class SessionStart(unittest.TestCase):
    def test_names_the_helpers_when_plugin_root_is_set(self):
        result = run({"CLAUDE_PLUGIN_ROOT": "/plug"})
        context = result["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(result["hookSpecificOutput"]["hookEventName"], "SessionStart")
        self.assertIn("/plug/lib/mlpipeline_guard.py", context)
        self.assertIn("/plug/lib/mlpipeline_canary.py", context)
        self.assertIn("ml_pipeline/guard.py", context)

    def test_silent_without_plugin_root(self):
        self.assertEqual(run({}), {})


if __name__ == "__main__":
    unittest.main()
