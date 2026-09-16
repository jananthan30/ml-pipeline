"""Tests for hooks/post_tool_warn.py: python3 -m unittest tests/test_post_tool_warn.py"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import post_tool_warn as hook  # noqa: E402


@contextlib.contextmanager
def _stdin(text: str):
    old, sys.stdin = sys.stdin, io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = old


class Warnings(unittest.TestCase):
    def test_too_good_metric(self):
        out = hook.warnings("", "Test accuracy: 0.995\n")
        self.assertEqual(len(out), 1)
        self.assertIn("too good", out[0])

    def test_perfect_auc_percent(self):
        self.assertTrue(hook.warnings("", "AUC = 100%"))

    def test_honest_metric_is_quiet(self):
        self.assertEqual(hook.warnings("", "accuracy: 0.81, auc 0.87"), [])

    def test_split_without_stratify(self):
        out = hook.warnings("X_tr, X_te = train_test_split(X, y, test_size=0.2)", "")
        self.assertTrue(any("stratify" in w for w in out))

    def test_split_on_dated_data(self):
        code = "df['order_date'] = pd.to_datetime(df.order_date)\ntrain_test_split(X, y, stratify=y)"
        out = hook.warnings(code, "")
        self.assertTrue(any("chronological" in w for w in out))
        self.assertFalse(any("stratify=" in w for w in out))

    def test_safe_split_is_quiet(self):
        self.assertEqual(hook.warnings("train_test_split(X, y, stratify=y)", ""), [])


class MainRoundTrip(unittest.TestCase):
    def _run(self, payload) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), _stdin(payload):
            hook.main()
        return json.loads(out.getvalue())

    def test_warns_in_hook_json(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "python eval.py"},
                              "tool_response": "f1: 0.99"})
        result = self._run(payload)
        self.assertEqual(result["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        self.assertIn("ml-pipeline:", result["hookSpecificOutput"]["additionalContext"])

    def test_quiet_is_empty_object(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": "a\nb"})
        self.assertEqual(self._run(payload), {})

    def test_fails_open(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), _stdin("nope"):
            hook.main()
        self.assertIn("hook error", json.loads(out.getvalue())["systemMessage"])


if __name__ == "__main__":
    unittest.main()
