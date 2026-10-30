"""Tests for hooks/record_approval.py (the UserPromptSubmit hook): python3 -m pytest tests/test_record_approval.py"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import record_approval as hook  # noqa: E402

ASK_B = "Here is the Gate B report: ... Do you approve Gate B so I can start the baseline?"
ASK_OVERRIDE = ("Skipping to training risks leakage. Do you want to override Gate B and train now?")


class Parse(unittest.TestCase):
    def test_explicit_approvals(self):
        for prompt, gate in (("approve gate A", "A"), ("Gate B approved, go on", "B"), ("yes, go ahead with gate C", "C"),
                             ("LGTM gate a", "A"), ("gate D is signed off", "D")):
            with self.subTest(prompt=prompt):
                self.assertEqual(hook.parse(prompt)[0], [gate])

    def test_negated_is_not_approval(self):
        for prompt in ("don't approve gate B yet", "do not approve gate A", "not ready to approve gate C",
                       "wait, gate B approved? no", "I reject gate A"):
            with self.subTest(prompt=prompt):
                self.assertEqual(hook.parse(prompt)[0], [])

    def test_bare_yes_uses_the_gate_claude_asked_about(self):
        self.assertEqual(hook.parse("yes", ASK_B), (["B"], []))
        self.assertEqual(hook.parse("ok, proceed", ASK_B), (["B"], []))

    def test_bare_yes_without_a_question_is_nothing(self):
        self.assertEqual(hook.parse("yes", "Gate B report attached."), ([], []))
        self.assertEqual(hook.parse("yes"), ([], []))

    def test_questions_and_mentions_are_not_approvals(self):
        for prompt in ("Is gate B ok?", "ok, what does gate C require?", "Yes please explain gate B first",
                       "go ahead and fix the bug, then we'll talk about gate B",
                       "approval for gate B depends on the split, show me the figures"):
            with self.subTest(prompt=prompt):
                self.assertEqual(hook.parse(prompt), ([], []))

    def test_bare_yes_needs_an_approval_question(self):
        self.assertEqual(hook.parse("yes", "Steps 4-8 come next, then Gate B. Shall I start cleaning?"), ([], []))
        self.assertEqual(hook.parse("ok", "I won't override red flag group-split unless you say so. Want me to continue?"), ([], []))

    def test_long_or_negated_yes_is_nothing(self):
        self.assertEqual(hook.parse("yes but not yet, first check the outliers", ASK_B), ([], []))

    def test_overrides(self):
        self.assertEqual(hook.parse("override gate B")[1], ["gate b"])
        self.assertEqual(hook.parse("ok override red flag random-split-temporal")[1], ["red flag random-split-temporal"])
        self.assertEqual(hook.parse("override final test please")[1], ["final test"])
        self.assertEqual(hook.parse("yes", ASK_OVERRIDE), ([], ["gate b"]))


def run(prompt: str, transcript: list[dict] | None = None, pipeline: bool = True, ledger: str = "# Pipeline\n",
        record: dict | None = None) -> tuple[dict, dict | None]:
    with tempfile.TemporaryDirectory() as tmp:
        if pipeline:
            (Path(tmp) / "ml_pipeline").mkdir()
            (Path(tmp) / "ml_pipeline" / "PIPELINE.md").write_text(ledger)
            if record is not None:
                (Path(tmp) / "ml_pipeline" / ".approvals.json").write_text(json.dumps(record))
        tpath = Path(tmp) / "t.jsonl"
        tpath.write_text("\n".join(json.dumps(e) for e in (transcript or [])))
        payload = json.dumps({"prompt": prompt, "cwd": tmp, "transcript_path": str(tpath), "session_id": "s1"})
        out = io.StringIO()
        saved = sys.stdin
        os.environ.pop("ML_PIPELINE_FILE", None)
        sys.stdin = io.StringIO(payload)
        try:
            with contextlib.redirect_stdout(out):
                hook.main()
        finally:
            sys.stdin = saved
        record_path = Path(tmp) / "ml_pipeline" / ".approvals.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
    return json.loads(out.getvalue()), record


class Main(unittest.TestCase):
    def test_records_explicit_approval(self):
        result, record = run("approve gate A")
        self.assertIn("A", record["gates"])
        self.assertIn("Gate A: approved", result["hookSpecificOutput"]["additionalContext"])

    def test_reads_the_transcript_for_a_bare_yes(self):
        transcript = [{"type": "user", "message": {"content": "train it"}},
                      {"type": "assistant", "message": {"content": [{"type": "text", "text": ASK_B}]}}]
        _, record = run("yes", transcript)
        self.assertIn("B", record["gates"])

    def test_nothing_to_record(self):
        result, record = run("what does the outlier plot mean?")
        self.assertEqual(result, {})
        self.assertIsNone(record)

    def test_override_is_appended(self):
        _, record = run("override final test")
        self.assertEqual([o["what"] for o in record["overrides"]], ["final test"])

    def test_pilot_and_partial_data_overrides(self):
        self.assertEqual(hook.parse("override pilot, 5,000 rows is fine")[1], ["pilot"])
        self.assertEqual(hook.parse("ok, override partial data for this demo")[1], ["partial data"])
        self.assertEqual(hook.parse("don't override pilot")[1], [])

    def test_approvals_belong_to_the_run_they_were_given_in(self):
        _, record = run("approve gate A")
        self.assertEqual(record["gates"]["A"]["run"], "")
        ledger = "# Pipeline\n- Gate A: approved 2026-09-16\n## Run: retrain on new data (2026-10-01)\n"
        _, record = run("approve gate A", ledger=ledger, record=record)
        self.assertEqual(record["gates"]["A"]["run"], "## Run: retrain on new data (2026-10-01)")
        _, again = run("approve gate A", ledger=ledger, record=record)
        self.assertEqual(again["gates"]["A"]["at"], record["gates"]["A"]["at"])  # the first approval in a run is kept

    def test_bare_yes_outside_a_pipeline_is_ignored(self):
        transcript = [{"type": "assistant", "message": {"content": [{"type": "text", "text": ASK_B}]}}]
        result, record = run("yes", transcript, pipeline=False)
        self.assertEqual(result, {})
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
