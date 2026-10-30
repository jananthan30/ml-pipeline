"""Ticket 08: signs of test-set misuse - proven from logs, suspected from code.

Run:  python -m pytest tests/test_audit_test_misuse.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402


def misuse(f: dict) -> list[dict]:
    return [x for x in f["findings"] if x["family"] == "test-misuse"]


class TestMisuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeated_final_tests_in_the_ledger_are_proven(self):
        _, _, f = fx.run_audit(fx.repeated_test_ledger(self.root / "p"))
        hits = misuse(f)
        self.assertEqual([h["code"] for h in hits], ["repeated-test-scoring"])
        self.assertEqual((hits[0]["severity"], hits[0]["evidence"]), ("high", "proven"))
        self.assertEqual(hits[0]["counts"]["ledger_final_tests"], 3)
        self.assertEqual(hits[0]["counts"]["final_test_overrides"], 2)
        self.assertEqual(fx.find(f, "claim-mismatch"), [])

    def test_a_run_log_with_repeated_test_scores_is_proven(self):
        _, _, f = fx.run_audit(fx.repeated_test_runs(self.root / "p"))
        hits = misuse(f)
        self.assertEqual([h["code"] for h in hits], ["repeated-test-scoring"])
        self.assertEqual(hits[0]["counts"]["runs_with_test_metrics"], 4)

    def test_test_data_in_tuning_code_is_suspected(self):
        _, _, f = fx.run_audit(fx.test_in_tuning(self.root / "p"))
        hits = misuse(f)
        self.assertEqual([(h["code"], h["evidence"]) for h in hits], [("test-in-tuning", "suspected")])
        self.assertEqual(hits[0]["files"], ["tune.py"])
        self.assertIn("tune.py:9", hits[0]["source"])
        self.assertEqual(f["verdict"], "trustworthy")  # a suspected high does not decide the verdict

    def test_no_run_log_or_ledger_is_could_not_check(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"))
        check = next(c for c in f["checks"] if c["id"] == "test-use-log")
        self.assertEqual(check["status"], "could not check")
        self.assertIn("no run log or ledger", check["reason"])
        self.assertEqual(misuse(f), [])

    def test_a_single_honest_final_test_gives_no_finding(self):
        _, _, f = fx.run_audit(fx.honest_final_test(self.root / "p"))
        self.assertEqual(misuse(f), [])
        check = next(c for c in f["checks"] if c["id"] == "test-use-log")
        self.assertEqual(check["status"], "ran")


if __name__ == "__main__":
    unittest.main()
