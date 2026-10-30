"""Done-condition 1: every planted problem gives its finding and verdict; no clean control raises a critical or high.

Run:  python -m pytest tests/test_audit_table.py      (the table itself: python tests/audit_fixture_table.py)
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixture_table as table  # noqa: E402


class FixtureTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.sklearn = table._sklearn()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_planted_problems(self):
        for name, build, args, code, sev, ev, verdict, needs in table.PLANTED:
            with self.subTest(case=name):
                if needs and not self.sklearn:
                    self.skipTest("scikit-learn not installed")
                f = table.run_case(self.root, name, build, args)
                ok, got = table.judge_planted(f, code, sev, ev, verdict)
                self.assertTrue(ok, f"{name}: {got}; verdict {f['verdict']} ({f['verdict_reason']})")

    def test_clean_controls(self):
        for name, build, args, verdict, needs in table.CLEAN:
            with self.subTest(case=name):
                if needs and not self.sklearn:
                    self.skipTest("scikit-learn not installed")
                f = table.run_case(self.root, name, build, args)
                ok, got = table.judge_clean(f, verdict)
                self.assertTrue(ok, f"{name}: {got}; verdict {f['verdict']} ({f['verdict_reason']})")


if __name__ == "__main__":
    unittest.main()
