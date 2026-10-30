"""Ticket 03: data problems, reported with column names and counts only.

Run:  python -m pytest tests/test_audit_data.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

DATA_CODES = {"missing-values", "duplicate-rows", "constant-column", "conflicting-labels", "invalid-labels"}


def data_codes(f: dict) -> set[str]:
    return {x["code"] for x in f["findings"] if x["family"] == "data-problems"}


class DataProblems(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_planted_defect_gives_exactly_its_finding(self):
        cases = [(fx.missing_values, "missing-values", "medium", {"x2": 360}),
                 (fx.duplicate_rows, "duplicate-rows", "medium", {"duplicate_rows": 60}),
                 (fx.constant_column, "constant-column", "info", None),
                 (fx.conflicting_labels, "conflicting-labels", "medium", {"conflicting_rows": 40}),
                 (fx.invalid_labels, "invalid-labels", "high", {"missing_label": 25})]
        for build, code, severity, counts in cases:
            with self.subTest(code=code):
                _, _, f = fx.run_audit(build(self.root / code))
                self.assertEqual(data_codes(f), {code})
                hit = fx.find(f, code)[0]
                self.assertEqual((hit["severity"], hit["evidence"]), (severity, "proven"))
                self.assertEqual(hit["files"], ["data/train.csv"])
                for key, value in (counts or {}).items():
                    self.assertEqual(hit["counts"][key], value)

    def test_the_clean_control_has_no_data_findings_and_nothing_serious(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "clean"))
        self.assertEqual(data_codes(f), set())
        self.assertEqual(fx.serious(f), [])
        check = next(c for c in f["checks"] if c["id"] == "data-problems")
        self.assertEqual(check["status"], "ran")

    def test_missing_value_threshold_on_both_sides(self):
        _, _, below = fx.run_audit(fx.missing_values(self.root / "below", frac=0.19))
        _, _, above = fx.run_audit(fx.missing_values(self.root / "above", frac=0.21))
        self.assertEqual(fx.find(below, "missing-values")[0]["severity"], "info")
        self.assertEqual(fx.find(above, "missing-values")[0]["severity"], "medium")

    def test_label_conflict_threshold_on_both_sides(self):
        # 1,200 rows plus n flipped twins: 2n conflicting rows; high from 5%
        _, _, below = fx.run_audit(fx.conflicting_labels(self.root / "below", twins_n=30))  # 60 / 1,230 = 4.9%
        _, _, above = fx.run_audit(fx.conflicting_labels(self.root / "above", twins_n=32))  # 64 / 1,232 = 5.2%
        self.assertEqual(fx.find(below, "conflicting-labels")[0]["severity"], "medium")
        self.assertEqual(fx.find(above, "conflicting-labels")[0]["severity"], "high")

    def test_a_large_file_is_streamed_and_counts_stay_exact(self):
        proj = fx.large_file(self.root / "big")
        _, _, f = fx.run_audit(proj, "--sample-rows", "1000")
        hit = fx.find(f, "missing-values")[0]
        self.assertEqual(hit["counts"]["x3"], 1234)  # exact over all 6,000 rows, read in batches of 1,000
        self.assertIsNone(hit["sampled_rows"])
        check = next(c for c in f["checks"] if c["id"] == "data-problems")
        self.assertIn("every row", check["reason"])
        self.assertIn("batches", check["reason"])
        sampled = [c for c in f["checks"] if c.get("sampled")]
        for c in sampled:
            self.assertLessEqual(c["sampled"], 1000)
            self.assertIn("sample", c["reason"])

    def test_no_raw_row_values_reach_the_outputs(self):
        proj = fx.unique_secrets(self.root / "secret")
        _, out, f = fx.run_audit(proj)
        self.assertTrue({"missing-values", "duplicate-rows"} <= data_codes(f))
        for name in ("report.md", "findings.json", "inventory.json"):
            text = (out / name).read_text()
            self.assertNotIn("SECRET-ROW", text, name)
            for value in ("0.123456", "north", "south"):  # no numbers or levels from rows either
                self.assertNotIn(f'"{value}"', text, name)


if __name__ == "__main__":
    unittest.main()
