"""Ticket 07: weak subgroups with their size and interval, and calibration of predicted probabilities.

Run:  python -m pytest tests/test_audit_weak_spots.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402


class WeakSpots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_planted_weak_subgroup_is_found_with_its_size_and_interval(self):
        _, out, f = fx.run_audit(fx.weak_subgroup(self.root / "p"))
        hits = fx.find(f, "weak-subgroup")
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual((hit["severity"], hit["evidence"], hit["columns"]), ("medium", "proven", ["region"]))
        self.assertGreaterEqual(hit["counts"]["group_rows"], 100)
        low, high = hit["counts"]["gap_interval_95"]
        self.assertGreater(low, 0)
        self.assertGreater(high, low)
        # the level is named by its frequency rank; its value is only in a local-only file
        for name in ("report.md", "findings.json"):
            self.assertNotIn("west", (out / name).read_text())
        levels = __import__("json").loads((out / "local_only" / "subgroup_levels.json").read_text())
        self.assertEqual(levels["region"][str(hit["counts"]["level_rank"])], "west")
        self.assertIn("local_only", (out / "report.md").read_text())
        self.assertEqual(f["verdict"], "trustworthy")  # a weak spot is reported, it does not make the score false

    def test_groups_under_the_minimum_size_are_not_reported(self):
        _, _, small = fx.run_audit(fx.small_weak_group(self.root / "small", 29))
        _, _, big = fx.run_audit(fx.small_weak_group(self.root / "big", 31))
        # only 'site' is the subject here (the forced errors also shift other columns' levels a little)
        site = lambda f: [h for h in fx.find(f, "weak-subgroup") if h["columns"] == ["site"]]  # noqa: E731
        self.assertEqual(site(small), [])
        self.assertEqual([h["counts"]["group_rows"] for h in site(big)], [31])
        check = next(c for c in small["checks"] if c["id"] == "subgroups")
        self.assertIn("fewer than 30", check["reason"])

    def test_the_clean_control_has_no_weak_spot(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"))
        self.assertEqual(fx.find(f, "weak-subgroup"), [])
        self.assertEqual(fx.find(f, "miscalibrated"), [])
        self.assertEqual(next(c for c in f["checks"] if c["id"] == "calibration")["status"], "ran")

    def test_a_miscalibrated_model_is_flagged(self):
        _, _, f = fx.run_audit(fx.miscalibrated(self.root / "p"))
        hit = fx.find(f, "miscalibrated")[0]
        self.assertEqual((hit["severity"], hit["evidence"]), ("medium", "proven"))
        self.assertGreater(hit["counts"]["expected_calibration_error"], 0.1)

    def test_calibration_threshold_on_both_sides(self):
        _, _, below = fx.run_audit(fx.constant_probability(self.root / "below", 0.09))
        _, _, above = fx.run_audit(fx.constant_probability(self.root / "above", 0.11))
        self.assertEqual(fx.find(below, "miscalibrated"), [])
        self.assertAlmostEqual(fx.find(above, "miscalibrated")[0]["counts"]["expected_calibration_error"], 0.11, places=6)

    def test_regression_skips_calibration_with_a_note(self):
        _, _, f = fx.run_audit(fx.regression_project(self.root / "p"))
        check = next(c for c in f["checks"] if c["id"] == "calibration")
        self.assertEqual(check["status"], "not applicable")
        self.assertIn("regression", check["reason"])
        self.assertEqual(f["verdict"], "trustworthy", f["verdict_reason"])
        self.assertEqual(next(c for c in f["checks"] if c["id"] == "subgroups")["status"], "ran")


if __name__ == "__main__":
    unittest.main()
