"""Ticket 04: leakage - rows and entities shared across splits, features that give the target away, time order.

Run:  python -m pytest tests/test_audit_leakage.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402


def leak_codes(f: dict) -> set[str]:
    return {x["code"] for x in f["findings"] if x["family"] == "leakage"}


class Leakage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_rows_across_splits_are_proven_critical(self):
        _, _, f = fx.run_audit(fx.split_overlap(self.root / "p"))
        self.assertEqual(leak_codes(f), {"split-overlap"})
        hit = fx.find(f, "split-overlap")[0]
        self.assertEqual((hit["severity"], hit["evidence"]), ("critical", "proven"))
        self.assertEqual(hit["counts"]["shared_rows"], 30)
        self.assertEqual(f["verdict"], "not trustworthy")

    def test_overlap_threshold_on_both_sides(self):
        # 600 test rows; critical from 1% (6 rows)
        _, _, below = fx.run_audit(fx.split_overlap(self.root / "below", copies=5))
        _, _, above = fx.run_audit(fx.split_overlap(self.root / "above", copies=7))
        self.assertEqual(fx.find(below, "split-overlap")[0]["severity"], "high")
        self.assertEqual(fx.find(above, "split-overlap")[0]["severity"], "critical")

    def test_shared_entities_are_flagged_when_an_entity_column_is_known(self):
        _, out, f = fx.run_audit(fx.shared_entities(self.root / "p"))
        self.assertEqual(leak_codes(f), {"entity-overlap"})
        hit = fx.find(f, "entity-overlap")[0]
        self.assertEqual(hit["columns"], ["patient_id"])
        self.assertEqual(hit["evidence"], "proven")
        self.assertGreater(hit["counts"]["shared_entities"], 100)

    def test_a_planted_near_perfect_feature_is_flagged(self):
        _, _, f = fx.run_audit(fx.target_leak(self.root / "p"))
        self.assertEqual(leak_codes(f), {"target-leak-feature"})
        hit = fx.find(f, "target-leak-feature")[0]
        self.assertEqual((hit["severity"], hit["evidence"], hit["columns"]), ("critical", "proven", ["post_visit_score"]))
        self.assertEqual(f["verdict"], "not trustworthy")

    def test_association_threshold_on_both_sides(self):
        _, _, below = fx.run_audit(fx.flipped_copy_feature(self.root / "below", 0.94))
        _, _, above = fx.run_audit(fx.flipped_copy_feature(self.root / "above", 0.96))
        self.assertEqual(fx.find(below, "target-leak-feature"), [])
        self.assertEqual(fx.find(above, "target-leak-feature")[0]["columns"], ["flag"])

    def test_a_strong_honest_feature_is_not_flagged(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"))
        self.assertEqual(fx.find(f, "target-leak-feature"), [])  # x0 drives the label, but not alone

    def test_time_order_only_with_a_time_column(self):
        _, _, bad = fx.run_audit(fx.time_order(self.root / "bad"))
        self.assertEqual(leak_codes(bad), {"time-order"})
        self.assertEqual(fx.find(bad, "time-order")[0]["columns"], ["event_date"])
        _, _, good = fx.run_audit(fx.time_order(self.root / "good", violate=False))
        self.assertEqual(leak_codes(good), set())
        _, _, none = fx.run_audit(fx.clean_project(self.root / "none"))
        check = next(c for c in none["checks"] if c["id"] == "time-order")
        self.assertEqual(check["status"], "could not check")
        self.assertIn("no time column", check["reason"])

    def test_a_fully_clean_project_is_trustworthy_and_lists_what_was_not_checked(self):
        _, out, f = fx.run_audit(fx.clean_project(self.root / "p"))
        self.assertEqual(f["verdict"], "trustworthy", f["verdict_reason"])
        self.assertEqual(fx.serious(f), [])
        self.assertTrue(all(c["status"] == "ran" for c in f["checks"] if c["required"]))
        report = (out / "report.md").read_text()
        self.assertIn("**Verdict: trustworthy**", report)
        self.assertIn("could not check", report)  # e.g. time order: there is no time column
        self.assertIn("covers only what was checked", report)

    def test_the_association_check_states_its_sample(self):
        _, _, f = fx.run_audit(fx.large_leak(self.root / "big"), "--sample-rows", "1000")
        check = next(c for c in f["checks"] if c["id"] == "feature-target-leakage")
        self.assertEqual(check["sampled"], 1000)
        hit = fx.find(f, "target-leak-feature")[0]
        self.assertEqual(hit["sampled_rows"], 1000)
        self.assertIn("sample of 1,000", hit["detail"])


if __name__ == "__main__":
    unittest.main()
