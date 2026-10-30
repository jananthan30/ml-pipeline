"""Ticket 02: claimed scores are recomputed from saved predictions (no scikit-learn in the tool).

Run:  python -m pytest tests/test_audit_scores.py
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

spec = importlib.util.spec_from_file_location("mlpipeline_audit", fx.TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class Metrics(unittest.TestCase):
    """Implemented with numpy only; checked against hand-computed values and, when installed, scikit-learn."""

    y = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.4, 0.35, 0.8, 0.7, 0.6, 0.2, 0.65, 0.4])

    def test_known_values(self):
        self.assertAlmostEqual(tool.metric_value("accuracy", self.y, self.p), 0.7)
        # tp=4 fp=1 fn=2 -> precision 0.8, recall 4/6, f1 = 8/11
        self.assertAlmostEqual(tool.metric_value("f1", self.y, self.p, positive=1), 8 / 11)
        # ties in scores (0.4 twice: one positive, one positive) and pairs counted by hand: 22 of 24 -> 0.9166..
        self.assertAlmostEqual(tool.metric_value("auc", self.y, None, scores=self.s, positive=1), 22 / 24)
        self.assertAlmostEqual(tool.metric_value("rmse", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])), (4 / 3) ** 0.5)
        self.assertAlmostEqual(tool.metric_value("mae", np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])), 1.0)

    def test_auc_ties_count_half(self):
        self.assertAlmostEqual(tool.metric_value("auc", np.array([0, 1]), None, scores=np.array([0.5, 0.5]), positive=1), 0.5)

    def test_macro_f1_for_three_classes(self):
        y = np.array(["a", "a", "b", "b", "c", "c"])
        p = np.array(["a", "b", "b", "b", "c", "a"])
        # per class f1: a 0.5, b 0.8, c 2/3
        self.assertAlmostEqual(tool.metric_value("f1", y, p), (0.5 + 0.8 + 2 / 3) / 3)

    def test_against_scikit_learn_when_installed(self):
        try:
            from sklearn import metrics as m
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 500)
        s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1).round(2)
        p = (s > 0.5).astype(int)
        self.assertAlmostEqual(tool.metric_value("accuracy", y, p), m.accuracy_score(y, p))
        self.assertAlmostEqual(tool.metric_value("f1", y, p, positive=1), m.f1_score(y, p))
        self.assertAlmostEqual(tool.metric_value("auc", y, None, scores=s, positive=1), m.roc_auc_score(y, s))
        yr, pr = rng.normal(size=300), rng.normal(size=300)
        self.assertAlmostEqual(tool.metric_value("rmse", yr, pr), m.mean_squared_error(yr, pr) ** 0.5)
        self.assertAlmostEqual(tool.metric_value("mae", yr, pr), m.mean_absolute_error(yr, pr))

    def test_tolerance_follows_the_stated_precision(self):
        self.assertAlmostEqual(tool.claim_tolerance("0.93"), 0.005 + tool.SCORE_FLOAT_SLACK)
        self.assertAlmostEqual(tool.claim_tolerance("0.930"), 0.0005 + tool.SCORE_FLOAT_SLACK)
        self.assertAlmostEqual(tool.claim_tolerance("93"), 0.5 + tool.SCORE_FLOAT_SLACK)


class Claims(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_wrong_claim_is_proven_critical_and_not_trustworthy(self):
        _, _, f = fx.run_audit(fx.wrong_claim(self.root / "p"))
        hits = fx.find(f, "claim-mismatch")
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0]["severity"], hits[0]["evidence"]), ("critical", "proven"))
        self.assertEqual(f["verdict"], "not trustworthy")
        self.assertIn("outputs/metrics.json", hits[0]["source"])

    def test_the_correct_claim_gives_no_finding(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"))
        self.assertEqual(fx.find(f, "claim-mismatch"), [])
        check = next(c for c in f["checks"] if c["id"] == "score-recomputation")
        self.assertEqual(check["status"], "ran")

    def test_rounding_tolerance_on_both_sides(self):
        # 0.93 stated with 2 decimals covers [0.925, 0.935]
        for correct, expect in ((9349, False), (9351, True), (9251, False), (9249, True)):
            with self.subTest(accuracy=correct / 10000):
                _, _, f = fx.run_audit(fx.exact_accuracy_project(self.root / f"p{correct}", correct, 10000, "0.93"))
                self.assertEqual(bool(fx.find(f, "claim-mismatch")), expect)

    def test_trailing_zero_tightens_the_tolerance(self):
        _, _, f = fx.run_audit(fx.exact_accuracy_project(self.root / "p", 9340, 10000, "0.930"))
        self.assertTrue(fx.find(f, "claim-mismatch"))
        _, _, f = fx.run_audit(fx.exact_accuracy_project(self.root / "q", 9304, 10000, "0.930"))
        self.assertFalse(fx.find(f, "claim-mismatch"))

    def test_a_claim_without_predictions_could_not_be_checked(self):
        _, out, f = fx.run_audit(fx.claim_without_predictions(self.root / "p"))
        check = next(c for c in f["checks"] if c["id"] == "score-recomputation")
        self.assertEqual(check["status"], "could not check")
        self.assertIn("no predictions", check["reason"])
        self.assertEqual(f["verdict"], "not enough evidence")
        self.assertEqual(fx.find(f, "claim-mismatch"), [])

    def test_a_score_no_better_than_the_trivial_baseline_is_flagged(self):
        _, _, f = fx.run_audit(fx.baseline_model(self.root / "p"))
        hits = fx.find(f, "no-better-than-baseline")
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0]["severity"], hits[0]["evidence"]), ("high", "proven"))
        self.assertEqual(fx.find(f, "claim-mismatch"), [])
        _, _, clean = fx.run_audit(fx.clean_project(self.root / "c"))
        self.assertEqual(fx.find(clean, "no-better-than-baseline"), [])

    def test_baseline_margin_on_both_sides(self):
        # 600 test rows: 5 extra correct rows = +0.0083 (flagged), 7 = +0.0117 (not flagged); margin 0.01
        _, _, below = fx.run_audit(fx.near_baseline(self.root / "below", 5 / 600))
        _, _, above = fx.run_audit(fx.near_baseline(self.root / "above", 7 / 600))
        self.assertEqual(len(fx.find(below, "no-better-than-baseline")), 1)
        self.assertEqual(fx.find(above, "no-better-than-baseline"), [])

    def test_a_test_claim_computed_on_validation_rows_is_caught(self):
        _, _, f = fx.run_audit(fx.wrong_split_claim(self.root / "p"))
        hits = fx.find(f, "claim-wrong-split")
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0]["severity"], hits[0]["evidence"]), ("critical", "proven"))
        self.assertEqual(f["verdict"], "not trustworthy")

    def test_a_prose_claim_is_checked_like_any_other(self):
        proj = fx.clean_project(self.root / "p")
        (proj / "outputs" / "metrics.json").unlink()
        _, _, f = fx.run_audit(proj, "--claim", "metric=accuracy,value=0.99,split=test,source=README.md:3")
        hits = fx.find(f, "claim-mismatch")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["source"], "README.md:3 (claim passed in)")
        _, _, ok = fx.run_audit(proj, "--claim", "metric=accuracy,value=0.5,split=test,source=README.md:3")
        self.assertEqual(fx.find(ok, "claim-mismatch")[0]["severity"], "medium")  # understated: not inflated

    def test_percent_claims(self):
        proj = fx.exact_accuracy_project(self.root / "p", 9300, 10000, "93")
        _, _, f = fx.run_audit(proj)
        self.assertFalse(fx.find(f, "claim-mismatch"))

    def test_ledger_and_run_log_claims_are_parsed(self):
        proj = fx.wrong_claim(self.root / "p")
        (proj / "outputs" / "metrics.json").unlink()
        (proj / "ml_pipeline").mkdir()
        (proj / "ml_pipeline" / "PIPELINE.md").write_text(
            "# Pipeline\n- Step 14 final test: accuracy_score=0.9900 (touch 1, 2026-10-01)\n")
        (proj / "ml_pipeline" / "runs.jsonl").write_text(json.dumps({"name": "gbm", "kind": "candidate",
                                                                     "metrics": {"accuracy": 0.81}}) + "\n")
        _, _, f = fx.run_audit(proj)
        hits = fx.find(f, "claim-mismatch")
        self.assertEqual(len(hits), 1)
        self.assertIn("ml_pipeline/PIPELINE.md", hits[0]["source"])
        check = next(c for c in f["checks"] if c["id"] == "score-recomputation")
        self.assertIn("validation", check["reason"])  # the run log's validation metric could not be recomputed

    def test_every_finding_reproduces_through_the_guard(self):
        sys.path.insert(0, str(fx.ROOT / "hooks"))
        import guard_training as hook
        for build in (fx.wrong_claim, fx.baseline_model, fx.wrong_split_claim):
            proj = build(self.root / build.__name__)
            _, _, f = fx.run_audit(proj)
            for finding in f["findings"]:
                with self.subTest(fixture=build.__name__, finding=finding["id"]):
                    command = finding["reproduce"]
                    self.assertTrue(hook.audit_problem({"command": command}, str(proj))[0])
                    self.assertIsNone(hook.audit_problem({"command": command}, str(proj))[1])
                    proc = subprocess.run(shlex.split(command), capture_output=True, text=True, cwd=str(proj))
                    out = Path(next(l.split(": ", 1)[1] for l in proc.stdout.splitlines() if l.startswith("audit folder: ")))
                    again = json.loads((out / "findings.json").read_text())
                    same = [g for g in again["findings"] if g["id"] == finding["id"]]
                    self.assertEqual(len(same), 1)
                    for key in ("code", "severity", "evidence", "counts", "files", "columns"):
                        self.assertEqual(same[0][key], finding[key])


if __name__ == "__main__":
    unittest.main()
