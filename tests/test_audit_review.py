"""Ticket 09: the agent's review findings are labelled, never proven, and never decide 'not trustworthy' alone.

Run:  python -m pytest tests/test_audit_review.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

CRITICAL = "family=leakage,severity=critical,text=the scaler is fit on all rows, before the split,source=train.py:41"
HIGH = "family=honest-scores,severity=high,text=accuracy is the wrong metric for a 2% minority class,source=README.md:5"


class Review(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_findings_are_labelled_and_keep_their_text(self):
        _, out, f = fx.run_audit(fx.clean_project(self.root / "p"), "--review", CRITICAL, "--review", HIGH)
        hits = fx.find(f, "agent-review")
        self.assertEqual(len(hits), 2)
        self.assertEqual({h["evidence"] for h in hits}, {"review"})
        crit = next(h for h in hits if h["severity"] == "critical")
        self.assertEqual(crit["family"], "leakage")
        self.assertIn("fit on all rows, before the split", crit["detail"])  # commas inside the text survive
        self.assertEqual(crit["source"], "train.py:41")
        self.assertIn("agent review (not proven by the tool)", (out / "report.md").read_text())

    def test_a_review_critical_alone_cannot_make_it_not_trustworthy(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"), "--review", CRITICAL)
        self.assertEqual(f["verdict"], "not enough evidence")
        self.assertIn("review", f["verdict_reason"])

    def test_a_review_high_does_not_change_the_verdict(self):
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"), "--review", HIGH)
        self.assertEqual(f["verdict"], "trustworthy")

    def test_a_review_cannot_claim_to_be_proven_or_invent_a_family(self):
        bad = ["family=leakage,severity=critical,evidence=proven,text=x", "family=made-up,severity=high,text=x",
               "family=leakage,severity=fatal,text=x", "family=leakage,severity=high"]
        _, _, f = fx.run_audit(fx.clean_project(self.root / "p"), *[a for b in bad for a in ("--review", b)])
        hits = fx.find(f, "agent-review")
        self.assertEqual([h["evidence"] for h in hits], ["review"])  # only the first is kept, and as review
        self.assertEqual(len([n for n in f["notes"] if n.startswith("review ignored")]), 3)
        self.assertEqual(f["verdict"], "not enough evidence")

    def test_proven_findings_still_decide(self):
        _, _, f = fx.run_audit(fx.wrong_claim(self.root / "p"), "--review", HIGH)
        self.assertEqual(f["verdict"], "not trustworthy")


if __name__ == "__main__":
    unittest.main()
