"""Tests for lib/mlpipeline_canary.py (stdlib only): python3 -m unittest tests/test_mlpipeline_canary.py"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import mlpipeline_canary as canary  # noqa: E402


class Canary(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(canary.make_rows(50, seed=1), canary.make_rows(50, seed=1))
        self.assertNotEqual(canary.make_rows(50, seed=1), canary.make_rows(50, seed=2))

    def test_label_noise_rate_matches(self):
        rows = canary.make_rows(20000, noise=0.2, seed=3)
        clean = [1 if r[0] + r[1] * r[2] - 0.5 * r[3] > 0 else 0 for r in rows]
        flipped = sum(c != r[-1] for c, r in zip(clean, rows)) / len(rows)
        self.assertAlmostEqual(flipped, 0.2, delta=0.01)

    def test_verdict(self):
        leaked, why = canary.verdict(0.79, 2000)
        self.assertFalse(leaked)
        self.assertIn("within", why)
        leaked, why = canary.verdict(0.86, 2000)
        self.assertTrue(leaked)
        self.assertIn("leaked", why)

    def test_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            canary.write_csv(path, n=10)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["x0", "x1", "x2", "x3", "x4", "y"])
        self.assertEqual(len(rows), 11)


if __name__ == "__main__":
    unittest.main()
