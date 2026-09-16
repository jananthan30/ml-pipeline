"""Tests for lib/mlpipeline_guard.py. Run with the dev venv:
    .venv/bin/python -m unittest tests/test_mlpipeline_guard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - the stdlib suite skips this file
    raise unittest.SkipTest("pandas/numpy not installed; run with .venv/bin/python")
import mlpipeline_guard as guard  # noqa: E402


def frame(n=1000, seed=0, dates=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, size=n), "y": rng.integers(0, 2, size=n)})
    if dates:
        df["date"] = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


class Split(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sizes_and_state_file(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 1000)
        self.assertAlmostEqual(len(te) / 1000, 0.2, delta=0.02)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual(state["test_touches"], 0)
        self.assertEqual(state["n_test"], len(te))
        self.assertEqual(state["split"], "stratified")

    def test_stratified_keeps_class_balance(self):
        df = frame()
        _, _, te = guard.split(df, target="y", state_dir=self.state)
        self.assertAlmostEqual(te.y.mean(), df.y.mean(), delta=0.03)

    def test_temporal_split_is_chronological(self):
        tr, va, te = guard.split(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        self.assertLess(tr.date.max(), va.date.min())
        self.assertLess(va.date.max(), te.date.min())

    def test_datetime_column_without_time_col_raises(self):
        with self.assertRaises(guard.TemporalSplitRequired):
            guard.split(frame(dates=True), target="y", state_dir=self.state)

    def test_group_split_keeps_groups_together(self):
        df = frame()
        df["patient"] = df.index // 10
        tr, va, te = guard.split(df, target="y", group_col="patient", state_dir=self.state)
        self.assertFalse(set(tr.patient) & set(te.patient))
        self.assertFalse(set(va.patient) & set(te.patient))
        self.assertFalse(set(tr.patient) & set(va.patient))

    def test_duplicates_are_dropped_before_splitting(self):
        df = pd.concat([frame(200)] * 2, ignore_index=True)  # every row twice
        tr, va, te = guard.split(df, target="y", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 200)
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["duplicates_dropped"], 200)

    def test_check_no_overlap_raises(self):
        df = frame(50)
        with self.assertRaises(guard.OverlapLeakage):
            guard.check_no_overlap(df.iloc[:30], df.iloc[25:])

    def test_bad_sizes_rejected(self):
        with self.assertRaises(ValueError):
            guard.split(frame(), target="y", test_size=0.6, val_size=0.5, state_dir=self.state)


if __name__ == "__main__":
    unittest.main()
