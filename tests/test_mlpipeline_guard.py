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



class FinalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.md = self.state / "PIPELINE.md"
        self.tr, self.va, self.te = guard.split(frame(), target="y", state_dir=self.state)
        self.md.write_text("- Gate C: approved 2026-09-18\n")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def acc(y_true, y_pred):
        return float((y_true == y_pred).mean())

    @staticmethod
    def predict(X):
        return np.zeros(len(X), dtype=int)

    def _run(self, test):
        return guard.final_test(self.predict, test, target="y", metric_fn=self.acc,
                                state_dir=self.state, pipeline_md=self.md)

    def test_first_touch_scores_logs_and_counts(self):
        score = self._run(self.te)
        self.assertAlmostEqual(score, 1 - self.te.y.mean(), places=6)
        self.assertIn("Step 14 final test: acc=", self.md.read_text())
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["test_touches"], 1)

    def test_second_touch_raises(self):
        self._run(self.te)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)

    def test_override_line_permits_second_touch(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved a second evaluation 2026-09-18 - reason: new strategy\n")
        self._run(self.te)

    def test_tampered_test_set_raises(self):
        with self.assertRaises(guard.TestSetTampered):
            self._run(self.va)

    def test_no_split_state_raises(self):
        (self.state / guard.STATE_FILE).unlink()
        with self.assertRaises(guard.LeakageError):
            self._run(self.te)


class Profile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def _df(self, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        return pd.DataFrame({
            "age": rng.normal(50, 12, size=n),
            "zip": rng.integers(90000, 90400, size=n).astype(str),          # 400 levels -> high cardinality
            "flag": rng.integers(0, 2, size=n).astype(bool),
            "notes": ["x" * 40 + str(i) for i in range(n)],                 # long strings -> text
            "patient_id": np.arange(n),                                     # unique ints -> id
            "admitted_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": y,
        })

    def test_column_kinds(self):
        p = guard.profile(self._df(), target="y", time_col="admitted_at", state_dir=self.state)
        kinds = {c: p["columns"][c]["kind"] for c in p["columns"]}
        self.assertEqual(kinds["age"], "numeric")
        self.assertEqual(kinds["zip"], "categorical")
        self.assertEqual(kinds["flag"], "bool")
        self.assertEqual(kinds["notes"], "text")
        self.assertEqual(kinds["patient_id"], "id")
        self.assertEqual(kinds["admitted_at"], "datetime")

    def test_shape_target_and_file(self):
        p = guard.profile(self._df(), target="y", time_col="admitted_at", state_dir=self.state)
        self.assertEqual(p["shape"]["n_rows"], 1000)
        self.assertEqual(p["target"]["task"], "binary")
        self.assertAlmostEqual(sum(p["target"]["class_balance"].values()), 1.0, places=3)
        self.assertIn("zip", p["high_cardinality_categoricals"])
        self.assertIn("patient_id", p["entity_candidates"])
        self.assertEqual(json.loads((self.state / guard.PROFILE_FILE).read_text())["shape"]["n_rows"], 1000)

    def test_missing_and_duplicates(self):
        df = self._df(200)
        df.loc[:19, "age"] = np.nan
        df = pd.concat([df, df.iloc[:50]], ignore_index=True)
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        self.assertAlmostEqual(p["columns"]["age"]["missing_frac"], 40 / 250, places=3)  # 20 NaN + 20 in the duplicated rows
        self.assertAlmostEqual(p["duplicates"]["exact_row_frac"], 50 / 250, places=3)

    def test_regression_and_multiclass_targets(self):
        df = self._df()
        df["amount"] = np.random.default_rng(1).normal(size=len(df))
        self.assertEqual(guard.profile(df, target="amount", time_col="admitted_at", state_dir=self.state)["target"]["task"], "regression")
        df["cls"] = np.random.default_rng(2).integers(0, 4, size=len(df))
        self.assertEqual(guard.profile(df, target="cls", time_col="admitted_at", state_dir=self.state)["target"]["task"], "multiclass")

    def test_leakage_suspects(self):
        df = self._df()
        df["risk_final"] = df["y"] + np.random.default_rng(3).normal(0, 0.01, size=len(df))   # |corr| > 0.95
        df["discharge_outcome"] = "a"                                                            # name rule
        df["y_copy_cat"] = df["y"].astype(str)                                                   # purity rule
        df["recorded_at"] = df["admitted_at"] + pd.Timedelta(days=3)                              # later than time_col
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        flagged = {s["column"] for s in p["leakage_suspects"]}
        self.assertTrue({"risk_final", "discharge_outcome", "y_copy_cat", "recorded_at"} <= flagged, flagged)
        self.assertNotIn("age", flagged)

    def test_traits(self):
        df = self._df()
        df["y"] = (np.arange(len(df)) % 20 == 0).astype(int)   # 5% minority
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        t = p["traits"]
        self.assertTrue(t["small_data"])
        self.assertTrue(t["imbalanced"])
        self.assertTrue(t["has_datetime"])
        self.assertFalse(t["has_groups"])              # entity candidates never set has_groups
        self.assertEqual(t["n_features"], 6)
        p2 = guard.profile(df, target="y", time_col="admitted_at", group_col="patient_id", state_dir=self.state)
        self.assertTrue(p2["traits"]["has_groups"])

    def test_sampling_flag(self):
        saved = guard.SAMPLE_ROWS
        guard.SAMPLE_ROWS = 300
        try:
            p = guard.profile(self._df(1000), target="y", time_col="admitted_at", state_dir=self.state)
        finally:
            guard.SAMPLE_ROWS = saved
        self.assertTrue(p["shape"]["sampled"])
        self.assertEqual(p["shape"]["sample_rows"], 300)
        self.assertEqual(p["shape"]["n_rows"], 1000)


class Figures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.md = self.state / "PIPELINE.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_eda_figures_writes_required_set_with_ledger(self):
        paths = guard.eda_figures(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        names = sorted(p.name for p in paths)
        self.assertEqual(names, ["01_missingness.png", "02_correlations.png", "02_distributions.png",
                                 "02_target_balance.png", "02_temporal_coverage.png"])
        for p in paths:
            self.assertGreater(p.stat().st_size, 1024, p.name)
            self.assertIn(f"- Figure {p.name}: ", self.md.read_text())

    def test_no_temporal_figure_without_dates(self):
        names = {p.name for p in guard.eda_figures(frame(), target="y", state_dir=self.state)}
        self.assertNotIn("02_temporal_coverage.png", names)

    def test_fig_saves_and_ledgers(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = guard.fig(12, "ROC curve", f, "AUC 0.81 on validation, baseline 0.50.", state_dir=self.state)
        self.assertEqual(path.name, "12_roc_curve.png")
        self.assertGreater(path.stat().st_size, 1024)
        self.assertIn("- Figure 12_roc_curve.png: AUC 0.81", self.md.read_text())

    def test_fig_requires_explanation(self):
        import matplotlib.pyplot as plt
        with self.assertRaises(ValueError):
            guard.fig(4, "x", plt.figure(), "   ", state_dir=self.state)

    def test_missing_matplotlib_is_a_clear_error(self):
        import sys
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "matplotlib" or k.startswith("matplotlib.")}
        sys.modules["matplotlib"] = None  # makes `import matplotlib` raise ImportError
        try:
            with self.assertRaises(RuntimeError) as ctx:
                guard.eda_figures(frame(), target="y", state_dir=self.state)
            self.assertIn("pip install matplotlib", str(ctx.exception))
        finally:
            del sys.modules["matplotlib"]
            sys.modules.update(saved)

if __name__ == "__main__":
    unittest.main()
