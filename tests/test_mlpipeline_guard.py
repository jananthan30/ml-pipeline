"""Tests for skills/ml-pipeline/lib/mlpipeline_guard.py. Run with the dev venv:
    .venv/bin/python -m unittest tests/test_mlpipeline_guard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "ml-pipeline" / "lib"))
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
        self.assertIn("patient_id", [e["column"] for e in p["entity_candidates"]])
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
                                 "02_feature_overview.png", "02_feature_vs_target.png", "02_outliers.png",
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



# ====================================================================== v0.5: big data, dates, groups, leakage, CV, runs
class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()


def wide(n=3000, cols=60, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n) for i in range(cols)}
    data["cat"] = rng.choice(list("abcdef"), size=n)
    y = (data[f"f{min(7, cols - 1)}"] + 0.5 * data[f"f{min(42, cols - 1)}"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({**data, "y": y})


class BigData(Tmp):
    def test_profile_streams_a_parquet_file_in_batches(self):
        df = wide(5000)
        path = self.root / "data.parquet"
        df.to_parquet(path, index=False)
        saved = (guard.SAMPLE_ROWS, guard.BATCH_ROWS)
        guard.SAMPLE_ROWS, guard.BATCH_ROWS = 1000, 700
        try:
            p = guard.profile(str(path), target="y", state_dir=self.state)
        finally:
            guard.SAMPLE_ROWS, guard.BATCH_ROWS = saved
        self.assertEqual(p["shape"]["n_rows"], 5000)          # counted over every row
        self.assertTrue(p["shape"]["sampled"])
        self.assertEqual(p["columns"]["cat"]["n_unique"], 6)  # exact distinct count over every row
        self.assertEqual(p["duplicates"]["exact_row_frac"], 0.0)
        self.assertIn("f7", list(p["feature_signal"])[:3])    # the real signal ranks first
        self.assertTrue((self.state / guard.SUMMARY_FILE).is_file())

    def test_profile_from_csv_matches_frame(self):
        df = wide(1500, cols=5)
        path = self.root / "data.csv"
        df.to_csv(path, index=False)
        saved = guard.BATCH_ROWS
        guard.BATCH_ROWS = 400
        try:
            from_file = guard.profile(str(path), target="y", state_dir=self.state)
        finally:
            guard.BATCH_ROWS = saved
        from_frame = guard.profile(df, target="y", state_dir=self.state)
        self.assertEqual(from_file["shape"]["n_rows"], from_frame["shape"]["n_rows"])
        self.assertEqual(from_file["duplicates"], from_frame["duplicates"])

    def test_duplicates_counted_across_batches(self):
        df = wide(1000, cols=3)
        df = pd.concat([df, df.iloc[:100]], ignore_index=True)
        saved = guard.BATCH_ROWS
        guard.BATCH_ROWS = 300
        try:
            p = guard.profile(df, target="y", state_dir=self.state)
        finally:
            guard.BATCH_ROWS = saved
        self.assertAlmostEqual(p["duplicates"]["exact_row_frac"], 100 / 1100, places=3)

    def test_summary_is_short_on_wide_data(self):
        guard.profile(wide(500, cols=400), target="y", state_dir=self.state)
        summary = guard.summary(self.state)
        self.assertLess(len(summary.splitlines()), 120)
        self.assertIn("Strongest target signal", summary)

    def test_eda_covers_every_column_in_ranked_pages(self):
        guard.profile(wide(800, cols=60), target="y", state_dir=self.state)
        paths = guard.eda_figures(wide(800, cols=60), target="y", state_dir=self.state)
        names = {p.name for p in paths}
        self.assertIn("02_feature_overview.png", names)
        self.assertEqual(len([n for n in names if n.startswith("02_distributions_p")]), 4)   # 60 numeric / 16 per page
        self.assertIn("02_categorical_vs_target.png", names)
        self.assertIn("02_outliers.png", names)
        ledger = (self.state / "PIPELINE.md").read_text()
        self.assertIn("Every one of the 61 columns", ledger)
        self.assertIn("ranked by target signal", ledger)
        self.assertTrue((self.state / "correlations.csv").is_file())
        first_page = ledger.split("- Figure 02_distributions_p1.png:")[1].splitlines()[0]
        self.assertIn("columns 1-16 of 60 numeric", first_page)

    def test_eda_page_cap_is_explained(self):
        df = wide(300, cols=40)
        guard.eda_figures(df, target="y", state_dir=self.state, per_page=4, max_pages=2)
        self.assertIn("remaining 32 are summarized in data_profile.json", (self.state / "PIPELINE.md").read_text())

    def test_split_to_files_and_final_test_from_file(self):
        df = wide(4000, cols=4)
        df = pd.concat([df, df.iloc[:200]], ignore_index=True)   # duplicates must never cross splits
        src = self.root / "data.parquet"
        df.to_parquet(src, index=False)
        saved = guard.BATCH_ROWS
        guard.BATCH_ROWS = 900
        try:
            paths = guard.split_to_files(str(src), target="y", out_dir=self.state / "splits", state_dir=self.state)
            parts = {k: pd.read_parquet(v) for k, v in paths.items()}
            self.assertEqual(sum(len(v) for v in parts.values()), len(df))
            self.assertAlmostEqual(len(parts["test"]) / len(df), 0.2, delta=0.03)
            guard.check_no_overlap(parts["train"].drop_duplicates(), parts["val"].drop_duplicates(), parts["test"].drop_duplicates())
            md = self.state / "PIPELINE.md"
            score = guard.final_test(lambda X: np.zeros(len(X), dtype=int), str(paths["test"]), target="y",
                                     metric_fn=lambda a, b: float((a == b).mean()), state_dir=self.state, pipeline_md=md,
                                     batch_rows=250)
        finally:
            guard.BATCH_ROWS = saved
        self.assertAlmostEqual(score, 1 - parts["test"].y.mean(), places=6)
        with self.assertRaises(guard.TestSetTampered):
            guard.final_test(lambda X: np.zeros(len(X)), str(paths["val"]), target="y", metric_fn=lambda a, b: 0.0,
                             state_dir=self.state, pipeline_md=md)

    def test_split_to_files_by_time_respects_order(self):
        df = frame(3000, dates=True)
        src = self.root / "data.parquet"
        df.to_parquet(src, index=False)
        paths = guard.split_to_files(str(src), target="y", time_col="date", out_dir=self.state / "s", state_dir=self.state)
        parts = {k: pd.read_parquet(v) for k, v in paths.items()}
        self.assertLess(parts["train"].date.max(), parts["val"].date.min())
        self.assertLess(parts["val"].date.max(), parts["test"].date.min())

    def test_final_test_predicts_in_batches(self):
        tr, va, te = guard.split(frame(2000), target="y", state_dir=self.state)
        calls = []
        guard.final_test(lambda X: calls.append(len(X)) or np.zeros(len(X), dtype=int), te, target="y",
                         metric_fn=lambda a, b: float((a == b).mean()), state_dir=self.state,
                         pipeline_md=self.state / "PIPELINE.md", batch_rows=100)
        self.assertEqual(sum(calls), len(te))
        self.assertTrue(all(c <= 100 for c in calls))


class Dates(Tmp):
    def test_text_dates_are_detected_and_force_a_time_split(self):
        df = frame(500, dates=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
        self.assertEqual(guard.datetime_columns(df), {"date": "text"})
        with self.assertRaises(guard.TemporalSplitRequired):
            guard.split(df, target="y", state_dir=self.state)
        tr, va, te = guard.split(df, target="y", time_col="date", state_dir=self.state)
        self.assertLess(pd.to_datetime(tr.date).max(), pd.to_datetime(te.date).min())
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertTrue(p["traits"]["has_datetime"])
        self.assertTrue(p["traits"]["datetime_as_text"])
        self.assertEqual(p["columns"]["date"]["kind"], "datetime")

    def test_epoch_seconds_with_a_date_name_are_dates(self):
        df = frame(300)
        df["created_ts"] = 1_700_000_000 + np.arange(300) * 3600
        self.assertEqual(guard.datetime_columns(df).get("created_ts"), "epoch_s")

    def test_codes_and_amounts_are_not_dates(self):
        df = frame(300)
        df["zip"] = np.random.default_rng(0).integers(10000, 99999, size=300).astype(str)
        df["amount"] = np.random.default_rng(1).integers(1_000_000_000, 2_000_000_000, size=300)
        self.assertEqual(guard.datetime_columns(df), {})

    def test_reason_allows_a_random_split(self):
        df = frame(300, dates=True)
        guard.split(df, target="y", not_temporal_reason="signup date only, user agreed", state_dir=self.state)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual(state["params"]["not_temporal_reason"], "signup date only, user agreed")


class TimeSplit(Tmp):
    def test_ties_never_straddle_a_boundary(self):
        df = frame(1000)
        df["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(1000) // 50, unit="D")   # 50 rows per day
        tr, va, te = guard.split(df, target="y", time_col="date", state_dir=self.state)
        self.assertFalse(set(tr.date) & set(va.date))
        self.assertFalse(set(va.date) & set(te.date))

    def test_gap_leaves_an_embargo(self):
        df = frame(1000, dates=True)
        tr, va, te = guard.split(df, target="y", time_col="date", gap="24h", state_dir=self.state)
        self.assertGreaterEqual(va.date.min() - tr.date.max(), pd.Timedelta("24h"))
        self.assertGreaterEqual(te.date.min() - va.date.max(), pd.Timedelta("24h"))


class Groups(Tmp):
    def test_repeated_entity_forces_a_group_split(self):
        df = frame(1000)
        df["patient_id"] = np.arange(1000) // 5
        with self.assertRaises(guard.GroupSplitRequired):
            guard.split(df, target="y", state_dir=self.state)
        guard.split(df, target="y", not_grouped_reason="visits are independent, user agreed", state_dir=self.state)
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertTrue(p["traits"]["has_groups"])
        self.assertEqual(p["traits"]["group_candidates"], ["patient_id"])

    def test_amount_paid_is_not_an_entity(self):
        df = frame(1000)
        df["amount_paid"] = np.random.default_rng(0).integers(0, 50, size=1000)
        guard.split(df, target="y", state_dir=self.state)  # no GroupSplitRequired

    def test_one_huge_group_does_not_swallow_the_test_set(self):
        df = frame(2000)
        groups = np.arange(2000) // 10
        groups[:900] = -1                              # one group holds 45% of rows
        df["site"] = groups
        tr, va, te = guard.split(df, target="y", group_col="site", seed=3, state_dir=self.state)
        self.assertLess(len(te) / len(df), 0.35)
        self.assertGreater(len(va), 0)
        self.assertFalse(set(tr.site) & set(te.site))

    def test_too_few_groups(self):
        df = frame(100)
        df["g"] = np.arange(100) % 2
        with self.assertRaises(ValueError):
            guard.split(df, target="y", group_col="g", state_dir=self.state)


class Stratified(Tmp):
    def test_rare_classes_reach_every_split_or_are_reported(self):
        df = frame(1000)
        df.loc[:3, "y"] = 2          # 4 rows of class 2
        df.loc[4:5, "y"] = 3         # 2 rows of class 3
        tr, va, te = guard.split(df, target="y", state_dir=self.state)
        self.assertIn(2, set(te.y))
        self.assertIn(2, set(va.y))
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual(state["report"]["rare_classes_kept_in_train"], ["3"])

    def test_works_with_a_non_default_index(self):
        df = frame(500)
        df.index = np.arange(10_000, 10_500)
        tr, va, te = guard.split(df, target="y", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 500)


class Leakage(Tmp):
    def _df(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        return pd.DataFrame({"a": rng.normal(size=n), "y": y}), rng

    def test_nonlinear_leak_is_caught(self):
        df, rng = self._df()
        df["score"] = np.where(df.y == 1, rng.uniform(5, 6, len(df)), rng.uniform(0, 1, len(df))) ** 3  # monotone but not linear
        df["band"] = np.where(df.y == 1, rng.choice([-1, 1], len(df)) * 10, rng.normal(0, 1, len(df)))  # non-monotone
        flagged = {s["column"] for s in guard.profile(df, target="y", state_dir=self.state)["leakage_suspects"]}
        self.assertIn("band", flagged)
        self.assertIn("score", flagged)
        self.assertNotIn("a", flagged)

    def test_imbalanced_target_does_not_flag_every_category(self):
        df, rng = self._df(5000)
        df["y"] = (rng.random(len(df)) < 0.01).astype(int)
        df["region"] = rng.choice(list("abcd"), len(df))
        flagged = {s["column"] for s in guard.profile(df, target="y", state_dir=self.state)["leakage_suspects"]}
        self.assertNotIn("region", flagged)

    def test_label_conflicts_and_id_duplicates_are_counted(self):
        df, _ = self._df(400)
        df["row_id"] = np.arange(400)
        twin = df.iloc[:40].copy()
        twin["row_id"] = np.arange(400, 440)
        twin["y"] = 1 - twin["y"]
        p = guard.profile(pd.concat([df, twin], ignore_index=True), target="y", state_dir=self.state)
        self.assertAlmostEqual(p["duplicates"]["label_conflict_frac"], 40 / 440, places=3)
        self.assertEqual(p["duplicates"]["exact_row_frac"], 0.0)

    def test_sorted_ids_are_flagged(self):
        df, _ = self._df(1000)
        df = df.sort_values("y").reset_index(drop=True)
        df["record_id"] = np.arange(1000)
        flagged = {s["column"]: s["reason"] for s in guard.profile(df, target="y", state_dir=self.state)["leakage_suspects"]}
        self.assertIn("id order correlates", flagged.get("record_id", ""))

    def test_unique_integer_amount_is_numeric_not_id(self):
        df, rng = self._df(1000)
        df["income"] = rng.choice(np.arange(20_000, 900_000), size=1000, replace=False)
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertEqual(p["columns"]["income"]["kind"], "numeric")


class CrossValidation(Tmp):
    def test_stratified_folds_partition_train(self):
        df = frame(1000)
        folds = guard.cv_splits(df, target="y", n_splits=5, n_repeats=2)
        self.assertEqual(len(folds), 10)
        for fit_idx, score_idx in folds:
            self.assertFalse(set(fit_idx) & set(score_idx))
            self.assertEqual(len(fit_idx) + len(score_idx), 1000)

    def test_group_folds_keep_groups_together(self):
        df = frame(1000)
        df["g"] = np.arange(1000) // 7
        for fit_idx, score_idx in guard.cv_splits(df, target="y", group_col="g"):
            self.assertFalse(set(df.g.iloc[fit_idx]) & set(df.g.iloc[score_idx]))

    def test_time_folds_only_look_back(self):
        df = frame(1200, dates=True)
        for fit_idx, score_idx in guard.cv_splits(df, target="y", time_col="date", gap=10):
            self.assertLess(df.date.iloc[fit_idx].max(), df.date.iloc[score_idx].min())


class RunLog(Tmp):
    def test_log_run_records_data_and_environment(self):
        guard.split(frame(), target="y", state_dir=self.state)
        guard.log_run("dummy", kind="baseline", metrics={"ap": 0.11}, seed=0, state_dir=self.state)
        guard.log_run("gbm", kind="candidate", metrics={"ap": 0.42}, params={"depth": 3}, state_dir=self.state)
        table = guard.runs(self.state)
        self.assertEqual(list(table["kind"]), ["baseline", "candidate"])
        record = json.loads((self.state / guard.RUNS_FILE).read_text().splitlines()[0])
        self.assertIn("pandas", record["environment"]["packages"])
        self.assertIsNotNone(record["data"]["train_fingerprint"])
        with self.assertRaises(ValueError):
            guard.log_run("x", kind="best", metrics={"ap": 1}, state_dir=self.state)


class Manifest(Tmp):
    def test_manifest_groups_exact_duplicates(self):
        for label in ("cat", "dog"):
            (self.root / "imgs" / label).mkdir(parents=True)
        (self.root / "imgs" / "cat" / "p1_a.png").write_bytes(b"\x89PNG one")
        (self.root / "imgs" / "dog" / "p2_b.png").write_bytes(b"\x89PNG two")
        (self.root / "imgs" / "dog" / "p3_c.png").write_bytes(b"\x89PNG one")      # same bytes as p1_a
        m = guard.manifest(self.root / "imgs", kind="image", state_dir=self.state)
        self.assertEqual(sorted(m.label), ["cat", "dog", "dog"])
        self.assertEqual(m.group.nunique(), 2)
        by_source = guard.manifest(self.root / "imgs", kind="image", group_pattern=r"(p\d)_", state_dir=self.state)
        self.assertEqual(sorted(by_source.group), ["p1", "p2", "p3"])
        self.assertTrue((self.state / "manifest.csv").is_file())


class Overrides(Tmp):
    def test_user_record_controls_extra_test_touches(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        md = self.state / "PIPELINE.md"
        acc = lambda a, b: float((a == b).mean())  # noqa: E731
        pred = lambda X: np.zeros(len(X), dtype=int)  # noqa: E731
        (self.state / guard.APPROVALS_FILE).write_text(json.dumps({"gates": {"C": {}}, "overrides": []}))
        guard.final_test(pred, te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=md)
        md.write_text(md.read_text() + "- Override: final test - agent wrote this itself\n")
        with self.assertRaises(guard.TestSetAlreadyUsed):   # a ledger line alone is not the user's word
            guard.final_test(pred, te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=md)
        (self.state / guard.APPROVALS_FILE).write_text(json.dumps({"gates": {"C": {}}, "overrides": [{"what": "final test"}]}))
        guard.final_test(pred, te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=md)

    def test_resetting_state_does_not_reset_the_count(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        md = self.state / "PIPELINE.md"
        acc = lambda a, b: float((a == b).mean())  # noqa: E731
        guard.final_test(lambda X: np.zeros(len(X), dtype=int), te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=md)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        state["test_touches"] = 0
        (self.state / guard.STATE_FILE).write_text(json.dumps(state))
        with self.assertRaises(guard.TestSetAlreadyUsed):  # the ledger still shows one touch
            guard.final_test(lambda X: np.zeros(len(X), dtype=int), te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=md)

    def test_v04_state_files_still_verify(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        state.pop("fingerprint_version")
        state["test_fingerprint"] = guard._fingerprint_v1(te)
        (self.state / guard.STATE_FILE).write_text(json.dumps(state))
        guard.final_test(lambda X: np.zeros(len(X), dtype=int), te, target="y", metric_fn=lambda a, b: 0.5,
                         state_dir=self.state, pipeline_md=self.state / "PIPELINE.md")



# ====================================================================== review fixes
class ReviewFixes(Tmp):
    def test_mixed_utc_offsets_parse(self):
        df = frame(400)
        stamps = pd.date_range("2024-03-01", periods=400, freq="h", tz="America/New_York")   # crosses the DST change
        df["ts"] = [t.isoformat() for t in stamps]
        self.assertEqual(guard.datetime_columns(df), {"ts": "text"})
        p = guard.profile(df, target="y", time_col="ts", state_dir=self.state)
        self.assertEqual(p["columns"]["ts"]["kind"], "datetime")
        tr, va, te = guard.split(df, target="y", time_col="ts", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 400)
        guard.eda_figures(df, target="y", time_col="ts", state_dir=self.state)

    def test_missing_targets_are_refused_not_dropped(self):
        df = frame(400).astype({"y": float})
        df.loc[:39, "y"] = np.nan
        with self.assertRaises(ValueError):
            guard.split(df, target="y", state_dir=self.state)
        with self.assertRaises(ValueError):
            guard.cv_splits(df, target="y")

    def test_late_text_in_a_numeric_looking_csv_column(self):
        df = frame(1500)
        df["code"] = np.arange(1500).astype(str)
        df.loc[1200, "code"] = "A12"
        path = self.root / "late.csv"
        df.to_csv(path, index=False)
        saved = guard.BATCH_ROWS
        guard.BATCH_ROWS = 500
        try:
            p = guard.profile(str(path), target="y", state_dir=self.state)
            batches = list(guard.scan(str(path)))
        finally:
            guard.BATCH_ROWS = saved
        self.assertEqual(p["shape"]["n_rows"], 1500)
        self.assertEqual(batches[0]["b"].dtype, np.int64)       # integer columns stay integers
        self.assertEqual(batches[0]["code"].dtype, object)

    def test_inf_does_not_break_figures(self):
        df = frame(500)
        df.loc[3, "a"] = np.inf
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertGreater(p["columns"]["a"]["inf_frac"], 0)
        guard.eda_figures(df, target="y", state_dir=self.state)

    def test_split_to_files_unifies_late_types(self):
        rows = [{"x": i, "note": None, "y": i % 2} for i in range(300)] + \
               [{"x": i, "note": "text", "y": i % 2} for i in range(300, 600)]
        path = self.root / "late.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        saved = guard.BATCH_ROWS
        guard.BATCH_ROWS = 200
        try:
            paths = guard.split_to_files(str(path), target="y", out_dir=self.state / "s", state_dir=self.state)
        finally:
            guard.BATCH_ROWS = saved
        total = sum(len(pd.read_parquet(p)) for p in paths.values())
        self.assertEqual(total, 600)

    def test_split_to_files_refuses_missing_targets(self):
        df = frame(500).astype({"y": float})
        df.loc[10, "y"] = np.nan
        path = self.root / "d.parquet"
        df.to_parquet(path, index=False)
        with self.assertRaises(ValueError):
            guard.split_to_files(str(path), target="y", out_dir=self.state / "s", state_dir=self.state)

    def test_final_test_needs_gate_c_when_approvals_are_recorded(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        (self.state / guard.APPROVALS_FILE).write_text(json.dumps({"gates": {"A": {}, "B": {}}, "overrides": []}))
        with self.assertRaises(guard.LeakageError):
            guard.final_test(lambda X: np.zeros(len(X)), te, target="y", metric_fn=lambda a, b: 0.0, state_dir=self.state,
                             pipeline_md=self.state / "PIPELINE.md")

    def test_another_ledger_path_does_not_reset_the_count(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        acc = lambda a, b: float((a == b).mean())  # noqa: E731
        pred = lambda X: np.zeros(len(X), dtype=int)  # noqa: E731
        guard.final_test(pred, te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=self.state / "PIPELINE.md")
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        state["test_touches"] = 0
        (self.state / guard.STATE_FILE).write_text(json.dumps(state))
        with self.assertRaises(guard.TestSetAlreadyUsed):
            guard.final_test(pred, te, target="y", metric_fn=acc, state_dir=self.state, pipeline_md=self.root / "other.md")

    def test_empty_data_is_an_error(self):
        with self.assertRaises(ValueError):
            guard.profile(frame(10).iloc[:0], target="y", state_dir=self.state)


if __name__ == "__main__":
    unittest.main()
