"""Tests for lib/mlpipeline_guard.py. Run with the dev venv:
    .venv/bin/python -m unittest tests/test_mlpipeline_guard.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
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


def messy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(50, 12, n).round(1),
        "income": rng.lognormal(10, 1, n).round(0),
        "zip": rng.integers(0, 120, n).astype(str),          # ~119 levels -> high cardinality
        "plan": rng.choice(["basic", "plus", "pro"], n).astype(object),
        "smoker": rng.integers(0, 2, n).astype(bool),
        "patient_id": np.arange(n),
        "y": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 40, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "plan"] = np.nan
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

    def test_state_records_train_and_validation(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual((state["n_train"], state["n_val"]), (len(tr), len(va)))
        self.assertEqual((len(state["train_fingerprint"]), len(state["val_fingerprint"])), (64, 64))

    def test_resplitting_before_any_test_touch_is_fine(self):
        guard.split(frame(), target="y", state_dir=self.state)
        guard.split(frame(), target="y", state_dir=self.state)



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

    def test_resplit_after_final_test_raises(self):
        self._run(self.te)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            guard.split(frame(), target="y", state_dir=self.state)

    def test_resplit_allowed_with_final_test_override(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved a new test strategy 2026-09-24 - reason: new data\n")
        guard.split(frame(), target="y", state_dir=self.state)
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["test_touches"], 0)

    def test_override_must_be_the_strict_line(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() + "### Override requested by the agent for the final test\n")
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)

    def test_override_from_an_older_run_does_not_count(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved 2026-09-18 - reason: x\n## Run: retrain (2026-09-24)\n")
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)


class Sources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"
        self.csv = self.root / "raw.csv"
        frame(50).to_csv(self.csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def sources(self):
        return json.loads((self.state / guard.SOURCE_FILE).read_text())["sources"]

    def test_load_records_hash_rows_and_columns(self):
        df = guard.load(self.csv, state_dir=self.state)
        entry = self.sources()[str(self.csv)]
        self.assertEqual(len(df), 50)
        self.assertEqual(entry["sha256"], hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual((entry["kind"], entry["n_rows"], entry["n_cols"]), ("file", 50, 3))
        self.assertEqual(entry["columns"], ["a", "b", "y"])

    def test_partial_loads_are_refused(self):
        for kwargs in ({"nrows": 10}, {"skiprows": 5}, {"skipfooter": 1}, {"chunksize": 10},
                       {"iterator": True}, {"usecols": ["a"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(guard.PartialData):
                guard.load(self.csv, state_dir=self.state, **kwargs)
        self.assertFalse((self.state / guard.SOURCE_FILE).exists())

    def test_parquet_and_feather_subsets_are_refused(self):
        for kwargs in ({"columns": ["a"]}, {"filters": [("y", "=", 1)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(guard.PartialData):
                guard.load(self.csv, state_dir=self.state, **kwargs)

    def test_other_read_options_pass_through(self):
        self.assertEqual(str(guard.load(self.csv, state_dir=self.state, dtype={"b": "float64"})["b"].dtype), "float64")

    def test_tab_separated(self):
        tsv = self.root / "raw.tsv"
        frame(20).to_csv(tsv, sep="\t", index=False)
        self.assertEqual(list(guard.load(tsv, state_dir=self.state).columns), ["a", "b", "y"])

    def test_unsupported_suffix(self):
        odd = self.root / "raw.dat"
        odd.write_text("a\n1\n")
        with self.assertRaises(ValueError):
            guard.load(odd, state_dir=self.state)

    def test_derive_records_the_chain(self):
        df = guard.load(self.csv, state_dir=self.state)
        guard.derive(df.assign(c=1), "raw_plus_c", parents=[str(self.csv)], how="added a constant column c",
                     state_dir=self.state)
        entry = self.sources()["raw_plus_c"]
        self.assertEqual((entry["kind"], entry["parents"], entry["n_rows"], entry["n_cols"]),
                         ("derived", [str(self.csv)], 50, 4))

    def test_derive_needs_known_parents_and_a_reason(self):
        df = guard.load(self.csv, state_dir=self.state)
        with self.assertRaises(KeyError):
            guard.derive(df, "x", parents=["never_loaded.csv"], how="join", state_dir=self.state)
        with self.assertRaises(ValueError):
            guard.derive(df, "x", parents=[str(self.csv)], how="  ", state_dir=self.state)
        with self.assertRaises(ValueError):
            guard.derive(df, "x", parents=[], how="join", state_dir=self.state)


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

    def test_every_row_counts(self):
        n = 250_000
        df = pd.DataFrame({"c": np.zeros(n), "y": np.arange(n) % 2})
        df.loc[n - 10:, "c"] = 5.0
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertEqual(p["schema_version"], "2")
        self.assertEqual(p["columns"]["c"]["zeros"], n - 10)
        self.assertNotIn("sampled", p["shape"])
        self.assertFalse(hasattr(guard, "SAMPLE_ROWS"))

    def test_kind_specific_statistics(self):
        df = self._df()
        c = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)["columns"]
        common = {"kind", "dtype", "non_null", "missing", "missing_frac", "n_unique", "unique_frac", "memory_bytes"}
        levels = {"n_levels", "top", "top_counts", "rare_levels"}
        expected = {
            "age": {"mean", "std", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "skew",
                    "kurtosis", "zeros", "negatives", "outliers", "outlier_frac"},
            "zip": levels, "flag": levels, "admitted_at": {"start", "end", "span_days"},
            "notes": {"length_min", "length_median", "length_max"}, "patient_id": set(),
        }
        for col, keys in expected.items():
            with self.subTest(col=col):
                self.assertLessEqual(common | keys, set(c[col]))
        self.assertEqual(c["zip"]["n_levels"], df["zip"].nunique())
        self.assertEqual(sum(c["flag"]["top_counts"].values()), 1000)
        self.assertEqual(c["admitted_at"]["span_days"], 41)

    def test_all_missing_column(self):
        df = self._df()
        df["empty"] = np.nan
        info = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)["columns"]["empty"]
        self.assertEqual((info["missing"], info["non_null"]), (1000, 0))
        self.assertNotIn("mean", info)

    def test_integer_named_leakage_suspects_are_strings(self):
        n = 200
        df = pd.DataFrame({0: np.arange(n) % 2 + 0.001 * np.arange(n), 1: np.random.default_rng(0).normal(size=n),
                           "y": np.arange(n) % 2})
        suspects = guard.profile(df, target="y", state_dir=self.state)["leakage_suspects"]
        self.assertIn("0", [s["column"] for s in suspects])
        self.assertTrue(all(isinstance(s["column"], str) for s in suspects))

    def test_target_counts_and_regression_stats(self):
        df = self._df()
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        self.assertEqual(p["target"]["class_counts"], {str(k): int(v) for k, v in df["y"].value_counts().items()})
        df["amount"] = np.random.default_rng(1).normal(size=len(df))
        t = guard.profile(df, target="amount", time_col="admitted_at", state_dir=self.state)["target"]
        self.assertIsNone(t["class_counts"])
        self.assertAlmostEqual(t["mean"], round(float(df["amount"].mean()), 4))

    def _loaded(self, n=300):
        csv_path = Path(self.tmp.name) / "raw.csv"
        frame(n).to_csv(csv_path, index=False)
        return csv_path, guard.load(csv_path, state_dir=self.state)

    def test_full_data_when_the_frame_is_a_loaded_source(self):
        csv_path, df = self._loaded()
        self.assertEqual(guard.profile(df, target="y", state_dir=self.state)["source"],
                         {"name": str(csv_path), "n_rows": 300, "full_data": True})

    def test_a_sample_is_not_full_data(self):
        _, df = self._loaded()
        src = guard.profile(df.sample(frac=0.5, random_state=0), target="y", state_dir=self.state)["source"]
        self.assertFalse(src["full_data"])
        self.assertIn("matches no loaded source", src["reason"])

    def test_no_source_loaded(self):
        src = guard.profile(frame(), target="y", state_dir=self.state)["source"]
        self.assertFalse(src["full_data"])
        self.assertIn("guard.load()", src["reason"])

    def test_named_source_must_exist(self):
        self._loaded()
        src = guard.profile(frame(), target="y", source="nope.csv", state_dir=self.state)["source"]
        self.assertIn("not in source.json", src["reason"])


NOTE_CASES = json.loads((Path(__file__).resolve().parent / "note_rule_cases.json").read_text())


class NoteRule(unittest.TestCase):
    def test_shared_cases(self):
        for case in NOTE_CASES["cases"]:
            with self.subTest(note=case["note"]):
                stats = NOTE_CASES["stats"] if case["column"] else None
                self.assertEqual(guard.note_problem(case["note"], stats), case["problem"])


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
        path = guard.fig(12, "ROC curve", f, "AUC 0.81 on validation against a dummy baseline of 0.50.", state_dir=self.state)
        self.assertEqual(path.name, "12_roc_curve.png")
        self.assertGreater(path.stat().st_size, 1024)
        self.assertIn("- Figure 12_roc_curve.png: AUC 0.81", self.md.read_text())

    def ledger(self):
        return dict(re.findall(r"^- Figure (\S+): (.*)$", self.md.read_text(), re.M))

    def test_fig_rejects_an_explanation_without_a_number(self):
        import matplotlib.pyplot as plt
        with self.assertRaises(ValueError):
            guard.fig(4, "x", plt.figure(), "shows the distribution of values nicely here", state_dir=self.state)

    def test_eda_explanations_are_real(self):
        guard.eda_figures(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        for name, explanation in self.ledger().items():
            with self.subTest(figure=name):
                self.assertIsNone(guard.note_problem(explanation))

    def test_correlations_leave_out_the_target(self):
        guard.eda_figures(frame(), target="y", state_dir=self.state)
        self.assertIn("target is left out", self.ledger()["02_correlations.png"])

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


class Records(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.df = messy()
        guard.profile(self.df, target="y", state_dir=self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, name, state=None):
        with ((state or self.state) / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, name, rows, excel=False):
        with (self.state / name).open("w", newline="", encoding="utf-8-sig" if excel else "utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\r\n" if excel else "\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_dictionary_skeleton(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        rows = self.rows(guard.DICTIONARY_FILE)
        self.assertEqual([r["column"] for r in rows], list(self.df.columns))
        by = {r["column"]: r for r in rows}
        self.assertEqual(by["y"]["role"], "target")
        self.assertEqual(by["zip"]["kind"], "categorical")
        self.assertIn("median", by["age"]["observed"])
        self.assertEqual({by["age"][f] for f in ("meaning", "unit", "role", "figure_note")}, {""})

    def test_dictionary_keeps_words_and_survives_excel(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        rows = self.rows(guard.DICTIONARY_FILE)
        rows[0]["meaning"] = "age at admission"
        self.write_rows(guard.DICTIONARY_FILE, rows, excel=True)
        guard.data_dictionary(self.df, state_dir=self.state)
        self.assertEqual(self.rows(guard.DICTIONARY_FILE)[0]["meaning"], "age at admission")

    def test_dictionary_drops_stale_rows_with_a_warning(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        with self.assertWarns(UserWarning) as ctx:
            guard.data_dictionary(self.df.drop(columns=["plan"]), state_dir=self.state)
        self.assertIn("plan", str(ctx.warning))
        self.assertNotIn("plan", [r["column"] for r in self.rows(guard.DICTIONARY_FILE)])

    def test_dictionary_needs_a_profile(self):
        (self.state / guard.PROFILE_FILE).unlink()
        with self.assertRaises(guard.LeakageError):
            guard.data_dictionary(self.df, state_dir=self.state)

    def test_awkward_column_names_round_trip(self):
        df = pd.DataFrame({0: [1.5, 2.5, 3.5, 4.5], 'weight, "kg"': [60.0, 70.0, 80.0, 90.0], "y": [0, 1, 0, 1]})
        guard.profile(df, target="y", state_dir=self.state)
        guard.data_dictionary(df, state_dir=self.state)
        self.assertEqual([r["column"] for r in self.rows(guard.DICTIONARY_FILE)], ["0", 'weight, "kg"', "y"])

    def test_semicolon_and_legacy_excel_saves_keep_the_words(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        rows = self.rows(guard.DICTIONARY_FILE)
        rows[0].update(meaning="age at admission", unit="years °")
        path = self.state / guard.DICTIONARY_FILE
        for encoding, delimiter in (("cp1252", ";"), ("cp1252", ","), ("utf-8", ";")):
            with self.subTest(encoding=encoding, delimiter=delimiter):
                with path.open("w", newline="", encoding=encoding) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(rows)
                guard.data_dictionary(self.df, state_dir=self.state)
                again = self.rows(guard.DICTIONARY_FILE)
                self.assertEqual((again[0]["meaning"], again[0]["unit"]), ("age at admission", "years °"))

    def test_an_unreadable_record_is_never_overwritten(self):
        path = self.state / guard.DICTIONARY_FILE
        path.write_text("colonne|sens\nage|âge\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            guard.data_dictionary(self.df, state_dir=self.state)
        self.assertEqual(path.read_text(encoding="utf-8"), "colonne|sens\nage|âge\n")

    def test_missingness_rows_counts_and_pattern(self):
        guard.missingness(self.df, state_dir=self.state)
        rows = {r["column"]: r for r in self.rows(guard.MISSINGNESS_FILE)}
        self.assertEqual(set(rows), {"income", "plan"})
        self.assertEqual((rows["income"]["n_missing"], rows["income"]["pct_missing"]), ("40", "6.67"))
        self.assertGreater((self.state / "figures" / "01_missing_pattern.png").stat().st_size, 1024)
        self.assertIn("- Figure 01_missing_pattern.png: 2 columns have gaps", (self.state / "PIPELINE.md").read_text())

    def test_missingness_single_column_and_no_gaps(self):
        guard.missingness(self.df.drop(columns=["plan"]), state_dir=self.state)
        self.assertIn("Only income has gaps", (self.state / "PIPELINE.md").read_text())
        clean = Path(self.tmp.name) / "clean"
        guard.missingness(self.df.dropna(), state_dir=clean)
        self.assertEqual(self.rows(guard.MISSINGNESS_FILE, state=clean), [])
        self.assertFalse((clean / "figures" / "01_missing_pattern.png").exists())

    def test_categorical_plan_rows(self):
        guard.categorical_plan(self.df, state_dir=self.state)
        rows = {r["column"]: r for r in self.rows(guard.PLAN_FILE)}
        self.assertEqual(set(rows), {"zip", "plan", "smoker"})
        self.assertEqual((rows["zip"]["high_cardinality"], rows["plan"]["n_levels"], rows["smoker"]["high_cardinality"]),
                         ("yes", "3", "no"))


class VariablePagesAndLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.df = messy()

    def tearDown(self):
        self.tmp.cleanup()

    def pages(self):
        return json.loads((self.state / "figures" / guard.PAGES_FILE).read_text())

    def test_pages_cover_every_column(self):
        paths = guard.variable_figures(self.df, target="y", per_page=3, state_dir=self.state)
        self.assertEqual([p.name for p in paths], ["02_variables_p01.png", "02_variables_p02.png", "02_variables_p03.png"])
        self.assertEqual(sorted(c for cols in self.pages().values() for c in cols), sorted(self.df.columns))
        ledger = dict(re.findall(r"^- Figure (\S+): (.*)$", (self.state / "PIPELINE.md").read_text(), re.M))
        for p in paths:
            self.assertGreater(p.stat().st_size, 1024)
            self.assertIsNone(guard.note_problem(ledger[p.name]))

    def test_wide_frame_pages(self):
        wide = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 30)), columns=[f"f{i}" for i in range(30)])
        wide["y"] = np.arange(50) % 2
        self.assertEqual(len(guard.variable_figures(wide, target="y", state_dir=self.state)), 3)  # 31 columns / 12

    def test_all_missing_column_gets_a_panel(self):
        guard.variable_figures(self.df.assign(empty=np.nan), target="y", state_dir=self.state)
        self.assertIn("empty", [c for cols in self.pages().values() for c in cols])

    def test_label_audit_template_and_counts(self):
        flipped = self.df.iloc[:5].assign(y=1 - self.df["y"].iloc[:5])
        df = pd.concat([self.df, flipped], ignore_index=True)
        guard.label_audit(df, target="y", state_dir=self.state)
        text = (self.state / guard.LABEL_AUDIT_FILE).read_text()
        counts = df["y"].value_counts()
        self.assertIn(f"0 = {counts[0]:,} (", text)
        self.assertIn(f"1 = {counts[1]:,} (", text)
        self.assertIn("- identical feature rows with different labels: 10 rows in 5 groups", text)
        for section in guard.LABEL_SECTIONS:
            self.assertIn(f"## {section}\n", text)

    def test_label_audit_refresh_keeps_the_writing(self):
        guard.label_audit(self.df, target="y", state_dir=self.state)
        path = self.state / guard.LABEL_AUDIT_FILE
        path.write_text(path.read_text().replace("## Source\n", "## Source\nClaims data from the hospital billing team.\n"))
        guard.label_audit(self.df.iloc[:300], target="y", state_dir=self.state)
        text = path.read_text()
        self.assertIn("Claims data from the hospital billing team.", text)
        self.assertIn("- rows: 300", text)
        self.assertEqual(text.count(guard._COMPUTED_START), 1)

    def test_label_audit_regression_line(self):
        guard.label_audit(self.df.assign(y=np.random.default_rng(3).normal(size=len(self.df))), target="y",
                          state_dir=self.state)
        self.assertIn("- target: min ", (self.state / guard.LABEL_AUDIT_FILE).read_text())


class CleaningAndAssociations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_cleaning_counts(self):
        before = messy()
        after = before.drop(index=range(10)).assign(age=before["age"].clip(upper=60))
        returned = guard.log_cleaning("clip age at 60 and drop 10 rows", before, after,
                                      reason="entry errors above 60", state_dir=self.state)
        self.assertIs(returned, after)
        with (self.state / guard.CLEANING_FILE).open(encoding="utf-8") as handle:
            row = list(csv.DictReader(handle))[0]
        changed = int((before["age"].drop(index=range(10)) > 60).sum())
        self.assertEqual((row["rows_before"], row["rows_after"], row["cells_changed"], row["columns_changed"]),
                         ("600", "590", str(changed), "age"))

    def test_log_cleaning_needs_a_reason(self):
        with self.assertRaises(ValueError):
            guard.log_cleaning("drop rows", messy(), messy(), reason=" ", state_dir=self.state)

    @staticmethod
    def signal_frame(n=800):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=n)
        weight = rng.normal(70, 10, size=n)
        return pd.DataFrame({"signal": signal, "noise": rng.normal(size=n), "weight": weight,
                             "bmi": weight * 0.3 + rng.normal(0, 0.1, size=n), "plan": rng.choice(["a", "b"], n),
                             "y": (signal + rng.normal(0, 0.3, size=n) > 0).astype(int)})

    def test_train_associations(self):
        tr, _, _ = guard.split(self.signal_frame(), target="y", state_dir=self.state)
        result = guard.train_associations(tr, target="y", state_dir=self.state)
        self.assertEqual(result["feature_target"][0]["column"], "signal")
        self.assertEqual((result["computed_on"]["split"], result["computed_on"]["n_rows"]), ("train", len(tr)))
        self.assertEqual([(p["a"], p["b"]) for p in result["collinear_pairs"]], [("weight", "bmi")])
        measures = {e["column"]: e["measure"] for e in result["feature_target"]}
        self.assertEqual((measures["signal"], measures["plan"]), ("eta", "cramers_v"))
        for name in ("07_correlations_train.png", "07_feature_target_train.png"):
            self.assertGreater((self.state / "figures" / name).stat().st_size, 1024)
        saved = json.loads((self.state / guard.ASSOCIATIONS_FILE).read_text())
        self.assertEqual(saved["computed_on"]["n_rows"], len(tr))

    def test_regression_target_measures(self):
        df = self.signal_frame()
        df["y"] = df["signal"] * 2 + np.random.default_rng(1).normal(0, 0.1, len(df))
        tr, _, _ = guard.split(df, target="y", state_dir=self.state)
        result = guard.train_associations(tr, target="y", state_dir=self.state)
        measures = {e["column"]: e["measure"] for e in result["feature_target"]}
        self.assertEqual((measures["signal"], measures["plan"]), ("spearman", "eta"))
        self.assertEqual(result["feature_target"][0]["column"], "signal")

    def test_associations_refuse_the_full_frame(self):
        df = self.signal_frame()
        guard.split(df, target="y", state_dir=self.state)
        with self.assertRaises(guard.NotTrainingSplit):
            guard.train_associations(df, target="y", state_dir=self.state)

    def test_associations_need_a_split(self):
        with self.assertRaises(guard.LeakageError):
            guard.train_associations(self.signal_frame(), target="y", state_dir=self.state)


class Stub:
    """A model that remembers how many rows it was fit on."""
    def fit(self, X, y=None):
        self.n_rows = len(X)
        return self


class TrainingAndImportance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        rng = np.random.default_rng(9)
        signal = rng.normal(size=600)
        df = pd.DataFrame({"signal": signal, "noise": rng.normal(size=600), "y": (signal > 0).astype(int)})
        self.tr, self.va, self.te = guard.split(df, target="y", state_dir=self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def runs(self):
        return [json.loads(line) for line in (self.state / guard.TRAINING_FILE).read_text().splitlines()]

    @staticmethod
    def predict(X):
        return (X["signal"] > 0).astype(int).to_numpy()

    @staticmethod
    def accuracy(y_true, y_pred):
        return float((y_true == y_pred).mean())

    def test_fit_records_a_full_run(self):
        model = guard.fit(Stub(), self.tr.drop(columns="y"), self.tr["y"], state_dir=self.state)
        self.assertEqual(model.n_rows, len(self.tr))
        run = self.runs()[0]
        self.assertEqual((run["model"], run["n_rows"], run["n_train"], run["full"]),
                         ("Stub", len(self.tr), len(self.tr), True))

    def test_a_slice_is_recorded_as_not_full(self):
        guard.fit(Stub(), self.tr.drop(columns="y").iloc[:50], self.tr["y"].iloc[:50], name="pilot",
                  state_dir=self.state)
        self.assertEqual((self.runs()[0]["model"], self.runs()[0]["full"]), ("pilot", False))

    def test_fit_needs_a_split_before_training(self):
        model = Stub()
        with self.assertRaises(guard.LeakageError):
            guard.fit(model, self.tr, state_dir=Path(self.tmp.name) / "nosplit")
        self.assertFalse(hasattr(model, "n_rows"))

    def test_record_training_measures_len(self):
        guard.record_training(self.tr, name="mlp", state_dir=self.state)
        self.assertTrue(self.runs()[0]["full"])
        with self.assertRaises(TypeError):
            guard.record_training(5, name="typed", state_dir=self.state)

    def test_sparse_like_inputs_are_counted_by_shape(self):
        class SparseLike:
            def __init__(self, n):
                self.shape = (n, 3)

            def __len__(self):
                raise TypeError("sparse array length is ambiguous")

        class ShapeStub:
            def fit(self, X, y=None):
                self.n_rows = X.shape[0]
                return self

        guard.fit(ShapeStub(), SparseLike(len(self.tr)), self.tr["y"], state_dir=self.state)
        self.assertEqual((self.runs()[0]["n_rows"], self.runs()[0]["full"]), (len(self.tr), True))

    def test_resampled_training_counts_its_source_rows(self):
        X, y = self.tr.drop(columns="y"), self.tr["y"]
        X_res, y_res = pd.concat([X, X.iloc[:100]]), pd.concat([y, y.iloc[:100]])
        guard.fit(Stub(), X_res, y_res, rows_from=X, state_dir=self.state)
        run = self.runs()[0]
        self.assertEqual((run["n_rows"], run["rows_from"], run["full"]), (len(X) + 100, len(X), True))

    def test_importance_ranks_the_signal_first(self):
        result = guard.importance(self.predict, self.va, target="y", metric_fn=self.accuracy, state_dir=self.state)
        self.assertEqual([f["column"] for f in result["features"]], ["signal", "noise"])
        self.assertGreater(result["features"][0]["importance_mean"], 0.3)
        self.assertEqual(result["features"][1]["importance_mean"], 0.0)
        self.assertEqual((result["computed_on"]["split"], result["computed_on"]["n_rows"]), ("validation", len(self.va)))
        self.assertGreater((self.state / "figures" / "12_permutation_importance.png").stat().st_size, 1024)

    def test_lower_is_better_metrics(self):
        def mse(y_true, y_pred):
            return float(((y_true - y_pred) ** 2).mean())
        result = guard.importance(self.predict, self.va, target="y", metric_fn=mse, greater_is_better=False,
                                  state_dir=self.state)
        self.assertGreater(result["features"][0]["importance_mean"], 0.3)

    def test_importance_refuses_other_frames(self):
        with self.assertRaises(guard.NotValidationSplit):
            guard.importance(self.predict, self.tr, target="y", metric_fn=self.accuracy, state_dir=self.state)

if __name__ == "__main__":
    unittest.main()
