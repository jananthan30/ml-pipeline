"""End to end: one project walks big data AND the rigorous-data-work records through the real hooks.

The data is a CSV file profiled, split, and scored in batches, with SAMPLE_ROWS and BATCH_ROWS made
small so every streaming path (sampled percentiles, out-of-core split, sampled associations and
importance) really runs. Every ledger line goes through the real PreToolUse hook, and every gate is
approved through the real UserPromptSubmit hook from a user message. The honest path records Gates
A-D; each shortcut is stopped. A second test runs the canary honestly and checks the final test score
against its known ceiling.

Ported from the v0.5-rigorous-data-work branch (tests/test_v05_end_to_end.py) and extended with the
big-data path and user-only approvals. Run with the dev requirements:
    python -m pytest tests/test_v05_end_to_end.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "skills" / "ml-pipeline" / "lib"))
sys.path.insert(0, str(ROOT / "hooks"))
try:
    import numpy as np
    import pandas as pd
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - the stdlib suite skips this file
    raise unittest.SkipTest("pandas/numpy/scikit-learn not installed; pip install -r tests/requirements-dev.txt")
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

import guard_training as pre_tool  # noqa: E402
import mlpipeline_canary as canary  # noqa: E402
import mlpipeline_guard as guard  # noqa: E402
import record_approval as user_prompt  # noqa: E402

N_ROWS = 3000
CONTRACT = ("Prediction contract:\n- target: y = 1 if readmitted within 30 days of discharge\n"
            "- unit: one row per admission\n- horizon: 30 days after discharge\n"
            "- available: features known at discharge\n- metric: balanced accuracy\n")
RATIONALE = ("Model rationale:\n- traits: binary, 3,000 rows (small), has_datetime, no repeating entity\n"
             "- baseline: majority-class dummy + logistic regression\n- candidates: gradient boosting with small trees\n"
             "- ruled out: neural nets (small data); random split (temporal)\n- metric: balanced accuracy\n")
SECTIONS = {
    "Definition": "Readmission within 30 days of discharge, counted once per admission.",
    "Source": "Claims data joined on patient number by the hospital data team.",
    "When labeled": "Known 30 days after discharge, after the prediction time at discharge.",
    "Noise and consistency": "Late claims can hide a few readmissions; conflicting rows are listed above.",
}
NUMERIC = ["age", "income", "prior_visits"]
CATEGORICAL = ["plan", "zip", "smoker"]
TRAIN_SCRIPT = "from sklearn.linear_model import LogisticRegression\nLogisticRegression().fit(X_train, y_train)\n"


def admissions(n=N_ROWS, seed=11):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    df = pd.DataFrame({
        "admitted_at": pd.date_range("2024-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M"),  # dates as text
        "age": rng.normal(55, 14, n).round(1),
        "income": rng.lognormal(10, 0.8, n).round(0),
        "zip": np.char.add("Z", rng.integers(0, 90, n).astype(str)),
        "plan": rng.choice(["basic", "plus", "pro"], n).astype(object),
        "smoker": rng.choice(["yes", "no"], n),
        "prior_visits": np.clip(np.round(signal * 2 + 3), 0, None),
        "patient_id": np.arange(100_000, 100_000 + n),
        "y": (signal + rng.normal(0, 0.8, n) > 0).astype(int),
    })
    df.loc[rng.choice(n, 90, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "plan"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "age"] = 140.0   # entry errors, clipped at step 4
    return df


def model_pipeline(estimator):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return make_pipeline(pre, estimator)


class Project(unittest.TestCase):
    """A temp project with the real hooks wired in-process, as Claude Code would call them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"
        self.md = self.state / "PIPELINE.md"
        self.csv = self.root / "admissions.csv"
        admissions().to_csv(self.csv, index=False)
        (self.root / "train.py").write_text(TRAIN_SCRIPT)
        self.small = mock.patch.multiple(guard, SAMPLE_ROWS=1000, BATCH_ROWS=700)
        self.small.start()
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        for key in ("ML_PIPELINE_ENFORCE", "ML_PIPELINE_FILE", "ML_PIPELINE_REQUIRE_USER_APPROVAL", "ML_PIPELINE_TEST_NAMES"):
            os.environ.pop(key, None)

    def tearDown(self):
        self.env.stop()
        self.small.stop()
        self.tmp.cleanup()

    # ------------------------------------------------------------------ the hooks
    def hook(self, tool_input, tool="Write"):
        """The real PreToolUse hook: (decision, reason)."""
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(self.root)})
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("sys.stdin", io.StringIO(payload)):
            pre_tool.main()
        result = json.loads(out.getvalue() or "{}").get("hookSpecificOutput", {})
        return result.get("permissionDecision", "allow"), result.get("permissionDecisionReason", "")

    def user_says(self, prompt):
        """The user's chat message through the real UserPromptSubmit hook."""
        payload = json.dumps({"prompt": prompt, "cwd": str(self.root), "session_id": "e2e"})
        with contextlib.redirect_stdout(io.StringIO()), mock.patch("sys.stdin", io.StringIO(payload)):
            user_prompt.main()

    def write_ledger(self, lines):
        """Append ``lines`` to PIPELINE.md through the hook; write them only when allowed."""
        existing = self.md.read_text() if self.md.is_file() else ""
        content = existing.rstrip("\n") + "\n" + lines if existing else lines
        decision, reason = self.hook({"file_path": str(self.md), "content": content})
        if decision == "allow":
            self.md.parent.mkdir(parents=True, exist_ok=True)
            self.md.write_text(content)
        return decision, reason

    def assert_recorded(self, lines):
        self.assertEqual(self.write_ledger(lines), ("allow", ""))

    def edit_csv(self, name, fill):
        path = self.state / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header, rows = reader.fieldnames, list(reader)
        for row in rows:
            fill(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def figure(self, step, name, text):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.bar(["a", "b"], [1, 2])
        return guard.fig(step, name, f, text, state_dir=self.state)

    # ------------------------------------------------------------------ phase 1: steps 1-3, Gate A
    def phase1(self, profiled=None):
        self.md.parent.mkdir(parents=True, exist_ok=True)
        self.md.write_text("# Pipeline: 30-day readmission\n")
        data = self.csv if profiled is None else profiled
        prof = guard.profile(data, target="y", time_col="admitted_at", state_dir=self.state)
        self.assert_recorded("- Step 1 done: profiled every row of admissions.csv\n")
        guard.eda_figures(data, target="y", time_col="admitted_at", state_dir=self.state)
        guard.data_dictionary(state_dir=self.state)
        roles = {"y": "target", "patient_id": "id"}

        def cite(info):
            """A figure of the column's own, as a person reading its variable page would note."""
            if info["kind"] == "numeric":
                return f"mean {info['mean']}"
            if info["kind"] in ("categorical", "bool"):
                return f"{info['n_levels']} levels"
            if info["kind"] == "datetime":
                return f"{info['span_days']} days covered"
            return f"{info['unique_frac']:.0%} unique"

        def fill(row):
            info = prof["columns"][row["column"]]
            row.update(meaning=f"{row['column']} as recorded at admission", unit="none", allowed="as observed",
                       source="admissions.csv", available_at_prediction="no" if row["column"] == "y" else "yes",
                       role=roles.get(row["column"], "feature"),
                       figure_note=f"{row['column']}: {cite(info)} in the full data, checked on its variable page")
        self.edit_csv(guard.DICTIONARY_FILE, fill)
        guard.missingness(data, state_dir=self.state)
        self.edit_csv(guard.MISSINGNESS_FILE, lambda r: r.update(
            mechanism="random", evidence="gaps are spread evenly across admission months"))
        guard.label_audit(data, target="y", state_dir=self.state)
        suspects = ", ".join(s["column"] for s in prof["leakage_suspects"]) or "none"
        bodies = dict(SECTIONS, **{"Leakage check": f"Profile leakage suspects: {suspects}; none is recorded after discharge."})
        audit = self.state / guard.LABEL_AUDIT_FILE
        text = audit.read_text()
        for name, body in bodies.items():
            text = re.sub(rf"(## {name}\n)<!--.*?-->", lambda m, body=body: m.group(1) + body, text, flags=re.S)
        audit.write_text(text)
        self.assert_recorded("- Step 2 done: every column drawn and noted\n")
        self.assert_recorded(CONTRACT + "- Step 3 done: prediction contract written\n")
        return prof

    def gate(self, letter):
        self.user_says(f"Thanks for the report. I approve gate {letter}.")
        return self.write_ledger(f"- Gate {letter}: approved 2026-10-30\n")

    # ------------------------------------------------------------------ phase 2: steps 4-8, Gate B
    def phase2(self):
        df = guard.load(self.csv, state_dir=self.state)
        cleaned = guard.log_cleaning("clip age to 18..110", df, df.assign(age=df["age"].clip(18, 110)),
                                     reason="ages above 110 are entry errors", state_dir=self.state)
        moved = int((df["age"] > 110).sum())
        self.figure(4, "cleaning", f"Clipping age kept all {len(cleaned):,} rows; {moved} values moved into 18..110.")
        self.assert_recorded("- Step 4 done: clipped 4 impossible ages\n")
        clean = self.root / "clean.parquet"
        guard.derive(cleaned, "clean.parquet", parents=[str(self.csv)], how="age clipped to 18..110",
                     state_dir=self.state).to_parquet(clean, index=False)
        self.assert_recorded("- Step 5 skipped: one table, nothing to join\n")
        paths = guard.split_to_files(clean, target="y", time_col="admitted_at", out_dir=self.root / "splits",
                                     state_dir=self.state)
        self.assert_recorded("- Step 6 done: chronological split frozen\n")
        self.edit_csv(guard.MISSINGNESS_FILE, lambda r: r.update(
            decision="impute-median" if r["column"] == "income" else "impute-mode", reason="few gaps and no pattern over time"))
        guard.categorical_plan(state_dir=self.state)
        self.edit_csv(guard.PLAN_FILE, lambda r: r.update(
            encoding="rare-grouping+one-hot" if r["high_cardinality"] == "yes" else "one-hot",
            reason="levels kept separate; rare zips grouped first"))
        guard.train_associations(paths["train"], target="y", state_dir=self.state)
        self.assert_recorded("- Step 7 done: categorical plan and train-only associations\n")
        self.assert_recorded("- Step 8 done: median/mode imputation and one-hot inside the model pipeline\n")
        self.assert_recorded(RATIONALE)
        return paths

    # ------------------------------------------------------------------ phase 3: steps 9-12, Gate C
    def phase3(self, paths, rows=None):
        train, val = pd.read_parquet(paths["train"]), pd.read_parquet(paths["val"])
        X, y = train.drop(columns=["y"]), train["y"]
        Xv, yv = val.drop(columns=["y"]), val["y"]
        if rows is not None:
            X, y = X.iloc[:rows], y.iloc[:rows]
        dummy = guard.fit(DummyClassifier(), X, y, state_dir=self.state)
        logistic = guard.fit(model_pipeline(LogisticRegression(max_iter=500)), X, y, state_dir=self.state)
        for name, model in (("dummy", dummy), ("logistic", logistic)):
            guard.log_run(name, kind="baseline", model=model, state_dir=self.state,
                          metrics={"balanced_accuracy": balanced_accuracy_score(yv, model.predict(Xv))})
        self.assert_recorded("- Step 9 done: dummy 0.50, logistic beats it on validation\n")
        boosting = guard.fit(model_pipeline(HistGradientBoostingClassifier(max_depth=3, random_state=0)), X, y,
                             state_dir=self.state)
        guard.log_run("boosting", kind="candidate", model=boosting, state_dir=self.state,
                      metrics={"balanced_accuracy": balanced_accuracy_score(yv, boosting.predict(Xv))})
        self.assert_recorded("- Step 10 done: boosting compared with logistic on validation\n")
        self.assert_recorded("- Step 11 skipped: defaults are within noise of each other\n")
        guard.importance(logistic.predict, paths["val"], target="y", metric_fn=balanced_accuracy_score, state_dir=self.state)
        coef = logistic[-1].coef_[0]
        self.figure(12, "attribution_coefficients",
                    f"The largest logistic coefficient is {abs(coef).max():.2f} (scaled units) across {len(coef)} inputs.")
        self.assert_recorded("- Step 12 done: importance on validation and coefficients drawn\n")
        return logistic

    # ------------------------------------------------------------------ the honest path
    def test_honest_path_records_every_gate(self):
        self.assertEqual(self.hook({"command": "python train.py"}, tool="Bash")[0], "deny")   # nothing before Gate A
        prof = self.phase1()
        self.assertTrue(prof["shape"]["sampled"], "the big-data path must really sample")
        self.assertEqual(prof["shape"]["n_rows"], N_ROWS)
        self.assertTrue(prof["source"]["full_data"])
        self.assertEqual(prof["columns"]["admitted_at"]["kind"], "datetime")
        self.assertEqual(self.gate("A"), ("allow", ""))
        self.assertEqual(self.hook({"command": "python train.py"}, tool="Bash")[0], "deny")   # training waits for B
        paths = self.phase2()
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual(state["n_train"] + state["n_val"] + state["n_test"], N_ROWS)
        self.assertEqual(json.loads((self.state / guard.ASSOCIATIONS_FILE).read_text())["computed_on"]["n_rows"],
                         state["n_train"])
        self.assertEqual(self.gate("B"), ("allow", ""))
        self.assertEqual(self.hook({"command": "python train.py"}, tool="Bash"), ("allow", ""))
        model = self.phase3(paths)
        self.assertEqual(self.gate("C"), ("allow", ""))
        self.figure(13, "errors", "Validation errors are spread evenly: no month differs by more than 4 points.")
        self.assert_recorded("- Step 13 done: no slice stands out\n")
        score = guard.final_test(model.predict, paths["test"], target="y", metric_fn=balanced_accuracy_score,
                                 state_dir=self.state, pipeline_md=self.md, batch_rows=300)
        self.assertGreater(score, 0.6)
        self.assertEqual(self.gate("D"), ("allow", ""))

    # ------------------------------------------------------------------ each shortcut is stopped
    def test_the_agent_cannot_approve_its_own_gate(self):
        self.phase1()
        decision, reason = self.write_ledger("- Gate A: approved 2026-10-30\n")   # no user message
        self.assertEqual(decision, "deny")
        self.assertIn("the user has not approved Gate A in chat", reason)
        self.assertEqual(self.hook({"file_path": str(self.state / ".approvals.json"),
                                    "content": json.dumps({"gates": {"A": {"run": ""}}})})[0], "deny")
        self.assertEqual(self.hook({"command": "export ML_PIPELINE_REQUIRE_USER_APPROVAL=0"}, tool="Bash")[0], "deny")
        self.assertEqual(self.hook({"command": "ML_PIPELINE_ENFORCE=0 python train.py"}, tool="Bash")[0], "deny")

    def test_a_partial_load_or_sampled_profile_is_stopped(self):
        with self.assertRaises(guard.PartialData):
            guard.load(self.csv, nrows=500, state_dir=self.state)
        df = guard.load(self.csv, state_dir=self.state)
        self.phase1(profiled=df.sample(frac=0.5, random_state=0))
        decision, reason = self.gate("A")
        self.assertEqual(decision, "deny")
        self.assertIn("does not cover the full data", reason)

    def test_a_missing_column_note_is_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE, lambda r: r.update(figure_note="") if r["column"] == "income" else None)
        decision, reason = self.gate("A")
        self.assertEqual(decision, "deny")
        self.assertIn("blank fields: income (figure_note)", reason)

    def test_copy_pasted_notes_are_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE,
                      lambda r: r.update(figure_note="numbers look fine, 3000 rows, nothing unusual in this one"))
        decision, reason = self.gate("A")
        self.assertEqual(decision, "deny")
        self.assertIn("columns share the same figure note", reason)

    def test_target_statistics_before_the_split_are_stopped(self):
        self.phase1()
        self.assertEqual(self.gate("A")[0], "allow")
        df = guard.load(self.csv, state_dir=self.state)
        guard.split_to_files(self.csv, target="y", time_col="admitted_at", out_dir=self.root / "splits", state_dir=self.state)
        with self.assertRaises(guard.NotTrainingSplit):
            guard.train_associations(df, target="y", state_dir=self.state)
        decision, reason = self.hook({"command": "python -c 'SelectKBest(k=3).fit(X, y)'"}, tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("target-stats-outside-train", reason)

    def test_pilot_training_and_skipping_the_baseline_are_stopped(self):
        self.phase1()
        self.assertEqual(self.gate("A")[0], "allow")
        paths = self.phase2()
        self.assertEqual(self.gate("B")[0], "allow")
        decision, reason = self.hook({"command": "python -c 'LogisticRegression().fit(X_train[:100], y_train[:100])'"},
                                     tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("slice of the data", reason)
        decision, reason = self.hook({"command": "python -c 'HistGradientBoostingClassifier().fit(X_train, y_train)'"},
                                     tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("baseline first", reason)
        self.phase3(paths, rows=100)
        decision, reason = self.gate("C")
        self.assertEqual(decision, "deny")
        self.assertIn("no full-data training run", reason)

    def test_the_test_set_is_locked_then_used_once(self):
        self.phase1()
        self.assertEqual(self.gate("A")[0], "allow")
        paths = self.phase2()
        self.assertEqual(self.gate("B")[0], "allow")
        model = self.phase3(paths)
        decision, reason = self.hook({"command": "python -c 'balanced_accuracy_score(y_test, clf.predict(X_test))'"},
                                     tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("locked until Gate C", reason)
        self.assertEqual(self.gate("C")[0], "allow")
        args = dict(target="y", metric_fn=balanced_accuracy_score, state_dir=self.state, pipeline_md=self.md)
        guard.final_test(model.predict, paths["test"], **args)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            guard.final_test(model.predict, paths["test"], **args)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            guard.split_to_files(self.root / "clean.parquet", target="y", time_col="admitted_at",
                                 out_dir=self.root / "splits2", state_dir=self.state)
        decision, reason = self.write_ledger("- Override: final test - user approved 2026-10-30 - reason: new data\n")
        self.assertEqual(decision, "deny")   # the user has not asked for it
        self.assertIn("has not asked for it in chat", reason)


class Canary(unittest.TestCase):
    """The honest path on the canary dataset scores under its known ceiling (beat it and you leaked)."""

    def test_final_test_stays_under_the_honest_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "ml_pipeline"
            df = guard.load(ROOT / "examples" / "canary" / "canary.csv", state_dir=state)
            prof = guard.profile(df, target="y", state_dir=state)
            self.assertTrue(prof["source"]["full_data"])
            train, val, test = guard.split(df, target="y", state_dir=state)
            (state / "PIPELINE.md").write_text("- Gate C: approved 2026-10-30\n")
            features = [c for c in train.columns if c != "y"]
            model = guard.fit(HistGradientBoostingClassifier(max_depth=3, random_state=0), train[features], train["y"],
                              state_dir=state)

            def accuracy(y_true, y_pred):
                return float((np.asarray(y_true) == np.asarray(y_pred)).mean())

            score = guard.final_test(lambda X: model.predict(X[features]), test, target="y", metric_fn=accuracy,
                                     state_dir=state, pipeline_md=state / "PIPELINE.md")
            leaked, why = canary.verdict(score, len(test))
        print(f"\ncanary: honest test accuracy {score:.4f} on {len(test)} rows; ceiling {canary.CEILING:.2f}; {why}")
        self.assertLess(score, canary.CEILING)
        self.assertFalse(leaked, why)


if __name__ == "__main__":
    unittest.main()
