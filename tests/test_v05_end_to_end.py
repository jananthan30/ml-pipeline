"""End to end: the honest v0.5 path records Gates A-C through the real hook; each shortcut is stopped.

Run with the dev venv: .venv/bin/python -m unittest tests/test_v05_end_to_end.py
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
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "hooks"))
try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - the stdlib suite skips this file
    raise unittest.SkipTest("pandas/numpy not installed; run with .venv/bin/python")
import guard_training as hook  # noqa: E402
import mlpipeline_guard as guard  # noqa: E402

RATIONALE = ("Model rationale:\n- traits: binary, 600 rows (small), has_datetime, no groups\n"
             "- baseline: majority-class dummy + logistic regression\n- candidates: gradient boosting with small trees\n"
             "- ruled out: neural nets (small data); random split (temporal)\n- metric: balanced accuracy\n")
SECTIONS = {
    "Definition": "Readmission within 30 days of discharge, counted once per admission.",
    "Source": "Claims data joined on patient number by the hospital data team.",
    "When labeled": "Known 30 days after discharge, after the prediction time at discharge.",
    "Noise and consistency": "Late claims can hide a few readmissions; conflicting rows are listed above.",
}


def admissions(n=600, seed=11):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    df = pd.DataFrame({
        "admitted_at": pd.date_range("2024-01-01", periods=n, freq="12h"),
        "age": rng.normal(55, 14, n).round(1),
        "income": rng.lognormal(10, 0.8, n).round(0),
        "zip": rng.integers(0, 90, n).astype(str),
        "plan": rng.choice(["basic", "plus", "pro"], n).astype(object),
        "smoker": rng.integers(0, 2, n).astype(bool),
        "prior_visits": np.clip(np.round(signal * 2 + 3), 0, None),
        "patient_id": np.arange(n),
        "y": (signal + rng.normal(0, 0.8, n) > 0).astype(int),
    })
    df.loc[rng.choice(n, 30, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "plan"] = np.nan
    return df


class Threshold:
    """Predicts readmission when prior visits exceed 3."""
    def fit(self, X, y=None):
        return self

    @staticmethod
    def predict_frame(X):
        return (X["prior_visits"] > 3).astype(int).to_numpy()


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"
        self.csv = self.root / "admissions.csv"
        admissions().to_csv(self.csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def hook(self, tool_input, tool="Write"):
        """The real PreToolUse hook, in-process, exactly as Claude Code calls it."""
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(self.root)})
        out = io.StringIO()
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(out), mock.patch("sys.stdin", io.StringIO(payload)):
            os.environ.pop("ML_PIPELINE_ENFORCE", None)
            os.environ.pop("ML_PIPELINE_FILE", None)
            hook.main()
        result = json.loads(out.getvalue() or "{}").get("hookSpecificOutput", {})
        return result.get("permissionDecision", "allow"), result.get("permissionDecisionReason", "")

    def record(self, gate, extra=""):
        """Ask the hook to append a gate line to PIPELINE.md; write it only when allowed."""
        md = self.state / "PIPELINE.md"
        content = (md.read_text() if md.is_file() else "") + extra + f"- Gate {gate}: approved 2026-09-24\n"
        decision, reason = self.hook({"file_path": str(md), "content": content})
        if decision == "allow":
            md.write_text(content)
        return decision, reason

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

    def phase1(self, profiled=None):
        df = guard.load(self.csv, state_dir=self.state, parse_dates=["admitted_at"])
        prof = guard.profile(df if profiled is None else profiled(df), target="y", time_col="admitted_at",
                             state_dir=self.state)
        guard.eda_figures(df, target="y", time_col="admitted_at", state_dir=self.state)
        guard.variable_figures(df, target="y", state_dir=self.state)
        guard.data_dictionary(df, state_dir=self.state)
        roles = {"y": "target", "patient_id": "id"}

        def fill(row):
            info = prof["columns"][row["column"]]
            row.update(meaning=f"{row['column']} as recorded at admission", unit="none", allowed="as observed",
                       source="admissions.csv", available_at_prediction="no" if row["column"] == "y" else "yes",
                       role=roles.get(row["column"], "feature"),
                       figure_note=(f"{row['column']}: {info['non_null']:,} recorded values and "
                                    f"{info['n_unique']:,} distinct ones in the full data"))
        self.edit_csv(guard.DICTIONARY_FILE, fill)
        guard.missingness(df, state_dir=self.state)
        self.edit_csv(guard.MISSINGNESS_FILE, lambda r: r.update(
            mechanism="random", evidence="gaps are spread evenly across admission months",
            decision="impute-median" if r["column"] == "income" else "impute-mode",
            reason="few gaps and no pattern over time"))
        guard.label_audit(df, target="y", state_dir=self.state)
        suspects = ", ".join(s["column"] for s in prof["leakage_suspects"]) or "none"
        bodies = dict(SECTIONS, **{"Leakage check": f"Profile leakage suspects: {suspects}; none is recorded after discharge."})
        audit = self.state / guard.LABEL_AUDIT_FILE
        text = audit.read_text()
        for name, body in bodies.items():
            text = re.sub(rf"(## {name}\n)<!--.*?-->", lambda m, body=body: m.group(1) + body, text, flags=re.S)
        audit.write_text(text)
        return df

    def phase2(self, df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        clipped = df.assign(age=df["age"].clip(18, 100))
        cleaned = guard.log_cleaning("clip age to 18..100", df, clipped, reason="ages outside 18..100 are entry errors",
                                     state_dir=self.state)
        train, val, test = guard.split(cleaned, target="y", time_col="admitted_at", state_dir=self.state)
        guard.categorical_plan(cleaned, state_dir=self.state)
        self.edit_csv(guard.PLAN_FILE, lambda r: r.update(
            encoding="rare-grouping+one-hot" if r["high_cardinality"] == "yes" else "one-hot",
            reason="levels kept separate; rare zips grouped first"))
        guard.train_associations(train, target="y", state_dir=self.state)
        moved = int(((df["age"] < 18) | (df["age"] > 100)).sum())
        f, ax = plt.subplots()
        ax.bar(["before", "after"], [len(df), len(cleaned)])
        guard.fig(4, "cleaning", f, f"Clipping age kept all {len(cleaned):,} rows; {moved} values moved into 18..100.",
                  state_dir=self.state)
        return train, val, test

    def phase3(self, train, val, rows=None):
        import matplotlib.pyplot as plt
        X, y = train.drop(columns=["y"]), train["y"]
        if rows is not None:
            X, y = X.iloc[:rows], y.iloc[:rows]
        guard.fit(Threshold(), X, y, state_dir=self.state)
        guard.importance(Threshold.predict_frame, val, target="y", metric_fn=accuracy, state_dir=self.state)
        f, ax = plt.subplots()
        ax.bar(["prior_visits"], [1.0])
        guard.fig(12, "attribution_rule", f, "The model predicts readmission when prior_visits exceeds 3, its only input.",
                  state_dir=self.state)

    def test_honest_path_records_every_gate(self):
        df = self.phase1()
        self.assertEqual(self.record("A"), ("allow", ""))
        train, val, _ = self.phase2(df)
        self.assertEqual(self.record("B", extra=RATIONALE), ("allow", ""))
        self.phase3(train, val)
        self.assertEqual(self.record("C"), ("allow", ""))

    def test_a_sampled_profile_is_stopped(self):
        self.phase1(profiled=lambda df: df.sample(frac=0.5, random_state=0))
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("does not cover the full data", reason)

    def test_a_missing_column_note_is_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE, lambda r: r.update(figure_note="") if r["column"] == "income" else None)
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("blank fields: income (figure_note)", reason)

    def test_copy_pasted_notes_are_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE,
                      lambda r: r.update(figure_note="numbers look fine, 600 rows, nothing unusual in this one"))
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("columns share the same figure note", reason)

    def test_associations_on_the_full_frame_are_refused(self):
        df = self.phase1()
        guard.split(df, target="y", time_col="admitted_at", state_dir=self.state)
        with self.assertRaises(guard.NotTrainingSplit):
            guard.train_associations(df, target="y", state_dir=self.state)

    def test_pilot_training_is_stopped(self):
        df = self.phase1()
        self.assertEqual(self.record("A")[0], "allow")
        train, val, _ = self.phase2(df)
        self.assertEqual(self.record("B", extra=RATIONALE)[0], "allow")
        decision, reason = self.hook({"command": "python -c 'model.fit(X_train[:100], y_train[:100])'"}, tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("slice of the data", reason)
        self.phase3(train, val, rows=100)
        decision, reason = self.record("C")
        self.assertEqual(decision, "deny")
        self.assertIn("no full-data training run", reason)


if __name__ == "__main__":
    unittest.main()
