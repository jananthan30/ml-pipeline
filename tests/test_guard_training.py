"""Behavioural tests for hooks/guard_training.py.

Run:  python3 -m unittest tests/test_guard_training.py
The hook is exercised end to end through main(), with stdin/stdout swapped in-process.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import guard_training as hook  # noqa: E402

GATE_A = "# Pipeline\n- Gate A: approved 2026-09-16\n"
GATE_AB = GATE_A + "- Gate B: approved 2026-09-17\n"
OVERRIDE = GATE_A + "- Override: Gate B - user approved skipping to training 2026-09-17 - reason: deadline\n"
GATE_ABC = GATE_AB + "- Gate C: approved 2026-09-18\n"
OVERRIDE_C = GATE_AB + "- Override: Gate C - user approved evaluating on the test set 2026-09-18 - reason: demo\n"


@contextlib.contextmanager
def _stdin(text: str):
    old, sys.stdin = sys.stdin, io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = old


def run(tool_input, pipeline_md=None, tool="Bash", env=None, cwd_sub="", files=None):
    """Run the hook against a temp project. Returns (decision, reason)."""
    with tempfile.TemporaryDirectory() as tmp:
        if pipeline_md is not None:
            (Path(tmp) / "ml_pipeline").mkdir(exist_ok=True)
            (Path(tmp) / "ml_pipeline" / "PIPELINE.md").write_text(pipeline_md)
        for rel, data in (files or {}).items():
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
        cwd = Path(tmp) / cwd_sub
        cwd.mkdir(parents=True, exist_ok=True)
        if "file_path" in tool_input and str(tool_input["file_path"]).startswith("./"):
            tool_input = {**tool_input, "file_path": str(Path(tmp) / tool_input["file_path"][2:])}
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)})
        saved = dict(os.environ)
        os.environ.pop("ML_PIPELINE_ENFORCE", None)
        os.environ.pop("ML_PIPELINE_FILE", None)
        os.environ.update(env or {})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), _stdin(payload):
                hook.main()
        finally:
            os.environ.clear()
            os.environ.update(saved)
    result = json.loads(out.getvalue() or "{}")
    specific = result.get("hookSpecificOutput", {})
    return specific.get("permissionDecision", "allow"), specific.get("permissionDecisionReason", "")


class NotMLAtAll(unittest.TestCase):
    def test_plain_command_allowed(self):
        self.assertEqual(run({"command": "ls -la && git status"})[0], "allow")

    def test_fit_only_in_comment_allowed(self):
        self.assertEqual(run({"command": "python -c '# model.fit(X_train, y_train)'"})[0], "allow")

    def test_markdown_edit_allowed(self):
        tool_input = {"file_path": "/x/README.md", "content": "call model.fit(X_train, y_train)"}
        self.assertEqual(run(tool_input, tool="Write")[0], "allow")

    def test_removing_a_fit_is_allowed(self):
        tool_input = {"file_path": "/x/train.py", "old_string": "clf.fit(X, y)", "new_string": "pass"}
        self.assertEqual(run(tool_input, tool="Edit")[0], "allow")

    def test_kill_switch(self):
        cmd = {"command": "python -c 'RandomForestClassifier().fit(X_train, y_train)'"}
        self.assertEqual(run(cmd, env={"ML_PIPELINE_ENFORCE": "0"})[0], "allow")


class NoPipelineFile(unittest.TestCase):
    def test_model_training_denied(self):
        decision, reason = run({"command": "clf = RandomForestClassifier()\nclf.fit(X_train, y_train)"})
        self.assertEqual(decision, "deny")
        self.assertIn("step 1", reason)

    def test_bare_model_fit_is_treated_as_training(self):
        self.assertEqual(run({"command": "model.fit(X_train, y_train)"})[0], "deny")

    def test_transformer_fit_allowed(self):
        self.assertEqual(run({"command": "StandardScaler().fit(X_train)"})[0], "allow")

    def test_torch_training_denied(self):
        self.assertEqual(run({"command": "loss.backward()\noptimizer.step()"})[0], "deny")

    def test_xgboost_denied(self):
        self.assertEqual(run({"command": "bst = xgb.train(params, dtrain, 100)"})[0], "deny")

    def test_write_tool_scans_content(self):
        tool_input = {"file_path": "/x/03_model.py", "content": "LogisticRegression().fit(X_train, y_train)"}
        self.assertEqual(run(tool_input, tool="Write")[0], "deny")


class BeforeGateA(unittest.TestCase):
    md = "# Pipeline\n- Gate A: todo\n"

    def test_transformer_fit_denied(self):
        decision, reason = run({"command": "scaler.fit(X_train)"}, self.md)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate A", reason)

    def test_model_fit_denied(self):
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, self.md)[0], "deny")

    def test_negated_approval_does_not_count(self):
        md = "- Gate A: not yet approved\n- Gate B: awaiting approval\n"
        self.assertEqual(run({"command": "scaler.fit(X_train)"}, md)[0], "deny")


class AfterGateA(unittest.TestCase):
    def test_transformer_class_allowed(self):
        self.assertEqual(run({"command": "scaler = StandardScaler()\nscaler.fit(X_train)"}, GATE_A)[0], "allow")

    def test_transformer_variable_allowed(self):
        self.assertEqual(run({"command": "enc.fit(X_train[cat_cols])"}, GATE_A)[0], "allow")

    def test_column_transformer_allowed(self):
        cmd = "preprocessor = ColumnTransformer([('num', num_pipe, num_cols)])\npreprocessor.fit(X_train)"
        self.assertEqual(run({"command": cmd}, GATE_A)[0], "allow")

    def test_model_denied(self):
        decision, reason = run({"command": "LogisticRegression().fit(X_train, y_train)"}, GATE_A)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate B", reason)

    def test_pipeline_containing_a_model_denied(self):
        cmd = "pipe = make_pipeline(StandardScaler(), LogisticRegression())\npipe.fit(X_train, y_train)"
        self.assertEqual(run({"command": cmd}, GATE_A)[0], "deny")


class AfterGateB(unittest.TestCase):
    def test_model_allowed(self):
        self.assertEqual(run({"command": "RandomForestClassifier().fit(X_train, y_train)"}, GATE_AB)[0], "allow")

    def test_override_unlocks_training(self):
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, OVERRIDE)[0], "allow")

    def test_fit_on_test_split_still_denied(self):
        decision, reason = run({"command": "clf.fit(X_test, y_test)"}, GATE_AB)
        self.assertEqual(decision, "deny")
        self.assertIn("test", reason)

    def test_fit_transform_on_test_denied(self):
        self.assertEqual(run({"command": "scaler.fit_transform(df_test)"}, GATE_AB)[0], "deny")

    def test_train_test_split_itself_is_fine(self):
        cmd = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nclf.fit(X_train, y_train)"
        self.assertEqual(run({"command": cmd}, GATE_AB)[0], "allow")


class SelectionLeakage(unittest.TestCase):
    """Evaluating on the test split before Gate C is selection leakage — the kind that matters most."""

    def test_metric_on_test_before_gate_c_denied(self):
        decision, reason = run({"command": "print(accuracy_score(y_test, clf.predict(X_test)))"}, GATE_AB)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate C", reason)

    def test_score_on_test_without_pipeline_denied(self):
        self.assertEqual(run({"command": "clf.score(X_test, y_test)"})[0], "deny")

    def test_metric_on_test_after_gate_c_allowed(self):
        self.assertEqual(run({"command": "accuracy_score(y_test, clf.predict(X_test))"}, GATE_ABC)[0], "allow")

    def test_override_c_allows(self):
        self.assertEqual(run({"command": "clf.score(X_test, y_test)"}, OVERRIDE_C)[0], "allow")

    def test_validation_evaluation_is_fine(self):
        self.assertEqual(run({"command": "roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])"}, GATE_AB)[0], "allow")

    def test_transforming_test_is_preprocessing_not_evaluation(self):
        self.assertEqual(run({"command": "X_test_s = scaler.transform(X_test)"}, GATE_A)[0], "allow")


PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2000          # > 1,024 bytes, fine for the hook's size check
TINY = b"\x89PNG\r\n\x1a\n" + b"\0" * 50
LEDGER_A = ("- Figure 01_missingness.png: none missing.\n"
            "- Figure 02_target_balance.png: 51% vs 49%.\n"
            "- Figure 02_distributions.png: two skewed.\n")
ARTIFACTS_A = {"ml_pipeline/data_profile.json": "{}", "ml_pipeline/figures/01_missingness.png": PNG,
               "ml_pipeline/figures/02_target_balance.png": PNG, "ml_pipeline/figures/02_distributions.png": PNG}
RATIONALE = ("Model rationale:\n- traits: binary, 1,000 rows (small)\n- baseline: dummy + logistic\n"
             "- candidates: boosting\n- ruled out: neural nets (small data)\n- metric: PR-AUC\n")


def write_md(content):
    return {"file_path": "./ml_pipeline/PIPELINE.md", "content": content}


class GateRecording(unittest.TestCase):
    """A gate line is denied until that gate's artifacts exist on disk."""

    def test_gate_a_denied_without_anything(self):
        decision, reason = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write")
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertIn("figures/01_*.png 0 found (need 1)", reason)

    def test_gate_a_allowed_when_complete(self):
        decision, _ = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=ARTIFACTS_A)
        self.assertEqual(decision, "allow")

    def test_png_without_ledger_line_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/02_extra.png": PNG}
        decision, reason = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("02_extra.png has no '- Figure 02_extra.png:' line", reason)

    def test_tiny_png_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/01_missingness.png": TINY}
        decision, reason = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("(empty?)", reason)

    def test_ledger_in_existing_file_counts(self):
        decision, _ = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md=LEDGER_A, tool="Write", files=ARTIFACTS_A)
        self.assertEqual(decision, "allow")

    def test_bash_append_is_checked_too(self):
        cmd = {"command": "echo '- Gate A: approved 2026-09-16' >> ml_pipeline/PIPELINE.md"}
        self.assertEqual(run(cmd, pipeline_md="")[0], "deny")

    def test_negated_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate A: not yet approved\n"), pipeline_md="", tool="Write")[0], "allow")

    def test_gate_b_needs_rationale_and_figure(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/04_cleaning.png": PNG}
        ledger = LEDGER_A + "- Figure 04_cleaning.png: 312 rows dropped.\n"
        decision, reason = run(write_md(ledger + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("Model rationale block missing lines: traits, baseline, candidates, ruled out, metric", reason)
        partial = "Model rationale:\n- traits: x\n- baseline: y\n"
        decision, reason = run(write_md(ledger + partial + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertIn("missing lines: candidates, ruled out, metric", reason)
        decision, _ = run(write_md(ledger + RATIONALE + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "allow")

    def test_gate_c_and_d(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/12_roc.png": PNG, "ml_pipeline/figures/13_errors.png": PNG}
        ledger = LEDGER_A + "- Figure 12_roc.png: AUC 0.8.\n- Figure 13_errors.png: worst slice.\n"
        self.assertEqual(run(write_md(ledger + "- Gate C: approved 2026-09-18\n"), pipeline_md="", tool="Write", files=files)[0], "allow")
        decision, reason = run(write_md(ledger + "- Gate D: approved 2026-09-19\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("no '- Step 14 final test:' line", reason)
        ledger += "- Step 14 final test: acc=0.79 (touch 1, 2026-09-19)\n"
        self.assertEqual(run(write_md(ledger + "- Gate D: approved 2026-09-19\n"), pipeline_md="", tool="Write", files=files)[0], "allow")


def profile_with(**traits):
    base = {"n_rows": 1000, "n_features": 5, "task": "binary", "minority_frac": 0.4, "has_datetime": False,
            "has_groups": False, "small_data": True, "imbalanced": False, "high_card_categoricals": []}
    base.update(traits)
    return {"ml_pipeline/data_profile.json": json.dumps({"traits": base})}


class RedFlags(unittest.TestCase):
    """Provably wrong model applications are denied; each has an override line."""

    def test_neural_net_on_small_data(self):
        decision, reason = run({"command": "clf = MLPClassifier()\nclf.fit(X_train, y_train)"}, GATE_AB, files=profile_with(small_data=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag neural-net-small-data", reason)

    def test_neural_net_without_training_call_is_fine(self):
        self.assertEqual(run({"command": "from sklearn.neural_network import MLPClassifier"}, GATE_AB, files=profile_with())[0], "allow")

    def test_random_split_on_temporal_data(self):
        decision, reason = run({"command": "train_test_split(X, y, test_size=0.2)"}, GATE_A, files=profile_with(has_datetime=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag random-split-temporal", reason)
        self.assertEqual(run({"command": "TimeSeriesSplit(n_splits=5)"}, GATE_A, files=profile_with(has_datetime=True))[0], "allow")

    def test_group_split(self):
        decision, reason = run({"command": "KFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag group-split", reason)
        self.assertEqual(run({"command": "GroupKFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True))[0], "allow")

    def test_accuracy_selection_on_imbalanced(self):
        decision, reason = run({"command": "GridSearchCV(clf, grid, scoring='accuracy')"}, GATE_AB, files=profile_with(imbalanced=True, minority_frac=0.05))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag accuracy-imbalanced", reason)
        self.assertEqual(run({"command": "GridSearchCV(clf, grid, scoring='average_precision')"}, GATE_AB, files=profile_with(imbalanced=True))[0], "allow")

    def test_resample_before_split(self):
        decision, reason = run({"command": "X_res, y_res = SMOTE().fit_resample(X, y)"}, GATE_AB, files=profile_with())
        self.assertEqual(decision, "deny")
        self.assertIn("red flag resample-before-split", reason)
        self.assertEqual(run({"command": "SMOTE().fit_resample(X_train, y_train)"}, GATE_AB, files=profile_with())[0], "allow")

    def test_override_lifts_one_flag(self):
        md = GATE_A + "- Override: red flag random-split-temporal - user approved a random split 2026-09-16 - reason: signup date only\n"
        self.assertEqual(run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True))[0], "allow")
        self.assertEqual(run({"command": "KFold(5)"}, md, files=profile_with(has_datetime=True, has_groups=True))[0], "deny")

    def test_no_profile_no_flags(self):
        self.assertEqual(run({"command": "train_test_split(X, y)"}, GATE_A)[0], "allow")


class PipelineDiscovery(unittest.TestCase):
    def test_found_from_a_subdirectory(self):
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, GATE_AB, cwd_sub="notebooks/sub")[0], "allow")
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, GATE_A, cwd_sub="notebooks/sub")[0], "deny")

    def test_env_var_points_at_pipeline_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "elsewhere.md"
            path.write_text(GATE_AB)
            decision, _ = run({"command": "clf.fit(X_train, y_train)"}, env={"ML_PIPELINE_FILE": str(path)})
        self.assertEqual(decision, "allow")


class FailOpen(unittest.TestCase):
    def test_garbage_stdin_allows_and_reports(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), _stdin("this is not json"):
            hook.main()
        result = json.loads(out.getvalue())
        self.assertNotIn("hookSpecificOutput", result)
        self.assertIn("hook error", result["systemMessage"])


if __name__ == "__main__":
    unittest.main()
