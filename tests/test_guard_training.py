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


def run(tool_input, pipeline_md=None, tool="Bash", env=None, cwd_sub=""):
    """Run the hook against a temp project. Returns (decision, reason)."""
    with tempfile.TemporaryDirectory() as tmp:
        if pipeline_md is not None:
            (Path(tmp) / "ml_pipeline").mkdir()
            (Path(tmp) / "ml_pipeline" / "PIPELINE.md").write_text(pipeline_md)
        cwd = Path(tmp) / cwd_sub
        cwd.mkdir(parents=True, exist_ok=True)
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
