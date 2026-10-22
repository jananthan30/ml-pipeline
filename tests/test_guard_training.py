"""Behavioural tests for hooks/guard_training.py.

Run:  python3 -m pytest tests/test_guard_training.py
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
STEP_9 = "- Step 9 done: dummy AP 0.11, logistic AP 0.31\n"
STEP_10 = "- Step 10 done: boosting AP 0.42 beats logistic\n"
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


def approvals_for(md: str) -> dict:
    """The record the UserPromptSubmit hook would have written had the user approved everything in ``md``."""
    return {"gates": {g: {"at": "2026-09-16"} for g in hook.approved_gates_in(md)},
            "overrides": [{"what": w} for w in hook.override_targets(md)]}


def run(tool_input, pipeline_md=None, tool="Bash", env=None, cwd_sub="", files=None, approvals="auto"):
    """Run the hook against a temp project. Returns (decision, reason).

    ``approvals='auto'`` mirrors the ledger (the user approved what it says); pass a dict to control it,
    or None for no record at all.
    """
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "ml_pipeline"
        if pipeline_md is not None:
            state.mkdir(exist_ok=True)
            (state / "PIPELINE.md").write_text(pipeline_md)
        record = approvals_for(pipeline_md or "") if approvals == "auto" else approvals
        if record is not None:
            state.mkdir(exist_ok=True)
            (state / ".approvals.json").write_text(json.dumps(record))
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
        for key in ("ML_PIPELINE_ENFORCE", "ML_PIPELINE_FILE", "ML_PIPELINE_REQUIRE_USER_APPROVAL", "ML_PIPELINE_TEST_NAMES"):
            os.environ.pop(key, None)
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

    def test_partial_fit_denied(self):
        self.assertEqual(run({"command": "clf.partial_fit(X_batch, y_batch)"})[0], "deny")


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

    def test_exploratory_fits_allowed_for_eda(self):
        for cmd in ("emb = TSNE(2).fit_transform(X)", "PCA(n_components=2).fit(X)", "km = KMeans(5).fit(X)",
                    "from scipy import stats\nstats.norm.fit(df.age)", "umap.UMAP().fit_transform(X)",
                    "iso = IsolationForest().fit(X)"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run({"command": cmd}, self.md)[0], "allow")

    def test_exploratory_fit_on_test_still_denied(self):
        self.assertEqual(run({"command": "PCA(2).fit(X_test)"}, GATE_ABC)[0], "deny")


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
    def test_baseline_allowed(self):
        self.assertEqual(run({"command": "LogisticRegression().fit(X_train, y_train)"}, GATE_AB)[0], "allow")
        self.assertEqual(run({"command": "DummyClassifier().fit(X_train, y_train)"}, GATE_AB)[0], "allow")

    def test_complex_model_needs_baseline_first(self):
        decision, reason = run({"command": "RandomForestClassifier().fit(X_train, y_train)"}, GATE_AB)
        self.assertEqual(decision, "deny")
        self.assertIn("baseline first", reason)
        self.assertEqual(run({"command": "RandomForestClassifier().fit(X_train, y_train)"}, GATE_AB + STEP_9)[0], "allow")
        self.assertEqual(run({"command": "bst = xgb.train(params, dtrain)"}, GATE_AB)[0], "deny")

    def test_baseline_first_override(self):
        md = GATE_AB + "- Override: red flag baseline-first - user approved 2026-09-17 - reason: known problem\n"
        self.assertEqual(run({"command": "RandomForestClassifier().fit(X_train, y_train)"}, md)[0], "allow")

    def test_tuning_needs_candidates_first(self):
        cmd = "search = GridSearchCV(clf, grid, scoring='average_precision')\nsearch.fit(X_train, y_train)"
        decision, reason = run({"command": cmd}, GATE_AB + STEP_9)
        self.assertEqual(decision, "deny")
        self.assertIn("tuning comes after", reason)
        self.assertEqual(run({"command": cmd}, GATE_AB + STEP_9 + STEP_10)[0], "allow")
        self.assertEqual(run({"command": "study = optuna.create_study()"}, GATE_AB + STEP_9)[0], "deny")

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


class OverrideBug(unittest.TestCase):
    """v0.4 unlocked training from any override line that mentioned 'model' or 'training'."""

    def test_final_test_override_does_not_unlock_training(self):
        md = GATE_A + "- Override: final test - user approved a second evaluation of the model 2026-10-02 - reason: x\n"
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, md)[0], "deny")

    def test_red_flag_override_does_not_unlock_training(self):
        md = GATE_A + "- Override: red flag accuracy-imbalanced - user approved model metric 2026-10-02 - reason: x\n"
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, md)[0], "deny")

    def test_final_test_override_does_not_unlock_gate_c(self):
        md = GATE_AB + "- Override: final test - user approved re-evaluating the test set 2026-10-02\n"
        self.assertEqual(run({"command": "clf.score(X_test, y_test)"}, md)[0], "deny")

    def test_gate_override_line_is_not_an_approval(self):
        self.assertEqual(hook.approved_gates_in("- Override: Gate B - user approved skipping 2026-09-17"), [])


class SelectionLeakage(unittest.TestCase):
    """Evaluating on the test split before Gate C is selection leakage - the kind that matters most."""

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

    def test_other_test_set_names_are_caught(self):
        for cmd in ("f1_score(y_holdout, clf.predict(X_holdout))", "clf.score(X_te, y_te)", "clf.predict(df_hold_out)",
                    "roc_auc_score(yTest, clf.predict_proba(XTest))", "clf.fit(X_holdout, y_holdout)"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run({"command": cmd}, GATE_AB + STEP_9)[0], "deny")

    def test_test_size_is_not_a_test_set(self):
        self.assertEqual(run({"command": "clf.fit(X_train, y_train, test_size=0.2)"}, GATE_AB)[0], "allow")

    def test_custom_test_names_from_env(self):
        cmd = {"command": "clf.score(X_lockbox, y_lockbox)"}
        self.assertEqual(run(cmd, GATE_AB)[0], "allow")
        self.assertEqual(run(cmd, GATE_AB, env={"ML_PIPELINE_TEST_NAMES": "X_lockbox, y_lockbox"})[0], "deny")


class HiddenFits(unittest.TestCase):
    def test_getattr_fit(self):
        self.assertEqual(run({"command": "getattr(clf, 'fit')(X, y)"}, GATE_A)[0], "deny")

    def test_aliased_fit(self):
        self.assertEqual(run({"command": "train = clf.fit\ntrain(X, y)"}, GATE_A)[0], "deny")

    def test_fitted_attribute_is_not_a_fit(self):
        self.assertEqual(run({"command": "print(clf.fitted_)"}, GATE_A)[0], "allow")


class ExecutedScripts(unittest.TestCase):
    """A command that runs a local file is judged by what the file contains."""

    def test_python_script_is_scanned(self):
        files = {"train.py": "from sklearn.ensemble import RandomForestClassifier\nRandomForestClassifier().fit(X, y)\n"}
        self.assertEqual(run({"command": "python train.py"}, GATE_A, files=files)[0], "deny")

    def test_code_hidden_in_a_txt_file_is_scanned_when_run(self):
        files = {"notes.txt": "clf.fit(X_train, y_train)\n"}
        self.assertEqual(run({"file_path": "./notes.txt", "content": "clf.fit(X_train, y_train)"}, GATE_A, tool="Write")[0], "allow")
        self.assertEqual(run({"command": "python3 notes.txt"}, GATE_A, files=files)[0], "deny")

    def test_exec_open_is_scanned(self):
        files = {"step.py": "clf.fit(X_train, y_train)\n"}
        self.assertEqual(run({"command": "python -c \"exec(open('step.py').read())\""}, GATE_A, files=files)[0], "deny")

    def test_local_imports_are_scanned(self):
        files = {"run.py": "import helpers\nhelpers.go()\n", "helpers.py": "def go():\n    clf.fit(X_train, y_train)\n"}
        self.assertEqual(run({"command": "python run.py"}, GATE_A, files=files)[0], "deny")

    def test_notebook_is_scanned(self):
        nb = json.dumps({"cells": [{"cell_type": "code", "source": ["clf.fit(X_train, y_train)\n"]}]})
        self.assertEqual(run({"command": "jupyter nbconvert --execute nb.ipynb"}, GATE_A, files={"nb.ipynb": nb})[0], "deny")

    def test_shell_script_chain_is_scanned(self):
        files = {"run.sh": "#!/bin/sh\npython fit_it.py\n", "fit_it.py": "clf.fit(X_train, y_train)\n"}
        self.assertEqual(run({"command": "bash run.sh"}, GATE_A, files=files)[0], "deny")

    def test_clean_script_allowed(self):
        files = {"eda.py": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())\n"}
        self.assertEqual(run({"command": "python eda.py"}, GATE_A, files=files)[0], "allow")

    def test_script_allowed_after_gate(self):
        files = {"train.py": "LogisticRegression().fit(X_train, y_train)\n"}
        self.assertEqual(run({"command": "python train.py"}, GATE_AB, files=files)[0], "allow")


class UserApprovalProof(unittest.TestCase):
    """The agent can't approve its own gates: a ledger line without the user's chat approval doesn't count."""

    def test_ledger_without_user_approval_denies(self):
        decision, reason = run({"command": "LogisticRegression().fit(X_train, y_train)"}, GATE_AB, approvals={"gates": {"A": {}}})
        self.assertEqual(decision, "deny")
        self.assertIn("the user has not approved it in chat", reason)

    def test_no_record_at_all_denies(self):
        self.assertEqual(run({"command": "scaler.fit(X_train)"}, GATE_A, approvals=None)[0], "deny")

    def test_override_needs_user_record(self):
        self.assertEqual(run({"command": "clf.fit(X_train, y_train)"}, OVERRIDE, approvals={"gates": {"A": {}}})[0], "deny")

    def test_red_flag_override_needs_user_record(self):
        md = GATE_A + "- Override: red flag random-split-temporal - user approved 2026-09-16\n"
        decision, _ = run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True),
                          approvals={"gates": {"A": {}}})
        self.assertEqual(decision, "deny")

    def test_proof_can_be_switched_off(self):
        env = {"ML_PIPELINE_REQUIRE_USER_APPROVAL": "0"}
        self.assertEqual(run({"command": "scaler.fit(X_train)"}, GATE_A, approvals=None, env=env)[0], "allow")


class TamperProtection(unittest.TestCase):
    def test_state_files_cannot_be_written(self):
        for tool, tool_input in (
            ("Write", {"file_path": "./ml_pipeline/.approvals.json", "content": "{\"gates\": {\"B\": {}}}"}),
            ("Write", {"file_path": "./ml_pipeline/.guard_state.json", "content": "{}"}),
            ("Bash", {"command": "python -c \"import json; json.dump({'test_touches': 0}, open('ml_pipeline/.guard_state.json','w'))\""}),
            ("Bash", {"command": "rm ml_pipeline/.approvals.json"}),
            ("Bash", {"command": "echo '{}' > ml_pipeline/.approvals.json"}),
        ):
            with self.subTest(tool_input=tool_input):
                self.assertEqual(run(tool_input, GATE_A, tool=tool)[0], "deny")

    def test_reading_state_is_fine(self):
        self.assertEqual(run({"command": "cat ml_pipeline/.guard_state.json"}, GATE_A)[0], "allow")

    def test_recorded_lines_cannot_be_deleted(self):
        md = GATE_AB + "- Step 14 final test: acc=0.71 (touch 1, 2026-09-19)\n"
        edit = {"file_path": "./ml_pipeline/PIPELINE.md", "old_string": "- Step 14 final test: acc=0.71 (touch 1, 2026-09-19)\n",
                "new_string": ""}
        self.assertEqual(run(edit, md, tool="Edit")[0], "deny")
        write = {"file_path": "./ml_pipeline/PIPELINE.md", "content": GATE_AB}
        self.assertEqual(run(write, md, tool="Write")[0], "deny")
        self.assertEqual(run({"command": "sed -i '/final test/d' ml_pipeline/PIPELINE.md"}, md)[0], "deny")
        self.assertEqual(run({"command": "rm ml_pipeline/PIPELINE.md"}, md)[0], "deny")

    def test_creating_a_fresh_ledger_is_fine(self):
        self.assertEqual(run({"command": "mkdir -p ml_pipeline && echo '# Pipeline' > ml_pipeline/PIPELINE.md"})[0], "allow")
        self.assertEqual(run({"command": "echo '# Pipeline' > ml_pipeline/PIPELINE.md"}, "# Pipeline\n- Figure a.png: x\n")[0], "allow")
        self.assertEqual(run({"command": "echo '# Pipeline' > ml_pipeline/PIPELINE.md"}, GATE_A)[0], "deny")

    def test_appending_and_reading_ledger_is_fine(self):
        self.assertEqual(run({"command": "cat ml_pipeline/PIPELINE.md"}, GATE_A)[0], "allow")
        self.assertEqual(run({"command": "cp ml_pipeline/PIPELINE.md /tmp/backup.md"}, GATE_A)[0], "allow")
        self.assertEqual(run({"command": "echo '- Figure x.png: y' >> ml_pipeline/PIPELINE.md"}, GATE_A)[0], "allow")


PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2000          # > 1,024 bytes, fine for the hook's size check
TINY = b"\x89PNG\r\n\x1a\n" + b"\0" * 50
LEDGER_A = ("- Figure 01_missingness.png: none missing.\n"
            "- Figure 02_target_balance.png: 51% vs 49%.\n"
            "- Figure 02_distributions.png: two skewed.\n")
CONTRACT = ("Prediction contract:\n- target: y = 1 if churned within 30 days\n- unit: one row per customer\n"
            "- horizon: 30 days after the snapshot date\n- available: features known on the snapshot date\n"
            "- metric: PR-AUC\n")
STEPS_A = "- Step 1 done: 2,000 x 6\n- Step 2 done: figures\n- Step 3 done: contract\n"
ARTIFACTS_A = {"ml_pipeline/data_profile.json": "{}", "ml_pipeline/figures/01_missingness.png": PNG,
               "ml_pipeline/figures/02_target_balance.png": PNG, "ml_pipeline/figures/02_distributions.png": PNG}
RATIONALE = ("Model rationale:\n- traits: binary, 1,000 rows (small)\n- baseline: dummy + logistic\n"
             "- candidates: boosting\n- ruled out: neural nets (small data)\n- metric: PR-AUC\n")
STEPS_B = "".join(f"- Step {s} done: ok\n" for s in range(4, 9))
STEPS_C = "".join(f"- Step {s} done: ok\n" for s in range(9, 13))
ALL_GATES = {"gates": {g: {} for g in "ABCD"}, "overrides": []}
RUN_BASELINE = json.dumps({"kind": "baseline"}) + "\n"
RUN_BOTH = RUN_BASELINE + json.dumps({"kind": "candidate"}) + "\n"


def write_md(content):
    return {"file_path": "./ml_pipeline/PIPELINE.md", "content": content}


class GateRecording(unittest.TestCase):
    """A gate line is denied until that gate's steps, artifacts, and user approval exist."""

    def test_gate_a_denied_without_anything(self):
        decision, reason = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertIn("figures/01_*.png 0 found (need 1)", reason)
        self.assertIn("step 1 not recorded", reason)
        self.assertIn("Prediction contract block missing lines", reason)

    def test_gate_a_allowed_when_complete(self):
        content = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n"
        self.assertEqual(run(write_md(content), pipeline_md="", tool="Write", files=ARTIFACTS_A, approvals=ALL_GATES)[0], "allow")

    def test_gate_a_denied_without_the_users_approval(self):
        content = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n"
        decision, reason = run(write_md(content), pipeline_md="", tool="Write", files=ARTIFACTS_A, approvals=None)
        self.assertEqual(decision, "deny")
        self.assertIn("the user has not approved Gate A in chat", reason)

    def test_png_without_ledger_line_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/02_extra.png": PNG}
        content = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n"
        decision, reason = run(write_md(content), pipeline_md="", tool="Write", files=files, approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("02_extra.png has no '- Figure 02_extra.png:' line", reason)

    def test_tiny_png_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/01_missingness.png": TINY}
        content = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n"
        decision, reason = run(write_md(content), pipeline_md="", tool="Write", files=files, approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("(empty?)", reason)

    def test_ledger_in_existing_file_counts(self):
        existing = LEDGER_A + STEPS_A + CONTRACT
        tool_input = {"file_path": "./ml_pipeline/PIPELINE.md", "old_string": CONTRACT,
                      "new_string": CONTRACT + "- Gate A: approved 2026-09-16\n"}
        self.assertEqual(run(tool_input, pipeline_md=existing, tool="Edit", files=ARTIFACTS_A, approvals=ALL_GATES)[0], "allow")

    def test_bash_append_is_checked_too(self):
        cmd = {"command": "echo '- Gate A: approved 2026-09-16' >> ml_pipeline/PIPELINE.md"}
        self.assertEqual(run(cmd, pipeline_md="", approvals=ALL_GATES)[0], "deny")

    def test_negated_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate A: not yet approved\n"), pipeline_md="", tool="Write")[0], "allow")

    def test_gate_b_needs_steps_rationale_figure_and_split(self):
        base = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n"
        files = {**ARTIFACTS_A, "ml_pipeline/figures/04_cleaning.png": PNG}
        ledger = base + "- Figure 04_cleaning.png: 312 rows dropped.\n"
        decision, reason = run(write_md(ledger + "- Gate B: approved 2026-09-17\n"), pipeline_md=base, tool="Write",
                               files=files, approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("Model rationale block missing lines: traits, baseline, candidates, ruled out, metric", reason)
        self.assertIn("step 4 not recorded", reason)
        self.assertIn("no frozen split", reason)
        files["ml_pipeline/.guard_state.json"] = "{}"
        decision, _ = run(write_md(ledger + STEPS_B + RATIONALE + "- Gate B: approved 2026-09-17\n"), pipeline_md=base + STEPS_B,
                          tool="Write", files=files, approvals=ALL_GATES)
        self.assertEqual(decision, "allow")

    def test_gate_c_and_d(self):
        base = LEDGER_A + STEPS_A + CONTRACT + GATE_AB + STEPS_B
        files = {**ARTIFACTS_A, "ml_pipeline/figures/12_roc.png": PNG, "ml_pipeline/figures/13_errors.png": PNG,
                 "ml_pipeline/runs.jsonl": RUN_BOTH, "ml_pipeline/.guard_state.json": "{}"}
        ledger = base + STEPS_C + "- Figure 12_roc.png: AUC 0.8.\n- Figure 13_errors.png: worst slice.\n"
        self.assertEqual(run(write_md(ledger + "- Gate C: approved 2026-09-18\n"), pipeline_md=ledger, tool="Write",
                             files=files, approvals=ALL_GATES)[0], "allow")
        ledger_c = ledger + "- Gate C: approved 2026-09-18\n- Step 13 done: worst slice is new customers\n"
        decision, reason = run(write_md(ledger_c + "- Gate D: approved 2026-09-19\n"), pipeline_md=ledger_c, tool="Write",
                               files=files, approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("no '- Step 14 final test:' line", reason)
        ledger_c += "- Step 14 final test: acc=0.79 (touch 1, 2026-09-19)\n"
        self.assertEqual(run(write_md(ledger_c + "- Gate D: approved 2026-09-19\n"), pipeline_md=ledger_c, tool="Write",
                             files=files, approvals=ALL_GATES)[0], "allow")


class StepOrder(unittest.TestCase):
    """One step at a time: a step line needs the steps before it, its phase gate, and its own artifacts."""

    def test_steps_in_order(self):
        decision, reason = run({"command": "echo '- Step 2 done: figures' >> ml_pipeline/PIPELINE.md"}, pipeline_md="",
                               files=ARTIFACTS_A, approvals=ALL_GATES)
        self.assertEqual(decision, "deny")
        self.assertIn("step 1 comes first", reason)

    def test_step_needs_its_artifacts(self):
        decision, reason = run({"command": "echo '- Step 1 done: 2,000 x 6' >> ml_pipeline/PIPELINE.md"}, pipeline_md="")
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertEqual(run({"command": "echo '- Step 1 done: 2,000 x 6' >> ml_pipeline/PIPELINE.md"}, pipeline_md="",
                             files=ARTIFACTS_A)[0], "allow")

    def test_step_needs_its_phase_gate(self):
        md = LEDGER_A + STEPS_A + CONTRACT
        decision, reason = run({"command": "echo '- Step 4 done: cleaned' >> ml_pipeline/PIPELINE.md"}, pipeline_md=md, files=ARTIFACTS_A)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate A is not approved by the user yet", reason)

    def test_mandatory_step_cannot_be_skipped(self):
        md = LEDGER_A + STEPS_A + CONTRACT + "- Gate A: approved 2026-09-16\n- Step 4 done: ok\n- Step 5 skipped: one table\n"
        decision, reason = run({"command": "echo '- Step 6 skipped: no time' >> ml_pipeline/PIPELINE.md"}, pipeline_md=md, files=ARTIFACTS_A)
        self.assertEqual(decision, "deny")
        self.assertIn("step 6 can't be skipped", reason)

    def test_baseline_step_needs_a_logged_baseline_run(self):
        md = LEDGER_A + STEPS_A + CONTRACT + GATE_AB + STEPS_B
        cmd = {"command": "echo '- Step 9 done: dummy 0.11, logistic 0.31' >> ml_pipeline/PIPELINE.md"}
        decision, reason = run(cmd, pipeline_md=md, files=ARTIFACTS_A)
        self.assertEqual(decision, "deny")
        self.assertIn("no 'baseline' run", reason)
        self.assertEqual(run(cmd, pipeline_md=md, files={**ARTIFACTS_A, "ml_pipeline/runs.jsonl": RUN_BASELINE})[0], "allow")


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
        decision, reason = run({"command": "KFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True, group_candidates=["patient_id"]))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag group-split", reason)
        self.assertIn("patient_id", reason)
        self.assertEqual(run({"command": "GroupKFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True))[0], "allow")

    def test_accuracy_selection_on_imbalanced(self):
        md = GATE_AB + STEP_9 + STEP_10
        decision, reason = run({"command": "GridSearchCV(clf, grid, scoring='accuracy')"}, md, files=profile_with(imbalanced=True, minority_frac=0.05))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag accuracy-imbalanced", reason)
        self.assertEqual(run({"command": "GridSearchCV(clf, grid, scoring='average_precision')"}, md, files=profile_with(imbalanced=True))[0], "allow")

    def test_resample_before_split(self):
        decision, reason = run({"command": "X_res, y_res = SMOTE().fit_resample(X, y)"}, GATE_AB, files=profile_with())
        self.assertEqual(decision, "deny")
        self.assertIn("red flag resample-before-split", reason)
        self.assertEqual(run({"command": "SMOTE().fit_resample(X_train, y_train)"}, GATE_AB, files=profile_with())[0], "allow")

    def test_override_lifts_one_flag(self):
        md = GATE_A + "- Override: red flag random-split-temporal - user approved a random split 2026-09-16 - reason: signup date only\n"
        self.assertEqual(run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True))[0], "allow")
        self.assertEqual(run({"command": "KFold(5)"}, md, files=profile_with(has_datetime=True, has_groups=True))[0], "deny")

    def test_red_flags_apply_to_executed_scripts(self):
        files = {**profile_with(has_datetime=True), "split.py": "train_test_split(X, y)\n"}
        self.assertEqual(run({"command": "python split.py"}, GATE_A, files=files)[0], "deny")

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
            (Path(tmp) / ".approvals.json").write_text(json.dumps(approvals_for(GATE_AB)))
            decision, _ = run({"command": "clf.fit(X_train, y_train)"}, env={"ML_PIPELINE_FILE": str(path)}, approvals=None)
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
