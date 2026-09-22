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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import guard_training as hook  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v05_fixtures as fx  # noqa: E402

GATE_A = "# Pipeline\n- Gate A: approved 2026-09-16\n"
GATE_AB = GATE_A + "- Gate B: approved 2026-09-17\n"
OVERRIDE = GATE_A + "- Override: Gate B - user approved skipping to training 2026-09-17 - reason: deadline\n"
GATE_ABC = GATE_AB + "- Gate C: approved 2026-09-18\n"
OVERRIDE_C = GATE_AB + "- Override: Gate C - user approved evaluating on the test set 2026-09-18 - reason: demo\n"
TRAIN = "model.fit(X_train, y_train)\n"
TRAIN_CMD = {"command": "clf.fit(X_train, y_train)"}
TEST_EVAL = {"command": "accuracy_score(y_test, clf.predict(X_test))"}


def notebook(*cells):
    """A minimal .ipynb; each cell is a (cell_type, source) pair."""
    return json.dumps({"cells": [{"cell_type": kind, "metadata": {}, "source": [source]} for kind, source in cells],
                       "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


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


TINY = b"\x89PNG\r\n\x1a\n" + b"\0" * 50


def write_md(content):
    return {"file_path": "./ml_pipeline/PIPELINE.md", "content": content}


class HookRobustness(unittest.TestCase):
    """Input the hook cannot read never switches enforcement off (review finding 1)."""

    def test_unresolvable_home_directory_is_not_an_off_switch(self):
        for cmd in ('cd ~nosuchuser_zz 2>/dev/null; python -c "model.fit(X_test, y_test)"',
                    'python ~nosuchuser_zz/x.py; python -c "model.fit(X_test, y_test)"'):
            with self.subTest(cmd=cmd):
                decision, reason = run({"command": cmd}, GATE_AB)
                self.assertEqual(decision, "deny")
                self.assertIn("fits on the test split", reason)

    def test_records_that_crash_the_check_fail_closed(self):
        p = fx.profile()
        p["columns"]["age"] = "not a mapping"
        files = {**fx.records("A"), fx.PROFILE: json.dumps(p)}
        decision, reason = run(write_md(fx.ledger("A") + "- Gate A: approved 2026-09-23\n"), pipeline_md="",
                               tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("could not check the Gate A records", reason)

    def test_gate_messages_say_overrides_need_the_records(self):
        self.assertIn("an override needs the same records", run(TRAIN_CMD, GATE_A)[1])
        self.assertIn("an override needs the same records", run(TEST_EVAL, GATE_AB)[1])


class GateRecording(unittest.TestCase):
    """A gate line is denied until that gate's records exist and explain every column."""

    def record(self, gate, files, ledger=None, existing=""):
        text = (fx.ledger(gate if gate in "ABC" else "C") if ledger is None else ledger)
        return run(write_md(text + f"- Gate {gate}: approved 2026-09-23\n"), pipeline_md=existing, tool="Write",
                   files=files)

    def test_gate_a_denied_without_anything(self):
        decision, reason = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write")
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertIn("figures/01_*.png 0 found (need 1)", reason)

    def test_gate_a_allowed_when_complete(self):
        self.assertEqual(self.record("A", fx.records("A")), ("allow", ""))

    def test_missing_records_named_in_the_denial(self):
        files = fx.records("A")
        del files[fx.DICTIONARY]
        decision, reason = self.record("A", files)
        self.assertEqual(decision, "deny")
        self.assertIn("data_dictionary.csv missing", reason)

    def test_png_without_ledger_line_denied(self):
        decision, reason = self.record("A", {**fx.records("A"), fx.FIG("02_extra.png"): fx.PNG})
        self.assertEqual(decision, "deny")
        self.assertIn("02_extra.png has no '- Figure 02_extra.png:' line", reason)

    def test_tiny_png_denied(self):
        decision, reason = self.record("A", {**fx.records("A"), fx.FIG("01_missingness.png"): TINY})
        self.assertEqual(decision, "deny")
        self.assertIn("(empty?)", reason)

    def test_vague_figure_explanation_denied(self):
        ledger = fx.ledger("A").replace(
            "y=0 is 70.0% of 200 rows, so a majority-class dummy scores 70.0%.",
            "shows the class balance of the target variable nicely")
        decision, reason = self.record("A", fx.records("A"), ledger=ledger)
        self.assertEqual(decision, "deny")
        self.assertIn("02_target_balance.png: explanation contains no number", reason)

    def test_ledger_in_existing_file_counts(self):
        decision, _ = run(write_md("- Gate A: approved 2026-09-23\n"), pipeline_md=fx.ledger("A"), tool="Write",
                          files=fx.records("A"))
        self.assertEqual(decision, "allow")

    def test_recorded_gates_are_not_rechecked(self):
        rows = fx.dictionary_rows()
        rows[1]["figure_note"] = "plan: fine"                    # Gate A's records have gone stale since
        files = {**fx.records("B"), fx.DICTIONARY: fx.dictionary_csv(rows)}
        existing = fx.ledger("B") + "- Gate A: approved 2026-09-23\n"
        edit = {"file_path": "./ml_pipeline/PIPELINE.md", "old_string": "- Gate A: approved 2026-09-23\n",
                "new_string": "- Gate A: approved 2026-09-23\n- Gate B: approved 2026-09-24\n"}
        self.assertEqual(run(edit, pipeline_md=existing, tool="Edit", files=files), ("allow", ""))
        whole = {"file_path": "./ml_pipeline/PIPELINE.md", "content": existing + "- Gate B: approved 2026-09-24\n"}
        self.assertEqual(run(whole, pipeline_md=existing, tool="Write", files=files), ("allow", ""))
        new_run = {"file_path": "./ml_pipeline/PIPELINE.md",
                   "content": existing + "## Run: again (2026-09-25)\n- Gate A: approved 2026-09-25\n"}
        self.assertEqual(run(new_run, pipeline_md=existing, tool="Write", files=files)[0], "deny")

    def test_bash_append_is_checked_too(self):
        cmd = {"command": "echo '- Gate A: approved 2026-09-16' >> ml_pipeline/PIPELINE.md"}
        self.assertEqual(run(cmd, pipeline_md="")[0], "deny")

    def test_negated_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate A: not yet approved\n"), pipeline_md="", tool="Write")[0], "allow")

    def test_gate_b_needs_rationale_and_records(self):
        decision, reason = self.record("B", fx.records("B"), ledger=fx.ledger("B").replace(fx.RATIONALE, ""))
        self.assertEqual(decision, "deny")
        self.assertIn("Model rationale block missing lines: traits, baseline, candidates, ruled out, metric", reason)
        files = fx.records("B")
        del files[fx.ASSOCIATIONS]
        decision, reason = self.record("B", files)
        self.assertIn("associations.json missing", reason)
        self.assertEqual(self.record("B", fx.records("B")), ("allow", ""))

    def test_gate_c_and_d(self):
        self.assertEqual(self.record("C", fx.records("C")), ("allow", ""))
        files = {**fx.records("C"), fx.FIG("13_errors.png"): fx.PNG}
        ledger = fx.ledger("C") + "- Figure 13_errors.png: the worst slice is plan=pro with 12 of 40 validation rows wrong.\n"
        decision, reason = self.record("D", files, ledger=ledger)
        self.assertEqual(decision, "deny")
        self.assertIn("no '- Step 14 final test:' line", reason)
        ledger += "- Step 14 final test: accuracy=0.7900 (touch 1, 2026-09-25)\n"
        self.assertEqual(self.record("D", files, ledger=ledger)[0], "allow")


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

    def test_target_stats_outside_the_training_split(self):
        for code in ("scores = mutual_info_classif(X, y)", "f_classif(X_all, y_all)",
                     "sel = SelectKBest(k=5)\nsel.fit(X, y)", "SelectKBest(chi2, k=5).fit_transform(X, y)"):
            with self.subTest(code=code):
                decision, reason = run({"command": code}, GATE_A, files=profile_with())
                self.assertEqual(decision, "deny")
                self.assertIn("red flag target-stats-outside-train", reason)
        for code in ("mutual_info_classif(X_train, y_train)", "SelectKBest(k=5).fit(X_train, y_train)",
                     "sel = SelectKBest(k=5).fit(X_train, y_train)\nmodel.fit(X_tr_sel, y_tr)"):
            with self.subTest(code=code):
                self.assertEqual(run({"command": code}, GATE_AB, files=profile_with())[0], "allow")

    def test_target_stats_override(self):
        md = GATE_A + "- Override: red flag target-stats-outside-train - user approved 2026-09-24 - reason: inside CV\n"
        self.assertEqual(run({"command": "mutual_info_classif(X, y)"}, md, files=profile_with())[0], "allow")


class FollowsExecution(unittest.TestCase):
    """Running a file trains whatever the file says, so the file is scanned as well as the command."""

    def test_python_script(self):
        self.assertEqual(run({"command": "python train.py"}, files={"train.py": TRAIN})[0], "deny")

    def test_python_module(self):
        self.assertEqual(run({"command": "python3 -m train_mod"}, files={"train_mod.py": TRAIN})[0], "deny")

    def test_package_main_module(self):
        self.assertEqual(run({"command": "python -m pkg.trainer"}, files={"pkg/trainer/__main__.py": TRAIN})[0], "deny")

    def test_other_launchers(self):
        for cmd in ("uv run train.py", "uv run python train.py", ".venv/bin/python -u train.py",
                    "marimo export html train.py -o out.html", "cd . && python train.py"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run({"command": cmd}, files={"train.py": TRAIN})[0], "deny")

    def test_follows_cd(self):
        self.assertEqual(run({"command": "cd scripts && python train.py"}, files={"scripts/train.py": TRAIN})[0], "deny")

    def test_executed_notebook_code_cells(self):
        nb = notebook(("markdown", "# Train"), ("code", TRAIN))
        for cmd in ("jupyter nbconvert --to notebook --execute nb.ipynb", "jupyter execute nb.ipynb",
                    "papermill nb.ipynb out.ipynb"):
            with self.subTest(cmd=cmd):
                self.assertEqual(run({"command": cmd}, files={"nb.ipynb": nb})[0], "deny")

    def test_notebook_markdown_cells_are_not_code(self):
        nb = notebook(("markdown", "Later we call " + TRAIN), ("code", "df.describe()"))
        self.assertEqual(run({"command": "papermill nb.ipynb out.ipynb"}, files={"nb.ipynb": nb})[0], "allow")

    def test_converting_without_executing_is_not_a_run(self):
        nb = notebook(("code", TRAIN))
        self.assertEqual(run({"command": "jupyter nbconvert --to html nb.ipynb"}, files={"nb.ipynb": nb})[0], "allow")

    def test_harmless_script_allowed(self):
        self.assertEqual(run({"command": "python clean.py"}, files={"clean.py": "df = df.dropna()\n"})[0], "allow")

    def test_missing_script_allowed(self):
        self.assertEqual(run({"command": "python nowhere.py"})[0], "allow")

    def test_script_allowed_after_gate_b(self):
        self.assertEqual(run({"command": "python train.py"}, GATE_AB, files={"train.py": TRAIN})[0], "allow")


class WiderDetection(unittest.TestCase):
    """Training spelled in ways the v0.4 patterns missed."""

    def test_getattr_fit(self):
        self.assertEqual(run({"command": 'getattr(model, "fit")(X, y)'})[0], "deny")

    def test_bound_method_called_later(self):
        self.assertEqual(run({"command": "m = model.fit\nm(X, y)"})[0], "deny")

    def test_bound_transformer_is_still_preprocessing(self):
        self.assertEqual(run({"command": "f = scaler.fit\nf(X_train)"}, GATE_A)[0], "allow")

    def test_spaced_dot(self):
        self.assertEqual(run({"command": "model . fit(X, y)"})[0], "deny")

    def test_functions_that_train_internally(self):
        for call in ("cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5)",
                     "cross_validate(clf, X_train, y_train)", "cross_val_predict(clf, X_train, y_train)",
                     "learning_curve(clf, X_train, y_train)",
                     "validation_curve(clf, X_train, y_train, param_name='C', param_range=grid)",
                     "permutation_importance(clf, X_val, y_val)"):
            with self.subTest(call=call):
                self.assertEqual(run({"command": call}, GATE_A)[0], "deny")

    def test_torch_loop_with_any_names(self):
        self.assertEqual(run({"command": "total_loss = crit(out, y)\ntotal_loss.backward()\nopt.step()"})[0], "deny")

    def test_any_trainer_class(self):
        self.assertEqual(run({"command": "sft = SFTTrainer(model=m, train_dataset=ds)\nsft.train()"})[0], "deny")

    def test_defining_a_trainer_class_is_not_constructing_one(self):
        self.assertEqual(run({"command": "class MyTrainer(Trainer):\n    pass"})[0], "allow")


class OnlyPythonIsScanned(unittest.TestCase):
    """The plugin is user-scoped and runs in every project: other languages and prose are left alone."""

    def test_typescript_fit_addon(self):
        self.assertEqual(run({"file_path": "/x/term.ts", "content": "fitAddon.fit();\n"}, tool="Write")[0], "allow")

    def test_javascript_view_fit(self):
        self.assertEqual(run({"file_path": "/x/map.js", "content": "view.fit(extent);\n"}, tool="Write")[0], "allow")

    def test_fit_code_in_a_txt_file_is_still_code(self):
        self.assertEqual(run({"file_path": "/x/train.txt", "content": TRAIN}, tool="Write")[0], "deny")

    def test_read_only_grep_for_a_gate_line(self):
        self.assertEqual(run({"command": 'grep -n "Gate B: approved" ml_pipeline/PIPELINE.md'})[0], "allow")

    def test_docs_page_quoting_the_approval_format(self):
        tool_input = {"file_path": "./docs/howto.md", "content": "- Gate B: approved 2026-09-17\n"}
        self.assertEqual(run(tool_input, tool="Write")[0], "allow")

    def test_heredoc_into_the_ledger_is_a_recording(self):
        cmd = "cat >> ml_pipeline/PIPELINE.md <<'EOF'\n- Gate A: approved 2026-09-16\nEOF"
        self.assertEqual(run({"command": cmd}, pipeline_md="")[0], "deny")

    def test_tee_into_the_ledger_is_a_recording(self):
        cmd = "echo '- Gate A: approved 2026-09-16' | tee -a ml_pipeline/PIPELINE.md"
        self.assertEqual(run({"command": cmd}, pipeline_md="")[0], "deny")


class NoSelfDisable(unittest.TestCase):
    """Only the user switches enforcement off; deny messages never teach the agent how."""

    def assert_denied_without_naming_the_switch(self, tool_input, tool="Bash"):
        decision, reason = run(tool_input, tool=tool)
        self.assertEqual(decision, "deny")
        self.assertIn("ask the user", reason.lower())
        self.assertNotIn("ML_PIPELINE", reason)

    def test_settings_file(self):
        tool_input = {"file_path": "/x/.claude/settings.json", "content": '{"env": {"ML_PIPELINE_ENFORCE": "0"}}\n'}
        self.assert_denied_without_naming_the_switch(tool_input, tool="Write")

    def test_shell(self):
        for cmd in ("export ML_PIPELINE_ENFORCE=0", "ML_PIPELINE_ENFORCE=0 python run.py",
                    "echo 'export ML_PIPELINE_ENFORCE=off' >> ~/.zshrc", "launchctl setenv ML_PIPELINE_ENFORCE 0"):
            with self.subTest(cmd=cmd):
                self.assert_denied_without_naming_the_switch({"command": cmd})

    def test_python_environ(self):
        self.assert_denied_without_naming_the_switch(
            {"command": "python -c 'import os; os.environ[\"ML_PIPELINE_ENFORCE\"] = \"0\"'"})

    def test_pointing_at_another_ledger(self):
        self.assert_denied_without_naming_the_switch({"command": "export ML_PIPELINE_FILE=/tmp/fake.md"})

    def test_reading_the_setting_is_fine(self):
        for cmd in ("echo $ML_PIPELINE_ENFORCE", "echo ${ML_PIPELINE_ENFORCE:-1}",
                    '[ "$ML_PIPELINE_ENFORCE" == "0" ] && echo off'):
            with self.subTest(cmd=cmd):
                self.assertEqual(run({"command": cmd})[0], "allow")

    def test_no_pipeline_message_does_not_name_the_switch(self):
        decision, reason = run(TRAIN_CMD)
        self.assertEqual(decision, "deny")
        self.assertIn("ask the user", reason.lower())
        self.assertNotIn("ML_PIPELINE", reason)


class StrictLedger(unittest.TestCase):
    """Only '- Gate X: approved ...' and '- Override: Gate X - ...' lines count, in the newest run only."""

    def trains(self, md):
        return run(TRAIN_CMD, md)[0]

    def test_lines_that_only_mention_approval_do_not_count(self):
        for line in ("- Gate B: to be approved by the user", "- Gate B needs an approved model rationale first",
                     "- Gate B: never approved", "### Gate B approved 2026-09-17", "**Gate B approved:** yes",
                     "| — | **GATE B** | **APPROVED** 2026-09-17 |", "- [x] **Gate B: approved 2026-09-17**"):
            with self.subTest(line=line):
                self.assertEqual(self.trains(GATE_A + line + "\n"), "deny")

    def test_other_overrides_never_unlock_training(self):
        for line in ("- Override: red flag accuracy-imbalanced - user approved accuracy for model selection 2026-09-20 - reason: demo",
                     "- Override: final test - user approved a second evaluation 2026-09-21 - reason: model retrained"):
            with self.subTest(line=line):
                self.assertEqual(self.trains(GATE_A + line + "\n"), "deny")

    def test_a_trailing_note_never_revokes(self):
        self.assertEqual(self.trains(GATE_A + "- Gate B: approved 2026-09-17 - split is not stratified by design\n"), "allow")

    def test_any_case_and_no_date(self):
        self.assertEqual(self.trains(GATE_A + "- Gate B: APPROVED\n"), "allow")

    def test_newest_line_per_gate_wins(self):
        self.assertEqual(self.trains(GATE_AB + "- Gate B: hold - run finished 2026-09-20\n"), "deny")
        self.assertEqual(self.trains(GATE_A + "- Gate B: pending\n- Gate B: approved 2026-09-21\n"), "allow")

    def test_a_new_run_starts_locked(self):
        run2 = "\n## Run: second experiment (2026-09-23)\n"
        self.assertEqual(self.trains(GATE_AB + run2), "deny")
        self.assertEqual(self.trains(GATE_AB + run2 + "- Gate A: approved 2026-09-23\n- Gate B: approved 2026-09-23\n"), "allow")

    def test_an_old_run_gate_c_does_not_unlock_the_test_set(self):
        md = GATE_ABC + "\n## Run: retrain (2026-09-23)\n- Gate A: approved 2026-09-23\n- Gate B: approved 2026-09-23\n"
        self.assertEqual(run(TEST_EVAL, md)[0], "deny")
        self.assertEqual(self.trains(md), "allow")

    def test_an_old_run_red_flag_override_does_not_carry_over(self):
        md = (GATE_A + "- Override: red flag random-split-temporal - user approved a random split 2026-09-16 - reason: x\n"
              "## Run: next (2026-09-23)\n- Gate A: approved 2026-09-23\n")
        self.assertEqual(run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True))[0], "deny")

    def test_a_red_flag_override_needs_the_documented_line(self):
        md = GATE_A + "## Override of red flag random-split-temporal requested by the agent\n"
        self.assertEqual(run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True))[0], "deny")

    def test_writing_a_non_approval_gate_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate B: to be approved by the user\n"), pipeline_md="", tool="Write")[0], "allow")


# The shapes of a real multi-experiment ledger (its Gate B approval line kept verbatim, the rest reworded):
# old approvals written as headings, table rows and bold prose, and a newest run that approves B and stops
# before C.
REAL_LEDGER = (
    "| — | **GATE A** | **APPROVED** 2026-08-19 | user asked to continue |\n"
    "### 🔏 OWNER OVERRIDE 2026-08-22 — Gate B sample-size bar waived for an\n"
    "### 🔏 GATE C APPROVED — owner override (2026-08-25)\n"
    '**Gate C approved:** owner said "go ahead" for the error analysis.\n'
    "## Model training and evaluation — Gate B approved 2026-09-09\n"
    "- Gate B: APPROVED for this complete development run; record above is the permission\n"
    "  to fit, not permission to call the result final.\n"
    "- Gate C: stop after the result report. Test-set use and new data\n"
    "  are outside this approval.\n"
)


class RealLedgerRegression(unittest.TestCase):
    def test_parses_as_gate_b_only(self):
        self.assertEqual(hook.gate_state(REAL_LEDGER),
                         {"A": False, "B": True, "C": False, "D": False, "override": False, "override_c": False,
                          "pilot": False})

    def test_training_allowed_but_the_test_set_stays_locked(self):
        self.assertEqual(run(TRAIN_CMD, REAL_LEDGER)[0], "allow")
        decision, reason = run(TEST_EVAL, REAL_LEDGER)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate C", reason)


class PilotRule(unittest.TestCase):
    """After Gate B, training on a slice needs the user's pilot override; Gate C still needs a full run."""
    PILOT = "- Override: pilot - 5,000 rows - user approved 2026-09-24 - reason: smoke test\n"

    def test_slices_in_a_fit_call_denied(self):
        for code in ("model.fit(X_train[:5000], y_train[:5000])", "model.fit(X_train.iloc[:500], y_train.iloc[:500])",
                     "model.fit(X_train.sample(1000), y)", "guard.fit(clf, X_train.head(500), y_train.head(500))",
                     "df = pd.read_csv('d.csv', nrows=1000)\nmodel.fit(X_train, y_train)"):
            with self.subTest(code=code):
                decision, reason = run({"command": code}, GATE_AB)
                self.assertEqual(decision, "deny")
                self.assertIn("slice of the data", reason)

    def test_whole_split_and_column_selection_allowed(self):
        for code in ("model.fit(X_train, y_train)", "model.fit(X_train.loc[:, features], y_train)",
                     "model.fit(X_train[:, :5], y_train)"):
            with self.subTest(code=code):
                self.assertEqual(run({"command": code}, GATE_AB)[0], "allow")

    def test_override_allows_a_pilot(self):
        self.assertEqual(run({"command": "model.fit(X_train[:5000], y_train[:5000])"}, GATE_AB + self.PILOT)[0], "allow")

    def test_override_from_an_older_run_does_not_count(self):
        md = GATE_AB + self.PILOT + "## Run: next (2026-09-25)\n- Gate A: approved 2026-09-25\n- Gate B: approved 2026-09-25\n"
        self.assertEqual(run({"command": "model.fit(X_train[:5000], y_train[:5000])"}, md)[0], "deny")

    def test_transformer_slices_are_not_pilots(self):
        self.assertEqual(run({"command": "scaler.fit(X_train[:1000])"}, GATE_AB)[0], "allow")


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
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(out), _stdin("this is not json"):
            os.environ.pop("ML_PIPELINE_ENFORCE", None)  # a developer session may run with enforcement off
            hook.main()
        result = json.loads(out.getvalue())
        self.assertNotIn("hookSpecificOutput", result)
        self.assertIn("hook error", result["systemMessage"])


if __name__ == "__main__":
    unittest.main()
