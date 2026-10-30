"""The training guard's audit exception: only the exact invocation of the plugin's own audit tool is let through.

Run:  python3 -m unittest tests.test_audit_hook      (standard library only, like the other hook tests)

The guard denies scoring on test data before Gate C, and the audit must do exactly that. So the hook skips its
checks for one command shape only: ``<python> <plugin>/skills/ml-audit/lib/mlpipeline_audit.py <folder> [flags]``,
one plain command, the plugin's own copy, known flags only. Everything else that names the tool is denied, and
look-alikes are judged like any other script.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hooks"))
import guard_training as hook  # noqa: E402

TOOL = ROOT / "skills" / "ml-audit" / "lib" / "mlpipeline_audit.py"
F = "." + "fit"
SCORE_TEST = "acc = accuracy_score(y_test, model.predict(X_test))\n"
TRAIN = f"model = RandomForestClassifier()\nmodel{F}(X_train, y_train)\n"


@contextlib.contextmanager
def _stdin(text: str):
    old, sys.stdin = sys.stdin, io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = old


def decide(command: str | None = None, files: dict | None = None, ledger: str | None = "# Pipeline\n",
           approvals: dict | None = None, audit_approvals: dict | None = None, tool: str = "Bash",
           tool_input: dict | None = None, setup=None) -> tuple[str, str]:
    """Run the real hook on ``command`` in a temp project ({proj} in the command is that folder)."""
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        proj.mkdir()
        if ledger is not None:
            (proj / "ml_pipeline").mkdir()
            (proj / "ml_pipeline" / "PIPELINE.md").write_text(ledger)
            (proj / "ml_pipeline" / ".approvals.json").write_text(json.dumps(approvals or {"gates": {}, "overrides": []}))
        if audit_approvals is not None:
            (proj / "ml_audit").mkdir()
            (proj / "ml_audit" / ".audit_approvals.json").write_text(json.dumps(audit_approvals))
        for rel, body in (files or {}).items():
            path = proj / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
        if setup:
            setup(proj)
        if tool_input is None:
            tool_input = {"command": command.format(proj=proj, tool=TOOL, tmp=tmp, reltool=os.path.relpath(TOOL, proj))}
        else:
            tool_input = json.loads(json.dumps(tool_input).replace("{proj}", str(proj)))
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(proj)})
        saved = dict(os.environ)
        for key in list(os.environ):
            if key.startswith("ML_PIPELINE_"):
                os.environ.pop(key)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), _stdin(payload):
                hook.main()
        finally:
            os.environ.clear()
            os.environ.update(saved)
        result = json.loads(out.getvalue() or "{}")
        spec = result.get("hookSpecificOutput", {})
        return ("deny" if spec.get("permissionDecision") == "deny" else "allow"), spec.get("permissionDecisionReason", "")


EXACT = "python3 {tool} {proj}"


class ExactInvocation(unittest.TestCase):
    def test_the_tool_exists_where_the_hook_expects_it(self):
        self.assertTrue(TOOL.is_file())
        self.assertEqual(hook.AUDIT_TOOL.resolve(), TOOL.resolve())

    def test_exact_invocation_is_allowed_before_any_gate(self):
        self.assertEqual(decide(EXACT)[0], "allow")

    def test_exact_invocation_is_allowed_with_no_pipeline_at_all(self):
        self.assertEqual(decide(EXACT, ledger=None)[0], "allow")

    def test_known_flags_and_quoted_data_are_allowed(self):
        cmd = (EXACT + " --claim 'metric=accuracy,value=0.93,split=test,source=README.md:12'"
               " --check honest-scores --sample-rows 5000 --timeout 60"
               " --review 'family=leakage,severity=high,text=model.predict(X_test) is called inside the tuning loop'")
        self.assertEqual(decide(cmd)[0], "allow")

    def test_flag_equals_form_and_safe_interpreter_flags(self):
        self.assertEqual(decide("python3 -B -I {tool} {proj} --check=leakage")[0], "allow")

    def test_absolute_interpreter_path_to_a_real_binary(self):
        self.assertEqual(decide(f"{shlex.quote(os.path.realpath(sys.executable))} {{tool}} {{proj}}")[0], "allow")

    def test_relative_tool_path_resolving_to_the_plugin_copy(self):
        self.assertEqual(decide("python3 {reltool} .")[0], "allow")


class LookAlikes(unittest.TestCase):
    def test_a_project_script_with_the_same_name_is_denied(self):
        files = {"mlpipeline_audit.py": "import pandas\n" + SCORE_TEST}
        self.assertEqual(decide("python3 mlpipeline_audit.py .", files=files)[0], "deny")

    def test_same_name_under_a_copied_plugin_layout_is_denied(self):
        files = {"skills/ml-audit/lib/mlpipeline_audit.py": "x = 1\n"}
        self.assertEqual(decide("python3 {proj}/skills/ml-audit/lib/mlpipeline_audit.py {proj}", files=files)[0], "deny")

    def test_a_look_alike_copying_the_tools_docstring_is_denied(self):
        head = TOOL.read_text().split('"""')[1]
        files = {"lib/mlpipeline_audit.py": f'"""{head}"""\n' + SCORE_TEST}
        self.assertEqual(decide("python3 lib/mlpipeline_audit.py .", files=files)[0], "deny")

    def test_a_symlink_that_points_elsewhere_is_denied(self):
        def link(proj):
            (proj / "evil.py").write_text(SCORE_TEST)
            (proj / "mlpipeline_audit.py").symlink_to(proj / "evil.py")
        self.assertEqual(decide("python3 {proj}/mlpipeline_audit.py {proj}", setup=link)[0], "deny")

    def test_a_training_script_that_mentions_audit_is_denied(self):
        files = {"train_audit.py": "# audit-ready training script, see the ml audit\n" + TRAIN}
        self.assertEqual(decide("python3 train_audit.py --audit", files=files)[0], "deny")

    def test_a_scoring_script_named_audit_is_denied_before_gate_c(self):
        files = {"audit.py": "# audit the model\n" + SCORE_TEST}
        self.assertEqual(decide("python3 audit.py .", files=files)[0], "deny")

    def test_a_local_python_wrapper_script_is_denied(self):
        files = {"python3": "#!/bin/sh\nexec /usr/bin/env python3 \"$@\"\n"}
        self.assertEqual(decide("./python3 {tool} {proj}", files=files)[0], "deny")

    def test_a_runner_script_that_wraps_the_tool_is_denied(self):
        files = {"run.py": f"import subprocess\nsubprocess.run(['python3', '{TOOL}', '.'])\n"}
        self.assertEqual(decide("python3 run.py", files=files)[0], "deny")


class Chained(unittest.TestCase):
    def test_chained_commands_are_denied(self):
        for tail in (" && echo done", " ; python3 train.py", " || true", " | tee out.txt", " > out.txt",
                     " &", "\npython3 train.py", " 2>&1"):
            with self.subTest(tail=tail):
                self.assertEqual(decide(EXACT + tail, files={"train.py": TRAIN})[0], "deny")

    def test_a_command_before_the_tool_is_denied(self):
        for head in ("cd {proj} && ", "echo hi; ", "true && "):
            with self.subTest(head=head):
                self.assertEqual(decide(head + EXACT)[0], "deny")

    def test_substitution_and_expansion_are_denied(self):
        for arg in ("--claim \"$(cat secret)\"", "--claim `id`", "--claim \"$HOME\"", "--claim x*",
                    "--claim ~/x", "--claim 'unbalanced"):
            with self.subTest(arg=arg):
                self.assertEqual(decide(EXACT + " " + arg)[0], "deny")

    def test_environment_prefixes_and_wrappers_are_denied(self):
        for head in ("PYTHONPATH=. ", "ML_AUDIT=1 ", "env ", "nohup ", "timeout 5 ", "sudo "):
            with self.subTest(head=head):
                self.assertEqual(decide(head + EXACT)[0], "deny")

    def test_interpreter_code_and_module_flags_are_denied(self):
        for cmd in ("python3 -c 'import runpy' {tool} {proj}", "python3 -m mlpipeline_audit {proj}",
                    "python3 -X dev {tool} {proj}", "python3 -W ignore {tool} {proj}"):
            with self.subTest(cmd=cmd):
                self.assertEqual(decide(cmd)[0], "deny")

    def test_unknown_flags_and_extra_positionals_are_denied(self):
        for tail in (" --exec 'print(1)'", " --import os", " other_folder", " -v", " --check not-a-check",
                     " --sample-rows many", " --timeout"):
            with self.subTest(tail=tail):
                self.assertEqual(decide(EXACT + tail)[0], "deny")


class Tampering(unittest.TestCase):
    def test_writing_over_the_tool_is_denied(self):
        for tool in ("Write", "Edit"):
            with self.subTest(tool=tool):
                ti = {"file_path": str(TOOL), "content": "x = 1\n", "old_string": "a", "new_string": "b"}
                self.assertEqual(decide(tool=tool, tool_input=ti)[0], "deny")

    def test_copying_over_the_tool_with_the_shell_is_denied(self):
        self.assertEqual(decide("cp evil.py {tool}", files={"evil.py": "x = 1\n"})[0], "deny")

    def test_reading_the_tool_is_allowed(self):
        self.assertEqual(decide("cat {tool}")[0], "allow")
        self.assertEqual(decide("grep -n verdict {tool}")[0], "allow")

    def test_the_audit_approval_file_is_protected(self):
        ti = {"file_path": "{proj}/ml_audit/.audit_approvals.json", "content": "{}"}
        self.assertEqual(decide(tool="Write", tool_input=ti)[0], "deny")
        self.assertEqual(decide("echo '{{}}' > ml_audit/.audit_approvals.json")[0], "deny")


if __name__ == "__main__":
    unittest.main()
