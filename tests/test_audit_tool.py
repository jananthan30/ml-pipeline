"""The audit tool end to end: run it on a generated folder and assert on findings.json, the verdict, and the files.

Run:  python -m pytest tests/test_audit_tool.py      (needs pandas and numpy, like the guard library)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

VERDICTS = {"trustworthy", "not trustworthy", "not enough evidence"}

# Runs the tool under an audit hook that logs every file it opens (path, mode, flags).
OPEN_LOGGER = r"""
import json, runpy, sys
log = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)) or (event == "open" and hasattr(args[0], "__fspath__")):
        log.append([str(args[0]), str(args[1]), int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 0])
sys.addaudithook(hook)
tool, out = sys.argv[1], sys.argv[2]
sys.argv = [tool] + sys.argv[3:]
try:
    runpy.run_path(tool, run_name="__main__")
except SystemExit:
    pass
snapshot = list(log)
with open(out, "w") as fh:
    fh.write(json.dumps(snapshot))
"""


class Skeleton(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()  # the tool reports resolved paths; macOS temp is a symlink

    def tearDown(self):
        self.tmp.cleanup()

    def project(self, name="proj"):
        proj = fx.clean_project(self.root / name, with_model=True)
        fx.noise_folders(proj)
        fx.guard_state(proj)
        return proj

    def test_inventory_lists_every_kind_and_skips_environment_folders(self):
        proj = self.project()
        proc, out, _ = fx.run_audit(proj)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        inv = json.loads((out / "inventory.json").read_text())
        kinds = {f["path"]: (f["kind"], f.get("role")) for f in inv["files"]}
        self.assertEqual(kinds["data/train.csv"], ("data", "train"))
        self.assertEqual(kinds["data/test.csv"], ("data", "test"))
        self.assertEqual(kinds["outputs/test_predictions.csv"][0], "predictions")
        self.assertEqual(kinds["models/plain.pkl"][0], "model")
        self.assertEqual(kinds["train.py"][0], "code")
        self.assertEqual(kinds["outputs/metrics.json"][0], "claims")
        self.assertEqual(kinds["ml_pipeline/PIPELINE.md"][0], "claims")
        self.assertEqual(kinds["README.md"][0], "docs")
        self.assertEqual(inv["target"], "y")
        listed = " ".join(kinds)
        for skipped in (".venv", "venv/", "node_modules", ".git/", "__pycache__", "env/", "envs/py", ".ipynb_checkpoints"):
            self.assertNotIn(skipped, listed)
        self.assertNotIn("ml_pipeline/.approvals.json", kinds)
        self.assertNotIn("ml_pipeline/.guard_state.json", kinds)
        self.assertTrue({".venv", "node_modules", ".git"} <= set(inv["skipped_folders"]))

    def test_the_user_can_correct_the_inventory_and_rerun(self):
        proj = self.project()
        _, out, _ = fx.run_audit(proj)
        inv = json.loads((out / "inventory.json").read_text())
        self.assertFalse(inv["corrected"])
        (proj / "ml_audit" / "inventory.json").write_text(json.dumps(
            {"files": [{"path": "data/test.csv", "role": "val"}], "target": "x3", "time_column": None}))
        _, out2, _ = fx.run_audit(proj)
        inv2 = json.loads((out2 / "inventory.json").read_text())
        self.assertTrue(inv2["corrected"])
        self.assertEqual(inv2["target"], "x3")
        self.assertEqual({f["path"]: f.get("role") for f in inv2["files"]}["data/test.csv"], "val")
        self.assertIn({"what": "target", "guessed": "y", "corrected": "x3"}, inv2["corrections"])
        self.assertIn({"what": "data/test.csv role", "guessed": "test", "corrected": "val"}, inv2["corrections"])
        self.assertEqual(len(inv2["corrections"]), 2)  # time_column was already null: not a change
        report = (out2 / "report.md").read_text()
        self.assertIn("target: `y` -> `x3`", report)
        self.assertIn("data/test.csv role: `test` -> `val`", report)
        self.assertNotIn("by the user", report)

    def test_output_goes_to_a_new_dated_subfolder_each_run(self):
        proj = self.project()
        _, out1, f1 = fx.run_audit(proj)
        first = fx.hash_tree(out1)
        _, out2, f2 = fx.run_audit(proj)
        self.assertNotEqual(out1, out2)
        self.assertEqual(out1.parent, proj / "ml_audit")
        self.assertEqual(out2.parent, proj / "ml_audit")
        self.assertRegex(out1.name, r"^\d{4}-\d{2}-\d{2}_\d{6}(-\d+)?$")
        for out in (out1, out2):
            self.assertEqual({p.name for p in out.iterdir()} >= {"report.md", "findings.json", "inventory.json"}, True)
        self.assertEqual(fx.hash_tree(out1), first)  # earlier audits are kept as they were
        self.assertIn(f1["verdict"], VERDICTS)
        self.assertIn(f2["verdict"], VERDICTS)

    def test_only_one_new_audit_folder_changes(self):
        proj = self.project()
        before = fx.hash_tree(proj)
        _, out, _ = fx.run_audit(proj)
        after = fx.hash_tree(proj)
        prefix = str(out.relative_to(proj)) + os.sep
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        self.assertTrue(changed)
        self.assertEqual({k for k in changed if not k.startswith(prefix)}, set())
        self.assertEqual({k for k in after if k.startswith("ml_audit" + os.sep) and not k.startswith(prefix)}, set())

    def test_the_guard_state_and_approval_files_are_never_opened(self):
        proj = self.project()
        log_path = self.root / "opens.json"
        script = self.root / "logger.py"
        script.write_text(OPEN_LOGGER)
        proc = subprocess.run([sys.executable, str(script), str(fx.TOOL), str(log_path), str(proj)],
                              capture_output=True, text=True, cwd=str(proj), timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        opens = json.loads(log_path.read_text())
        self.assertTrue(opens)
        names = [Path(p).name for p, _, _ in opens]
        self.assertNotIn(".approvals.json", names)
        self.assertNotIn(".guard_state.json", names)
        out = next(line.split(": ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("audit folder: "))
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND
        for path, mode, flags in opens:
            writes = any(c in mode for c in "wax+") or bool(flags & write_flags)
            if writes and str(Path(path).resolve()).startswith(str(proj.resolve())):
                self.assertTrue(str(Path(path).resolve()).startswith(str(Path(out).resolve())), path)

    def test_the_ledger_gains_no_lines(self):
        proj = self.project()
        ledger = (proj / "ml_pipeline" / "PIPELINE.md").read_text()
        fx.run_audit(proj)
        self.assertEqual((proj / "ml_pipeline" / "PIPELINE.md").read_text(), ledger)

    def test_the_guards_record_files_are_pipeline_records_not_data(self):
        proj = fx.pipeline_project(self.root / "p")
        _, out, f = fx.run_audit(proj)
        inv = json.loads((out / "inventory.json").read_text())
        kinds = {x["path"]: x["kind"] for x in inv["files"]}
        for name in fx.GUARD_RECORDS:
            self.assertEqual(kinds["ml_pipeline/" + name], "pipeline record", name)
        self.assertEqual(inv["target"], "y")
        self.assertEqual(f["verdict"], "trustworthy", f["verdict_reason"])

    def test_every_record_file_the_guard_library_names_is_known_to_the_tool(self):
        import importlib.util
        import re
        spec = importlib.util.spec_from_file_location("audit_tool", fx.TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        guard_src = (fx.ROOT / "skills" / "ml-pipeline" / "lib" / "mlpipeline_guard.py").read_text()
        names = set(re.findall(r'^\w+_FILE = "([^"]+)"', guard_src, re.M))
        names |= set(re.findall(r'state_dir / "(\w+\.\w+)"', guard_src))
        names -= {".guard_state.json", ".approvals.json", "runs.jsonl", "PIPELINE.md"}  # never opened / claims
        self.assertTrue(names)
        self.assertEqual(names - set(tool.GUARD_RECORD_FILES), set())

    def test_required_checks_never_say_ran_without_comparing_anything(self):
        _, _, f = fx.run_audit(fx.xy_split_files(self.root / "xy"))
        status = {c["id"]: (c["status"], c["reason"]) for c in f["checks"]}
        self.assertEqual(status["train-test-overlap"][0], "could not check")
        self.assertIn("inventory", status["train-test-overlap"][1])
        self.assertEqual(status["feature-target-leakage"][0], "could not check")
        self.assertNotEqual(f["verdict"], "trustworthy")

    def test_a_folder_with_nothing_to_check_is_not_enough_evidence(self):
        proj = self.root / "empty"
        proj.mkdir()
        (proj / "README.md").write_text("# nothing here\n")
        proc, out, findings = fx.run_audit(proj)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(findings["verdict"], "not enough evidence")
        report = (out / "report.md").read_text()
        self.assertIn("not enough evidence", report)
        self.assertIn("could not check", report)
        required = [c for c in findings["checks"] if c["required"]]
        self.assertEqual({c["id"] for c in required}, {"score-recomputation", "train-test-overlap", "feature-target-leakage"})

    def test_a_missing_folder_is_an_error(self):
        proc = subprocess.run([sys.executable, str(fx.TOOL), str(self.root / "nope")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)

    def test_two_runs_in_the_same_second_get_separate_folders(self):
        proj = self.root / "fast"
        proj.mkdir()
        procs = [subprocess.Popen([sys.executable, str(fx.TOOL), str(proj)], stdout=subprocess.PIPE, text=True)
                 for _ in range(3)]
        outs = {p.communicate(timeout=120)[0] for p in procs}
        folders = {line for o in outs for line in o.splitlines() if line.startswith("audit folder: ")}
        self.assertEqual(len(folders), 3)
        time.sleep(0)


class WithoutPandas(unittest.TestCase):
    def test_a_python_without_pandas_still_gets_a_report(self):
        """The inventory, model scan and log checks are standard library; the rest say they could not run."""
        with tempfile.TemporaryDirectory() as tmp:
            proj = fx.model_project(Path(tmp) / "p", {"models/a.pkl": fx.DANGEROUS_PICKLE})
            blocker = Path(tmp) / "block"
            blocker.mkdir()
            for name in ("pandas", "numpy"):
                (blocker / f"{name}.py").write_text("raise ImportError('blocked for this test')\n")
            proc = subprocess.run([sys.executable, str(fx.TOOL), str(proj)], capture_output=True, text=True,
                                  env={**os.environ, "PYTHONPATH": str(blocker)}, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = Path(next(l.split(": ", 1)[1] for l in proc.stdout.splitlines() if l.startswith("audit folder: ")))
            f = json.loads((out / "findings.json").read_text())
            self.assertEqual(f["verdict"], "not enough evidence")
            self.assertEqual(fx.find(f, "risky-imports")[0]["severity"], "high")
            for check in f["checks"]:
                if check["id"] not in ("model-scan", "test-use-log", "test-in-tuning-code"):
                    self.assertNotEqual(check["status"], "ran", check)
            self.assertIn("pandas", next(c for c in f["checks"] if c["id"] == "data-problems")["reason"])


class SkillBoundaries(unittest.TestCase):
    def test_codex_and_kimi_installs_receive_the_skill_and_say_rules_are_instructions_only(self):
        with tempfile.TemporaryDirectory() as home:
            for d in (".codex", ".kimi-code"):
                Path(home, d).mkdir()
            subprocess.run(["bash", str(fx.ROOT / "install-other-tools.sh")], check=True, capture_output=True,
                           env={**os.environ, "HOME": home})
            for d in (".codex", ".kimi-code"):
                skill = Path(home, d, "skills", "ml-audit")
                self.assertTrue((skill / "SKILL.md").is_file())
                self.assertTrue((skill / "lib" / "mlpipeline_audit.py").is_file())
                self.assertTrue(Path(home, d, "skills", "ml-pipeline", "lib", "mlpipeline_guard.py").is_file())
                agents = Path(home, d, "AGENTS.md").read_text()
                self.assertIn("ML audit (read-only)", agents)
                self.assertIn("instructions only", agents)
                self.assertIn("instructions only", (skill / "SKILL.md").read_text())


    def test_the_pipeline_skill_excludes_audits_and_the_audit_skill_is_narrow(self):
        root = fx.ROOT
        pipeline = (root / "skills" / "ml-pipeline" / "SKILL.md").read_text().split("---")[1]
        self.assertIn("audit", pipeline.lower())
        self.assertRegex(pipeline, r"(?i)not .*audit|except.*audit|audit.*use the ml-audit")
        audit = (root / "skills" / "ml-audit" / "SKILL.md").read_text().split("---")[1]
        self.assertIn("name: ml-audit", audit)
        self.assertNotRegex(audit.lower(), r"\btrain(ing)? a model\b")


if __name__ == "__main__":
    unittest.main()
