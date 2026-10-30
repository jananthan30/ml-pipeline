"""Ticket 10: the retrain check runs only with the user's approval, only on copies inside the audit folder.

Run:  python -m pytest tests/test_audit_retrain.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import audit_fixtures as fx  # noqa: E402
import guard_training as hook  # noqa: E402


class Retrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()  # the tool reports resolved paths; macOS temp is a symlink

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_approval_it_is_listed_with_its_reason_and_does_not_run(self):
        _, out, f = fx.run_audit(fx.near_copies(self.root / "proj"))
        check = next(c for c in f["checks"] if c["id"] == "shuffled-label")
        self.assertEqual(check["status"], "needs approval")
        self.assertIn("approve audit retrain for proj", check["reason"])
        self.assertEqual(fx.find(f, "shuffled-label-above-chance"), [])
        self.assertFalse((out / "retrain").exists())
        self.assertIn("needs approval", (out / "report.md").read_text())

    def test_the_guard_denies_the_flag_without_the_users_message(self):
        proj = fx.near_copies(self.root / "proj")
        command = f"python3 {fx.TOOL} {proj} --retrain proj"
        exact, reason = hook.audit_problem({"command": command}, str(proj))
        self.assertTrue(exact)
        self.assertIn("approve audit retrain for proj", reason)

    def test_a_planted_leak_scores_well_above_chance_and_a_clean_control_does_not(self):
        _, _, leak = fx.run_audit(fx.near_copies(self.root / "proj"), "--retrain", "proj")
        self.assertEqual(fx.find(leak, "split-overlap"), [])  # the exact check cannot see jittered copies
        hit = fx.find(leak, "shuffled-label-above-chance")[0]
        self.assertEqual((hit["severity"], hit["evidence"]), ("critical", "proven"))
        self.assertGreater(hit["counts"]["accuracy"], hit["counts"]["chance"] + 0.15)
        self.assertEqual(leak["verdict"], "not trustworthy")
        _, _, clean = fx.run_audit(fx.clean_project(self.root / "clean"), "--retrain", "clean")
        self.assertEqual(fx.find(clean, "shuffled-label-above-chance"), [])
        check = next(c for c in clean["checks"] if c["id"] == "shuffled-label")
        self.assertEqual(check["status"], "ran")
        self.assertEqual(clean["verdict"], "trustworthy")

    def test_project_files_gates_and_test_touches_are_unchanged(self):
        proj = fx.near_copies(self.root / "proj")
        fx.guard_state(proj)
        before = fx.hash_tree(proj)
        _, out, _ = fx.run_audit(proj, "--retrain", "proj")
        after = fx.hash_tree(proj)
        prefix = str(out.relative_to(proj)) + os.sep
        self.assertEqual({k for k in set(before) | set(after) if before.get(k) != after.get(k) and not k.startswith(prefix)},
                         set())
        self.assertFalse((out / "retrain").exists())  # the copies are removed once the check is done

    def test_retrain_must_name_this_project(self):
        _, _, f = fx.run_audit(fx.near_copies(self.root / "proj"), "--retrain", "other")
        check = next(c for c in f["checks"] if c["id"] == "shuffled-label")
        self.assertEqual(check["status"], "needs approval")


class NeverFits(unittest.TestCase):
    def test_the_guards_own_detector_finds_no_training_in_the_tool(self):
        found = hook.analyse(fx.TOOL.read_text())
        self.assertEqual({k: v for k, v in found.items() if v}, {})


if __name__ == "__main__":
    unittest.main()
