"""Ticket 06: a model the user trusts is loaded in a child process with the project's interpreter and scored.

Run:  python -m pytest tests/test_audit_trust.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

try:
    import joblib  # noqa: F401
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

FAKE_TORCH = '''import json, os
from collections import OrderedDict
def load(path, map_location=None, weights_only=None):
    with open(os.environ["FAKE_TORCH_LOG"], "a") as fh:
        fh.write(json.dumps({"weights_only": weights_only}) + "\\n")
    return OrderedDict(weight=[0.0])
'''


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn and joblib write the trusted model")
class TrustedScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_trust_the_model_is_not_loaded_and_its_checks_could_not_run(self):
        proj = fx.trusted_project(self.root / "proj")
        _, out, f = fx.run_audit(proj)
        trust = next(c for c in f["checks"] if c["id"] == "trusted-scoring")
        self.assertEqual(trust["status"], "could not check")
        self.assertIn("--trust", trust["reason"])
        score = next(c for c in f["checks"] if c["id"] == "score-recomputation")
        self.assertEqual(score["status"], "could not check")
        self.assertEqual(f["verdict"], "not enough evidence")
        self.assertFalse((out / "trusted_predictions").exists())
        self.assertFalse(f["viewed_test_labels"])

    def test_with_trust_the_model_is_scored_and_its_claim_checked(self):
        proj = fx.trusted_project(self.root / "proj")
        before = fx.hash_tree(proj)
        _, out, f = fx.run_audit(proj, "--trust", "proj")
        trust = next(c for c in f["checks"] if c["id"] == "trusted-scoring")
        self.assertEqual(trust["status"], "ran", trust["reason"])
        score = next(c for c in f["checks"] if c["id"] == "score-recomputation")
        self.assertEqual(score["status"], "ran", score["reason"])
        self.assertEqual(fx.find(f, "claim-mismatch"), [])
        self.assertEqual(f["verdict"], "trustworthy", f["verdict_reason"])
        self.assertTrue(f["viewed_test_labels"])
        report = (out / "report.md").read_text()
        self.assertIn("viewed the test labels", report)
        self.assertIn("must not be used to choose", report)
        after = fx.hash_tree(proj)
        prefix = str(out.relative_to(proj)) + os.sep
        self.assertEqual({k for k in set(before) | set(after) if before.get(k) != after.get(k) and not k.startswith(prefix)}, set())
        self.assertFalse(any("__pycache__" in k for k in after))

    def test_a_trusted_model_exposes_a_wrong_claim(self):
        _, _, f = fx.run_audit(fx.trusted_project(self.root / "proj", claim_offset=0.1), "--trust", "proj")
        self.assertEqual(fx.find(f, "claim-mismatch")[0]["severity"], "critical")
        self.assertEqual(f["verdict"], "not trustworthy")

    def test_trust_must_name_this_project(self):
        proj = fx.trusted_project(self.root / "proj")
        _, _, f = fx.run_audit(proj, "--trust", "other")
        trust = next(c for c in f["checks"] if c["id"] == "trusted-scoring")
        self.assertEqual(trust["status"], "could not check")
        self.assertIn("names 'other'", trust["reason"])

    def test_a_crash_while_loading_is_a_could_not_check_finding(self):
        proj = fx.missing_class_project(self.root / "proj")
        proc, _, f = fx.run_audit(proj, "--trust", "proj")
        self.assertEqual(proc.returncode, 0)
        hit = fx.find(f, "trusted-scoring-failed")[0]
        self.assertEqual((hit["evidence"], hit["files"]), ("could-not-check", ["models/gone.pkl"]))
        self.assertIn("ModuleNotFoundError", hit["detail"])

    def test_a_timeout_is_a_could_not_check_finding(self):
        proj = fx.slow_model_project(self.root / "proj")
        proc, _, f = fx.run_audit(proj, "--trust", "proj", "--timeout", "3")
        self.assertEqual(proc.returncode, 0)
        hit = fx.find(f, "trusted-scoring-failed")[0]
        self.assertIn("timed out after 3 s", hit["detail"])

    def test_dangerous_models_are_not_loaded_even_with_trust(self):
        marker = self.root / "marker"
        proj = fx.model_project(self.root / "proj", {"models/evil.pkl": fx.marker_pickle(marker)})
        _, _, f = fx.run_audit(proj, "--trust", "proj")
        self.assertFalse(marker.exists())
        trust = next(c for c in f["checks"] if c["id"] == "trusted-scoring")
        self.assertIn("not loaded", trust["reason"])

    def test_the_code_free_mode_is_tried_first(self):
        proj = fx.clean_project(self.root / "proj")
        (proj / "models").mkdir()
        (proj / "models" / "net.pt").write_bytes(fx.torch_zip_bytes())
        fake = self.root / "fake_site" / "torch"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text(FAKE_TORCH)
        log = self.root / "torch_calls.jsonl"
        env = {**os.environ, "FAKE_TORCH_LOG": str(log), "PYTHONPATH": str(fake.parent)}
        import subprocess
        proc = subprocess.run([sys.executable, str(fx.TOOL), str(proj), "--trust", "proj"], capture_output=True,
                              text=True, env=env, cwd=str(proj), timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(calls[0], {"weights_only": True})
        out = Path(next(l.split(": ", 1)[1] for l in proc.stdout.splitlines() if l.startswith("audit folder: ")))
        f = json.loads((out / "findings.json").read_text())
        hit = fx.find(f, "trusted-scoring-failed")[0]
        self.assertIn("weights only", hit["detail"])


if __name__ == "__main__":
    unittest.main()
