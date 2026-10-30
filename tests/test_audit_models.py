"""Ticket 05: model files are scanned without being run, and never loaded without the user's trust.

Run:  python -m pytest tests/test_audit_models.py
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fixtures as fx  # noqa: E402

spec = importlib.util.spec_from_file_location("mlpipeline_audit", fx.TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def model_findings(f: dict, path: str) -> list[dict]:
    return [x for x in f["findings"] if x["family"] == "model-safety" and path in x["files"]]


class Scanner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_marker_pickle_runs_when_loaded_but_not_when_audited(self):
        # Positive control, deliberately: loading this pickle creates the marker, so "no marker" below means something.
        control = self.root / "control_marker"
        pkl = self.root / "control.pkl"
        pkl.write_bytes(fx.marker_pickle(control))
        subprocess.run([sys.executable, "-c", "import pickle, sys; pickle.load(open(sys.argv[1], 'rb'))", str(pkl)],
                       check=True, capture_output=True)
        self.assertTrue(control.exists())
        # The audit, without trust, must leave no marker.
        marker = self.root / "audit_marker"
        proj = fx.model_project(self.root / "p", {"models/model.pkl": fx.marker_pickle(marker)})
        _, _, f = fx.run_audit(proj)
        self.assertFalse(marker.exists())
        hit = model_findings(f, "models/model.pkl")[0]
        self.assertEqual(hit["code"], "risky-imports")
        self.assertIn("builtins.open", hit["detail"])

    def test_dangerous_imports_are_named(self):
        cases = {"models/a.pkl": (fx.DANGEROUS_PICKLE, "os.system"),
                 "models/b.pkl": (fx.STACK_GLOBAL_DANGEROUS, "subprocess.Popen"),
                 "models/c.pkl": (fx.MEMO_STACK_GLOBAL, "os.system")}
        proj = fx.model_project(self.root / "p", {k: v[0] for k, v in cases.items()})
        _, _, f = fx.run_audit(proj)
        for path, (_, name) in cases.items():
            with self.subTest(path=path):
                hit = model_findings(f, path)[0]
                self.assertEqual((hit["code"], hit["severity"], hit["evidence"]), ("risky-imports", "high", "proven"))
                self.assertIn(name, hit["detail"])
                self.assertEqual(hit["counts"]["dangerous_imports"], 1)

    def test_ordinary_sklearn_joblib_and_torch_files_are_not_flagged(self):
        try:
            import joblib  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn and joblib are needed to write the ordinary fixtures")
        proj = fx.clean_project(self.root / "p")
        paths = fx.sklearn_models(proj / "models")
        (proj / "models" / "net.pt").write_bytes(fx.torch_zip_bytes())
        _, _, f = fx.run_audit(proj)
        for path in [p.relative_to(proj).as_posix() for p in paths] + ["models/net.pt"]:
            with self.subTest(path=path):
                hits = model_findings(f, path)
                self.assertEqual([h["code"] for h in hits], ["scan-clean"], [h["detail"] for h in hits])
                self.assertEqual(hits[0]["severity"], "info")
                self.assertIn("does not prove", hits[0]["detail"])

    def test_a_dangerous_import_after_a_joblib_array_is_still_found(self):
        try:
            import joblib
            import numpy as np
        except ImportError:
            self.skipTest("joblib is needed to write this fixture")
        marker = self.root / "never"

        class Opens:
            def __reduce__(self):
                return (open, (str(marker), "w"))
        proj = fx.clean_project(self.root / "p")
        (proj / "models").mkdir()
        joblib.dump({"weights": np.arange(1000, dtype=np.float64), "hook": Opens()}, proj / "models" / "m.joblib")
        _, _, f = fx.run_audit(proj)
        hit = model_findings(f, "models/m.joblib")[0]
        self.assertEqual((hit["code"], hit["severity"]), ("risky-imports", "high"))
        self.assertIn("_io.open", hit["detail"])  # open() pickles under its C module
        self.assertFalse(marker.exists())

    def test_opcodes_disguised_as_joblib_array_bytes_are_still_found(self):
        proj = fx.model_project(self.root / "p", {"models/disguised.joblib": fx.joblib_array_disguise(),
                                                  "models/disguised.pkl": fx.joblib_array_disguise()})
        _, _, f = fx.run_audit(proj)
        for path in ("models/disguised.joblib", "models/disguised.pkl"):
            with self.subTest(path=path):
                hit = model_findings(f, path)[0]
                self.assertEqual((hit["code"], hit["severity"]), ("risky-imports", "high"))
                self.assertIn("os.system", hit["detail"])

    def test_corrupt_or_unparseable_files_are_unverified_never_safe(self):
        files = {"models/truncated.pkl": fx.TRUNCATED_PICKLE, "models/garbage.joblib": b"\x00\x01not a pickle at all",
                 "models/empty.pkl": b"", "models/broken.pt": b"PK\x03\x04broken zip",
                 "models/ext.pkl": b"\x80\x02\x82\x01)R.",  # EXT1: a name from the copyreg registry, unresolvable here
                 "models/lz4.joblib": b'\x04"M\x18' + b"\x00" * 20}
        proj = fx.model_project(self.root / "p", files)
        _, _, f = fx.run_audit(proj)
        for path in files:
            with self.subTest(path=path):
                hits = model_findings(f, path)
                self.assertEqual([h["code"] for h in hits], ["unverified-model"])
                self.assertNotIn("scan-clean", [h["code"] for h in hits])

    def test_code_free_formats_are_reported_as_such(self):
        header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        st = len(header).to_bytes(8, "little") + header + b"\x00" * 4
        proj = fx.model_project(self.root / "p", {"models/w.safetensors": st})
        _, _, f = fx.run_audit(proj)
        self.assertEqual([h["code"] for h in model_findings(f, "models/w.safetensors")], ["code-free-format"])

    def test_the_report_says_a_clean_scan_is_not_proof(self):
        proj = fx.clean_project(self.root / "p", with_model=True)
        _, out, f = fx.run_audit(proj)
        self.assertIn("does not prove", (out / "report.md").read_text())
        check = next(c for c in f["checks"] if c["id"] == "model-scan")
        self.assertEqual(check["status"], "ran")
        self.assertIn("not loaded", check["reason"])


class NoLoadingWithoutTrust(unittest.TestCase):
    def test_no_code_path_in_the_tool_loads_a_model(self):
        """Every model load lives in the trusted child runner (a string run in a child process), never in the tool."""
        tree = ast.parse(fx.TOOL.read_text())
        loads = {"load", "loads", "Unpickler", "load_model", "InferenceSession"}
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                owner = getattr(fn.value, "id", "") if isinstance(fn, ast.Attribute) else ""
                if name in loads and owner not in ("json",):
                    bad.append((node.lineno, owner, name))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if any(n.split(".")[0] in ("pickle", "joblib", "torch", "dill", "cloudpickle") for n in names):
                    bad.append((node.lineno, "import", names))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
