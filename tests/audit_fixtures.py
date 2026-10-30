"""Generated project folders for the audit tests: clean controls and planted problems, built by code.

Each builder writes a small, self-contained project into a folder and returns it. Nothing here is checked-in data,
and no row holds anything but synthetic numbers and labels. Builders are deterministic (fixed seeds).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "skills" / "ml-audit" / "lib" / "mlpipeline_audit.py"


# ------------------------------------------------------------------ running the tool
def run_audit(folder: Path, *args: str, cwd: Path | None = None, timeout: int = 300) -> tuple[subprocess.CompletedProcess, Path | None, dict]:
    """Run the audit tool as the agent would; return (process, output folder, findings.json)."""
    proc = subprocess.run([sys.executable, str(TOOL), str(folder), *args], capture_output=True, text=True,
                          cwd=str(cwd or folder), timeout=timeout)
    out = None
    for line in proc.stdout.splitlines():
        if line.startswith("audit folder: "):
            out = Path(line.split(": ", 1)[1].strip())
    findings = json.loads((out / "findings.json").read_text()) if out and (out / "findings.json").is_file() else {}
    return proc, out, findings


def codes(findings: dict, evidence: str | None = None) -> list[str]:
    return [f["code"] for f in findings.get("findings", []) if evidence is None or f["evidence"] == evidence]


def find(findings: dict, code: str) -> list[dict]:
    return [f for f in findings.get("findings", []) if f["code"] == code]


def serious(findings: dict) -> list[dict]:
    return [f for f in findings.get("findings", []) if f["severity"] in ("critical", "high")]


def hash_tree(folder: Path) -> dict[str, str]:
    out = {}
    for dirpath, _, names in os.walk(folder):
        for name in names:
            path = Path(dirpath) / name
            if path.is_symlink():
                out[str(path.relative_to(folder))] = "link:" + os.readlink(path)
                continue
            out[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------ small numeric helpers (fixture side only)
def _accuracy(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ------------------------------------------------------------------ the clean classification project
def classification_frame(n: int, seed: int, id_start: int = 0) -> pd.DataFrame:
    """Binary target with an honest, moderately strong signal; no leaks, no duplicates."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    cat = rng.choice(["a", "b", "c"], size=n)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    logit = 1.5 * x[:, 0] - 1.0 * x[:, 1] + 0.6 * (cat == "a")
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"row_id": np.arange(id_start, id_start + n), "x0": x[:, 0].round(6), "x1": x[:, 1].round(6),
                         "x2": x[:, 2].round(6), "x3": x[:, 3].round(6), "cat": cat, "region": region, "y": y,
                         "_prob": prob})


def predictions_for(test: pd.DataFrame, seed: int, noise: float = 0.0) -> pd.DataFrame:
    """A 'model' that knows the true probability (so it is calibrated), optionally with noise."""
    rng = np.random.default_rng(seed)
    prob = np.clip(test["_prob"].to_numpy() + rng.normal(0, noise, len(test)), 0.001, 0.999) if noise else test["_prob"].to_numpy()
    return pd.DataFrame({"row_id": test["row_id"].to_numpy(), "y_true": test["y"].to_numpy(),
                         "y_pred": (prob > 0.5).astype(int), "y_prob": prob.round(6)})


def clean_project(root: Path, *, n_train: int = 1200, n_test: int = 600, seed: int = 0, with_model: bool = False,
                  extras: bool = True) -> Path:
    """Train/test files, test predictions, a correct metrics file, code, and a README; nothing wrong."""
    root.mkdir(parents=True, exist_ok=True)
    train = classification_frame(n_train, seed)
    test = classification_frame(n_test, seed + 1, id_start=n_train)
    preds = predictions_for(test, seed + 2)
    _write(root / "data" / "train.csv", train.drop(columns="_prob"))
    _write(root / "data" / "test.csv", test.drop(columns="_prob"))
    _write(root / "outputs" / "test_predictions.csv", preds)
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred), 3)}})
    if extras:
        (root / "train.py").write_text("import pandas as pd\n# trains the model (not run by the audit)\n")
        (root / "README.md").write_text("# Demo project\n\nA classifier for y.\n")
    if with_model:
        import pickle
        (root / "models").mkdir(exist_ok=True)
        (root / "models" / "plain.pkl").write_bytes(pickle.dumps({"weights": [0.1, 0.2], "bias": 0.0}))
    return root


def noise_folders(root: Path) -> None:
    """Folders the inventory must skip: environments, dependencies, version control, caches, earlier audits."""
    for rel in (".venv/lib/site.py", "venv/bin/activate", "node_modules/pkg/index.js", ".git/HEAD",
                "__pycache__/m.cpython-313.pyc", "env/pyvenv.cfg", ".ipynb_checkpoints/n.ipynb"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
    (root / "envs" / "py").mkdir(parents=True, exist_ok=True)
    (root / "envs" / "py" / "pyvenv.cfg").write_text("home = /usr\n")
    (root / "envs" / "py" / "data.csv").write_text("a,b\n1,2\n")


def guard_state(root: Path) -> None:
    """A live pipeline: ledger, run log, and the guard's approval and state files (never opened by the audit)."""
    state = root / "ml_pipeline"
    state.mkdir(parents=True, exist_ok=True)
    (state / "PIPELINE.md").write_text("# Pipeline\n- Gate A: approved 2026-10-01\n")
    (state / ".approvals.json").write_text(json.dumps({"gates": {"A": {"at": "2026-10-01", "run": ""}}, "overrides": []}))
    (state / ".guard_state.json").write_text(json.dumps({"test_touches": 0}))


# ------------------------------------------------------------------ ticket 02: honest scores
def wrong_claim(root: Path) -> Path:
    """The metrics file claims 8 points more accuracy than the saved predictions give."""
    clean_project(root)
    preds = pd.read_csv(root / "outputs" / "test_predictions.csv")
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred) + 0.08, 3)}})
    return root


def claim_without_predictions(root: Path) -> Path:
    """A claimed test score with nothing to recompute it from."""
    clean_project(root)
    (root / "outputs" / "test_predictions.csv").unlink()
    return root


def baseline_model(root: Path) -> Path:
    """The 'model' predicts the majority class for every row, and the claim is honest about it."""
    clean_project(root)
    test = pd.read_csv(root / "data" / "test.csv")
    train = pd.read_csv(root / "data" / "train.csv")
    majority = int(train["y"].mode()[0])
    preds = pd.DataFrame({"row_id": test.row_id, "y_true": test.y, "y_pred": majority, "y_prob": 0.5})
    _write(root / "outputs" / "test_predictions.csv", preds)
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred), 3)}})
    return root


def wrong_split_claim(root: Path) -> Path:
    """The 'test' predictions were made on the validation rows."""
    clean_project(root)
    val = classification_frame(500, 77, id_start=5000)
    _write(root / "data" / "val.csv", val.drop(columns="_prob"))
    preds = predictions_for(val, 78)
    _write(root / "outputs" / "test_predictions.csv", preds)
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred), 3)}})
    return root


def exact_accuracy_project(root: Path, correct: int, total: int, claim: str) -> Path:
    """Predictions whose accuracy is exactly correct/total, and a metrics file stating ``claim`` verbatim."""
    clean_project(root)
    test = pd.read_csv(root / "data" / "test.csv")
    test = pd.concat([test] * (total // len(test) + 1), ignore_index=True).iloc[:total].copy()
    test["row_id"] = np.arange(len(test)) + 10_000
    _write(root / "data" / "test.csv", test)
    y = test["y"].to_numpy()
    pred = y.copy()
    pred[correct:] = 1 - pred[correct:]
    _write(root / "outputs" / "test_predictions.csv",
           pd.DataFrame({"row_id": test.row_id, "y_true": y, "y_pred": pred, "y_prob": np.where(pred == 1, 0.8, 0.2)}))
    path = root / "outputs" / "metrics.json"
    path.write_text('{"test": {"accuracy": ' + claim + "}}\n")
    return root
