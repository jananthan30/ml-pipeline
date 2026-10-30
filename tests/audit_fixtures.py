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


# ------------------------------------------------------------------ ticket 03: data problems (one defect each)
def _edit_train(root: Path, edit) -> Path:
    clean_project(root)
    train = pd.read_csv(root / "data" / "train.csv")
    _write(root / "data" / "train.csv", edit(train))
    return root


def missing_values(root: Path, frac: float = 0.3) -> Path:
    def edit(t):
        t = t.copy()
        t.loc[t.sample(frac=frac, random_state=1).index, "x2"] = np.nan
        return t
    return _edit_train(root, edit)


def duplicate_rows(root: Path) -> Path:
    return _edit_train(root, lambda t: pd.concat([t, t.iloc[:60]], ignore_index=True))


def constant_column(root: Path) -> Path:
    def edit(t):
        t = t.copy()
        t.insert(5, "source_batch", "batch-1")
        return t
    return _edit_train(root, edit)


def conflicting_labels(root: Path, twins_n: int = 20) -> Path:
    def edit(t):
        twins = t.iloc[:twins_n].copy()
        twins["y"] = 1 - twins["y"]
        return pd.concat([t, twins], ignore_index=True)
    return _edit_train(root, edit)


def invalid_labels(root: Path) -> Path:
    def edit(t):
        t = t.copy().astype({"y": "float"})
        t.loc[t.index[:25], "y"] = np.nan
        return t
    return _edit_train(root, edit)


def unique_secrets(root: Path) -> Path:
    """Every row carries a unique marker string; none of them may appear in the audit's output."""
    def edit(t):
        t = t.copy()
        t["note"] = [f"SECRET-ROW-{i:06d}" for i in range(len(t))]
        t.loc[t.index[:30], "x1"] = np.nan
        return pd.concat([t, t.iloc[:5]], ignore_index=True)
    return _edit_train(root, edit)


def large_file(root: Path, n: int = 6000, missing: int = 1234) -> Path:
    """A training file several times the sampling limit used by the test, with an exact number of gaps."""
    clean_project(root)
    train = classification_frame(n, 5)
    train.loc[train.index[:missing], "x3"] = np.nan
    _write(root / "data" / "train.csv", train.drop(columns="_prob"))
    return root


# ------------------------------------------------------------------ ticket 04: leakage
def _both(root: Path, edit_train, edit_test) -> Path:
    clean_project(root)
    for name, edit in (("train", edit_train), ("test", edit_test)):
        frame = pd.read_csv(root / "data" / f"{name}.csv")
        _write(root / "data" / f"{name}.csv", edit(frame))
    return root


def split_overlap(root: Path, copies: int = 30) -> Path:
    """``copies`` training rows reappear in the test file under fresh row ids."""
    clean_project(root)
    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test.csv")
    copied = train.iloc[:copies].copy()
    copied["row_id"] = np.arange(copies) + 90_000
    test = pd.concat([test.iloc[: len(test) - copies], copied], ignore_index=True)
    _write(root / "data" / "test.csv", test)
    preds = predictions_for(test.assign(_prob=1 / (1 + np.exp(-(1.5 * test.x0 - test.x1 + 0.6 * (test.cat == "a"))))), 2)
    _write(root / "outputs" / "test_predictions.csv", preds)
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred), 3)}})
    return root


def shared_entities(root: Path) -> Path:
    """Rows of the same patients land in both splits (a row-level split of repeated entities)."""
    rng = np.random.default_rng(9)
    def add(frame):
        frame = frame.copy()
        frame.insert(1, "patient_id", rng.integers(0, 150, len(frame)))
        return frame
    return _both(root, add, add)


def target_leak(root: Path) -> Path:
    """A column that is the label plus a little noise: it gives the answer away."""
    rng = np.random.default_rng(4)
    def add(frame):
        frame = frame.copy()
        frame["post_visit_score"] = (frame["y"] + rng.normal(0, 0.05, len(frame))).round(4)
        return frame
    return _both(root, add, add)


def flipped_copy_feature(root: Path, reduction: float) -> Path:
    """A binary column equal to the label with exactly enough flips for a chosen single-feature error reduction."""
    clean_project(root)
    train = pd.read_csv(root / "data" / "train.csv")
    m = float(train["y"].value_counts(normalize=True).iloc[0])
    flips = int(round((1 - reduction) * (1 - m) * len(train)))
    flag = train["y"].to_numpy().copy()
    flag[:flips] = 1 - flag[:flips]
    train["flag"] = flag
    _write(root / "data" / "train.csv", train)
    test = pd.read_csv(root / "data" / "test.csv")
    test["flag"] = np.random.default_rng(1).integers(0, 2, len(test))
    _write(root / "data" / "test.csv", test)
    return root


def time_order(root: Path, violate: bool = True) -> Path:
    """Dated rows; with ``violate`` the training dates run past the start of the test dates."""
    rng = np.random.default_rng(12)
    start_train = pd.Timestamp("2024-01-01")
    start_test = pd.Timestamp("2024-07-01") if violate else pd.Timestamp("2025-02-01")
    def dates(frame, start, days):
        frame = frame.copy()
        frame.insert(1, "event_date", (start + pd.to_timedelta(np.sort(rng.integers(0, days, len(frame))), unit="D")).strftime("%Y-%m-%d"))
        return frame
    return _both(root, lambda t: dates(t, start_train, 365), lambda t: dates(t, start_test, 180))


def large_leak(root: Path) -> Path:
    large_file(root, missing=0)
    rng = np.random.default_rng(4)
    for name in ("train", "test"):
        frame = pd.read_csv(root / "data" / f"{name}.csv")
        frame["post_visit_score"] = (frame["y"] + rng.normal(0, 0.05, len(frame))).round(4)
        _write(root / "data" / f"{name}.csv", frame)
    return root


# ------------------------------------------------------------------ ticket 05: model files
# Pickles are written as raw opcodes where the content matters, so the test suite never imports what they name.
def _binunicode(text: str) -> bytes:
    raw = text.encode()
    return b"X" + len(raw).to_bytes(4, "little") + raw


def marker_pickle(marker: Path) -> bytes:
    """Loading this pickle calls builtins.open(marker, 'w'): a harmless stand-in for code execution."""
    return b"\x80\x02cbuiltins\nopen\n" + _binunicode(str(marker)) + _binunicode("w") + b"\x86R."


DANGEROUS_PICKLE = b"\x80\x02cos\nsystem\n" + _binunicode("true") + b"\x85R."   # never loaded by any test
STACK_GLOBAL_DANGEROUS = (b"\x80\x04" + b"\x8c\x0asubprocess\x94" + b"\x8c\x05Popen\x94" + b"\x93\x94" +
                          b"\x8c\x04true\x94\x85\x94R\x94.")                    # never loaded by any test
MEMO_STACK_GLOBAL = (b"\x80\x04" + b"\x8c\x02os\x940" + b"\x8c\x06system\x940" + b"h\x00h\x01\x93" +
                     b"\x8c\x04true\x85R.")                                     # operands fetched from the memo
TRUNCATED_PICKLE = b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x07collect"


def torch_zip_bytes() -> bytes:
    """A zip-format torch state dict, assembled by hand (no torch needed): one 4-float tensor in an OrderedDict."""
    import io
    import zipfile
    tensor = (b"ctorch._utils\n_rebuild_tensor_v2\n(" + b"(" + _binunicode("storage") + b"ctorch\nFloatStorage\n" +
              _binunicode("0") + _binunicode("cpu") + b"K\x04tQ" + b"K\x00" + b"K\x04\x85" + b"K\x01\x85" + b"\x89" +
              b"ccollections\nOrderedDict\n)R" + b"tR")
    data_pkl = b"\x80\x02ccollections\nOrderedDict\n)R" + _binunicode("weight") + tensor + b"s."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", data_pkl)
        zf.writestr("archive/data/0", np.arange(4, dtype=np.float32).tobytes())
        zf.writestr("archive/version", "3\n")
    return buf.getvalue()


def model_project(root: Path, files: dict[str, bytes]) -> Path:
    clean_project(root)
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def sklearn_models(folder: Path) -> list[Path]:
    """Ordinary scikit-learn models saved with joblib (plain, compressed, object arrays) and pickle."""
    import pickle

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    folder.mkdir(parents=True, exist_ok=True)
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    enc = OneHotEncoder().fit(np.array([["a"], ["b"], [None]], dtype=object))
    out = []
    for name, obj, kw in (("lr.joblib", lr, {}), ("rf_z3.joblib", rf, {"compress": 3}),
                          ("rf_gz.joblib", rf, {"compress": ("gzip", 3)}), ("rf_xz.joblib", rf, {"compress": ("xz", 3)}),
                          ("enc_objects.joblib", enc, {})):
        joblib.dump(obj, folder / name, **kw)
        out.append(folder / name)
    (folder / "lr.pkl").write_bytes(pickle.dumps(lr))
    out.append(folder / "lr.pkl")
    return out


# ------------------------------------------------------------------ ticket 06: trusted scoring
# Every pickle loaded under trust is written by pickle/joblib from an object this code built; none is hand-crafted.
SLOW_MODULE = "import time\n\nclass Slow:\n    def predict(self, X):\n        time.sleep(60)\n        return [0] * len(X)\n"
GONE_MODULE = "class Gone:\n    def predict(self, X):\n        return [0] * len(X)\n"


def trusted_project(root: Path, claim_offset: float = 0.0) -> Path:
    """A saved scikit-learn model and a claimed test accuracy, but no saved predictions."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    clean_project(root)
    (root / "outputs" / "test_predictions.csv").unlink()
    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test.csv")
    feats = ["x0", "x1", "x2", "x3"]
    model = LogisticRegression().fit(train[feats], train["y"])
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(model, root / "models" / "model.joblib")
    acc = _accuracy(test["y"], model.predict(test[feats]))
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(acc + claim_offset, 3)}})
    return root


def _pickle_project_object(root: Path, module: str, source: str, cls: str, rel: str) -> None:
    """Pickle an instance of a class defined in the project's own module (as a project would)."""
    import importlib
    import pickle
    (root / f"{module}.py").write_text(source)
    sys.path.insert(0, str(root))
    try:
        mod = importlib.import_module(module)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(pickle.dumps(getattr(mod, cls)()))
    finally:
        sys.path.remove(str(root))
        sys.modules.pop(module, None)


def slow_model_project(root: Path) -> Path:
    clean_project(root)
    _pickle_project_object(root, "slowmodel", SLOW_MODULE, "Slow", "models/slow.pkl")
    return root


def missing_class_project(root: Path) -> Path:
    """The model's class lived in a module that is no longer there: loading it fails."""
    clean_project(root)
    _pickle_project_object(root, "gonemodel", GONE_MODULE, "Gone", "models/gone.pkl")
    (root / "gonemodel.py").unlink()
    return root


# ------------------------------------------------------------------ ticket 07: weak spots
def _rewrite_predictions(root: Path, preds: pd.DataFrame) -> None:
    _write(root / "outputs" / "test_predictions.csv", preds)
    _json(root / "outputs" / "metrics.json", {"test": {"accuracy": round(_accuracy(preds.y_true, preds.y_pred), 3)}})


def weak_subgroup(root: Path) -> Path:
    """The model guesses at random for one region; the claim is honest about the overall accuracy."""
    clean_project(root)
    test = pd.read_csv(root / "data" / "test.csv")
    preds = pd.read_csv(root / "outputs" / "test_predictions.csv")
    west = (test["region"] == "west").to_numpy()
    preds.loc[west, "y_pred"] = np.random.default_rng(3).integers(0, 2, int(west.sum()))
    _rewrite_predictions(root, preds)
    return root


def small_weak_group(root: Path, size: int) -> Path:
    """A 'site' level of ``size`` test rows on which every prediction is wrong."""
    clean_project(root)
    test = pd.read_csv(root / "data" / "test.csv")
    train = pd.read_csv(root / "data" / "train.csv")
    site = np.where(np.arange(len(test)) < size, "s9", np.where(np.arange(len(test)) % 2, "s1", "s2"))
    test.insert(6, "site", site)
    train.insert(6, "site", np.where(np.arange(len(train)) % 2, "s1", "s2"))
    _write(root / "data" / "test.csv", test)
    _write(root / "data" / "train.csv", train)
    preds = pd.read_csv(root / "outputs" / "test_predictions.csv")
    preds.loc[: size - 1, "y_pred"] = 1 - preds.loc[: size - 1, "y_true"]
    _rewrite_predictions(root, preds)
    return root


def miscalibrated(root: Path) -> Path:
    """Over-confident probabilities: same ranking and labels, pushed towards 0 and 1."""
    clean_project(root)
    preds = pd.read_csv(root / "outputs" / "test_predictions.csv")
    p = preds["y_prob"].clip(1e-6, 1 - 1e-6)
    preds["y_prob"] = (1 / (1 + np.exp(-4 * np.log(p / (1 - p))))).round(6)
    _rewrite_predictions(root, preds)
    return root


def constant_probability(root: Path, offset: float) -> Path:
    """Exactly half the test rows are positive and every probability is 0.5 + offset: calibration error = offset."""
    clean_project(root)
    preds = pd.read_csv(root / "outputs" / "test_predictions.csv")
    n = len(preds)
    preds["y_true"] = np.arange(n) % 2
    test = pd.read_csv(root / "data" / "test.csv")
    test["y"] = preds["y_true"].to_numpy()
    _write(root / "data" / "test.csv", test)
    preds["y_prob"] = 0.5 + offset
    preds["y_pred"] = np.where(np.arange(n) % 4 == 0, 1 - preds["y_true"], preds["y_true"])
    _rewrite_predictions(root, preds)
    return root


def regression_project(root: Path, *, n_train: int = 1000, n_test: int = 500, seed: int = 21) -> Path:
    """A clean regression project: honest predictions and a correct RMSE claim."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    def frame(n, start):
        x = rng.normal(size=(n, 3))
        y = 2.0 * x[:, 0] - x[:, 1] + rng.normal(0, 1.0, n)
        return pd.DataFrame({"row_id": np.arange(start, start + n), "x0": x[:, 0].round(6), "x1": x[:, 1].round(6),
                             "x2": x[:, 2].round(6), "group": rng.choice(["g1", "g2", "g3"], n), "target": y.round(6)})
    train, test = frame(n_train, 0), frame(n_test, n_train)
    _write(root / "data" / "train.csv", train)
    _write(root / "data" / "test.csv", test)
    pred = (2.0 * test.x0 - test.x1 + rng.normal(0, 0.3, n_test)).round(6)
    preds = pd.DataFrame({"row_id": test.row_id, "y_true": test.target, "y_pred": pred})
    _write(root / "outputs" / "test_predictions.csv", preds)
    rmse = float(np.sqrt(np.mean((preds.y_true - preds.y_pred) ** 2)))
    _json(root / "outputs" / "metrics.json", {"test": {"rmse": round(rmse, 3)}})
    return root
