"""Audit tool for the ml-audit skill: inspect an existing ML project folder and report whether its results can be trusted.

    python3 <plugin>/skills/ml-audit/lib/mlpipeline_audit.py <project folder> [--claim ...] [--review ...]
            [--trust <project name>] [--retrain <project name>] [--check <family>] [--sample-rows N] [--timeout S]

It inventories the folder, runs the checks it has evidence for, and writes one new dated subfolder
``<project>/ml_audit/<YYYY-MM-DD_HHMMSS>/`` holding ``report.md`` (for people), ``findings.json`` (for machines) and
``inventory.json`` (what it found). It writes nothing else, anywhere: no ledger lines, no guard state, no approvals.
The verdict is one of: trustworthy, not trustworthy, not enough evidence. There is no numeric score.

Arguments are data only - a folder, claim and review records, project names, check-family names, and numbers.
The tool takes no code, expressions, or import paths. It never fits a model, except in the retrain check, which runs
only with the user's recorded approval and only on copies inside the audit folder. Saved models are never loaded
unless the user trusts the project (``--trust``, backed by the user's own chat message), and then only in a child
process. Requires nothing beyond the standard library for the inventory, model scan and log checks; the data and score
checks use pandas and numpy through the guard library, like the rest of the plugin.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True  # never leave __pycache__ behind, in the plugin or anywhere else

TOOL = Path(__file__).resolve()
GUARD_LIB = TOOL.parents[2] / "ml-pipeline" / "lib"   # skills/ml-pipeline/lib, same layout in Codex installs
AUDIT_DIR = "ml_audit"
INVENTORY_FILE = "inventory.json"
AUDIT_APPROVALS_FILE = ".audit_approvals.json"
VERSION = 1

FAMILIES = ("inventory", "honest-scores", "data-problems", "leakage", "model-safety", "weak-spots", "test-misuse",
            "retrain")
SEVERITIES = ("critical", "high", "medium", "info")
# proven: the tool computed it. suspected: the tool saw a sign, not a proof. review: the agent's reading of code and prose.
# could-not-check: a check that was attempted on a specific item and failed (crash, timeout, missing input).
EVIDENCE = ("proven", "suspected", "review", "could-not-check")
VERDICTS = ("trustworthy", "not trustworthy", "not enough evidence")

# ------------------------------------------------------------------ inventory
SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "env", ".env", ".tox", ".nox", "node_modules", "__pycache__",
             ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "site-packages", "dist-packages",
             ".conda", ".idea", ".vscode", ".cache", ".eggs", "build", "dist", AUDIT_DIR}
ENV_MARKERS = ("pyvenv.cfg", "conda-meta")
# The guard's approval and state files, and the audit's own approvals: never opened, never listed.
NEVER_OPEN = {".approvals.json", ".guard_state.json", AUDIT_APPROVALS_FILE}
DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".jsonl", ".ndjson", ".csv.gz", ".tsv.gz")
MODEL_SUFFIXES = (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".h5", ".hdf5",
                  ".keras", ".cbm", ".ubj")
CODE_SUFFIXES = (".py", ".ipynb", ".r")
DOC_SUFFIXES = (".md", ".markdown", ".rst", ".txt")
CLAIM_NAME = re.compile(r"metric|result|score|eval|perf|report", re.I)
RUN_LOG_NAMES = {"runs.jsonl"}
LEDGER_NAME = "PIPELINE.md"
ROLE_WORDS = {"train": "train", "training": "train", "trn": "train",
              "val": "val", "valid": "val", "validation": "val", "dev": "val",
              "test": "test", "testing": "test", "holdout": "test", "tst": "test"}
PREDICTION_WORDS = {"pred", "preds", "prediction", "predictions", "predicted", "oof", "submission", "yhat"}
TARGET_NAMES = ("target", "label", "labels", "y", "class", "outcome")
Y_TRUE_NAMES = ("y_true", "true", "actual", "label", "target", "y")
Y_PRED_NAMES = ("y_pred", "pred", "prediction", "predicted", "yhat", "y_hat", "preds")
Y_PROB_NAMES = ("y_prob", "prob", "proba", "probability", "y_score", "pred_proba", "p1", "prob_1", "score")
DATE_NAME = re.compile(r"(date|time|timestamp|datetime|epoch|_at$|_ts$|^ts$|_dt$|^dt$)", re.I)
ENTITY_NAME = re.compile(r"(^|_)(id|key|uuid|guid|patient|user|customer|client|account|device|session|subject|"
                         r"member|site|store|hospital|household)(_|$)", re.I)


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _suffix(name: str) -> str:
    low = name.lower()
    for multi in (".csv.gz", ".tsv.gz"):
        if low.endswith(multi):
            return multi
    return Path(low).suffix


def _is_env_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in ENV_MARKERS)


def walk(root: Path) -> tuple[list[Path], list[str]]:
    """Every file under ``root`` the audit may open, and the folders it skipped (relative names)."""
    files, skipped = [], set()
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIRS or _is_env_dir(here / d) or (here / d).is_symlink():
                skipped.add(str((here / d).relative_to(root)))
            else:
                keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            if name in NEVER_OPEN:
                continue
            path = here / name
            if path.is_file():
                files.append(path)
    return files, sorted(skipped)


def header(path: Path) -> list[str] | None:
    """Column names of a data file (first line or schema), or None when they can't be read cheaply."""
    suffix = _suffix(path.name)
    try:
        if suffix in (".csv", ".tsv", ".csv.gz", ".tsv.gz"):
            opener = gzip.open if suffix.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
                row = next(csv.reader(fh, delimiter="\t" if "tsv" in suffix else ","), None)
            return [c.strip() for c in row] if row else None
        if suffix in (".jsonl", ".ndjson"):
            with open(path, encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
            obj = json.loads(first) if first.strip() else None
            return list(obj) if isinstance(obj, dict) else None
        if suffix in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            return list(pq.read_schema(str(path)).names)
        if suffix in (".feather", ".arrow"):
            import pyarrow.feather as feather
            return list(feather.read_table(str(path), memory_map=True).schema.names)
    except Exception:  # noqa: BLE001 - an unreadable header is reported as such, not fatal
        return None
    return None


def _pick(columns: list[str], names: tuple[str, ...], exclude=()) -> str | None:
    lowered = {c.lower(): c for c in columns if c not in exclude}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def classify(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).as_posix()
    entry: dict = {"path": rel, "size": path.stat().st_size}
    suffix = _suffix(path.name)
    words = _tokens(Path(path.name[: len(path.name) - len(suffix)] if suffix else path.name).name)
    role = next((ROLE_WORDS[w] for w in sorted(words) if w in ROLE_WORDS), None)
    if path.name in RUN_LOG_NAMES:
        entry.update(kind="claims", source="run log")
    elif path.name == LEDGER_NAME and path.parent.name == "ml_pipeline":
        entry.update(kind="claims", source="ledger")
    elif suffix in DATA_SUFFIXES:
        cols = header(path)
        entry["columns"] = cols
        is_pred = bool(words & PREDICTION_WORDS) or bool(cols and _pick(cols, Y_PRED_NAMES[:1] + ("yhat", "y_hat")))
        if is_pred:
            entry.update(kind="predictions", split=role)
            if cols:
                y_pred = _pick(cols, Y_PRED_NAMES)
                y_prob = _pick(cols, Y_PROB_NAMES, exclude=(y_pred,))
                entry.update(y_true=_pick(cols, Y_TRUE_NAMES, exclude=(y_pred, y_prob)), y_pred=y_pred, y_prob=y_prob)
        else:
            entry.update(kind="data", role=role or "all")
    elif suffix in MODEL_SUFFIXES:
        entry["kind"] = "model"
    elif suffix in CODE_SUFFIXES:
        entry["kind"] = "code"
    elif suffix == ".json" and CLAIM_NAME.search(path.stem):
        entry.update(kind="claims", source="metrics file")
    elif suffix in DOC_SUFFIXES:
        entry["kind"] = "docs"
    else:
        entry["kind"] = "other"
    return entry


def guess_columns(files: list[dict]) -> dict:
    """Target, time and entity columns guessed from headers (columns shared by the split files)."""
    data = [f for f in files if f["kind"] == "data" and f.get("columns")]
    splits = [f for f in data if f.get("role") in ("train", "val", "test")] or data
    if not splits:
        return {"target": None, "time_column": None, "entity_column": None}
    common = [c for c in splits[0]["columns"] if all(c in f["columns"] for f in splits[1:])]
    target = _pick(common, TARGET_NAMES)
    if target is None:  # the predictions' truth column may share a name with a data column
        for pred in (f for f in files if f["kind"] == "predictions" and f.get("y_true")):
            if pred["y_true"] in common:
                target = pred["y_true"]
                break
    time_col = next((c for c in common if c != target and DATE_NAME.search(c)), None)
    entity = next((c for c in common if c not in (target, time_col) and ENTITY_NAME.search(c)), None)
    return {"target": target, "time_column": time_col, "entity_column": entity}


INVENTORY_KEYS = ("target", "task", "positive_label", "time_column", "entity_column", "group_columns", "allowed_labels")


def build_inventory(root: Path) -> dict:
    paths, skipped = walk(root)
    files = [classify(p, root) for p in paths]
    inv = {"version": VERSION, "project": root.name, "corrected_by_user": False, "files": files,
           "skipped_folders": skipped, "not_opened": sorted(NEVER_OPEN),
           "task": None, "positive_label": None, "group_columns": None, "allowed_labels": None}
    inv.update(guess_columns(files))
    user = root / AUDIT_DIR / INVENTORY_FILE
    if user.is_file():
        inv = apply_correction(inv, json.loads(user.read_text(encoding="utf-8")), root)
    return inv


def apply_correction(inv: dict, user: dict, root: Path) -> dict:
    """The user's ``ml_audit/inventory.json`` overrides what was guessed: top-level keys, and file entries by path."""
    inv = dict(inv, corrected_by_user=True, correction_notes=[])
    for key in INVENTORY_KEYS:
        if key in user:
            inv[key] = user[key]
    by_path = {f["path"]: f for f in inv["files"]}
    for entry in user.get("files") or []:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        current = by_path.get(entry["path"])
        if current is None:
            if (root / entry["path"]).is_file() and Path(entry["path"]).name not in NEVER_OPEN:
                current = classify(root / entry["path"], root)
                inv["files"].append(current)
                by_path[entry["path"]] = current
            else:
                inv["correction_notes"].append(f"{entry['path']}: listed in the correction but not found")
                continue
        current.update({k: v for k, v in entry.items() if k not in ("path", "size", "columns")})
    if user.get("exclude"):
        excluded = set(user["exclude"])
        inv["files"] = [f for f in inv["files"] if f["path"] not in excluded]
    return inv


# ------------------------------------------------------------------ the audit record
class Audit:
    def __init__(self, root: Path, args: argparse.Namespace, inventory: dict):
        self.root, self.args, self.inv = root, args, inventory
        self.checks: dict[str, dict] = {}
        self.findings: list[dict] = []
        self.notes: list[str] = []
        self.viewed_test_labels = False

    # -- checks
    def check(self, check_id: str, family: str, title: str, *, required: bool = False) -> dict:
        entry = self.checks.setdefault(check_id, {"id": check_id, "family": family, "title": title, "required": required,
                                                  "status": "not run", "reason": "", "sampled": None})
        return entry

    def ran(self, check_id: str, note: str = "", sampled: int | None = None) -> None:
        entry = self.checks[check_id]
        entry["status"], entry["reason"] = "ran", note
        if sampled is not None:
            entry["sampled"] = sampled

    def could_not(self, check_id: str, reason: str) -> None:
        entry = self.checks[check_id]
        if entry["status"] != "ran":
            entry["status"], entry["reason"] = "could not check", reason

    # -- findings
    def add(self, code: str, family: str, severity: str, evidence: str, title: str, detail: str, *,
            counts: dict | None = None, files=(), columns=(), sampled: int | None = None, source: str | None = None) -> dict:
        assert severity in SEVERITIES and evidence in EVIDENCE and family in FAMILIES + ("review",)
        n = sum(1 for f in self.findings if f["code"] == code) + 1
        finding = {"id": f"{family}/{code}/{n}", "code": code, "family": family, "severity": severity,
                   "evidence": evidence, "title": title, "detail": detail, "counts": counts or {},
                   "files": sorted(set(files)), "columns": sorted(set(map(str, columns))),
                   "sampled_rows": sampled, "source": source,
                   "reproduce": self.reproduce(family) if family in FAMILIES else None}
        self.findings.append(finding)
        return finding

    def reproduce(self, family: str) -> str:
        """The exact command that reruns this check family (the guard allows it: one plain command)."""
        a = self.args
        argv = [os.path.realpath(sys.executable), str(TOOL), str(self.root)]
        for claim in a.claim or []:
            argv += ["--claim", claim]
        if a.trust:
            argv += ["--trust", a.trust]
        if a.retrain and family == "retrain":
            argv += ["--retrain", a.retrain]
        if a.sample_rows:
            argv += ["--sample-rows", str(a.sample_rows)]
        if a.timeout:
            argv += ["--timeout", str(a.timeout)]
        argv += ["--check", family]
        return " ".join(shlex.quote(x) for x in argv)

    # -- files by kind
    def files(self, kind: str, role: str | None = None) -> list[dict]:
        return [f for f in self.inv["files"] if f["kind"] == kind and (role is None or f.get("role") == role)]

    def wants(self, family: str) -> bool:
        return not self.args.check or family in self.args.check


REQUIRED = {"score-recomputation": ("honest-scores", "Claimed scores recomputed from saved predictions"),
            "train-test-overlap": ("leakage", "Identical rows and entities across splits"),
            "feature-target-leakage": ("leakage", "Features that give away the target")}


def verdict(audit: Audit) -> tuple[str, str]:
    """The verdict rule: one proven critical finding decides; otherwise every required check must have run cleanly."""
    proven = [f for f in audit.findings if f["evidence"] == "proven"]
    if any(f["severity"] == "critical" for f in proven):
        worst = next(f for f in proven if f["severity"] == "critical")
        return "not trustworthy", f"proven critical finding: {worst['title']}"
    missing = [c["id"] for c in audit.checks.values() if c["required"] and c["status"] != "ran"]
    if missing:
        return "not enough evidence", "required checks could not run: " + ", ".join(missing)
    open_critical = [f for f in audit.findings if f["evidence"] in ("suspected", "review") and f["severity"] == "critical"]
    if open_critical:
        return "not enough evidence", f"unresolved {open_critical[0]['evidence']} critical finding: {open_critical[0]['title']}"
    high = [f for f in proven if f["severity"] == "high"]
    if high:
        return "not enough evidence", f"proven high finding: {high[0]['title']}"
    return "trustworthy", "every required check ran and found no proven critical or high finding"


# ------------------------------------------------------------------ checks (filled in by later sections)
def run_checks(audit: Audit) -> None:
    for check_id, (family, title) in REQUIRED.items():
        audit.check(check_id, family, title, required=True)
    for fn in CHECK_FUNCTIONS:
        fn(audit)
    for check_id in REQUIRED:
        if audit.checks[check_id]["status"] == "not run":
            family = audit.checks[check_id]["family"]
            audit.could_not(check_id, "not requested (--check)" if not audit.wants(family) else "no check available")


CHECK_FUNCTIONS: list = []


def check_function(fn):
    CHECK_FUNCTIONS.append(fn)
    return fn


# ------------------------------------------------------------------ data access (pandas through the guard library)
_GUARD = None


class NeedsPandas(RuntimeError):
    pass


def guard():
    """The plugin's guard library (streaming readers, row hashes, column kinds). Needs pandas and numpy."""
    global _GUARD
    if _GUARD is None:
        if str(GUARD_LIB) not in sys.path:
            sys.path.insert(0, str(GUARD_LIB))
        try:
            import mlpipeline_guard
        except ImportError as exc:
            raise NeedsPandas(f"this check needs pandas and numpy ({exc}); run the tool with the project's interpreter")
        _GUARD = mlpipeline_guard
    return _GUARD


def read_columns(audit: "Audit", rel: str, columns=None):
    """A data file's columns as one DataFrame, read in batches (only the named columns are kept)."""
    g = guard()
    import pandas as pd
    parts = list(g.scan(str(audit.root / rel), columns=list(columns) if columns else None))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns or []))


# ------------------------------------------------------------------ metrics (no scikit-learn)
SCORE_FLOAT_SLACK = 1e-9      # added to the rounding tolerance so float arithmetic never decides a mismatch
BASELINE_MARGIN = 0.01        # accuracy/F1/AUC: a model must beat the trivial baseline by at least this much
BASELINE_REL_MARGIN = 0.01    # RMSE/MAE: a model's error must be at least 1% below the trivial baseline's
MAX_CLASSES = 20              # an integer target with more distinct values than this is treated as regression
METRIC_ALIASES = {
    "accuracy": "accuracy", "acc": "accuracy", "accuracy_score": "accuracy", "test_accuracy": "accuracy",
    "f1": "f1", "f1_score": "f1", "f1score": "f1", "test_f1": "f1",
    "auc": "auc", "roc_auc": "auc", "roc_auc_score": "auc", "auroc": "auc", "rocauc": "auc", "test_auc": "auc",
    "rmse": "rmse", "root_mean_squared_error": "rmse", "test_rmse": "rmse",
    "mae": "mae", "mean_absolute_error": "mae", "test_mae": "mae",
}
BOUNDED = {"accuracy", "f1", "auc"}
METRIC_LABEL = {"accuracy": "accuracy", "f1": "F1", "auc": "AUC", "rmse": "RMSE", "mae": "MAE"}


def _labels(a):
    import numpy as np
    a = np.asarray(a)
    if a.dtype.kind in "biuf":
        return a.astype(float)
    return a.astype(str)


def positive_label(y, configured=None):
    """The class counted as positive for F1/AUC: the inventory's choice, else 1 / True / 'yes', else the minority class."""
    import numpy as np
    y = _labels(y)
    values = list(np.unique(y))
    if configured is not None:
        return float(configured) if y.dtype.kind == "f" else str(configured)
    for cand in (1.0, "1", "True", "true", "yes", "Yes", "positive", "pos"):
        if cand in values:
            return cand
    counts = {v: int((y == v).sum()) for v in values}
    return min(counts, key=lambda v: (counts[v], str(v))) if counts else None


def _f1(y, p, positive) -> float:
    tp = float(((y == positive) & (p == positive)).sum())
    fp = float(((y != positive) & (p == positive)).sum())
    fn = float(((y == positive) & (p != positive)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def _average_ranks(s):
    import numpy as np
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2.0)[inverse]


def metric_value(metric: str, y, p, *, scores=None, positive=None) -> float | None:
    """accuracy, f1 (binary on ``positive``, macro over classes otherwise), auc (binary), rmse, mae; None if undefined."""
    import numpy as np
    if metric in ("rmse", "mae"):
        y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
        err = y - p
        return float(np.sqrt(np.mean(err ** 2))) if metric == "rmse" else float(np.mean(np.abs(err)))
    y = _labels(y)
    if metric == "auc":
        if scores is None:
            return None
        s = np.asarray(scores, dtype=float)
        classes = np.unique(y)
        if len(classes) != 2:
            return None
        positive = positive if positive is not None else positive_label(y)
        pos = y == positive
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if not n_pos or not n_neg:
            return None
        ranks = _average_ranks(s)
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    p = _labels(p)
    if y.dtype != p.dtype:
        y, p = y.astype(str), p.astype(str)
    if metric == "accuracy":
        return float(np.mean(y == p))
    if metric == "f1":
        classes = np.unique(np.concatenate([y, p]))
        if len(classes) <= 2:
            positive = positive if positive is not None else positive_label(y)
            return _f1(y, p, type(y[0])(positive) if len(y) else positive)
        return float(np.mean([_f1(y, p, c) for c in classes]))
    return None


def claim_tolerance(text: str) -> float:
    """Half a unit in the last stated digit: '0.93' covers 0.925..0.935, '0.930' covers 0.9295..0.9305."""
    text = text.strip().rstrip("%")
    mantissa = text.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(text.lower().split("e")[1]) if "e" in text.lower() else 0
    return 0.5 * 10.0 ** (exponent - decimals) + SCORE_FLOAT_SLACK


def infer_task(y, configured=None) -> str:
    import numpy as np
    if configured in ("classification", "regression"):
        return configured
    y = np.asarray(y)
    if y.dtype.kind not in "biuf":
        return "classification"
    values = y[~np.isnan(y.astype(float))] if y.dtype.kind == "f" else y
    if not len(values):
        return "classification"
    whole = bool(np.all(np.mod(values.astype(float), 1) == 0))
    return "classification" if whole and len(np.unique(values)) <= MAX_CLASSES else "regression"


# ------------------------------------------------------------------ claims
class _Num(float):
    """A JSON number that remembers how it was written, so its stated precision is known."""
    text = ""


def _num(text: str) -> _Num:
    n = _Num(float(text))
    n.text = text
    return n


def _metric_key(key: str) -> str | None:
    return METRIC_ALIASES.get(re.sub(r"[^a-z0-9_]", "", str(key).lower().replace("-", "_").replace(" ", "_")))


def _split_from(words: set[str]) -> str | None:
    for word in sorted(words):
        if word in ROLE_WORDS:
            return ROLE_WORDS[word]
    return None


def _claim(metric, number, split, source, origin) -> dict | None:
    text = number.text if isinstance(number, _Num) and number.text else repr(float(number))
    value, tol = float(number), claim_tolerance(text)
    percent = metric in BOUNDED and 1.0 < value <= 100.0
    if percent:
        value, tol = value / 100.0, tol / 100.0
    return {"metric": metric, "value": value, "text": text + ("%" if percent else ""), "tolerance": tol,
            "split": split, "source": source, "origin": origin}


def _json_claims(obj, path: list[str], file_words: set[str], rel: str, out: list[dict]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _json_claims(value, path + [str(key)], file_words, rel, out)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _json_claims(value, path + [str(i)], file_words, rel, out)
    elif isinstance(obj, float) and path:
        metric = _metric_key(path[-1])
        if metric:
            words = set().union(*(_tokens(p) for p in path)) | file_words
            out.append(_claim(metric, obj, _split_from(words), f"{rel}: {'.'.join(path)}", "metrics file"))


FINAL_TEST = re.compile(r"^\s*-\s*Step 14 final test:\s*([\w.]+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I | re.M)


def collect_claims(audit: "Audit") -> tuple[list[dict], list[str]]:
    """Every claimed score: metrics files, the run log, the ledger's final-test lines, and claims passed in."""
    claims, ignored = [], []
    for f in audit.files("claims"):
        path = audit.root / f["path"]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f.get("source") == "metrics file":
            try:
                obj = json.loads(text, parse_float=_num, parse_int=_num)
            except ValueError:
                ignored.append(f"{f['path']}: not valid JSON")
                continue
            _json_claims(obj, [], _tokens(Path(f["path"]).stem), f["path"], claims)
        elif f.get("source") == "run log":
            for n, line in enumerate(text.splitlines(), 1):
                try:
                    run = json.loads(line, parse_float=_num, parse_int=_num)
                except ValueError:
                    continue
                for key, value in (run.get("metrics") or {}).items() if isinstance(run, dict) else []:
                    metric = _metric_key(key)
                    if metric and isinstance(value, float):
                        split = "test" if "test" in _tokens(key) else "val"  # log_run records validation metrics
                        claims.append(_claim(metric, value, split, f"{f['path']}:{n} ({run.get('name', 'run')})", "run log"))
        elif f.get("source") == "ledger":
            for m in FINAL_TEST.finditer(text):
                metric = _metric_key(m.group(1))
                line = text.count("\n", 0, m.start()) + 1
                if metric:
                    claims.append(_claim(metric, _num(m.group(2)), "test", f"{f['path']}:{line}", "ledger"))
                else:
                    ignored.append(f"{f['path']}:{line}: metric '{m.group(1)}' is not one the tool recomputes")
    for raw in audit.args.claim or []:
        fields = dict(part.split("=", 1) for part in raw.split(",") if "=" in part)
        metric = _metric_key(fields.get("metric", ""))
        value = fields.get("value", "").strip()
        if not metric or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?", value):
            ignored.append(f"--claim '{raw}': needs metric= (accuracy, f1, auc, rmse, mae) and a numeric value=")
            continue
        split = fields.get("split", "").strip().lower()
        split = ROLE_WORDS.get(split, split or None)
        source = (fields.get("source") or "stated in prose").strip() + " (claim passed in)"
        claims.append(_claim(metric, _num(value.rstrip("%")) if not value.endswith("%") else _num(value[:-1]),
                             split, source, "prose"))
    return claims, ignored


# ------------------------------------------------------------------ check family: honest scores
def _prediction_file_for(claim: dict, preds: list[dict]) -> dict | None:
    same = [p for p in preds if p.get("split") == claim["split"]]
    if same:
        return same[0]
    if len(preds) == 1 and (claim["split"] is None or preds[0].get("split") is None):
        return preds[0]
    return None


def load_predictions(audit: "Audit", pf: dict):
    """(y_true, y_pred, y_prob, n) arrays from a predictions file (any may be None)."""
    cols = [c for c in (pf.get("y_true"), pf.get("y_pred"), pf.get("y_prob")) if c]
    frame = read_columns(audit, pf["path"], cols)
    get = lambda c: frame[c].to_numpy() if c else None  # noqa: E731
    return get(pf.get("y_true")), get(pf.get("y_pred")), get(pf.get("y_prob")), len(frame)


def _better(metric: str, a: float, b: float) -> bool:
    return a > b if metric in BOUNDED else a < b


def _split_target(audit: "Audit", role: str):
    target = audit.inv.get("target")
    files = [f for f in audit.files("data", role) if target and f.get("columns") and target in f["columns"]]
    if not files:
        return None
    import numpy as np
    return np.concatenate([read_columns(audit, f["path"], [target])[target].to_numpy() for f in files])


def baseline_score(metric: str, y, train_y, positive) -> tuple[float | None, str]:
    """The score of a predictor that ignores the features (taken from the training labels when available)."""
    import numpy as np
    ref = train_y if train_y is not None and len(train_y) else y
    where = "the training labels" if ref is train_y else "these labels"
    if metric == "accuracy":
        values, counts = np.unique(_labels(ref), return_counts=True)
        majority = values[np.argmax(counts)]
        return metric_value("accuracy", y, np.full(len(y), majority)), f"always predicting the majority class of {where}"
    if metric == "f1":
        y_l = _labels(y)
        if len(np.unique(y_l)) > 2:
            return None, ""
        return metric_value("f1", y_l, np.full(len(y_l), positive), positive=positive), "predicting the positive class for every row"
    if metric == "auc":
        return 0.5, "a constant score (AUC 0.5)"
    centre = float(np.mean(np.asarray(ref, dtype=float))) if metric == "rmse" else float(np.median(np.asarray(ref, dtype=float)))
    kind = "mean" if metric == "rmse" else "median"
    return metric_value(metric, y, np.full(len(y), centre)), f"always predicting the {kind} of {where}"


def _id_columns(pf: dict, data: dict, target: str | None) -> list[str]:
    skip = {pf.get("y_true"), pf.get("y_pred"), pf.get("y_prob"), target}
    common = [c for c in (pf.get("columns") or []) if c in (data.get("columns") or []) and c not in skip]
    return [c for c in common if ENTITY_NAME.search(c) or c.lower() in ("id", "index", "idx", "row", "row_id")]


def _hashes(frame) -> "object":
    import numpy as np
    g = guard()
    return np.unique(g._row_hashes(frame.reset_index(drop=True)))


def check_prediction_rows(audit: "Audit", pf: dict) -> None:
    """A prediction file labelled test must be made on the test file's rows."""
    import numpy as np
    target = audit.inv.get("target")
    tests = audit.files("data", "test")
    if not tests:
        return
    others = {role: audit.files("data", role) for role in ("val", "train")}
    test = tests[0]
    ids = _id_columns(pf, test, target)
    if ids:
        pred_ids = _hashes(read_columns(audit, pf["path"], ids))
        test_ids = _hashes(read_columns(audit, test["path"], ids))
        in_test = int(np.isin(pred_ids, test_ids).sum())
        if in_test == len(pred_ids):
            return
        where = {}
        for role, files in others.items():
            for f in files:
                if all(c in (f.get("columns") or []) for c in ids):
                    where[role] = where.get(role, 0) + int(np.isin(pred_ids, _hashes(read_columns(audit, f["path"], ids))).sum())
        audit.add("claim-wrong-split", "honest-scores", "critical", "proven",
                  "Predictions labelled 'test' were not made on the test rows",
                  f"{len(pred_ids) - in_test} of {len(pred_ids)} distinct prediction rows (matched on "
                  f"{', '.join(ids)}) are not in {test['path']}" +
                  "".join(f"; {n} are in the {role} split" for role, n in where.items() if n) +
                  ". A score computed on them is not a test score.",
                  counts={"prediction_rows": len(pred_ids), "in_test": in_test, **{f"in_{r}": n for r, n in where.items()}},
                  files=[pf["path"], test["path"]], columns=ids)
        return
    if not (target and pf.get("y_true")):
        return
    truth = load_predictions(audit, pf)[0]
    test_y = _split_target(audit, "test")
    if test_y is not None and len(test_y) == len(truth) and np.array_equal(_labels(test_y), _labels(truth)):
        return
    for role in ("val", "train"):
        other = _split_target(audit, role)
        if other is not None and len(other) == len(truth) and np.array_equal(_labels(other), _labels(truth)):
            audit.add("claim-wrong-split", "honest-scores", "critical", "proven",
                      "Predictions labelled 'test' line up with the wrong split",
                      f"the {len(truth)} true labels in {pf['path']} match the {role} split row for row, not the test split.",
                      counts={"prediction_rows": len(truth)}, files=[pf["path"], test["path"]], columns=[target])
            return


@check_function
def check_honest_scores(audit: "Audit") -> None:
    if not audit.wants("honest-scores"):
        return
    audit.check("baseline-comparison", "honest-scores", "Scores compared with a trivial baseline")
    claims, ignored = collect_claims(audit)
    audit.notes += [f"claim ignored: {why}" for why in ignored]
    preds = [p for p in audit.files("predictions") if p.get("y_true") and (p.get("y_pred") or p.get("y_prob"))]
    if not claims:
        audit.could_not("score-recomputation", "no claimed scores found (metrics files, run log, ledger; prose claims "
                                               "are passed in with --claim)")
    try:
        loaded = {p["path"]: load_predictions(audit, p) for p in preds}
    except NeedsPandas as exc:
        audit.could_not("score-recomputation", str(exc))
        audit.could_not("baseline-comparison", str(exc))
        return
    checked, unchecked, used, rows_checked = 0, [], {}, set()
    for claim in claims:
        pf = _prediction_file_for(claim, preds)
        label = f"{claim['split'] or 'unlabelled'} {METRIC_LABEL[claim['metric']]} {claim['text']} ({claim['source']})"
        if pf is None:
            why = "no predictions for this split" + (" (validation metrics are logged without predictions)"
                                                     if claim["split"] == "val" else "")
            unchecked.append((claim, f"{label}: {why}"))
            continue
        y, p, s, n = loaded[pf["path"]]
        task = infer_task(y, audit.inv.get("task"))
        positive = positive_label(y, audit.inv.get("positive_label")) if task == "classification" else None
        if claim["metric"] in BOUNDED and task == "regression" or claim["metric"] in ("rmse", "mae") and task == "classification":
            unchecked.append((claim, f"{label}: metric does not fit a {task} target"))
            continue
        source = p if claim["metric"] != "auc" else None
        value = metric_value(claim["metric"], y, source, scores=s, positive=positive) if (source is not None or s is not None) else None
        if value is None:
            unchecked.append((claim, f"{label}: {pf['path']} lacks the column this metric needs"))
            continue
        checked += 1
        used.setdefault(pf["path"], set()).add(claim["metric"])
        if (claim["split"] == "test" or pf.get("split") == "test") and pf["path"] not in rows_checked:
            rows_checked.add(pf["path"])
            check_prediction_rows(audit, pf)
        diff = claim["value"] - value
        if abs(diff) <= claim["tolerance"]:
            continue
        overstated = _better(claim["metric"], claim["value"], value)
        name = METRIC_LABEL[claim["metric"]]
        audit.add("claim-mismatch", "honest-scores", "critical" if overstated else "medium", "proven",
                  f"Claimed {claim['split'] or ''} {name} is {'better' if overstated else 'worse'} than the predictions give".replace("  ", " "),
                  f"claimed {claim['text']}, recomputed {value:.4f} from {n:,} rows of {pf['path']}; the difference "
                  f"{abs(diff):.4f} is beyond the claim's rounding tolerance ({claim['tolerance']:.4g})." +
                  ("" if overstated else " The claim understates the result; the number is still not reproducible."),
                  counts={"claimed": claim["value"], "recomputed": round(value, 6), "rows": n,
                          "tolerance": round(claim["tolerance"], 9)},
                  files=[pf["path"]] + ([claim["source"].split(":")[0]] if claim["origin"] != "prose" else []),
                  columns=[c for c in (pf.get("y_true"), pf.get("y_pred") if claim["metric"] != "auc" else pf.get("y_prob")) if c],
                  source=claim["source"])
    blocking = [why for claim, why in unchecked if claim["split"] in ("test", None)]
    if checked and not blocking:
        audit.ran("score-recomputation", f"{checked} claim(s) recomputed" +
                  (f"; not checkable: {'; '.join(why for _, why in unchecked)}" if unchecked else ""))
    elif claims:
        audit.could_not("score-recomputation", "; ".join(why for _, why in unchecked) or "no claim could be recomputed")
    # Baseline: every predictions file, on the claimed metrics (or accuracy / RMSE when nothing is claimed).
    for pf in preds:
        y, p, s, n = loaded[pf["path"]]
        task = infer_task(y, audit.inv.get("task"))
        metrics = used.get(pf["path"]) or ({"accuracy"} if task == "classification" else {"rmse"})
        train_y = _split_target(audit, "train")
        positive = positive_label(y, audit.inv.get("positive_label")) if task == "classification" else None
        for metric in sorted(metrics):
            model = metric_value(metric, y, p if metric != "auc" else None, scores=s, positive=positive) \
                if (p is not None or metric == "auc") else None
            base, how = baseline_score(metric, y, train_y, positive)
            if model is None or base is None:
                continue
            audit.ran("baseline-comparison")
            margin = BASELINE_MARGIN if metric in BOUNDED else BASELINE_REL_MARGIN * base
            beats = model > base + margin if metric in BOUNDED else model < base - margin
            if not beats:
                name = METRIC_LABEL[metric]
                audit.add("no-better-than-baseline", "honest-scores", "high", "proven",
                          f"The model's {name} is no better than a trivial baseline",
                          f"{name} {model:.4f} on {n:,} rows of {pf['path']} vs. {base:.4f} for {how}; a useful model "
                          f"must beat it by more than {margin:.4g}.",
                          counts={"model": round(model, 6), "baseline": round(base, 6), "rows": n},
                          files=[pf["path"]], columns=[c for c in (pf.get("y_true"), pf.get("y_pred")) if c])
    if not preds:
        audit.could_not("baseline-comparison", "no predictions file with true and predicted values")


# ------------------------------------------------------------------ output
def new_audit_folder(root: Path) -> Path:
    base = root / AUDIT_DIR
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for n in range(1, 1000):
        out = base / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            out.mkdir()
            return out
        except FileExistsError:
            continue
    raise RuntimeError("could not create a new audit folder")


SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def _sorted_findings(findings: list[dict]) -> list[dict]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f["severity"]], EVIDENCE.index(f["evidence"]), f["id"]))


def report_markdown(audit: Audit, result: str, why: str) -> str:
    inv = audit.inv
    lines = [f"# Audit of {inv['project']}", "", f"**Verdict: {result}** - {why}.", "",
             "The verdict covers only what was checked below. 'Could not check' is not a pass. There is no numeric score.",
             ""]
    lines += ["## What was checked", "", "| Check | Required | Status | Note |", "|---|---|---|---|"]
    for c in sorted(audit.checks.values(), key=lambda c: (not c["required"], FAMILIES.index(c["family"]) if c["family"] in FAMILIES else 99, c["id"])):
        note = c["reason"] + (f" (on a sample of {c['sampled']:,} rows)" if c.get("sampled") else "")
        lines.append(f"| {c['title']} | {'yes' if c['required'] else 'no'} | {c['status']} | {note} |")
    lines += ["", "## Findings", ""]
    if not audit.findings:
        lines.append("No findings.")
    for f in _sorted_findings(audit.findings):
        label = "agent review (not proven by the tool)" if f["evidence"] == "review" else f["evidence"]
        lines += [f"### {f['severity'].upper()} - {f['title']}", "",
                  f"- id: `{f['id']}`; evidence: {label}", f"- {f['detail']}"]
        if f["counts"]:
            lines.append("- counts: " + ", ".join(f"{k} = {v}" for k, v in f["counts"].items()))
        if f["files"]:
            lines.append("- files: " + ", ".join(f"`{p}`" for p in f["files"]))
        if f["columns"]:
            lines.append("- columns: " + ", ".join(f"`{c}`" for c in f["columns"]))
        if f["sampled_rows"]:
            lines.append(f"- computed on a sample of {f['sampled_rows']:,} rows")
        if f["source"]:
            lines.append(f"- source: {f['source']}")
        if f["reproduce"]:
            lines.append(f"- reproduce: `{f['reproduce']}`")
        lines.append("")
    lines += ["## Inventory", ""]
    if inv.get("corrected_by_user"):
        lines.append("The inventory was corrected by the user (`ml_audit/inventory.json`).")
        lines += [f"- {n}" for n in inv.get("correction_notes") or []]
    else:
        lines.append("Guessed from file names and headers. To correct it, copy this audit's `inventory.json` to "
                     "`ml_audit/inventory.json`, edit it, and run the audit again.")
    lines += ["", "| File | Kind | Role |", "|---|---|---|"]
    for f in inv["files"]:
        lines.append(f"| `{f['path']}` | {f['kind']} | {f.get('role') or f.get('split') or f.get('source') or ''} |")
    lines += ["", f"Target column: `{inv.get('target')}`; time column: `{inv.get('time_column')}`; "
                  f"entity column: `{inv.get('entity_column')}`.",
              f"Skipped folders: {', '.join(inv['skipped_folders']) or 'none'}. Never opened: {', '.join(inv['not_opened'])}.", ""]
    lines += ["## Limits", ""] + [f"- {n}" for n in audit.notes + LIMITS]
    return "\n".join(lines) + "\n"


LIMITS = [
    "The report holds column names and counts, never raw row values.",
    "A model-file scan reads pickle instructions without running them. It lowers risk; it does not prove a file is safe.",
    "Proven findings are computed by the tool. Review findings are the agent's reading of code and prose and can never "
    "make the verdict 'not trustworthy' on their own.",
]


def write_outputs(audit: Audit, out: Path) -> tuple[str, str]:
    result, why = verdict(audit)
    findings = {"version": VERSION, "project": audit.inv["project"], "created": datetime.now().isoformat(timespec="seconds"),
                "verdict": result, "verdict_reason": why, "viewed_test_labels": audit.viewed_test_labels,
                "checks": list(audit.checks.values()), "findings": _sorted_findings(audit.findings), "notes": audit.notes}
    (out / "findings.json").write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    (out / INVENTORY_FILE).write_text(json.dumps(audit.inv, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(audit, result, why), encoding="utf-8")
    return result, why


# ------------------------------------------------------------------ command line
NAME = re.compile(r"^[\w.-]{1,200}$")


def _name(value: str) -> str:
    if not NAME.match(value):
        raise argparse.ArgumentTypeError("a project name: letters, digits, '.', '_' or '-'")
    return value


def _positive(value: str) -> int:
    if not re.fullmatch(r"\d{1,9}", value) or int(value) <= 0:
        raise argparse.ArgumentTypeError("a positive whole number")
    return int(value)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlpipeline_audit", description="Audit an ML project folder (read-only).",
                                allow_abbrev=False)
    p.add_argument("folder")
    p.add_argument("--claim", action="append", help="a claimed score from prose: metric=..,value=..,split=..,source=..")
    p.add_argument("--review", action="append", help="an agent review finding: family=..,severity=..,text=..,source=..")
    p.add_argument("--trust", type=_name, help="the project's folder name: load its models (needs the user's approval)")
    p.add_argument("--retrain", type=_name, help="the project's folder name: run the retrain checks (needs approval)")
    p.add_argument("--check", action="append", choices=FAMILIES, help="run only these check families")
    p.add_argument("--sample-rows", type=_positive, help="rows kept for sampled checks (default: the guard's limit)")
    p.add_argument("--timeout", type=_positive, help="seconds allowed for loading and scoring one trusted model")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.folder).resolve()
    if not root.is_dir():
        print(f"ml-audit: {args.folder} is not a folder", file=sys.stderr)
        return 2
    inventory = build_inventory(root)
    audit = Audit(root, args, inventory)
    run_checks(audit)
    out = new_audit_folder(root)
    result, why = write_outputs(audit, out)
    print(f"audit folder: {out}")
    print(f"verdict: {result} - {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
