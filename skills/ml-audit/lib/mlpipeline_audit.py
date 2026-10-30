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
ROW_NUMBER = re.compile(r"^(row_?(id|num|number)?|id|index|idx|record_?id)$", re.I)  # numbers rows, names no entity
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
    entity = next((c for c in common if c not in (target, time_col) and ENTITY_NAME.search(c)
                   and not ROW_NUMBER.match(c)), None)
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
        self.out: Path | None = None
        self.scans: dict[str, dict] = {}

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

    def not_applicable(self, check_id: str, reason: str) -> None:
        self.checks[check_id]["status"], self.checks[check_id]["reason"] = "not applicable", reason

    # -- findings
    def add(self, code: str, family: str, severity: str, evidence: str, title: str, detail: str, *,
            counts: dict | None = None, files=(), columns=(), sampled: int | None = None, source: str | None = None) -> dict:
        assert severity in SEVERITIES and evidence in EVIDENCE and family in FAMILIES + ("review",)
        n = sum(1 for f in self.findings if f["code"] == code) + 1
        finding = {"id": f"{family}/{code}/{n}", "code": code, "family": family, "severity": severity,
                   "evidence": evidence, "title": title, "detail": detail, "counts": counts or {},
                   "files": sorted(set(files)), "columns": sorted(set(map(str, columns))),
                   "sampled_rows": sampled, "source": source,
                   "reproduce": self.reproduce(family) if family in FAMILIES and evidence != "review" else None}
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
    trusted_scoring(audit)  # first: its predictions feed the score checks
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


# ------------------------------------------------------------------ records passed in (--claim, --review)
FIELD_SPLIT = re.compile(r",\s*(?=(?:family|severity|text|source|metric|value|split|evidence)\s*=)", re.I)
REVIEW_FAMILIES = ("honest-scores", "data-problems", "leakage", "weak-spots", "test-misuse", "model-safety")


def parse_fields(raw: str) -> dict[str, str]:
    """'key=value,key=value' where a value may itself hold commas (only known keys start a new field)."""
    out = {}
    for part in FIELD_SPLIT.split(raw):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip().lower()] = value.strip()
    return out


@check_function
def add_reviews(audit: "Audit") -> None:
    """The agent's review findings: labelled 'review', never proven, never able to make the verdict on their own."""
    for raw in audit.args.review or []:
        fields = parse_fields(raw)
        family, severity, text = fields.get("family", ""), fields.get("severity", "").lower(), fields.get("text", "")
        if family not in REVIEW_FAMILIES or severity not in SEVERITIES or not text:
            audit.notes.append(f"review ignored (needs family= one of {', '.join(REVIEW_FAMILIES)}, severity= one of "
                               f"{', '.join(SEVERITIES)}, and text=): {raw[:120]}")
            continue
        audit.add("agent-review", family, severity, "review", text if len(text) <= 90 else text[:87] + "...",
                  text, source=fields.get("source") or None)


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
        fields = parse_fields(raw)
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
            audit.ran("baseline-comparison", "claimed metrics (or accuracy / RMSE) against a predictor that ignores the features")
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


# ------------------------------------------------------------------ check family: data problems
MISSING_MEDIUM_FRAC = 0.20        # a column missing in at least this share of rows is a medium finding (info below)
LABEL_CONFLICT_HIGH_FRAC = 0.05   # identical features with different labels in this share of rows is high (medium below)
LABEL_SET_CAP = 1000              # distinct labels remembered per file for the unseen-class comparison
ID_COLUMN = re.compile(r"(^|_)(id|uuid|guid|key|index|idx)$|^id", re.I)


def feature_columns(columns, inv: dict) -> list[str]:
    """Columns that describe a row (not its label, not its identifier)."""
    skip = {inv.get("target"), inv.get("entity_column")}
    return [c for c in columns if c not in skip and not ID_COLUMN.search(str(c))]


def _label_key(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _conflicts(feature_hashes, label_hashes) -> tuple[int, int]:
    """(groups, rows) of identical feature rows that carry more than one label."""
    import numpy as np
    if not len(feature_hashes):
        return 0, 0
    order = np.lexsort((label_hashes, feature_hashes))
    f, y = feature_hashes[order], label_hashes[order]
    new_group = np.concatenate([[True], f[1:] != f[:-1]])
    group = np.cumsum(new_group) - 1
    changed = np.concatenate([[False], (y[1:] != y[:-1]) & ~new_group[1:]])
    bad = np.unique(group[changed])
    sizes = np.bincount(group)
    return int(len(bad)), int(sizes[bad].sum())


@check_function
def check_data_problems(audit: "Audit") -> None:
    if not audit.wants("data-problems"):
        return
    audit.check("data-problems", "data-problems", "Missing values, duplicates, constant columns, label problems")
    files = audit.files("data")
    if not files:
        audit.could_not("data-problems", "no data files in the inventory")
        return
    try:
        g = guard()
    except NeedsPandas as exc:
        audit.could_not("data-problems", str(exc))
        return
    import numpy as np
    target, allowed = audit.inv.get("target"), audit.inv.get("allowed_labels")
    allowed_keys = {_label_key(v) for v in allowed} if allowed else None
    batch = audit.args.sample_rows or g.BATCH_ROWS
    total, batches, label_sets = 0, 0, {}
    for f in files:
        n, missing, distinct, seen = 0, {}, g._Distinct(), {}
        pairs, label_missing, label_inf, outside, labels, exact_pairs = [], 0, 0, 0, set(), True
        try:
            for chunk in g.scan(str(audit.root / f["path"]), batch_rows=batch):
                batches += 1
                n += len(chunk)
                for col, count in chunk.isna().sum().items():
                    if count:
                        missing[col] = missing.get(col, 0) + int(count)
                distinct.add(g._row_hashes(chunk))
                for col in chunk.columns:
                    known = seen.setdefault(col, set())
                    if len(known) < 2:
                        known.update(_label_key(v) for v in chunk[col].dropna().unique()[:2].tolist())
                if target and target in chunk.columns:
                    y = chunk[target]
                    has = y.notna().to_numpy()
                    label_missing += int((~has).sum())
                    if y.dtype.kind == "f":
                        label_inf += int(np.isinf(y.to_numpy()).sum())
                    values = [_label_key(v) for v in y[has].unique()[:LABEL_SET_CAP].tolist()]
                    if allowed_keys is not None:
                        outside += int((~y[has].map(_label_key).isin(allowed_keys)).sum())
                    if len(labels) < LABEL_SET_CAP:
                        labels.update(values)
                    feats = feature_columns(chunk.columns, audit.inv)
                    if exact_pairs and n <= g.EXACT_ROWS and feats:
                        pairs.append((g._row_hashes(chunk.loc[has, feats]),
                                      g._row_hashes(chunk.loc[has, [target]])))
                    else:
                        exact_pairs = False
        except Exception as exc:  # noqa: BLE001 - one unreadable file is reported, the rest still run
            audit.notes.append(f"{f['path']}: could not be read for the data checks ({type(exc).__name__})")
            continue
        total += n
        label_sets[f.get("role")] = labels | label_sets.get(f.get("role"), set())
        gaps = {c: v for c, v in missing.items() if c != target}
        if gaps:
            worst = max(v / n for v in gaps.values())
            audit.add("missing-values", "data-problems", "medium" if worst >= MISSING_MEDIUM_FRAC else "info", "proven",
                      f"Missing values in {len(gaps)} column(s) of {f['path']}",
                      f"{n:,} rows; the most-affected column is missing in {worst:.1%} of rows (medium from "
                      f"{MISSING_MEDIUM_FRAC:.0%}).", counts={c: gaps[c] for c in sorted(gaps)},
                      files=[f["path"]], columns=gaps)
        unique, exact = distinct.count()
        if n - unique > 0:
            audit.add("duplicate-rows", "data-problems", "medium", "proven", f"Duplicate rows in {f['path']}",
                      f"{n - unique:,} of {n:,} rows repeat an earlier row exactly" +
                      ("" if exact else " (estimated: the file is beyond the exact-count limit)") + ".",
                      counts={"duplicate_rows": n - unique, "rows": n}, files=[f["path"]])
        constant = sorted(c for c, values in seen.items() if len(values) <= 1)
        if constant:
            audit.add("constant-column", "data-problems", "info", "proven", f"Constant columns in {f['path']}",
                      f"{len(constant)} column(s) hold one value (or none) in all {n:,} rows; they carry no information.",
                      counts={"constant_columns": len(constant)}, files=[f["path"]], columns=constant)
        if pairs:
            groups, rows = _conflicts(np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]))
            if groups:
                frac = rows / n
                audit.add("conflicting-labels", "data-problems", "high" if frac >= LABEL_CONFLICT_HIGH_FRAC else "medium",
                          "proven", f"Identical features with different labels in {f['path']}",
                          f"{groups:,} groups of rows have the same feature values but different '{target}' labels "
                          f"({rows:,} rows, {frac:.1%}); identifier columns are left out of the comparison.",
                          counts={"conflicting_groups": groups, "conflicting_rows": rows}, files=[f["path"]],
                          columns=[target])
        elif target in seen and not exact_pairs:
            audit.notes.append(f"{f['path']}: label conflicts not checked above {g.EXACT_ROWS:,} rows")
        bad = {k: v for k, v in (("missing_label", label_missing), ("non_finite_label", label_inf),
                                 ("outside_allowed_labels", outside)) if v}
        if bad:
            audit.add("invalid-labels", "data-problems", "high", "proven", f"Invalid labels in {f['path']}",
                      f"{sum(bad.values()):,} of {n:,} rows have a missing, non-finite, or disallowed '{target}' value; "
                      "they can be neither trained on nor scored.", counts=bad, files=[f["path"]], columns=[target])
    train_labels = label_sets.get("train")
    if train_labels and len(train_labels) <= MAX_CLASSES:
        for role in ("val", "test"):
            unseen = label_sets.get(role, set()) - train_labels
            if unseen:
                audit.add("invalid-labels", "data-problems", "medium", "proven",
                          f"Classes in the {role} split never seen in training",
                          f"{len(unseen)} class(es) of '{target}' occur in the {role} split but not in the training "
                          "split; no model trained on it can predict them.", counts={"unseen_in_train": len(unseen)},
                          files=[x["path"] for x in audit.files("data", role)], columns=[target])
    audit.ran("data-problems", f"every row streamed: {total:,} rows in {batches} batches of up to {batch:,}")


# ------------------------------------------------------------------ check family: leakage
OVERLAP_CRITICAL_FRAC = 0.01   # identical rows shared with training in at least 1% of the later split: critical (high below)
LEAK_ASSOCIATION = 0.95        # one feature alone removes this share of the label error (or explains this share of variance)
SPLIT_PAIRS = (("train", "test"), ("train", "val"), ("val", "test"))


def _file_hashes(audit: "Audit", files: list[dict], columns: list[str]):
    """Sorted distinct row hashes over ``columns`` of every file in ``files`` (streamed), and the row count."""
    import numpy as np
    g = guard()
    parts, n = [], 0
    for f in files:
        for chunk in g.scan(str(audit.root / f["path"]), columns=columns, batch_rows=audit.args.sample_rows or None):
            n += len(chunk)
            parts.append(g._row_hashes(chunk[columns]))
    return (g._sorted_unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)), n


def _hashes_per_row(audit: "Audit", files: list[dict], columns: list[str]):
    import numpy as np
    g = guard()
    parts = [g._row_hashes(chunk[columns]) for f in files
             for chunk in g.scan(str(audit.root / f["path"]), columns=columns, batch_rows=audit.args.sample_rows or None)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)


def _common(files_a: list[dict], files_b: list[dict]) -> list[str]:
    cols = [c for c in (files_a[0].get("columns") or [])]
    return [c for c in cols if all(c in (f.get("columns") or []) for f in files_a + files_b)]


def check_overlap(audit: "Audit") -> None:
    import numpy as np
    roles = {r: audit.files("data", r) for r in ("train", "val", "test")}
    if not roles["train"] or not roles["test"]:
        audit.could_not("train-test-overlap", "needs separate training and test data files (correct the inventory "
                                              "if they are there under other names)")
        return
    entity = audit.inv.get("entity_column")
    audit.check("entity-overlap", "leakage", "Entities shared across splits")
    entity_ran = False
    for early, late in SPLIT_PAIRS:
        if not roles[early] or not roles[late]:
            continue
        cols = [c for c in _common(roles[early], roles[late]) if not ID_COLUMN.search(c) and c != entity]
        if not cols:
            continue
        a, _ = _file_hashes(audit, roles[early], cols)
        later = _hashes_per_row(audit, roles[late], cols)
        shared = int(np.isin(later, a).sum())
        if shared:
            frac = shared / len(later)
            files = [f["path"] for f in roles[early] + roles[late]]
            audit.add("split-overlap", "leakage", "critical" if frac >= OVERLAP_CRITICAL_FRAC else "high", "proven",
                      f"Rows of the {late} split also appear in the {early} split",
                      f"{shared:,} of {len(later):,} {late} rows ({frac:.2%}) are identical to a {early} row on the "
                      f"{len(cols)} shared columns (identifier columns left out). A model scores those rows from memory, "
                      f"so the {late} score is inflated. Critical from {OVERLAP_CRITICAL_FRAC:.0%}.",
                      counts={"shared_rows": shared, f"{late}_rows": len(later)}, files=files)
        if entity and all(entity in (f.get("columns") or []) for f in roles[early] + roles[late]):
            entity_ran = True
            ea, _ = _file_hashes(audit, roles[early], [entity])
            eb, _ = _file_hashes(audit, roles[late], [entity])
            shared_e = int(np.isin(eb, ea).sum())
            if shared_e:
                audit.add("entity-overlap", "leakage", "high", "proven",
                          f"The same {entity} values occur in the {early} and {late} splits",
                          f"{shared_e:,} of {len(eb):,} distinct '{entity}' values in the {late} split also occur in the "
                          f"{early} split. Rows of one entity are alike, so the {late} score overstates performance on "
                          "new entities. If the split is by time on purpose, set entity_column to null in the inventory.",
                          counts={"shared_entities": shared_e, f"{late}_entities": len(eb)},
                          files=[f["path"] for f in roles[early] + roles[late]], columns=[entity])
    audit.ran("train-test-overlap", "every row of every split compared (identifier columns left out)")
    if entity_ran:
        audit.ran("entity-overlap", f"entity column '{entity}'")
    else:
        audit.could_not("entity-overlap", "no entity column identified (set entity_column in the inventory)")


def check_target_association(audit: "Audit") -> None:
    import pandas as pd
    g = guard()
    target = audit.inv.get("target")
    files = [f for f in audit.files("data", "train") or audit.files("data", "all") if target in (f.get("columns") or [])]
    if not target or not files:
        audit.could_not("feature-target-leakage", "needs a training (or full) data file with a known target column")
        return
    limit = audit.args.sample_rows or g.SAMPLE_ROWS
    data, total = g.load_sample(str(audit.root / files[0]["path"]), limit)
    data = data[data[target].notna()]
    sampled = len(data) if total > limit else None
    task = infer_task(data[target].to_numpy(), audit.inv.get("task"))
    y = data[target]
    if task == "classification":
        majority = float(y.astype(str).value_counts(normalize=True).iloc[0]) if len(y) else 1.0
    skip = {target, audit.inv.get("time_column"), audit.inv.get("entity_column")}
    for col in data.columns:
        if col in skip or ID_COLUMN.search(str(col)):
            continue
        kind = g._column_kind(data[col])
        if kind not in ("numeric", "categorical", "bool"):
            continue
        codes = g._codes(data[col], "numeric" if kind == "numeric" else "categorical")
        if task == "classification":
            if majority >= 1.0:
                continue
            table = pd.crosstab(codes, y.astype(str).to_numpy())
            purity = float(table.max(axis=1).sum() / len(y))
            strength = (purity - majority) / (1 - majority)
        else:
            strength = g._eta2(codes, y.astype(float).to_numpy()[:, None])
        if strength >= LEAK_ASSOCIATION:
            what = "removes" if task == "classification" else "explains"
            share = "of the label error left by guessing the majority class" if task == "classification" else \
                "of the target's variance"
            audit.add("target-leak-feature", "leakage", "critical", "proven",
                      f"Column '{col}' alone nearly determines the target",
                      f"on its own, '{col}' {what} {strength:.1%} {share} in the training data" +
                      (f" (a sample of {sampled:,} of {total:,} rows)" if sampled else f" ({total:,} rows)") +
                      f". Near-perfect single-feature association (from {LEAK_ASSOCIATION:.0%}) means the column "
                      "carries the answer: recorded after the outcome, derived from the label, or a copy of it.",
                      counts={"association": round(strength, 4), "rows": len(y)}, files=[files[0]["path"]],
                      columns=[col], sampled=sampled)
    audit.ran("feature-target-leakage", f"{task} target '{target}'; single-feature association on the training data" +
              (f"; on a uniform sample of {sampled:,} of {total:,} rows" if sampled else ""), sampled=sampled)


def check_time_order(audit: "Audit") -> None:
    g = guard()
    time_col = audit.inv.get("time_column")
    audit.check("time-order", "leakage", "Training data earlier than test data")
    train, test = audit.files("data", "train"), audit.files("data", "test")
    if not time_col:
        audit.could_not("time-order", "no time column (set time_column in the inventory if there is one)")
        return
    if not train or not test or not all(time_col in (f.get("columns") or []) for f in train + test):
        audit.could_not("time-order", f"time column '{time_col}' is not in both the training and test files")
        return
    def times(files):
        import pandas as pd
        series = pd.concat([read_columns(audit, f["path"], [time_col])[time_col] for f in files], ignore_index=True)
        return g.as_datetime(series).dropna()
    t_train, t_test = times(train), times(test)
    if not len(t_train) or not len(t_test):
        audit.could_not("time-order", f"'{time_col}' could not be read as dates")
        return
    later = int((t_train > t_test.min()).sum())
    if later:
        audit.add("time-order", "leakage", "high", "proven", "Training data is later than test data",
                  f"{later:,} of {len(t_train):,} training rows are dated after the earliest test row (column "
                  f"'{time_col}'). A model that will predict the future was trained on it; for forecasting this is a "
                  "leak. If time order does not matter for this problem, set time_column to null in the inventory.",
                  counts={"training_rows_after_test_start": later, "training_rows": len(t_train)},
                  files=[f["path"] for f in train + test], columns=[time_col])
    audit.ran("time-order", f"time column '{time_col}'")


@check_function
def check_leakage(audit: "Audit") -> None:
    if not audit.wants("leakage"):
        return
    try:
        guard()
    except NeedsPandas as exc:
        for check_id in ("train-test-overlap", "feature-target-leakage"):
            audit.could_not(check_id, str(exc))
        return
    for step in (check_overlap, check_target_association, check_time_order):
        try:
            step(audit)
        except Exception as exc:  # noqa: BLE001 - a failed check is reported as such, the rest still run
            audit.notes.append(f"{step.__name__}: {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ check family: model safety (static scan)
# A pickle is a small program: GLOBAL / STACK_GLOBAL / INST import a callable and REDUCE / OBJ / NEWOBJ call it. The scan
# walks the opcodes with pickletools.genops (which only decodes them) and a stack of inert placeholders, so it learns
# every name the file would import without importing or calling anything.
SAFE_GLOBALS = re.compile(
    r"^(?:numpy(?:\.(?!load$|lib\.npyio|f2py|distutils|testing|ctypeslib)[\w.]+)?|"
    r"sklearn\.[\w.]+|scipy\.(?:sparse|stats|special|interpolate|spatial|linalg|optimize)[\w.]*|"
    r"pandas\.(?:core|_libs|arrays)[\w.]*|pandas\.(?:DataFrame|Series|Index|Categorical|Timestamp|Timedelta)|"
    r"xgboost\.[\w.]+|lightgbm\.[\w.]+|catboost\.[\w.]+|joblib\.numpy_pickle\.NumpyArrayWrapper|"
    r"collections\.(?:OrderedDict|defaultdict|Counter|deque)|copyreg\._reconstructor|_codecs\.encode|"
    r"builtins\.(?:set|frozenset|slice|range|complex|bytearray|bytes|dict|list|tuple|int|float|str|bool|object)|"
    r"datetime\.(?:datetime|date|time|timedelta|timezone)|decimal\.Decimal|"
    r"torch\._utils\._rebuild_(?:tensor|tensor_v2|parameter|parameter_with_state)|torch\._tensor\._rebuild_from_type_v2|"
    r"torch\.\w*Storage|torch\.storage\.(?:UntypedStorage|TypedStorage)|torch\.(?:Size|device|dtype)|"
    r"torch\.(?:float|int|uint|bool|bfloat|complex|half|double|long|short|cfloat|cdouble)\w*|"
    r"torch\.nn\.(?:modules\.[\w.]+|parameter\.Parameter))$")
DANGEROUS_MODULES = {"os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "runpy", "importlib", "pickle",
                     "_pickle", "marshal", "ctypes", "pty", "webbrowser", "code", "codeop", "http", "urllib", "requests",
                     "multiprocessing", "asyncio", "pathlib", "io", "tempfile", "glob", "signal", "commands", "platform",
                     # the C modules behind them, which is how some callables pickle (open -> _io.open)
                     "_io", "_posixsubprocess", "_socket", "_thread", "_winapi", "_ctypes", "_signal", "posixpath",
                     "ntpath", "_osx_support"}
DANGEROUS_BUILTINS = {"eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "__import__", "globals",
                      "locals", "vars", "input", "breakpoint", "help", "exit", "quit", "memoryview", "super"}
MAX_DECOMPRESSED = 1 << 30
CODE_FREE = {".safetensors": "safetensors holds tensors and a JSON header only",
             ".onnx": "ONNX is a protobuf graph of standard operators (custom-operator libraries aside)",
             ".cbm": "CatBoost's binary format holds trees, not code", ".ubj": "XGBoost's UBJSON format holds trees, not code"}
ZIP_MAGIC = b"PK\x03\x04"


class Unverified(Exception):
    pass


class _Mark:
    pass


class _Global:
    def __init__(self, name: str):
        self.name = name


class _Call:
    def __init__(self, func, args=None):
        self.func, self.args, self.state = func, args, None


def _pop_mark(stack: list) -> list:
    for i in range(len(stack) - 1, -1, -1):
        if isinstance(stack[i], _Mark):
            items = stack[i + 1:]
            del stack[i:]
            return items
    raise Unverified("MARK missing")


_PUSH_ARG = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "STRING", "BINSTRING", "SHORT_BINSTRING",
             "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8", "UNICODE", "SHORT_BINUNICODE", "BINUNICODE",
             "BINUNICODE8", "FLOAT", "BINFLOAT"}
_CONST = {"NONE": None, "NEWTRUE": True, "NEWFALSE": False}
_ITEMSIZE = re.compile(r"^[<>|=]?([a-zA-Z])(\d*)$")


def _joblib_array_bytes(state) -> tuple[int | None, bool, bool]:
    """(raw byte count, object dtype?, padded?) of the array joblib writes after a NumpyArrayWrapper."""
    if not isinstance(state, dict):
        raise Unverified("joblib array header unreadable")
    shape, dtype = state.get("shape"), state.get("dtype")
    if not isinstance(shape, tuple) or not all(isinstance(n, int) for n in shape):
        raise Unverified("joblib array shape unreadable")
    if not (isinstance(dtype, _Call) and isinstance(dtype.args, tuple) and dtype.args and isinstance(dtype.args[0], str)):
        raise Unverified("joblib array dtype unreadable")
    m = _ITEMSIZE.match(dtype.args[0])
    if not m:
        raise Unverified(f"joblib array dtype {dtype.args[0]!r} not understood")
    kind, size = m.group(1), m.group(2)
    if kind == "O":
        return None, True, False
    if not size:
        size = {"b": "1", "?": "1", "B": "1", "e": "2", "f": "4", "d": "8", "g": "16"}.get(kind) or ""
        if not size:
            raise Unverified(f"joblib array dtype {dtype.args[0]!r} not understood")
    itemsize = int(size) * (4 if kind == "U" else 1)
    count = 1
    for n in shape:
        count *= n
    return count * itemsize, False, "numpy_array_alignment_bytes" in state


def scan_pickle(stream, found: set, depth: int = 0) -> None:
    """Walk one pickle from ``stream`` to its STOP, adding every imported name to ``found``. Raises Unverified."""
    import pickletools
    if depth > 3:
        raise Unverified("nested pickles too deep")
    stack, memo = [], {}
    while True:
        restart = False
        try:
            for op, arg, _ in pickletools.genops(stream):
                name = op.name
                if name in _PUSH_ARG:
                    stack.append(arg)
                elif name in _CONST:
                    stack.append(_CONST[name])
                elif name in ("PROTO", "FRAME"):
                    continue
                elif name == "STOP":
                    return
                elif name == "EMPTY_TUPLE":
                    stack.append(())
                elif name == "EMPTY_LIST":
                    stack.append([])
                elif name == "EMPTY_DICT":
                    stack.append({})
                elif name == "EMPTY_SET":
                    stack.append(set())
                elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                    n = int(name[-1])
                    items = stack[-n:]
                    del stack[-n:]
                    stack.append(tuple(items))
                elif name == "TUPLE":
                    stack.append(tuple(_pop_mark(stack)))
                elif name == "LIST":
                    stack.append(list(_pop_mark(stack)))
                elif name == "DICT":
                    items = _pop_mark(stack)
                    stack.append({_key(items[i]): items[i + 1] for i in range(0, len(items) - 1, 2)})
                elif name in ("FROZENSET",):
                    _pop_mark(stack)
                    stack.append(frozenset())
                elif name == "MARK":
                    stack.append(_Mark())
                elif name == "POP":
                    stack.pop()
                elif name == "POP_MARK":
                    _pop_mark(stack)
                elif name == "DUP":
                    stack.append(stack[-1])
                elif name in ("APPEND",):
                    value = stack.pop()
                    if isinstance(stack[-1], list):
                        stack[-1].append(value)
                elif name == "APPENDS":
                    items = _pop_mark(stack)
                    if isinstance(stack[-1], list):
                        stack[-1].extend(items)
                elif name == "SETITEM":
                    value, key = stack.pop(), stack.pop()
                    if isinstance(stack[-1], dict):
                        stack[-1][_key(key)] = value
                elif name == "SETITEMS":
                    items = _pop_mark(stack)
                    if isinstance(stack[-1], dict):
                        for i in range(0, len(items) - 1, 2):
                            stack[-1][_key(items[i])] = items[i + 1]
                elif name == "ADDITEMS":
                    _pop_mark(stack)
                elif name in ("GET", "BINGET", "LONG_BINGET"):
                    if arg not in memo:
                        raise Unverified("memo reference to nothing")
                    stack.append(memo[arg])
                elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                    memo[arg] = stack[-1]
                elif name == "MEMOIZE":
                    memo[len(memo)] = stack[-1]
                elif name in ("GLOBAL", "INST"):
                    module, _, attr = str(arg).partition(" ")
                    found.add(f"{module}.{attr}")
                    if name == "INST":
                        stack.append(_Call(_Global(f"{module}.{attr}"), tuple(_pop_mark(stack))))
                    else:
                        stack.append(_Global(f"{module}.{attr}"))
                elif name == "STACK_GLOBAL":
                    attr, module = stack.pop(), stack.pop()
                    if not isinstance(module, str) or not isinstance(attr, str):
                        raise Unverified("an import whose name is built at load time")
                    found.add(f"{module}.{attr}")
                    stack.append(_Global(f"{module}.{attr}"))
                elif name in ("EXT1", "EXT2", "EXT4"):
                    raise Unverified("an import through the copyreg extension registry")
                elif name == "REDUCE":
                    args, func = stack.pop(), stack.pop()
                    stack.append(_Call(func, args))
                elif name == "NEWOBJ":
                    args, cls = stack.pop(), stack.pop()
                    stack.append(_Call(cls, args))
                elif name == "NEWOBJ_EX":
                    stack.pop()
                    args, cls = stack.pop(), stack.pop()
                    stack.append(_Call(cls, args))
                elif name == "OBJ":
                    items = _pop_mark(stack)
                    stack.append(_Call(items[0] if items else None, tuple(items[1:])))
                elif name == "BUILD":
                    state = stack.pop()
                    obj = stack[-1]
                    if isinstance(obj, _Call):
                        obj.state = state
                        func = obj.func
                        if isinstance(func, _Global) and func.name == "joblib.numpy_pickle.NumpyArrayWrapper":
                            nbytes, objects, padded = _joblib_array_bytes(state)
                            if objects:
                                scan_pickle(stream, found, depth + 1)  # joblib pickles object arrays as a nested pickle
                            else:
                                if padded:
                                    pad = stream.read(1)
                                    if len(pad) != 1:
                                        raise Unverified("file ends inside a joblib array")
                                    stream.read(pad[0])
                                skipped = stream.read(nbytes)
                                if len(skipped) != nbytes:
                                    raise Unverified("file ends inside a joblib array")
                            restart = True
                            break
                elif name == "PERSID":
                    stack.append(_Call(None, arg))
                elif name == "BINPERSID":
                    stack.append(_Call(None, stack.pop()))
                elif name in ("NEXT_BUFFER",):
                    stack.append(None)
                elif name in ("READONLY_BUFFER",):
                    continue
                else:
                    raise Unverified(f"opcode {name} not understood")
        except Unverified:
            raise
        except (ValueError, IndexError, KeyError, TypeError, EOFError) as exc:
            raise Unverified(f"not a readable pickle ({type(exc).__name__}: {str(exc)[:80]})") from None
        if not restart:
            raise Unverified("the pickle has no end (truncated)")


def _key(value):
    try:
        hash(value)
        return value
    except TypeError:
        return id(value)


def _decompressed(head: bytes, path: Path) -> bytes | None:
    """The decompressed bytes of a compressed joblib file, or None when the file is not compressed."""
    import bz2
    import lzma
    import zlib
    data = None
    if head.startswith(b"\x1f\x8b"):
        with gzip.open(path, "rb") as fh:
            data = fh.read(MAX_DECOMPRESSED + 1)
    elif head.startswith(b"BZh"):
        with bz2.open(path, "rb") as fh:
            data = fh.read(MAX_DECOMPRESSED + 1)
    elif head.startswith(b"\xfd7zXZ\x00") or head.startswith(b"]\x00\x00"):
        with lzma.open(path, "rb") as fh:
            data = fh.read(MAX_DECOMPRESSED + 1)
    elif head[:1] == b"x" and len(head) > 1 and head[1] in (0x01, 0x5E, 0x9C, 0xDA):
        d = zlib.decompressobj()
        data = d.decompress(path.read_bytes(), MAX_DECOMPRESSED + 1)
    elif head.startswith(b'\x04"M\x18'):
        raise Unverified("lz4-compressed (no decompressor in the standard library)")
    elif head.startswith(b"ZF"):
        raise Unverified("an old joblib format")
    if data is not None and len(data) > MAX_DECOMPRESSED:
        raise Unverified("decompressed size over the scan limit")
    return data


def scan_model(path: Path) -> dict:
    """{'status': 'clean'|'risky'|'unverified'|'code-free', 'imports': [...], 'dangerous': [...], 'unlisted': [...], 'why'}"""
    suffix = path.suffix.lower()
    if suffix in CODE_FREE:
        return {"status": "code-free", "imports": [], "why": CODE_FREE[suffix]}
    found: set = set()
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
        if not head:
            raise Unverified("the file is empty")
        if head.startswith(ZIP_MAGIC):
            import zipfile
            try:
                with zipfile.ZipFile(path) as zf:
                    members = [m for m in zf.namelist() if m.endswith(".pkl")]
                    config = [m for m in zf.namelist() if m.endswith("config.json")]
                    if not members and not config:
                        raise Unverified("a zip file with no pickle or model config inside")
                    for member in members:
                        with zf.open(member) as fh:
                            scan_pickle(io.BytesIO(fh.read()), found)
                    for member in config:
                        text = zf.read(member).decode("utf-8", "replace")
                        if re.search(r'"class_name"\s*:\s*"(?:Lambda|TFOpLambda)"|"function_type"', text):
                            found.add("keras.layers.Lambda (code stored in the model config)")
            except zipfile.BadZipFile as exc:
                raise Unverified(f"a damaged zip file ({exc})") from None
        elif suffix in (".h5", ".hdf5", ".keras"):
            raise Unverified("HDF5/Keras files are not scanned (Lambda layers can hold code)")
        else:
            data = _decompressed(head, path)
            stream = io.BytesIO(data) if data is not None else path.open("rb")
            try:
                if (data[:1] if data is not None else head[:1]) != b"\x80" and head[:1] not in (b"(", b"c", b"]", b"}"):
                    raise Unverified("not a pickle (unknown format)")
                scan_pickle(stream, found)
                while True:  # further pickles after the first STOP (legacy torch files) must parse too
                    rest = stream.read(1)
                    if not rest:
                        break
                    stream.seek(-1, 1)
                    scan_pickle(stream, found)
            finally:
                stream.close()
    except Unverified as exc:
        return {"status": "unverified", "imports": sorted(found), "why": str(exc)}
    except OSError as exc:
        return {"status": "unverified", "imports": sorted(found), "why": f"could not be read ({exc.strerror})"}
    found = {"builtins." + n.split(".", 1)[1] if n.startswith("__builtin__.") else n for n in found}  # Python 2 name
    dangerous = sorted(n for n in found if n.split(".")[0] in DANGEROUS_MODULES or
                       (n.startswith("builtins.") and n.split(".", 1)[1] in DANGEROUS_BUILTINS) or "Lambda" in n)
    unlisted = sorted(n for n in found if n not in dangerous and not SAFE_GLOBALS.match(n))
    status = "risky" if dangerous or unlisted else "clean"
    return {"status": status, "imports": sorted(found), "dangerous": dangerous, "unlisted": unlisted, "why": ""}


NOT_PROOF = "A clean scan lowers the risk; it does not prove the file is safe to load."


@check_function
def check_model_safety(audit: "Audit") -> None:
    if not audit.wants("model-safety"):
        return
    audit.check("model-scan", "model-safety", "Model files scanned without loading them")
    models = audit.files("model")
    if not models:
        audit.could_not("model-scan", "no model files in the inventory")
        return
    for f in models:
        result = audit.scans.get(f["path"]) or scan_model(audit.root / f["path"])
        audit.scans[f["path"]] = result
        if result["status"] == "code-free":
            audit.add("code-free-format", "model-safety", "info", "proven", f"{f['path']} is a code-free format",
                      f"{result['why']}; loading it runs no stored code.", files=[f["path"]])
        elif result["status"] == "unverified":
            audit.add("unverified-model", "model-safety", "medium", "could-not-check", f"{f['path']} could not be verified",
                      f"the scan could not read it: {result['why']}. Unverified is not safe: treat it as able to run code.",
                      files=[f["path"]])
        elif result["status"] == "risky":
            d, u = result["dangerous"], result["unlisted"]
            audit.add("risky-imports", "model-safety", "high" if d else "medium", "proven",
                      f"{f['path']} imports {'dangerous names' if d else 'names outside the safe list'}",
                      (f"loading it would import {', '.join(d)} - these can run commands or touch files. " if d else "") +
                      (f"It also imports names outside the safe list: {', '.join(u[:12])}" +
                       (f" and {len(u) - 12} more" if len(u) > 12 else "") +
                       " (often the project's own classes; they run code on load)." if u else "") +
                      " The file was not loaded.",
                      counts={"dangerous_imports": len(d), "unlisted_imports": len(u)}, files=[f["path"]])
        else:
            audit.add("scan-clean", "model-safety", "info", "proven", f"{f['path']}: no risky imports found",
                      f"every one of its {len(result['imports'])} imports is on the safe list. {NOT_PROOF} "
                      "The file was not loaded.", counts={"imports": len(result["imports"])}, files=[f["path"]])
    audit.ran("model-scan", f"{len(models)} file(s) scanned statically and not loaded. {NOT_PROOF}")


# ------------------------------------------------------------------ check family: hidden weak spots
MIN_GROUP_SIZE = 30            # a subgroup smaller than this is never reported as a problem
MAX_GROUP_LEVELS = 20          # columns with more levels than this are not used to form subgroups
SUBGROUP_GAP = 0.05            # accuracy gap (or MAE gap relative to the rest) a subgroup must show, with its interval above 0
CALIBRATION_ECE = 0.10         # expected calibration error (10 equal-width bins) from which probabilities are flagged
MIN_CALIBRATION_ROWS = 100


def _aligned_frame(audit: "Audit", pf: dict):
    """The predictions joined to the test rows' columns: by a shared id column, else row by row when the labels line up."""
    import numpy as np
    target = audit.inv.get("target")
    preds = read_columns(audit, pf["path"])
    tests = audit.files("data", "test")
    if not tests:
        return preds, "the predictions file only"
    test = tests[0]
    ids = _id_columns(pf, test, target)
    data = read_columns(audit, test["path"])
    extra = [c for c in data.columns if c not in preds.columns]
    if ids:
        joined = preds.merge(data[ids + extra].drop_duplicates(ids), on=ids, how="left")
        return joined, f"joined to {test['path']} on {', '.join(ids)}"
    if target and pf.get("y_true") and len(data) == len(preds) and \
            np.array_equal(_labels(data[target].to_numpy()), _labels(preds[pf["y_true"]].to_numpy())):
        for c in extra:
            preds[c] = data[c].to_numpy()
        return preds, f"aligned row by row with {test['path']} (the labels match in order)"
    return preds, "the predictions file only (no id column to join on the test file)"


def _group_columns(audit: "Audit", frame, pf: dict) -> list[str]:
    g = guard()
    configured = audit.inv.get("group_columns")
    if configured:
        return [c for c in configured if c in frame.columns]
    skip = {pf.get("y_true"), pf.get("y_pred"), pf.get("y_prob"), audit.inv.get("target"), audit.inv.get("time_column")}
    out = []
    for col in frame.columns:
        if col in skip or ID_COLUMN.search(str(col)):
            continue
        kind = g._column_kind(frame[col])
        levels = frame[col].nunique(dropna=True)
        if kind in ("categorical", "bool") and 2 <= levels <= MAX_GROUP_LEVELS:
            out.append(col)
    return out


def check_subgroups(audit: "Audit", pf: dict, frame, how: str, task: str) -> None:
    import numpy as np
    y, p = frame[pf["y_true"]].to_numpy(), frame[pf["y_pred"]].to_numpy()
    if task == "classification":
        per_row = (_labels(y) == _labels(p)).astype(float) if _labels(y).dtype == _labels(p).dtype else \
            (_labels(y).astype(str) == _labels(p).astype(str)).astype(float)
        name, worse = "accuracy", -1.0     # lower accuracy is worse
    else:
        per_row = np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))
        name, worse = "mean absolute error", 1.0
    from statistics import NormalDist
    columns = _group_columns(audit, frame, pf)
    groups = [(col, level, idx) for col in columns for level, idx in frame.groupby(col, dropna=True).indices.items()]
    judged = sum(1 for _, _, idx in groups if len(idx) >= MIN_GROUP_SIZE and len(frame) - len(idx) >= MIN_GROUP_SIZE)
    # Bonferroni: the interval holds for all judged groups at once, so testing many groups does not manufacture a finding
    z = NormalDist().inv_cdf(1 - 0.025 / max(judged, 1))
    small, tested = 0, 0
    for col, level, idx in groups:
        n_g = len(idx)
        rest = np.ones(len(frame), dtype=bool)
        rest[idx] = False
        if n_g < MIN_GROUP_SIZE:
            small += 1
            continue
        if rest.sum() < MIN_GROUP_SIZE:
            continue
        tested += 1
        g_vals, r_vals = per_row[idx], per_row[rest]
        gap = worse * (g_vals.mean() - r_vals.mean())        # > 0: the group is worse than the rest
        se = float(np.sqrt(g_vals.var(ddof=1) / n_g + r_vals.var(ddof=1) / len(r_vals)))
        low, high = gap - z * se, gap + z * se
        needed = SUBGROUP_GAP if task == "classification" else SUBGROUP_GAP * r_vals.mean()
        if gap >= needed and low > 0:
            audit.add("weak-subgroup", "weak-spots", "medium", "proven",
                      f"The model is weaker where {col} = {level}",
                      f"{name} {g_vals.mean():.3f} on {n_g:,} rows where {col} = {level}, against {r_vals.mean():.3f} "
                      f"on the other {len(r_vals):,} rows; gap {gap:.3f}, 95% interval {low:.3f} to {high:.3f} "
                      f"(Bonferroni-corrected over {judged} groups) "
                      f"(predictions {how}). Groups under {MIN_GROUP_SIZE} rows are never reported.",
                      counts={"group_rows": n_g, "group_metric": round(float(g_vals.mean()), 6),
                              "rest_metric": round(float(r_vals.mean()), 6), "gap": round(float(gap), 6),
                              "gap_interval_95": [round(float(low), 6), round(float(high), 6)]},
                      files=[pf["path"]], columns=[col])
    note = f"{tested} subgroup(s) of {len(columns)} column(s) compared ({how})"
    if small:
        note += f"; {small} group(s) with fewer than {MIN_GROUP_SIZE} rows not judged"
    if columns:
        audit.ran("subgroups", note)
    else:
        audit.could_not("subgroups", f"no categorical column with 2-{MAX_GROUP_LEVELS} levels next to the predictions "
                                     f"({how}); set group_columns in the inventory")


def expected_calibration_error(y_pos, prob, bins: int = 10) -> float:
    import numpy as np
    prob = np.asarray(prob, dtype=float)
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.sum() / len(prob) * abs(prob[m].mean() - y_pos[m].mean())
    return float(total)


def check_calibration(audit: "Audit", pf: dict, frame, task: str) -> None:
    import numpy as np
    if task == "regression":
        audit.not_applicable("calibration", "regression target: calibration applies to predicted class probabilities")
        return
    if not pf.get("y_prob"):
        audit.could_not("calibration", f"{pf['path']} has no probability column")
        return
    y = frame[pf["y_true"]].to_numpy()
    if len(np.unique(_labels(y))) != 2:
        audit.not_applicable("calibration", "more than two classes: per-class calibration is not checked")
        return
    if len(y) < MIN_CALIBRATION_ROWS:
        audit.could_not("calibration", f"fewer than {MIN_CALIBRATION_ROWS} rows with probabilities")
        return
    positive = positive_label(y, audit.inv.get("positive_label"))
    prob = frame[pf["y_prob"]].to_numpy(dtype=float)
    ece = expected_calibration_error((_labels(y) == positive).astype(float), prob)
    if ece >= CALIBRATION_ECE:
        audit.add("miscalibrated", "weak-spots", "medium", "proven", "Predicted probabilities do not match how often the outcome happens",
                  f"expected calibration error {ece:.3f} over {len(y):,} rows of {pf['path']} (10 equal-width bins; "
                  f"flagged from {CALIBRATION_ECE:.2f}). The labels may still be right, but the probabilities cannot be "
                  "read as chances.", counts={"expected_calibration_error": round(ece, 6), "rows": len(y)},
                  files=[pf["path"]], columns=[pf["y_prob"]])
    audit.ran("calibration", f"{pf['path']}: expected calibration error {ece:.3f} on {len(y):,} rows")


@check_function
def check_weak_spots(audit: "Audit") -> None:
    if not audit.wants("weak-spots"):
        return
    audit.check("subgroups", "weak-spots", "Subgroups where the model is weaker")
    audit.check("calibration", "weak-spots", "Calibration of predicted probabilities")
    preds = [p for p in audit.files("predictions") if p.get("y_true") and p.get("y_pred")]
    preds = [p for p in preds if p.get("split") == "test"] or preds
    if not preds:
        for check_id in ("subgroups", "calibration"):
            audit.could_not(check_id, "no predictions file with true and predicted values")
        return
    try:
        pf = preds[0]
        frame, how = _aligned_frame(audit, pf)
        task = infer_task(frame[pf["y_true"]].to_numpy(), audit.inv.get("task"))
        check_subgroups(audit, pf, frame, how, task)
        check_calibration(audit, pf, frame, task)
    except NeedsPandas as exc:
        for check_id in ("subgroups", "calibration"):
            audit.could_not(check_id, str(exc))


# ------------------------------------------------------------------ check family: test-set misuse
OVERRIDE_FINAL_TEST = re.compile(r"^\s*-\s*Override\s*:\s*final\s+test\b", re.I | re.M)
TUNING_CODE = re.compile(r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|HalvingRandomSearchCV|BayesSearchCV|"
                         r"TuneSearchCV|ParameterGrid|ParameterSampler)\b|\boptuna\.create_study\b|\bhyperopt\b|"
                         r"\bfmin\s*\(|\btune\.(?:run|Tuner)\b")
TEST_NAMES = r"(?:X_test|y_test|x_test|test_X|test_y|Xtest|ytest|df_test|test_df|test_data|X_holdout|y_holdout|holdout\w*)"
TEST_IN_CALL = re.compile(r"\b(?:predict(?:_proba)?|score|evaluate|fit|\w+_score|\w+_error|\w+_loss)\s*\([^)\n]*\b"
                          + TEST_NAMES + r"\b")


def _code_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".ipynb":
        try:
            cells = json.loads(text).get("cells", [])
            return "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                             for c in cells if c.get("cell_type") == "code")
        except ValueError:
            return text
    return text


@check_function
def check_test_misuse(audit: "Audit") -> None:
    if not audit.wants("test-misuse"):
        return
    audit.check("test-use-log", "test-misuse", "Repeated test scoring in the run log and ledger")
    audit.check("test-in-tuning-code", "test-misuse", "Test data referenced in tuning code")
    logs = [f for f in audit.files("claims") if f.get("source") in ("run log", "ledger")]
    if not logs:
        audit.could_not("test-use-log", "no run log or ledger (ml_pipeline/runs.jsonl, ml_pipeline/PIPELINE.md) to read; "
                                        "repeated test use cannot be ruled out")
    else:
        finals = overrides = runs_with_test = 0
        files = []
        for f in logs:
            text = (audit.root / f["path"]).read_text(encoding="utf-8", errors="replace")
            if f["source"] == "ledger":
                n, k = len(FINAL_TEST.findall(text)), len(OVERRIDE_FINAL_TEST.findall(text))
                finals, overrides = finals + n, overrides + k
                if n > 1 or k:
                    files.append(f["path"])
            else:
                count = 0
                for line in text.splitlines():
                    try:
                        run = json.loads(line)
                    except ValueError:
                        continue
                    metrics = run.get("metrics") if isinstance(run, dict) else None
                    if isinstance(metrics, dict) and any("test" in _tokens(str(k)) for k in metrics):
                        count += 1
                runs_with_test += count
                if count > 1:
                    files.append(f["path"])
        if finals > 1 or overrides or runs_with_test > 1:
            parts = []
            if finals > 1 or overrides:
                parts.append(f"the ledger records {finals} final-test scorings and {overrides} final-test override(s)")
            if runs_with_test > 1:
                parts.append(f"{runs_with_test} runs in the run log carry a test metric")
            audit.add("repeated-test-scoring", "test-misuse", "high", "proven", "The test set was scored more than once",
                      "; ".join(parts) + ". Each extra look lets choices follow the test set, so the final test number "
                      "is likely optimistic.", counts={"ledger_final_tests": finals, "final_test_overrides": overrides,
                                                       "runs_with_test_metrics": runs_with_test}, files=files)
        audit.ran("test-use-log", ", ".join(f["path"] for f in logs) + " read (the guard's state file is never opened)")
    code = audit.files("code")
    if not code:
        audit.could_not("test-in-tuning-code", "no code files")
        return
    for f in code:
        try:
            text = _code_text(audit.root / f["path"])
        except OSError:
            continue
        if not TUNING_CODE.search(text):
            continue
        lines = [text.count("\n", 0, m.start()) + 1 for m in TEST_IN_CALL.finditer(text)]
        if lines:
            audit.add("test-in-tuning", "test-misuse", "high", "suspected", f"Test data appears in tuning code ({f['path']})",
                      f"{f['path']} runs a hyperparameter search and passes test-split data to a fit, predict, or score "
                      f"call on {len(lines)} line(s). If those scores guide the search or the choice of model, the test "
                      "set was used for tuning. Read the code to confirm (the tool only sees the pattern).",
                      counts={"lines": len(lines)}, files=[f["path"]],
                      source=", ".join(f"{f['path']}:{n}" for n in lines[:10]))
    audit.ran("test-in-tuning-code", f"{len(code)} code file(s) read as text (never run)")


# ------------------------------------------------------------------ trusted scoring (only with the user's trust)
DEFAULT_TIMEOUT = 300
PROJECT_INTERPRETERS = (".venv/bin/python", "venv/bin/python", "env/bin/python", ".venv/Scripts/python.exe",
                        "venv/Scripts/python.exe")
# Run in a child process with the project's interpreter. It is the only place a model is ever loaded. Formats with a
# code-free mode use it first (torch weights_only=True, ONNX Runtime); a pickle has none, so trust is required.
TRUSTED_RUNNER = r"""
import csv, json, os, sys
args = json.loads(sys.argv[1])
sys.path.insert(0, args["project"])
def fail(message):
    print(json.dumps({"ok": False, "error": message}))
    sys.exit(3)
try:
    import pandas as pd
except ImportError:
    fail("pandas is not installed in the project's interpreter")
path = args["model"]
suffix = os.path.splitext(path)[1].lower()
mode = ""
if suffix in (".pt", ".pth", ".ckpt"):
    import torch
    try:
        model = torch.load(path, map_location="cpu", weights_only=True)
        mode = "weights only (no code run)"
    except Exception:
        model = torch.load(path, map_location="cpu", weights_only=False)
        mode = "full load (runs stored code)"
elif suffix == ".onnx":
    import onnxruntime
    model = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    mode = "ONNX Runtime (no code run)"
else:
    try:
        import joblib
        model = joblib.load(path)
    except ImportError:
        import pickle
        with open(path, "rb") as fh:
            model = pickle.load(fh)
    mode = "pickle (runs stored code)"
frames = []
for data in args["data"]:
    low = data.lower()
    if low.endswith((".parquet", ".pq")):
        frames.append(pd.read_parquet(data))
    elif low.endswith((".feather", ".arrow")):
        frames.append(pd.read_feather(data))
    elif low.endswith((".jsonl", ".ndjson")):
        frames.append(pd.read_json(data, lines=True))
    else:
        frames.append(pd.read_csv(data, sep="	" if ".tsv" in low else ","))
frame = pd.concat(frames, ignore_index=True)
y = frame[args["target"]]
X = frame.drop(columns=[c for c in args["drop"] if c in frame.columns])
names = getattr(model, "feature_names_in_", None)
if names is not None:
    X = X[list(names)]
prob = None
if suffix == ".onnx":
    inp = model.get_inputs()[0].name
    outs = model.run(None, {inp: X.to_numpy(dtype="float32")})
    pred = outs[0]
elif hasattr(model, "predict"):
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if getattr(p, "ndim", 0) == 2 and p.shape[1] == 2:
            prob = p[:, 1]
elif hasattr(model, "forward") and hasattr(model, "eval"):
    import torch
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X.to_numpy(dtype="float32")))
    out = out.numpy()
    pred = out.argmax(axis=1) if out.ndim == 2 and out.shape[1] > 1 else out.reshape(-1)
else:
    fail("loaded with " + mode + ", but the object (" + type(model).__name__ + ") has no predict method: "
         + ("a weights only file holds no model to score" if "weights" in mode else "nothing to score"))
with open(args["out"], "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["y_true", "y_pred"] + (["y_prob"] if prob is not None else []))
    for i in range(len(y)):
        w.writerow([y.iloc[i], pred[i]] + ([prob[i]] if prob is not None else []))
print(json.dumps({"ok": True, "rows": len(y), "mode": mode}))
"""


def project_interpreter(root: Path) -> str:
    for rel in PROJECT_INTERPRETERS:
        if (root / rel).is_file():
            return str(root / rel)
    return sys.executable


def trusted_scoring(audit: "Audit") -> None:
    """Load each model the user trusts in a child process and predict the test file; failures are findings."""
    import subprocess
    if not (audit.wants("honest-scores") or audit.wants("model-safety")):
        return
    audit.check("trusted-scoring", "honest-scores", "Trusted models scored on the test data")
    models = audit.files("model")
    if not models:
        audit.could_not("trusted-scoring", "no model files in the inventory")
        return
    if not audit.args.trust:
        audit.could_not("trusted-scoring", "models are not loaded without the user's trust: ask them, and on their "
                                           f"'trust models in {audit.root.name}' rerun with --trust {audit.root.name}")
        return
    if audit.args.trust != audit.root.name:
        audit.could_not("trusted-scoring", f"--trust names '{audit.args.trust}', but this project is "
                                           f"'{audit.root.name}'; trust is granted per project")
        return
    target = audit.inv.get("target")
    tests = [f for f in audit.files("data", "test") if target and target in (f.get("columns") or [])]
    if not tests:
        audit.could_not("trusted-scoring", "needs a test data file with the target column")
        return
    python = project_interpreter(audit.root)
    drop = sorted({target, audit.inv.get("entity_column"), audit.inv.get("time_column")} - {None} |
                  {c for c in tests[0].get("columns") or [] if ID_COLUMN.search(c)})
    folder = audit.out / "trusted_predictions"
    scored, skipped = 0, []
    timeout = audit.args.timeout or DEFAULT_TIMEOUT
    for f in models:
        scan = audit.scans.get(f["path"]) or scan_model(audit.root / f["path"])
        audit.scans[f["path"]] = scan
        if scan["status"] == "risky" and scan.get("dangerous"):
            skipped.append(f"{f['path']} not loaded: it imports {', '.join(scan['dangerous'])}")
            continue
        if scan["status"] == "code-free" and Path(f["path"]).suffix.lower() != ".onnx":
            skipped.append(f"{f['path']} holds weights only; nothing to score")
            continue
        folder.mkdir(exist_ok=True)
        out = folder / (re.sub(r"[^\w.-]", "_", f["path"]) + ".csv")
        payload = {"project": str(audit.root), "model": str(audit.root / f["path"]), "target": target, "drop": drop,
                   "data": [str(audit.root / t["path"]) for t in tests], "out": str(out)}
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        error = None
        try:
            proc = subprocess.run([python, "-B", "-c", TRUSTED_RUNNER, json.dumps(payload)], capture_output=True,
                                  text=True, timeout=timeout, cwd=str(audit.out), env=env)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            result = json.loads(lines[-1]) if lines else {}
            if proc.returncode != 0 or not result.get("ok"):
                tail = (proc.stderr.strip().splitlines() or [""])[-1]
                error = result.get("error") or tail[:300] or f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = f"timed out after {timeout} s"
        if error:
            audit.add("trusted-scoring-failed", "honest-scores", "info", "could-not-check",
                      f"{f['path']} could not be scored", f"loading or predicting in a child process failed: {error}.",
                      files=[f["path"]])
            continue
        scored += 1
        audit.viewed_test_labels = True
        columns = header(out) or []
        audit.inv["files"].append({"path": out.relative_to(audit.root).as_posix(), "kind": "predictions",
                                   "split": "test", "y_true": "y_true", "y_pred": "y_pred",
                                   "y_prob": "y_prob" if "y_prob" in columns else None, "columns": columns,
                                   "made_by_audit": f"{f['path']} ({result.get('mode')})"})
    note = f"{scored} of {len(models)} model(s) scored with {'the project interpreter' if python != sys.executable else 'this interpreter'}"
    if skipped:
        note += "; " + "; ".join(skipped)
    if scored:
        audit.ran("trusted-scoring", note)
        audit.notes.append("The audit viewed the test labels to score the trusted models. Its numbers must not be used "
                           "to choose, tune, or compare models: that would turn the test set into a validation set.")
    else:
        audit.could_not("trusted-scoring", note)


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
        lines += ["No findings.", ""]
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
    out = audit.out = new_audit_folder(root)
    run_checks(audit)
    result, why = write_outputs(audit, out)
    print(f"audit folder: {out}")
    print(f"verdict: {result} - {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
