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
