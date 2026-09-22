"""Gate-time checks for the v0.5 records: every profiled column has its complete, explained record.

Imported by guard_training.py. Each check reads the small files the guard library writes into
ml_pipeline/ and returns problems in the words the agent will read. Stdlib only.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

MIN_NOTE_CHARS = 40
LIST_CAP = 10
PROFILE_FILE = "data_profile.json"
SOURCE_FILE = "source.json"
DICTIONARY_FILE = "data_dictionary.csv"
PAGES_FILE = "figures/pages.json"
MISSINGNESS_FILE = "missingness.csv"
LABEL_AUDIT_FILE = "label_audit.md"
CLEANING_FILE = "cleaning_log.csv"
PLAN_FILE = "categorical_plan.csv"
ASSOCIATIONS_FILE = "associations.json"
TRAINING_FILE = "training_runs.jsonl"
IMPORTANCE_FILE = "importance.json"
STATE_FILE = ".guard_state.json"

PROFILE_KEYS = {
    "common": ("kind", "dtype", "non_null", "missing", "missing_frac", "n_unique", "unique_frac", "memory_bytes"),
    "numeric": ("mean", "std", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "skew",
                "kurtosis", "zeros", "negatives", "outliers", "outlier_frac"),
    "categorical": ("n_levels", "top", "top_counts", "rare_levels"),
    "bool": ("n_levels", "top", "top_counts", "rare_levels"),
    "datetime": ("start", "end", "span_days"),
    "text": ("length_min", "length_median", "length_max"),
    "id": (),
}
WORD_FIELDS = ("meaning", "unit", "allowed", "source", "available_at_prediction", "role", "figure_note")
ROLES = ("feature", "target", "id", "leak", "drop")
MECHANISMS = {"random", "depends-on-other-data", "depends-on-itself", "structural", "unknown", "mcar", "mar", "mnar"}
DECISIONS = ("drop-column", "drop-rows", "impute-median", "impute-mean", "impute-mode", "impute-constant",
             "impute-model", "indicator", "keep-native")
ENCODINGS = ("one-hot", "ordinal", "target-encoding-cv", "frequency", "rare-grouping", "hashing", "native", "drop")
LABEL_SECTIONS = ("Definition", "Source", "When labeled", "Leakage check", "Noise and consistency")
COMPUTED_START = "<!-- computed by guard.label_audit - do not edit -->"
COMPUTED_END = "<!-- end computed -->"
PARTIAL_DATA = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*partial data[ \t]*-", re.I | re.M)
CLEANING_NONE = re.compile(r"^[ \t]*-[ \t]*Cleaning[ \t]*:[ \t]*none needed[ \t]*-[ \t]*reason[ \t]*:[ \t]*\S",
                           re.I | re.M)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
CLASS_ITEM = re.compile(r"^(.*) = ([\d,]+) \(")

# ------------------------------------------------------------------ the real-explanation rule
# Same rule, same results as note_problem() in lib/mlpipeline_guard.py (tests/note_rule_cases.json).
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _stat_values(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)] if obj == obj else []
    if isinstance(obj, dict):
        return [v for value in obj.values() for v in _stat_values(value)]
    if isinstance(obj, (list, tuple)):
        return [v for value in obj for v in _stat_values(value)]
    return []


def _cites(note: str, stats: dict) -> bool:
    facts = _stat_values(stats)
    for token in _NUMBER.findall(note):
        percent = token.endswith("%")
        text = token.rstrip("%").replace(",", "")
        try:
            value = float(text)
        except ValueError:
            continue
        if value in (0.0, 1.0):
            continue
        decimals = len(text.split(".")[1]) if "." in text else 0
        for fact in facts:
            candidate = round(fact * 100, decimals) if percent else round(fact, decimals)
            if abs(candidate - value) < 1e-9:
                return True
    return False


def note_problem(note: str, stats: dict | None = None) -> str | None:
    """Why ``note`` is not a real explanation, or None. With ``stats`` it is a column note and must cite them."""
    text = " ".join(str(note or "").split())
    if len(text) < MIN_NOTE_CHARS:
        return f"shorter than {MIN_NOTE_CHARS} characters"
    if not re.search(r"\d", text):
        return "contains no number"
    if stats is not None and not _cites(text, stats):
        return "cites no statistic of this column"
    return None


# ------------------------------------------------------------------ helpers
def _listing(names) -> str:
    names = [str(n) for n in names]
    shown = ", ".join(names[:LIST_CAP])
    return shown + (f" and {len(names) - LIST_CAP} more" if len(names) > LIST_CAP else "")


def _read_text(path: Path) -> str:
    """A record's text: UTF-8 (with or without a byte-order mark), else Excel's legacy cp1252."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _json(path: Path):
    try:
        return json.loads(_read_text(path))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    try:
        lines = _read_text(path).splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _csv_rows(path: Path, key: str = "column") -> list[dict] | None:
    """Rows of a CSV record with stripped values; None when the file is absent.

    Files saved by Excel or Numbers parse too: a byte-order mark, CRLF, cp1252, or a ``;`` or tab
    delimiter (the delimiter is the one whose header has ``key``).
    """
    if not path.is_file():
        return None
    text = _read_text(path)
    for delimiter in (",", ";", "\t"):
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
        if reader.fieldnames and key in [f.strip() for f in reader.fieldnames]:
            return [{(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]
    return []


def _valid_combo(value: str, allowed) -> bool:
    parts = [p.strip() for p in value.lower().split("+")]
    return bool(value.strip()) and all(p in allowed for p in parts)


def _int(value) -> int | None:
    try:
        return int(float(str(value).replace(",", "")))
    except (ValueError, OverflowError):
        return None


def _v2_profile(state_dir: Path) -> dict | None:
    profile = _json(state_dir / PROFILE_FILE)
    return profile if isinstance(profile, dict) and profile.get("schema_version") == "2" else None


# ------------------------------------------------------------------ Gate A
def _source_problems(state_dir: Path, profile: dict, run_text: str) -> list[str]:
    if PARTIAL_DATA.search(run_text):
        return []
    if not (state_dir / SOURCE_FILE).is_file():
        return ["source.json missing: load the data with guard.load() so the profile can be tied to the full file"]
    source = profile.get("source") or {}
    if not source.get("full_data"):
        reason = source.get("reason") or "no loaded source matches it"
        return [f"the profile does not cover the full data ({reason}); profile the whole loaded frame, or record "
                "'- Override: partial data - ...' if the user agreed"]
    return []


def _profile_problems(profile: dict) -> list[str]:
    columns = profile.get("columns") or {}
    problems = []
    n_cols = (profile.get("shape") or {}).get("n_cols")
    if n_cols is not None and n_cols != len(columns):
        problems.append(f"data_profile.json lists {len(columns)} of {n_cols} columns: re-run guard.profile()")
    incomplete = []
    for name, info in columns.items():
        required = list(PROFILE_KEYS["common"])
        if info.get("non_null"):
            required += PROFILE_KEYS.get(info.get("kind", ""), ())
        if any(key not in info for key in required):
            incomplete.append(name)
    if incomplete:
        problems.append(f"data_profile.json lacks describe/info statistics for {_listing(incomplete)}: "
                        "re-run guard.profile()")
    return problems


def _dictionary_problems(rows: list[dict] | None, profile: dict) -> list[str]:
    if rows is None:
        return ["data_dictionary.csv missing: run guard.data_dictionary(df) and fill in every column"]
    columns = profile.get("columns") or {}
    target = str((profile.get("target") or {}).get("name"))
    names = [r.get("column", "") for r in rows]
    problems = []
    missing = [c for c in columns if c not in names]
    extra = [c for c in names if c not in columns]
    dupes = sorted({c for c in names if names.count(c) > 1})
    if missing:
        problems.append(f"data_dictionary.csv has no row for {_listing(missing)}")
    if extra:
        problems.append(f"data_dictionary.csv has rows for columns not in the profile: {_listing(extra)}")
    if dupes:
        problems.append(f"data_dictionary.csv has more than one row for {_listing(dupes)}")
    blank = [f"{r.get('column')} ({', '.join(f for f in WORD_FIELDS if not r.get(f))})"
             for r in rows if any(not r.get(f) for f in WORD_FIELDS)]
    if blank:
        problems.append("data_dictionary.csv has blank fields: " + _listing(blank))
    bad_available = [r.get("column") for r in rows
                     if r.get("available_at_prediction") and r["available_at_prediction"].lower() not in ("yes", "no")]
    if bad_available:
        problems.append(f"available_at_prediction must be yes or no: {_listing(bad_available)}")
    bad_role = [r.get("column") for r in rows if r.get("role") and r["role"].lower() not in ROLES]
    if bad_role:
        problems.append(f"role must be one of {', '.join(ROLES)}: {_listing(bad_role)}")
    targets = [r.get("column") for r in rows if r.get("role", "").lower() == "target"]
    if targets != [target]:
        problems.append(f"data_dictionary.csv must mark exactly one target row, {target} "
                        f"(found: {_listing(targets) or 'none'})")
    unavailable = [r.get("column") for r in rows if r.get("role", "").lower() == "feature"
                   and r.get("available_at_prediction", "").lower() == "no"]
    if unavailable:
        problems.append(f"features not available at prediction time: {_listing(unavailable)} - mark them leak or drop")
    weak = []
    for r in rows:
        stats = columns.get(r.get("column"))
        if r.get("figure_note") and stats is not None:
            why = note_problem(r["figure_note"], stats)
            if why:
                weak.append(f"{r.get('column')} ({why})")
    if weak:
        problems.append("figure notes that are not real explanations: " + _listing(weak))
    groups: dict[str, list[str]] = {}
    for r in rows:
        key = " ".join(r.get("figure_note", "").lower().split())
        if key:
            groups.setdefault(key, []).append(r.get("column", ""))
    shared = [" = ".join(cols) for cols in groups.values() if len(cols) > 1]
    if shared:
        problems.append("columns share the same figure note: " + _listing(shared))
    return problems


def _pages_problems(state_dir: Path, columns: dict) -> list[str]:
    pages = _json(state_dir / PAGES_FILE)
    if not isinstance(pages, dict):
        return ["figures/pages.json missing: run guard.variable_figures(df, target=...)"]
    on_pages = {c for cols in pages.values() for c in cols}
    problems = []
    absent = [c for c in columns if c not in on_pages]
    if absent:
        problems.append(f"columns not on any figure page: {_listing(absent)}")
    missing_files = [p for p in pages if not (state_dir / "figures" / p).is_file()]
    if missing_files:
        problems.append(f"pages listed in pages.json but not on disk: {_listing(missing_files)}")
    return problems


def _missingness_problems(state_dir: Path, columns: dict, stage: str) -> list[str]:
    rows = _csv_rows(state_dir / MISSINGNESS_FILE)
    if rows is None:
        return ["missingness.csv missing: run guard.missingness(df)"]
    gappy = {name: info for name, info in columns.items() if (info.get("missing") or 0) > 0}
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r.get("column", ""), []).append(r)
    problems = []
    unlisted = [c for c in gappy if len(by_name.get(c, [])) != 1]
    if unlisted:
        problems.append(f"missingness.csv must list each column with gaps once: {_listing(unlisted)}")
    listed = {c: by_name[c][0] for c in gappy if len(by_name.get(c, [])) == 1}
    wrong = [c for c, r in listed.items() if _int(r.get("n_missing")) != gappy[c]["missing"]]
    if wrong:
        problems.append(f"missingness.csv counts differ from the profile for {_listing(wrong)}: re-run guard.missingness(df)")
    if stage == "A":
        bad = [c for c, r in listed.items() if r.get("mechanism", "").lower() not in MECHANISMS or not r.get("evidence")]
        if bad:
            problems.append("missingness.csv needs a likely cause (random, depends-on-other-data, depends-on-itself, "
                            f"structural, unknown) and evidence for {_listing(bad)}")
        if gappy and not (state_dir / "figures" / "01_missing_pattern.png").is_file():
            problems.append("figures/01_missing_pattern.png missing: run guard.missingness(df)")
    else:
        bad = [c for c, r in listed.items() if not _valid_combo(r.get("decision", ""), DECISIONS) or not r.get("reason")]
        if bad:
            problems.append(f"missingness.csv needs a decision ({', '.join(DECISIONS)}) and a reason for {_listing(bad)}")
    return problems


def _sections(text: str) -> dict[str, str]:
    """Markdown ``## `` sections -> their text with HTML comments removed."""
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out[current] = ""
        elif current is not None:
            out[current] += line + "\n"
    return {name: HTML_COMMENT.sub("", body) for name, body in out.items()}


def _class_counts(block: str) -> dict[str, int]:
    for line in block.splitlines():
        if line.startswith("- classes:"):
            counts = {}
            for item in line[len("- classes:"):].split(";"):
                match = CLASS_ITEM.match(item.strip())
                if match:
                    counts[match.group(1)] = int(match.group(2).replace(",", ""))
            return counts
    return {}


def _label_audit_problems(state_dir: Path, profile: dict, dictionary: list[dict] | None) -> list[str]:
    path = state_dir / LABEL_AUDIT_FILE
    if not path.is_file():
        return ["label_audit.md missing: run guard.label_audit(df, target=...) and write every section"]
    text = _read_text(path)
    problems = []
    start, end = text.find(COMPUTED_START), text.find(COMPUTED_END)
    if start == -1 or end == -1:
        problems.append("label_audit.md lost its computed block: re-run guard.label_audit(df, target=...)")
    else:
        block = text[start:end]
        target = profile.get("target") or {}
        if target.get("task") == "regression":
            if "- target:" not in block:
                problems.append("label_audit.md lost its target line: re-run guard.label_audit(df, target=...)")
        else:
            expected = {str(k): int(v) for k, v in (target.get("class_counts") or {}).items()}
            if _class_counts(block) != expected:
                problems.append("label_audit.md class counts differ from the profile: re-run guard.label_audit(df, target=...)")
    sections = _sections(text)
    thin = [name for name in LABEL_SECTIONS if len(" ".join(sections.get(name, "").split())) < MIN_NOTE_CHARS]
    if thin:
        problems.append(f"label_audit.md sections need at least {MIN_NOTE_CHARS} characters of writing: {_listing(thin)}")
    roles = {r.get("column"): r.get("role", "").lower() for r in (dictionary or [])}
    leak_text = sections.get("Leakage check", "").lower()
    kept = sorted({s.get("column") for s in profile.get("leakage_suspects") or []
                   if roles.get(s.get("column")) in ("feature", "id")})
    unnamed = [c for c in kept if str(c).lower() not in leak_text]
    if unnamed:
        problems.append("label_audit.md Leakage check must discuss each leakage suspect kept as a feature or id: "
                        f"{_listing(unnamed)}")
    return problems


def gate_a_problems(state_dir: Path, run_text: str) -> list[str]:
    """Everything Phase 1 still lacks. A missing profile is reported by the v0.4 check, not here."""
    raw = _json(state_dir / PROFILE_FILE)
    if raw is None:
        return []
    profile = _v2_profile(state_dir)
    if profile is None:
        return ["data_profile.json is from an older guard: re-run guard.profile()"]
    columns = profile.get("columns") or {}
    dictionary = _csv_rows(state_dir / DICTIONARY_FILE)
    return (_source_problems(state_dir, profile, run_text) + _profile_problems(profile)
            + _dictionary_problems(dictionary, profile) + _pages_problems(state_dir, columns)
            + _missingness_problems(state_dir, columns, stage="A")
            + _label_audit_problems(state_dir, profile, dictionary))


# ------------------------------------------------------------------ Gates B and C
def _split_record_problems(state_dir: Path, filename: str, split: str, key: str, how: str) -> list[str]:
    record = _json(state_dir / filename)
    if not isinstance(record, dict):
        return [f"{filename} missing: run {how}"]
    state = _json(state_dir / STATE_FILE) or {}
    on = record.get("computed_on") or {}
    if on.get("split") != split or key not in state or on.get("n_rows") != state.get(key):
        return [f"{filename} was not computed on the {split} split ({on.get('n_rows')} rows, the {split} split has "
                f"{state.get(key)}): run {how}"]
    return []


def _plan_problems(state_dir: Path, profile: dict) -> list[str]:
    roles = {r.get("column"): r.get("role", "").lower() for r in _csv_rows(state_dir / DICTIONARY_FILE) or []}
    target = str((profile.get("target") or {}).get("name"))
    needed = [c for c, info in (profile.get("columns") or {}).items()
              if info.get("kind") in ("categorical", "bool") and c != target and roles.get(c) == "feature"]
    if not needed:
        return []
    rows = _csv_rows(state_dir / PLAN_FILE)
    if rows is None:
        return ["categorical_plan.csv missing: run guard.categorical_plan(df)"]
    by_name = {r.get("column"): r for r in rows}
    unplanned = [c for c in needed if c not in by_name or not by_name[c].get("reason")
                 or not _valid_combo(by_name[c].get("encoding", ""), ENCODINGS)]
    if unplanned:
        return [f"categorical_plan.csv needs an encoding ({', '.join(ENCODINGS)}) and a reason for {_listing(unplanned)}"]
    return []


def gate_b_problems(state_dir: Path, run_text: str) -> list[str]:
    """Everything Phase 2 still lacks."""
    problems = []
    cleaning = _csv_rows(state_dir / CLEANING_FILE, key="rule") or []
    if not CLEANING_NONE.search(run_text) and not any(r.get("rule") and r.get("reason") for r in cleaning):
        problems.append("no cleaning recorded: log each rule with guard.log_cleaning(...), or record "
                        "'- Cleaning: none needed - reason: ...'")
    profile = _v2_profile(state_dir)
    if profile is None:
        problems.append("data_profile.json (schema 2) missing: run guard.profile() at step 1")
    else:
        problems += _missingness_problems(state_dir, profile.get("columns") or {}, stage="B")
        problems += _plan_problems(state_dir, profile)
    problems += _split_record_problems(state_dir, ASSOCIATIONS_FILE, "train", "n_train",
                                       "guard.train_associations(train, target=...)")
    for name in ("07_correlations_train.png", "07_feature_target_train.png"):
        if not (state_dir / "figures" / name).is_file():
            problems.append(f"figures/{name} missing: run guard.train_associations(train, target=...)")
    return problems


def gate_c_problems(state_dir: Path, run_text: str) -> list[str]:
    """Everything Phase 3 still lacks."""
    problems = []
    state = _json(state_dir / STATE_FILE) or {}
    runs = _jsonl(state_dir / TRAINING_FILE)
    if not any(r.get("full") is True and "n_train" in state and r.get("n_rows") == state["n_train"] for r in runs):
        problems.append("no full-data training run on record: train on the whole training split with guard.fit(...) "
                        "or record it with guard.record_training(...)")
    problems += _split_record_problems(state_dir, IMPORTANCE_FILE, "validation", "n_val",
                                       "guard.importance(model.predict, val, target=..., metric_fn=...)")
    figures = state_dir / "figures"
    if not (figures / "12_permutation_importance.png").is_file():
        problems.append("figures/12_permutation_importance.png missing: run guard.importance(...)")
    if not list(figures.glob("12_attribution_*.png")):
        problems.append("no figures/12_attribution_*.png: draw one attribution figure (SHAP, partial dependence, or "
                        "coefficients) and save it with guard.fig(12, 'attribution_<name>', ...)")
    return problems
