#!/usr/bin/env python3
"""PreToolUse hook: no model training until the pipeline gates say so.

Claude Code runs this before each matched tool call and pipes the call to stdin as
JSON. Every string in ``tool_input`` is scanned for fitting/training calls - and, when a
Bash command runs a Python file, module, or notebook, that file's code too - and the
call is denied unless ``ml_pipeline/PIPELINE.md`` records the gate that unlocks it:

    no gate approved  ->  nothing is fit; the data is still being understood
    Gate A approved   ->  preprocessing fits (scalers, encoders, imputers) allowed
    Gate B approved   ->  model training allowed
    Gate C approved   ->  evaluating on the test split allowed (once, at step 14)
    always            ->  fitting on the test split is denied

Only strict ledger lines count - ``- Gate X: approved ...`` and ``- Override: Gate X - ...`` -
and only in the newest run: everything after the last ``## Run:`` heading, or the whole
file when there is none. For each gate the last such line decides, so a later
``- Gate B: hold`` locks it again.

The hook fails open: an unexpected error allows the call and reports the error,
so a bug here can never lock anyone out of their own tools. The user (never the agent)
sets ``ML_PIPELINE_ENFORCE=0`` to switch it off for projects that are not ML pipelines.
Stdlib only; no subprocesses, no network.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

PIPELINE_REL = Path("ml_pipeline") / "PIPELINE.md"
SEARCH_UP = 4  # look in cwd and this many parent directories
PROSE_SUFFIXES = {".md", ".markdown", ".rst"}
# The plugin runs in every project; a fit/step/train call in these languages is not model training.
OTHER_LANGUAGE_SUFFIXES = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".html", ".htm", ".css", ".scss",
    ".java", ".kt", ".scala", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".rb",
    ".php", ".dart", ".lua", ".sql",
}
SKIP_KEYS = {"old_string", "file_path", "path", "notebook_path", "description", "url", "pattern", "glob", "cwd"}
SNIPPET = 80
MAX_SCRIPT_BYTES = 2_000_000  # read at most this much of a file a command executes

# ------------------------------------------------------------------ detection
FIT_NAME = r"fit(?:_transform|_predict)?"
RECEIVER = r"\b([A-Za-z_]\w*)\s*(?:\([^()]*\))?"
# `<receiver>.fit(` where receiver is `name` or `Name(...)`; also fit_transform / fit_predict, spaced dots too.
FIT_CALL = re.compile(r"(?:" + RECEIVER + r")?\s*\.\s*" + FIT_NAME + r"\s*\(")
# A fit method bound now and called later: `m = model.fit`, `partial(model.fit, ...)`.
FIT_REF = re.compile(r"(?:=|[(,])\s*" + RECEIVER + r"\s*\.\s*" + FIT_NAME + r"\b(?!\s*\()")
# A fit method looked up by name: `getattr(model, "fit")`.
GETATTR_FIT = re.compile(r"\bgetattr\s*\(\s*([A-Za-z_]\w*)[^,()]*,\s*[\"']" + FIT_NAME + r"[\"']")
# APIs that only ever train models. The sklearn helpers fit a fresh estimator per fold or point;
# permutation_importance needs a fitted model, so it belongs after the baseline too.
TRAIN_API = re.compile(
    r"\b(?:xgb|xgboost|lgb|lightgbm)\s*\.\s*train\s*\(|\btrainer\s*\.\s*train\s*\(|(?<!class )\b\w*Trainer\s*\(|"
    r"\.fit_generator\s*\(|\.fit_one_cycle\s*\(|\bloss\.backward\s*\(|\boptimizer\.step\s*\(|"
    r"\b(?:cross_val_score|cross_validate|cross_val_predict|learning_curve|validation_curve|"
    r"permutation_importance)\s*\("
)
# A gradient step under any names: a backward pass plus an optimizer step in the same code.
BACKWARD = re.compile(r"\.\s*backward\s*\(")
OPT_STEP = re.compile(r"\.\s*step\s*\(")
# Evaluating a model: predict/score methods and the usual sklearn metric functions.
EVAL_CALL = re.compile(
    r"(?:\.(?:predict|predict_proba|decision_function|score|evaluate)|"
    r"\b(?:accuracy_score|balanced_accuracy_score|roc_auc_score|average_precision_score|f1_score|"
    r"precision_score|recall_score|log_loss|mean_squared_error|mean_absolute_error|"
    r"root_mean_squared_error|r2_score|classification_report|confusion_matrix|cross_val_score|"
    r"cross_validate))\s*\("
)
ESTIMATOR = re.compile(
    r"\b\w*(?:Classifier|Regressor|Regression|Ridge|Lasso|ElasticNet|SVC|SVR|NB|KNeighbors|MLP|"
    r"XGB|LGBM|CatBoost|DecisionTree|RandomForest|GradientBoosting|ExtraTrees|AdaBoost|"
    r"Perceptron|SGD|GaussianProcess|Sequential|keras|Booster)\w*\b"
)
TRANSFORMER = re.compile(
    r"\b\w*(?:Scaler|Imputer|Encoder|Vectorizer|PCA|Normalizer|Binarizer|Discretizer|"
    r"Transformer|SelectKBest|TruncatedSVD|PolynomialFeatures)\w*\b"
)
TRANSFORMER_VAR = re.compile(
    r"^(?:scaler|scale|std|mms|imputer|imp|encoder|enc|ohe|ordinal|vectorizer|vec|tfidf|pca|"
    r"svd|preproc|preprocessor|prep|ct|transformer|selector|poly|binner|pipe_prep)\w*$",
    re.I,
)
TEST_SPLIT = re.compile(r"\b(?:\w+_test|test_\w+|test)\b")
COMMENT = re.compile(r"#[^\n]*")

# ------------------------------------------------------------------ following execution
PYTHON_EXE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|ipython3?|pypy3?)$")
PYTHON_OPTS_WITH_VALUE = {"-W", "-X", "--check-hash-based-pycs"}
SHELL_SEPARATOR = re.compile(r"&&|\|\||[;|\n]")

# ------------------------------------------------------------------ gate state
RUN_HEAD = re.compile(r"^[ \t]*##[ \t]+Run[ \t]*:", re.I | re.M)
GATE_RECORD = re.compile(r"^[ \t]*-[ \t]*Gate[ \t]+([A-D])[ \t]*:[ \t]*(\S*)", re.I)
OVERRIDE_GATE = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*Gate[ \t]+([A-D])\b", re.I)
APPROVED_WORD = re.compile(r"approved\b", re.I)
# Recording a gate: the approval or override shape at a line start or right after a quote,
# as in `echo '- Gate A: approved ...' >> ml_pipeline/PIPELINE.md`.
RECORDING = re.compile(
    r"(?:^|(?<=[\s'\"]))-[ \t]*(?:Gate[ \t]+([A-D])[ \t]*:[ \t]*approved\b|Override[ \t]*:[ \t]*Gate[ \t]+([A-D])\b)",
    re.I | re.M,
)
# A shell command that changes the ledger (a read-only grep of it does not).
LEDGER_WRITE = re.compile(
    r">\s*[\"']?[^\s;|&<>]*PIPELINE\.md|\btee\b[^;|&\n]*PIPELINE\.md|"
    r"\b(?:sed|perl)\b[^;|&\n]*\s-i[^;|&\n]*PIPELINE\.md|\b(?:cp|mv|install|dd|truncate)\b[^;|&\n]*PIPELINE\.md|"
    r"\b(?:open|write_text|write|writelines)\s*\("
)

# ------------------------------------------------------------------ self-disable
SWITCH = r"ML_PIPELINE_(?:ENFORCE|FILE)"
# Assigning the switch (`X=0`, `"X": "0"`, `environ["X"] = ...`, `setenv X`), never reading it
# (`$X`, `${X:-1}`, `"$X" == "0"`).
SELF_DISABLE = re.compile(
    SWITCH + r"[\"']?\]?\s*(?:=(?!=)|:(?!-))|\b(?:setenv|setdefault|putenv)\b\W*[\"']?" + SWITCH
)

# ------------------------------------------------------------------ messages
NO_PIPELINE_MSG = (
    "ml-pipeline: model training detected ({0!r}) but there is no ml_pipeline/PIPELINE.md. "
    "Training starts at step 1, not step 10 - run the ml-pipeline skill: inspect the data, do EDA, "
    "define the prediction problem, and get the user's approval at Gate A. "
    "Not an ML project? Ask the user to switch enforcement off."
)
GATE_A_MSG = (
    "ml-pipeline: nothing is fit before Gate A ({0!r}). Finish steps 1-3 (data inspection, EDA, "
    "prediction-problem contract), get the user's explicit approval, and record it in "
    "ml_pipeline/PIPELINE.md as `- Gate A: approved YYYY-MM-DD`."
)
GATE_B_MSG = (
    "ml-pipeline: model training is locked until Gate B ({0!r}). Finish steps 4-8 (cleaning, "
    "engineering, split, features, preprocessing), get the user's explicit approval, and record "
    "`- Gate B: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md. If the user explicitly chose to skip "
    "ahead, record `- Override: Gate B - user approved training YYYY-MM-DD - reason: ...` instead."
)
GATE_C_MSG = (
    "ml-pipeline: evaluating on the test split is locked until Gate C ({0!r}). The test set is touched "
    "exactly once, at step 14, after the user approves Gate C - tune and select on validation only. "
    "Record `- Gate C: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md first, or, if the user explicitly "
    "chose otherwise, `- Override: Gate C - user approved evaluating on the test set YYYY-MM-DD - reason: ...`."
)
TEST_MSG = (
    "ml-pipeline: this fits on the test split ({0!r}). The test set is touched exactly once, at "
    "step 14, and is never fit on. Fit on the training split only."
)
SELF_DISABLE_MSG = (
    "ml-pipeline: switching enforcement off, or pointing it at a different ledger, is the user's decision - "
    "an agent may not change that setting. If this project is not an ML pipeline, ask the user to switch "
    "enforcement off themselves."
)

# ------------------------------------------------------------------ gate recording
PROFILE_FILE = "data_profile.json"
FIGURES_DIR = "figures"
MIN_PNG_BYTES = 1024
GATE_REQUIREMENTS = {
    "A": {"profile": True, "figures": {"01": 1, "02": 2}},
    "B": {"figures": {"04": 1}, "rationale": True},
    "C": {"figures": {"12": 1}},
    "D": {"figures": {"13": 1}, "final_test_line": True},
}
RATIONALE_HEAD = re.compile(r"^\s*Model rationale:\s*$", re.I | re.M)
RATIONALE_BULLETS = ("traits", "baseline", "candidates", "ruled out", "metric")
LEDGER = re.compile(r"^\s*-\s*Figure\s+(\S+\.png):\s*\S", re.I | re.M)
FINAL_TEST_LINE = re.compile(r"^\s*-\s*Step 14 final test:", re.I | re.M)
GATE_RECORD_MSG = "ml-pipeline: Gate {0} can't be recorded: {1}."


def approved_gates_in(text: str) -> list[str]:
    """Gate letters that ``text`` records: a strict approval or override line for that gate."""
    return [(m.group(1) or m.group(2)).upper() for m in RECORDING.finditer(text)]


def rationale_missing(text: str) -> list[str]:
    """Bullets absent from the Model rationale block; all five when the block itself is absent."""
    head = RATIONALE_HEAD.search(text)
    if not head:
        return list(RATIONALE_BULLETS)
    window = "\n".join(text[head.end():].splitlines()[:12])
    return [b for b in RATIONALE_BULLETS if not re.search(r"^\s*-\s*" + re.escape(b) + r"\s*:", window, re.I | re.M)]


def gate_problems(gate: str, state_dir: Path, ledger_text: str) -> list[str]:
    """Everything still missing before ``gate`` may be recorded, in the words the agent will read."""
    required = GATE_REQUIREMENTS.get(gate, {})
    problems: list[str] = []
    if required.get("profile") and not (state_dir / PROFILE_FILE).is_file():
        problems.append(f"{PROFILE_FILE} missing")
    figures_dir = state_dir / FIGURES_DIR
    figures = sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else []
    for prefix, need in required.get("figures", {}).items():
        have = [p for p in figures if p.name.startswith(prefix + "_")]
        if len(have) < need:
            problems.append(f"{FIGURES_DIR}/{prefix}_*.png {len(have)} found (need {need})")
    ledgered = {m.group(1) for m in LEDGER.finditer(ledger_text)}
    for png in figures:
        size = png.stat().st_size
        if size < MIN_PNG_BYTES:
            problems.append(f"{FIGURES_DIR}/{png.name} is {size} bytes (empty?)")
        if png.name not in ledgered:
            problems.append(f"{png.name} has no '- Figure {png.name}:' line")
    if required.get("rationale"):
        missing = rationale_missing(ledger_text)
        if missing:
            problems.append("Model rationale block missing lines: " + ", ".join(missing))
    if required.get("final_test_line") and not FINAL_TEST_LINE.search(ledger_text):
        problems.append("no '- Step 14 final test:' line")
    return problems


def _state_dir_for(tool_input: dict, cwd: str) -> Path:
    """The ml_pipeline/ directory this call concerns: next to the edited PIPELINE.md, else found from cwd."""
    target = str(tool_input.get("file_path") or "")
    if target.endswith("PIPELINE.md"):
        return Path(target).resolve().parent
    found = find_pipeline(cwd)
    return found.parent if found else Path(cwd or ".").resolve() / "ml_pipeline"


def _target(tool_input: dict) -> str:
    return str(tool_input.get("file_path") or tool_input.get("notebook_path") or tool_input.get("path") or "")


def _writes_ledger(tool_input: dict) -> bool:
    """True when this call edits a PIPELINE.md: a file tool aimed at it, or a shell command writing to it."""
    target = _target(tool_input)
    if target:
        return Path(target).name == "PIPELINE.md"
    command = tool_input.get("command")
    return isinstance(command, str) and "PIPELINE.md" in command and bool(LEDGER_WRITE.search(command))


# ------------------------------------------------------------------ red flags (need a profile)
RED_FLAGS = [
    ("neural-net-small-data", lambda t: bool(t.get("small_data")),
     re.compile(r"\b(?:MLPClassifier|MLPRegressor|Sequential|keras|torch\.nn|nn\.Module|tensorflow|tf\.keras)\b"),
     "a neural net on {n_rows:,} rows overfits and hides it - start with a regularized linear model or gradient boosting"),
    ("random-split-temporal", lambda t: bool(t.get("has_datetime")),
     re.compile(r"\b(?:train_test_split|KFold|StratifiedKFold|ShuffleSplit|RepeatedKFold)\s*\("),
     "the data has a datetime column, so a random split leaks the future into training - use guard.split(time_col=...) or TimeSeriesSplit"),
    ("group-split", lambda t: bool(t.get("has_groups")),
     re.compile(r"\b(?:train_test_split|KFold|StratifiedKFold|ShuffleSplit|RepeatedKFold|TimeSeriesSplit)\s*\("),
     "rows share an entity, so any split that is not group-aware leaks - use guard.split(group_col=...) or GroupKFold"),
    ("accuracy-imbalanced", lambda t: bool(t.get("imbalanced")),
     re.compile(r"scoring\s*=\s*[\"']accuracy[\"']"),
     "the minority class is {minority_pct} of rows, so accuracy rewards ignoring it - select on average_precision, f1, or balanced_accuracy"),
]
RESAMPLE = re.compile(r"fit_resample\s*\(([^)]*)\)")
OVERRIDE_FLAG = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*red flag[ \t]+([a-z-]+)", re.I | re.M)
RED_FLAG_MSG = "ml-pipeline: red flag {0}: {1}. Override with '- Override: red flag {0} - ...' in ml_pipeline/PIPELINE.md."


def _load_profile(state_dir: Path) -> dict | None:
    path = state_dir / PROFILE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None


def overridden_flags(markdown: str) -> set[str]:
    """Red flags the newest run overrides with a strict ``- Override: red flag <name> - ...`` line."""
    return {m.group(1).lower() for m in OVERRIDE_FLAG.finditer(current_run(markdown))}


def red_flag(code: str, traits: dict, overrides: set[str]) -> str | None:
    """The first red flag the code trips given the profile's traits, or None."""
    values = {"n_rows": int(traits.get("n_rows") or 0),
              "minority_pct": f"{float(traits.get('minority_frac') or 0):.1%}"}
    found = analyse(code)
    trains = bool(found["model"] or found["transformer"])
    for name, applies, pattern, why in RED_FLAGS:
        if name in overrides or not applies(traits) or not pattern.search(code):
            continue
        if name == "neural-net-small-data" and not trains:
            continue
        return RED_FLAG_MSG.format(name, why.format(**values))
    if "resample-before-split" not in overrides:
        for match in RESAMPLE.finditer(code):
            if "train" not in match.group(1).lower():
                return RED_FLAG_MSG.format("resample-before-split",
                                           f"resampling must fit on the training split only ({match.group(0)[:60]}) - split first, then fit_resample(X_train, y_train)")
    return None


def _strings(obj, out: list[str] | None = None) -> list[str]:
    """Every string value in a tool_input tree, minus keys that describe rather than contain code."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in SKIP_KEYS:
                _strings(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _strings(value, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


# ------------------------------------------------------------------ following execution
def _python_target(args: list[str]) -> tuple[str, str] | None:
    """What ``python <args>`` runs: ('module', name) or ('file', path); None for -c, stdin, or a REPL."""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-m":
            return ("module", args[i + 1]) if i + 1 < len(args) else None
        if arg in ("-c", "-"):
            return None
        if arg.startswith("-"):
            i += 2 if arg in PYTHON_OPTS_WITH_VALUE else 1
            continue
        return ("file", arg)
    return None


def _exec_targets(tokens: list[str]) -> list[tuple[str, str]]:
    """Python files and modules one shell command segment executes."""
    targets: list[tuple[str, str]] = []
    for i, token in enumerate(tokens):
        name, rest = os.path.basename(token), tokens[i + 1:]
        files = [("file", t) for t in rest if t.lower().endswith((".py", ".ipynb"))]
        if PYTHON_EXE.match(name):
            target = _python_target(rest)
            targets += [target] if target else []
        elif name == "uv" and rest[:1] == ["run"]:
            module = next((rest[j + 1] for j, t in enumerate(rest[:-1]) if t in ("-m", "--module")), None)
            targets += [("module", module)] if module else files
        elif name == "marimo" or name == "papermill":
            targets += files
        elif name == "jupyter" and (rest[:1] in (["execute"], ["run"]) or "--execute" in rest):
            targets += files
    return targets


def _notebook_code(text: str) -> str:
    """The code cells of an .ipynb; the raw text when it is not a readable notebook."""
    try:
        cells = json.loads(text).get("cells") or []
    except (ValueError, AttributeError):
        return text
    code = []
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            source = cell.get("source") or ""
            code.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(code)


def _read_script(kind: str, name: str, base: Path) -> str:
    if kind == "module":
        rel = Path(*name.split("."))
        candidates = [base / (str(rel) + ".py"), base / rel / "__main__.py"]
    else:
        path = Path(name).expanduser()
        candidates = [path if path.is_absolute() else base / path]
    for path in candidates:
        if path.is_file():
            with path.open("rb") as handle:
                text = handle.read(MAX_SCRIPT_BYTES).decode("utf-8", errors="replace")
            return _notebook_code(text) if path.suffix.lower() == ".ipynb" else text
    return ""


def executed_code(command: str, cwd: str) -> str:
    """Source of every Python file, module, or notebook ``command`` runs, resolved against cwd and ``cd``."""
    base = Path(cwd or ".")
    chunks: list[str] = []
    for segment in SHELL_SEPARATOR.split(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        if tokens[:1] == ["cd"] and len(tokens) > 1:
            base = base / Path(tokens[1]).expanduser()
            continue
        for kind, name in _exec_targets(tokens):
            try:
                chunks.append(_read_script(kind, name, base))
            except OSError:  # one unreadable file must not fail the whole hook open
                continue
    return "\n".join(chunk for chunk in chunks if chunk)


# ------------------------------------------------------------------ analysis
def _args(text: str, start: int, limit: int = 400) -> str:
    """The argument list starting just after an opening paren, up to its matching close."""
    depth, i = 1, start
    while i < len(text) and i - start < limit:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start:i]


def _classify(receiver: str, before: str, args: str) -> str:
    """'transformer' for preprocessing fits, 'model' for anything that looks like training.

    Unknown receivers (``model.fit``, ``clf.fit``) count as models: the conservative
    reading is the one that keeps the test set honest.
    """
    if TRANSFORMER_VAR.match(receiver) or TRANSFORMER.search(receiver):
        return "transformer"
    if ESTIMATOR.search(receiver):
        return "model"
    context = before + args
    if TRANSFORMER.search(context) and not ESTIMATOR.search(context):
        return "transformer"
    return "model"


def analyse(text: str) -> dict[str, str | None]:
    """First matching snippet for each kind of finding, or None."""
    code = COMMENT.sub("", text)
    found: dict[str, str | None] = {"model": None, "transformer": None, "fit_on_test": None, "eval_on_test": None}
    api = TRAIN_API.search(code)
    if api:
        found["model"] = api.group(0)
    backward = BACKWARD.search(code)
    if found["model"] is None and backward and OPT_STEP.search(code):
        line_start = code.rfind("\n", 0, backward.start()) + 1
        found["model"] = code[line_start: backward.end() + 1][:SNIPPET]
    for match in FIT_CALL.finditer(code):
        args = _args(code, match.end())
        snippet = code[match.start(): match.end() + len(args) + 1].strip()[:SNIPPET]
        if TEST_SPLIT.search(args) and found["fit_on_test"] is None:
            found["fit_on_test"] = snippet
        kind = _classify(match.group(1) or "", code[max(0, match.start() - 200): match.start()], args)
        if found[kind] is None:
            found[kind] = snippet
    for match in (*FIT_REF.finditer(code), *GETATTR_FIT.finditer(code)):
        kind = _classify(match.group(1) or "", code[max(0, match.start() - 200): match.start()], "")
        if found[kind] is None:
            found[kind] = match.group(0).lstrip("=(, \t")[:SNIPPET]
    for match in EVAL_CALL.finditer(code):
        args = _args(code, match.end())
        if TEST_SPLIT.search(args) and found["eval_on_test"] is None:
            found["eval_on_test"] = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
    return found


# ------------------------------------------------------------------ ledger
def current_run(markdown: str) -> str:
    """The newest run: everything from the last ``## Run:`` heading on, or the whole file when there is none."""
    heads = list(RUN_HEAD.finditer(markdown))
    return markdown[heads[-1].start():] if heads else markdown


def gate_state(markdown: str) -> dict[str, bool]:
    """Which gates the newest run approves. For each gate the last strict line decides; nothing else counts."""
    last: dict[str, str] = {}
    for line in current_run(markdown).splitlines():
        gate = GATE_RECORD.match(line)
        if gate:
            last[gate.group(1).upper()] = "approved" if APPROVED_WORD.match(gate.group(2)) else "other"
            continue
        override = OVERRIDE_GATE.match(line)
        if override:
            last[override.group(1).upper()] = "override"
    state = {gate: last.get(gate) == "approved" for gate in "ABCD"}
    state["override"] = last.get("B") == "override"
    state["override_c"] = last.get("C") == "override"
    return state


def find_pipeline(cwd: str) -> Path | None:
    override = os.environ.get("ML_PIPELINE_FILE")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    here = Path(cwd or ".").resolve()
    for _ in range(SEARCH_UP + 1):
        candidate = here / PIPELINE_REL
        if candidate.is_file():
            return candidate
        if here.parent == here:
            break
        here = here.parent
    return None


def decide(found: dict[str, str | None], gates: dict[str, bool] | None) -> str | None:
    """A denial reason, or None to allow. ``gates`` is None when no PIPELINE.md exists."""
    if found["fit_on_test"]:
        return TEST_MSG.format(found["fit_on_test"])
    if found["eval_on_test"] and not (gates and (gates["C"] or gates["override_c"])):
        return GATE_C_MSG.format(found["eval_on_test"])
    if not (found["model"] or found["transformer"]):
        return None
    if gates is None:
        return NO_PIPELINE_MSG.format(found["model"]) if found["model"] else None
    if gates["B"] or gates["override"]:
        return None
    if gates["A"]:
        return GATE_B_MSG.format(found["model"]) if found["model"] else None
    return GATE_A_MSG.format(found["model"] or found["transformer"])


def _emit(reason: str | None) -> int:
    if reason is None:
        print("{}")
    else:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
            "systemMessage": reason,
        }))
    return 0


def main() -> int:
    try:
        if os.environ.get("ML_PIPELINE_ENFORCE", "1").strip().lower() in {"0", "false", "off", "no"}:
            return _emit(None)
        payload = json.load(sys.stdin)
        tool_input = payload.get("tool_input") or {}
        cwd = payload.get("cwd") or os.getcwd()
        text = "\n".join(_strings(tool_input))
        target = _target(tool_input)
        suffix = Path(target).suffix.lower() if target else ""
        # 1. Only the user switches enforcement off or points it at another ledger.
        if suffix not in PROSE_SUFFIXES and SELF_DISABLE.search(text):
            return _emit(SELF_DISABLE_MSG)
        # 2. Recording a gate in the ledger: that gate's artifacts must already exist.
        if _writes_ledger(tool_input):
            for gate in approved_gates_in(text):
                state_dir = _state_dir_for(tool_input, cwd)
                pipeline_md = state_dir / "PIPELINE.md"
                existing = pipeline_md.read_text(encoding="utf-8", errors="replace") if pipeline_md.is_file() else ""
                problems = gate_problems(gate, state_dir, existing + "\n" + text)
                if problems:
                    return _emit(GATE_RECORD_MSG.format(gate, "; ".join(problems)))
        # 3. Prose and other languages are not Python.
        if suffix in PROSE_SUFFIXES or suffix in OTHER_LANGUAGE_SUFFIXES:
            return _emit(None)
        # 4. A command that runs a file runs that file's code.
        command = tool_input.get("command")
        code = text + "\n" + executed_code(command, cwd) if isinstance(command, str) else text
        # 5. Red flags: wrong applications the data profile can prove.
        state_dir = _state_dir_for(tool_input, cwd)
        profile = _load_profile(state_dir)
        if profile is not None:
            pipeline_md = state_dir / "PIPELINE.md"
            markdown = pipeline_md.read_text(encoding="utf-8", errors="replace") if pipeline_md.is_file() else ""
            reason = red_flag(COMMENT.sub("", code), profile.get("traits") or {}, overridden_flags(markdown))
            if reason:
                return _emit(reason)
        # 6. Training and test-set evaluation need their gates.
        found = analyse(code)
        if not any(found.values()):
            return _emit(None)
        pipeline = find_pipeline(cwd)
        gates = gate_state(pipeline.read_text(encoding="utf-8", errors="replace")) if pipeline else None
        return _emit(decide(found, gates))
    except Exception as exc:  # fail open: never lock the user out because of us
        print(json.dumps({"systemMessage": f"ml-pipeline hook error (call allowed): {exc}"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
