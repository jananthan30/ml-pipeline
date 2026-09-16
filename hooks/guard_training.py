#!/usr/bin/env python3
"""PreToolUse hook: no model training until the pipeline gates say so.

Claude Code runs this before each matched tool call and pipes the call to stdin as
JSON. Every string in ``tool_input`` is scanned for fitting/training calls, and the
call is denied unless ``ml_pipeline/PIPELINE.md`` records the gate that unlocks it:

    no gate approved  ->  nothing is fit; the data is still being understood
    Gate A approved   ->  preprocessing fits (scalers, encoders, imputers) allowed
    Gate B approved   ->  model training allowed
    Gate C approved   ->  evaluating on the test split allowed (once, at step 14)
    always            ->  fitting on the test split is denied

The hook fails open: an unexpected error allows the call and reports the error,
so a bug here can never lock anyone out of their own tools. Set
``ML_PIPELINE_ENFORCE=0`` to switch it off for projects that are not ML pipelines.
Stdlib only; no subprocesses, no network.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PIPELINE_REL = Path("ml_pipeline") / "PIPELINE.md"
SEARCH_UP = 4  # look in cwd and this many parent directories
DOC_SUFFIXES = {".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
SKIP_KEYS = {"old_string", "file_path", "path", "notebook_path", "description", "url", "pattern", "glob", "cwd"}
SNIPPET = 80

# ------------------------------------------------------------------ detection
# `<receiver>.fit(` where receiver is `name` or `Name(...)`; also fit_transform / fit_predict.
FIT_CALL = re.compile(r"(?:\b([A-Za-z_]\w*)\s*(?:\([^()]*\))?)?\.fit(?:_transform|_predict)?\s*\(")
# APIs that only ever train models.
TRAIN_API = re.compile(
    r"\b(?:xgb|xgboost|lgb|lightgbm)\.train\s*\(|\btrainer\.train\s*\(|\bTrainer\s*\(|"
    r"\.fit_generator\s*\(|\.fit_one_cycle\s*\(|\bloss\.backward\s*\(|\boptimizer\.step\s*\("
)
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

# ------------------------------------------------------------------ gate state
GATE_LINE = re.compile(r"^.*\bgate\s*[\[(]?([abcd])\b.*$", re.I | re.M)
APPROVED = re.compile(r"\b(?:approved|approved-gate|overridden|override)\b", re.I)
NEGATED = re.compile(r"\b(?:not|pending|awaiting|waiting|todo|unapproved|blocked|in progress)\b", re.I)
OVERRIDE_LINE = re.compile(r"^.*\boverrid(?:e|den)\b.*\b(?:train|training|model|models|gate\s*b)\b.*$", re.I | re.M)
OVERRIDE_C_LINE = re.compile(r"^.*\boverrid(?:e|den)\b.*\b(?:gate\s*c|final test|test set)\b.*$", re.I | re.M)

# ------------------------------------------------------------------ messages
NO_PIPELINE_MSG = (
    "ml-pipeline: model training detected ({0!r}) but there is no ml_pipeline/PIPELINE.md. "
    "Training starts at step 1, not step 10 - run the ml-pipeline skill: inspect the data, do EDA, "
    "define the prediction problem, and get the user's approval at Gate A. "
    "Not an ML project? Set ML_PIPELINE_ENFORCE=0."
)
GATE_A_MSG = (
    "ml-pipeline: nothing is fit before Gate A ({0!r}). Finish steps 1-3 (data inspection, EDA, "
    "prediction-problem contract), get the user's explicit approval, and record it in "
    "ml_pipeline/PIPELINE.md as `Gate A: approved YYYY-MM-DD`."
)
GATE_B_MSG = (
    "ml-pipeline: model training is locked until Gate B ({0!r}). Finish steps 4-8 (cleaning, "
    "engineering, split, features, preprocessing), get the user's explicit approval, and record "
    "`Gate B: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md. If the user explicitly chose to skip "
    "ahead, record `Override: Gate B - user approved training YYYY-MM-DD - reason: ...` instead."
)
GATE_C_MSG = (
    "ml-pipeline: evaluating on the test split is locked until Gate C ({0!r}). The test set is touched "
    "exactly once, at step 14, after the user approves Gate C - tune and select on validation only. "
    "Record `Gate C: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md first, or, if the user explicitly "
    "chose otherwise, `Override: Gate C - user approved evaluating on the test set YYYY-MM-DD - reason: ...`."
)
TEST_MSG = (
    "ml-pipeline: this fits on the test split ({0!r}). The test set is touched exactly once, at "
    "step 14, and is never fit on. Fit on the training split only."
)


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
    for match in FIT_CALL.finditer(code):
        args = _args(code, match.end())
        snippet = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
        if TEST_SPLIT.search(args) and found["fit_on_test"] is None:
            found["fit_on_test"] = snippet
        kind = _classify(match.group(1) or "", code[max(0, match.start() - 200): match.start()], args)
        if found[kind] is None:
            found[kind] = snippet
    for match in EVAL_CALL.finditer(code):
        args = _args(code, match.end())
        if TEST_SPLIT.search(args) and found["eval_on_test"] is None:
            found["eval_on_test"] = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
    return found


def gate_state(markdown: str) -> dict[str, bool]:
    """Which gates PIPELINE.md records as approved. A negated line never counts."""
    state = {"A": False, "B": False, "C": False, "D": False, "override": False, "override_c": False}
    for match in GATE_LINE.finditer(markdown):
        line = match.group(0)
        if APPROVED.search(line) and not NEGATED.search(line):
            state[match.group(1).upper()] = True
    for match in OVERRIDE_LINE.finditer(markdown):
        if not NEGATED.search(match.group(0)):
            state["override"] = True
    for match in OVERRIDE_C_LINE.finditer(markdown):
        if not NEGATED.search(match.group(0)):
            state["override_c"] = True
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
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if target and Path(target).suffix.lower() in DOC_SUFFIXES:
            return _emit(None)  # prose about .fit() is not a fit
        text = "\n".join(_strings(tool_input))
        if ".fit" not in text and not TRAIN_API.search(text) and not EVAL_CALL.search(text):
            return _emit(None)  # fast path: nothing to look at
        found = analyse(text)
        if not any(found.values()):
            return _emit(None)
        pipeline = find_pipeline(payload.get("cwd") or os.getcwd())
        gates = gate_state(pipeline.read_text(encoding="utf-8", errors="replace")) if pipeline else None
        return _emit(decide(found, gates))
    except Exception as exc:  # fail open: never lock the user out because of us
        print(json.dumps({"systemMessage": f"ml-pipeline hook error (call allowed): {exc}"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
