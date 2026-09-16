#!/usr/bin/env python3
"""PostToolUse hook: warn when a result or a split looks like leakage. Never blocks.

Two checks, both cheap:
  - a reported metric that is too good to be honest (accuracy / AUC / F1 / R² >= 0.98)
  - train_test_split() without stratify=, or on code that mentions dates, where a random
    split leaks the future into training
Stdlib only; fails open.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard_training import _load_profile, _state_dir_for, _strings  # noqa: E402

METRIC = re.compile(
    r"\b(?:accuracy|acc|auc|roc[_ ]?auc|f1|r2|r\^2|precision|recall)\b[^0-9\n]{0,20}"
    r"(1\.0+\b|0\.9[89]\d*|100(?:\.0+)?\s*%|9[89](?:\.\d+)?\s*%)",
    re.I,
)
SPLIT = re.compile(r"train_test_split\s*\(([^)]*)\)")
DATEY = re.compile(r"\b(?:datetime|to_datetime|timestamp|date|time_col|\w+_date|\w+_at)\b", re.I)
LARGE_N = 50_000
SLOW_MODELS = re.compile(r"\b(?:KNeighbors\w+|SVC|SVR)\b")
ONE_HOT = re.compile(r"\b(?:OneHotEncoder|get_dummies)\b")
ACCURACY = re.compile(r"\baccuracy_score\s*\(")


def trait_warnings(code: str, traits: dict) -> list[str]:
    """Smells the profile makes visible. Advisory only - the red flags in guard_training deny."""
    found: list[str] = []
    n_rows = int(traits.get("n_rows") or 0)
    if n_rows > LARGE_N and SLOW_MODELS.search(code):
        found.append(f"k-NN / kernel SVM on {n_rows:,} rows scales badly - prefer gradient boosting or a linear model.")
    high_card = traits.get("high_card_categoricals") or []
    if high_card and ONE_HOT.search(code):
        found.append(f"one-hot on high-cardinality columns ({', '.join(map(str, high_card[:5]))}) explodes width - "
                     "use target/ordinal encoding fit on train, or CatBoost.")
    if traits.get("imbalanced") and ACCURACY.search(code):
        found.append("accuracy on an imbalanced target is misleading - report average precision, F1, or balanced accuracy alongside.")
    return found


def warnings(code: str, output: str, traits: dict | None = None) -> list[str]:
    found: list[str] = []
    metric = METRIC.search(output or "")
    if metric:
        found.append(
            f"metric looks too good to be honest ({metric.group(0).strip()}): check for leakage - "
            "duplicates across splits, a target-derived feature, or the test set used during tuning."
        )
    for call in SPLIT.finditer(code or ""):
        if "stratify" not in call.group(1):
            found.append("train_test_split() without stratify=: class balance may differ between splits; "
                         "prefer guard.split() from the ml-pipeline skill.")
        if DATEY.search(code):
            found.append("train_test_split() on data that mentions dates: a random split leaks the future "
                         "into training; use a chronological split (guard.split(..., time_col=...)).")
    found.extend(trait_warnings(code, traits or {}))
    return found


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        code = "\n".join(_strings(payload.get("tool_input") or {}))
        output = payload.get("tool_response")
        output = output if isinstance(output, str) else json.dumps(output) if output is not None else ""
        profile = _load_profile(_state_dir_for(payload.get("tool_input") or {}, payload.get("cwd") or os.getcwd()))
        found = warnings(code, output, (profile or {}).get("traits"))
        if not found:
            print("{}")
            return 0
        message = "ml-pipeline: " + " ".join(found)
        print(json.dumps({
            "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": message},
            "systemMessage": message,
        }))
        return 0
    except Exception as exc:  # fail open
        print(json.dumps({"systemMessage": f"ml-pipeline post-tool hook error (ignored): {exc}"}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
