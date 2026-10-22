#!/usr/bin/env python3
"""UserPromptSubmit hook: record gate approvals and overrides that the USER typed.

The PreToolUse hook only accepts a `- Gate X: approved` or `- Override: ...` line in PIPELINE.md
when this hook has recorded the matching approval in ``ml_pipeline/.approvals.json`` from a real
user message. The agent cannot write that file (the PreToolUse hook denies it), so it cannot
approve its own gates.

Recognized, case-insensitive, unless negated ("don't approve gate B", "not yet"):
  - explicit:  "approve gate B", "gate B approved", "yes, go ahead with gate C", "lgtm gate A"
  - overrides: "override gate B", "override red flag random-split-temporal", "override final test"
  - a short bare yes ("yes", "approved", "go ahead", "ok, proceed") when Claude's last message
    asked about a gate (or an override) - the gate it asked about is the one approved.
Stdlib only; fails open (never blocks the user's message).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard_training import APPROVALS_FILE, find_pipeline  # noqa: E402

YES = r"(?:approve[sd]?|approval|ok(?:ay)?|yes|yep|go ahead|proceed|continue|lgtm|accept(?:ed)?|confirm(?:ed)?|sign(?:ed)? off)"
EXPLICIT = re.compile(
    rf"\b{YES}\b[^.!?\n]{{0,40}}?\bgate\s*([abcd])\b|\bgate\s*([abcd])\b[^.!?\n]{{0,25}}?\b(?:is\s+)?"
    r"(?:approved?|ok(?:ay)?|yes|go|accepted?|lgtm|confirmed?|passed?|signed off)\b",
    re.I,
)
OVERRIDE = re.compile(r"\boverride\b[^.!?\n]{0,40}?\b(gate\s*[abcd]|red\s*flag\s+[a-z][a-z-]*[a-z]|final\s*test)\b", re.I)
NEGATION = re.compile(r"\b(?:not|don'?t|do not|no|never|wait|hold off|hold on|stop|reject(?:ed)?|deny|denied)\b", re.I)
BARE_YES = re.compile(
    r"^\s*(?:yes|yep|yeah|y|approved?|ok(?:ay)?|go(?:\s+ahead)?|proceed|continue|lgtm|sounds good|looks good|"
    r"confirm(?:ed)?|do it|agreed?|override)\b", re.I)
ASSISTANT_GATE = re.compile(r"\bgate\s*([abcd])\b", re.I)
ASSISTANT_OVERRIDE = re.compile(r"\boverrid(?:e|ing)\b[^.\n]{0,80}?\b(gate\s*[abcd]|red\s*flag\s+[a-z][a-z-]*[a-z]|final\s*test)\b", re.I)


def _norm(what: str) -> str:
    return re.sub(r"\s+", " ", what.strip().lower())


def _clause_negated(prompt: str, start: int, end: int) -> bool:
    clause_start = max(prompt.rfind(c, 0, start) for c in ".!?\n;") + 1
    return bool(NEGATION.search(prompt[max(clause_start, start - 30):end]))


def last_assistant_text(transcript_path: str | None) -> str:
    """Text of the last assistant message in the session transcript (JSON lines), or ''."""
    if not transcript_path:
        return ""
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-400:]):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("type") != "assistant" and entry.get("role") != "assistant":
            continue
        message = entry.get("message", entry)
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
            if text.strip():
                return text
    return ""


def parse(prompt: str, assistant_text: str = "") -> tuple[list[str], list[str]]:
    """(gates approved, overrides granted) by this user message."""
    gates, overrides = [], []
    for m in EXPLICIT.finditer(prompt):
        if not _clause_negated(prompt, m.start(), m.end()):
            gates.append((m.group(1) or m.group(2)).upper())
    for m in OVERRIDE.finditer(prompt):
        if not _clause_negated(prompt, m.start(), m.end()):
            overrides.append(_norm(m.group(1)))
    if not gates and not overrides and len(prompt) < 200 and BARE_YES.match(prompt) and not NEGATION.search(prompt):
        if "?" in assistant_text:
            asked_override = list(ASSISTANT_OVERRIDE.finditer(assistant_text))
            if asked_override:
                overrides.append(_norm(asked_override[-1].group(1)))
            else:
                asked = list(ASSISTANT_GATE.finditer(assistant_text))
                if asked:
                    gates.append(asked[-1].group(1).upper())
    return sorted(set(gates)), overrides


def record(state_dir: Path, gates: list[str], overrides: list[str], prompt: str, session: str | None) -> dict:
    path = state_dir / APPROVALS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("gates", {})
    data.setdefault("overrides", [])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    excerpt = " ".join(prompt.split())[:160]
    for gate in gates:
        data["gates"].setdefault(gate, {"at": now, "excerpt": excerpt, "session": session})
    for what in overrides:
        data["overrides"].append({"what": what, "at": now, "excerpt": excerpt, "session": session})
    state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def main() -> int:
    try:
        if os.environ.get("ML_PIPELINE_ENFORCE", "1").strip().lower() in {"0", "false", "off", "no"}:
            print("{}")
            return 0
        payload = json.load(sys.stdin)
        prompt = str(payload.get("prompt") or "")
        cwd = payload.get("cwd") or os.getcwd()
        pipeline = find_pipeline(cwd)
        state_dir = pipeline.parent if pipeline else Path(cwd).resolve() / "ml_pipeline"
        gates, overrides = parse(prompt, last_assistant_text(payload.get("transcript_path")))
        if not gates and not overrides:
            print("{}")
            return 0
        if pipeline is None and not state_dir.is_dir() and not re.search(r"\bgate\b|\boverride\b", prompt, re.I):
            print("{}")  # a bare "yes" in a project that never started the pipeline
            return 0
        record(state_dir, gates, overrides, prompt, payload.get("session_id"))
        today = datetime.now().date().isoformat()
        notes = [f"- Gate {g}: approved {today}" for g in gates] + [f"- Override: {o} - user approved {today} - reason: <why>" for o in overrides]
        context = ("ml-pipeline: recorded the user's approval from their message. Add to ml_pipeline/PIPELINE.md: "
                   + "; ".join(notes) + ".")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
        return 0
    except Exception as exc:  # fail open: never block the user's message
        print(json.dumps({"systemMessage": f"ml-pipeline approval hook error (ignored): {exc}"}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
