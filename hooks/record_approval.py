#!/usr/bin/env python3
"""UserPromptSubmit hook: record gate approvals and overrides that the USER typed.

The PreToolUse hook only accepts a `- Gate X: approved` or `- Override: ...` line in PIPELINE.md
when this hook has recorded the matching approval in ``ml_pipeline/.approvals.json`` from a real
user message. The agent cannot write that file (the PreToolUse hook denies it), so it cannot
approve its own gates.

Recognized, case-insensitive, unless negated ("don't approve gate B", "not yet"):
  - explicit:  "approve gate B", "gate B approved", "go ahead with gate C", "sign off on gate A"
  - overrides: "override gate B", "override red flag random-split-temporal", "override final test",
               "override pilot", "override partial data"
  - a bare yes ("yes", "approved", "go ahead") as the whole message, when Claude's last question
    asked for that gate's approval (or that override) - "Do you approve Gate B?" -> "yes".
Questions ("is gate B ok?") and passing mentions ("then we'll talk about gate B") never count.
Each approval is tagged with the newest ``## Run:`` heading in PIPELINE.md, so an approval given in
one run never counts for the next. Stdlib only; fails open (never blocks the user's message).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guard_training import APPROVALS_FILE, find_pipeline, run_label  # noqa: E402

# An approval names its gate next to an approving verb: "approve gate B", "gate B approved", "go ahead
# with gate C", "sign off on gate A". A question ("is gate B ok?") or a passing mention never counts.
APPROVE_GATE = re.compile(
    r"\b(?:approve[sd]?|approving|accept(?:ed)?|sign(?:ed)?\s+off(?:\s+on)?|lgtm(?:\s+(?:for|on))?|"
    r"(?:go\s+ahead|proceed|continue|move\s+on)\s+(?:with|past|through|to|beyond)|pass)\s+(?:the\s+)?gate\s*([abcd])\b",
    re.I,
)
GATE_APPROVED = re.compile(
    r"\bgate\s*([abcd])\s*(?:is\s+|:\s*)?(?:approved|accepted|signed\s+off|lgtm|good\s+to\s+go|ok(?:ay)?)\b", re.I)
_WHAT = r"(gate\s*[abcd]|red\s*flag\s+[a-z][a-z-]*[a-z]|final\s*test|pilot|partial\s+data)\b"
OVERRIDE = re.compile(r"\boverride\b\s+(?:the\s+)?" + _WHAT, re.I)
NEGATION = re.compile(r"\b(?:not|don'?t|do not|no|never|wait|hold off|hold on|stop|reject(?:ed)?|deny|denied|won'?t|unless|"
                      r"before|until|if)\b", re.I)
QUESTION = re.compile(r"^\s*(?:(?:yes|ok(?:ay)?|so|and|but|then)[,\s]+)?(?:is|are|was|were|what|why|how|when|where|which|who|"
                      r"should|shall|can|could|would|will|do|does|did|may|might|explain|tell\s+me|show\s+me|describe)\b", re.I)
_YES_WORD = (r"(?:yes|yep|yeah|y|approved?|ok(?:ay)?|go(?:\s+ahead)?|proceed|continue|lgtm|sounds good|looks good|"
             r"confirm(?:ed)?|do it|agreed?|override(?:\s+it)?|please|sure|thanks?|thank you)")
BARE_YES = re.compile(rf"^\s*{_YES_WORD}(?:[\s,!.]+{_YES_WORD})*\s*[.!]*\s*$", re.I)
ASK_GATE = re.compile(r"\bapprov\w*\b[^?]{0,60}?\bgate\s*([abcd])\b|\bgate\s*([abcd])\b[^?]{0,60}?\bapprov\w*", re.I)
ASK_OVERRIDE = re.compile(r"\boverrid(?:e|ing)\b[^?]{0,80}?\b" + _WHAT, re.I)
ASK_NEGATED = re.compile(r"\b(?:won'?t|will not|unless|not|don'?t|never|instead of)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?\n;])\s*", text) if p.strip()]


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") or bool(QUESTION.match(sentence))


def _norm(what: str) -> str:
    return re.sub(r"\s+", " ", what.strip().lower())


def _negated_before(sentence: str, start: int) -> bool:
    return bool(NEGATION.search(sentence[:start]))


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


def asked_in(assistant_text: str) -> tuple[str | None, str | None]:
    """(gate, override) that Claude's last message explicitly asked the user to approve, in a question."""
    for sentence in reversed(_sentences(assistant_text)):
        if not sentence.rstrip().endswith("?"):
            continue
        over = ASK_OVERRIDE.search(sentence)
        if over and not ASK_NEGATED.search(sentence):
            return None, _norm(over.group(1))
        gate = ASK_GATE.search(sentence)
        if gate and not ASK_NEGATED.search(sentence):
            return (gate.group(1) or gate.group(2)).upper(), None
        return None, None  # the last question asked something else
    return None, None


def parse(prompt: str, assistant_text: str = "") -> tuple[list[str], list[str]]:
    """(gates approved, overrides granted) by this user message."""
    gates, overrides = [], []
    for sentence in _sentences(prompt):
        if _is_question(sentence):
            continue
        for pattern in (APPROVE_GATE, GATE_APPROVED):
            for m in pattern.finditer(sentence):
                if not _negated_before(sentence, m.start()):
                    gates.append(m.group(1).upper())
        for m in OVERRIDE.finditer(sentence):
            if not _negated_before(sentence, m.start()):
                overrides.append(_norm(m.group(1)))
    if not gates and not overrides and BARE_YES.match(prompt):
        gate, override = asked_in(assistant_text)
        if override:
            overrides.append(override)
        elif gate:
            gates.append(gate)
    return sorted(set(gates)), overrides


def record(state_dir: Path, gates: list[str], overrides: list[str], prompt: str, session: str | None,
           run: str = "") -> dict:
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
        earlier = data["gates"].get(gate)
        if not isinstance(earlier, dict) or str(earlier.get("run") or "") != run:  # the first approval in this run
            data["gates"][gate] = {"at": now, "excerpt": excerpt, "session": session, "run": run}
    for what in overrides:
        data["overrides"].append({"what": what, "at": now, "excerpt": excerpt, "session": session, "run": run})
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
        ledger = state_dir / "PIPELINE.md"
        run = run_label(ledger.read_text(encoding="utf-8", errors="replace")) if ledger.is_file() else ""
        record(state_dir, gates, overrides, prompt, payload.get("session_id"), run)
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
