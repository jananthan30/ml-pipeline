"""Audit 2026-09-22 - feed crafted tool calls to the real ml-pipeline PreToolUse hook and report allow/deny.

Result on v0.4.0: 20 of 23 cases wrong; after fixes 1-6 (2026-09-22): 0 of 23. On v0.5-combined the
ledger cases also carry the user's chat approval, recorded by the real UserPromptSubmit hook
(hooks/record_approval.py), so each one tests the ledger parsing and not only the approval rule.
Fix plan, test-first (turn each XX row into a unit test):
1. Follow execution: when Bash runs python, uv, marimo, jupyter or papermill on a file or -m module,
   read that file (size-capped) and scan it as well as the command text.
2. Parse the ledger strictly: count only the documented approval and override line shapes from
   SKILL.md; red-flag and final-test overrides never unlock training; trailing notes never revoke one.
3. Scan only Python-like code (.py, .ipynb, python -c, heredocs); skip .ts/.js and prose; run the
   approval-artifact check only for writes that target PIPELINE.md.
4. Deny agent attempts to change ML_PIPELINE_ENFORCE; deny messages say "ask the user" instead of
   naming the switch.
5. Widen detection: cross_val_score, cross_validate, learning_curve, permutation_importance; any
   backward pass plus optimizer step calls; any custom Trainer class construction; getattr with
   "fit"; spaced dots; bound methods called later.
6. Add a CI job: python3 -m unittest for hooks, .venv/bin/python -m unittest for the guard library.
7. Later, separate design: record approvals only from the user's own message (UserPromptSubmit)
   so the agent cannot approve its own gates.

Develop in a session the user launched with enforcement off: the installed hook blocks edits to its
own test strings.

Trigger strings are assembled at runtime so this file itself trips nothing in the
installed hook. No model is trained: the hook only reads text and ledger files.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = str(Path(__file__).resolve().parents[1] / "hooks" / "guard_training.py")
APPROVAL_HOOK = str(Path(__file__).resolve().parents[1] / "hooks" / "record_approval.py")
ROOT = Path(tempfile.mkdtemp(prefix="mlp_probe_"))
F = "." + "fit"
APP = "appro" + "ved"
G = "Ga" + "te"
TRAIN = f"model{F}(X_train, y_train)\n"
CVS = "cross_val" + "_score"
BACK = "." + "backward"
SFT = "SFT" + "Trainer"
SWITCH = "ML_PIPELINE_" + "ENFORCE"


def project(name: str, ledger: list[str] | None = None, files: dict[str, str] | None = None) -> Path:
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    if ledger is not None:
        (d / "ml_pipeline").mkdir(exist_ok=True)
        (d / "ml_pipeline" / "PIPELINE.md").write_text("\n".join(ledger) + "\n")
    for rel, body in (files or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return d


def _env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("ML_PIPELINE_")}


def user_says(prompt: str, cwd: Path) -> None:
    """The user's chat message, through the real UserPromptSubmit hook (it records gate approvals)."""
    payload = {"prompt": prompt, "cwd": str(cwd), "session_id": "probe"}
    subprocess.run([sys.executable, APPROVAL_HOOK], input=json.dumps(payload), capture_output=True, text=True,
                   env=_env(), cwd=str(cwd), timeout=20, check=True)


def run(tool: str, tool_input: dict, cwd: Path) -> tuple[str, str]:
    payload = {"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)}
    p = subprocess.run([sys.executable, HOOK], input=json.dumps(payload), capture_output=True,
                       text=True, env=_env(), cwd=str(cwd), timeout=20)
    out = json.loads(p.stdout or "{}")
    deny = out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
    return ("DENY" if deny else "ALLOW"), (out.get("systemMessage") or "")[:70]


def write(path: Path, content: str) -> dict:
    return {"file_path": str(path), "content": content}


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    bare = project("bare", files={"train.py": TRAIN, "train_mod.py": TRAIN})
    cases = []

    # --- detection: code that trains but is not seen
    cases += [
        ("control: write .py with a fit call", "DENY", "Write", write(bare / "t0.py", TRAIN), bare),
        ("run an existing training script", "DENY", "Bash", {"command": "python train.py"}, bare),
        ("run it as a module", "DENY", "Bash", {"command": "python -m train_mod"}, bare),
        ("fit code saved as .txt", "DENY", "Write", write(bare / "train.txt", TRAIN), bare),
        ("getattr(model, <fit name>)(...)", "DENY", "Write", write(bare / "t1.py", 'getattr(model, "' + F[1:] + '")(X, y)\n'), bare),
        ("bound method, then called", "DENY", "Write", write(bare / "t2.py", f"m = model{F}\nm(X, y)\n"), bare),
        ("spaced dot before fit", "DENY", "Write", write(bare / "t3.py", "model . " + "fit(X, y)\n"), bare),
        ("cross_val_score trains models", "DENY", "Write",
         write(bare / "t4.py", f"scores = {CVS}(RandomForestClassifier(), X_train, y_train, cv=5)\n"), bare),
        ("PyTorch loop, common names", "DENY", "Write",
         write(bare / "t5.py", f"total_loss = crit(out, y)\ntotal_loss{BACK}()\nopt.step()\n"), bare),
        ("Hugging Face SFTTrainer", "DENY", "Write",
         write(bare / "t6.py", f"sft = {SFT}(model=m, train_dataset=ds)\nsft.train()\n"), bare),
    ]

    # --- false alarms outside ML
    cases += [
        ("xterm.js fitAddon call in a .ts file", "ALLOW", "Write", write(bare / "term.ts", f"fitAddon{F}();\n"), bare),
        ("OpenLayers view fit call in a .js file", "ALLOW", "Write", write(bare / "map.js", f"view{F}(extent);\n"), bare),
        ("grep for an approval line (read-only)", "ALLOW", "Bash",
         {"command": f'grep -n "{G} B: {APP}" ml_pipeline/PIPELINE.md'}, bare),
        ("docs page quoting the approval format", "ALLOW", "Write",
         write(bare / "docs" / "howto.md", f"- {G} B: {APP} 2026-09-17\n"), bare),
    ]

    # --- self-disable
    cases += [
        ("agent writes ENFORCE=0 into settings", "DENY", "Write",
         write(bare / ".claude" / "settings.json", '{"env": {"' + SWITCH + '": "0"}}\n'), bare),
    ]

    # --- ledger parsing: training must be denied unless B is really recorded.
    # Every case has the user's chat approval of Gates A and B (and of the override it names), so a DENY here
    # comes from the ledger line itself, never from a missing approval.
    a_ok = f"- {G} A: {APP} 2026-09-16"
    chat = "approve gate A. approve gate B."
    ledgers = [
        ("control: A only", "DENY", [a_ok], chat),
        ("control: A and B recorded", "ALLOW", [a_ok, f"- {G} B: {APP} 2026-09-17"], chat),
        ("'B: to be ... by the user'", "DENY", [a_ok, f"- {G} B: to be {APP} by the user"], chat),
        ("'B needs an ... rationale first'", "DENY", [a_ok, f"- {G} B needs an {APP} model rationale first"], chat),
        ("'B: never ...'", "DENY", [a_ok, f"- {G} B: never {APP}"], chat),
        ("red-flag override mentioning 'model'", "DENY",
         [a_ok, f"- Override: red flag accuracy-imbalanced - user {APP} accuracy for model selection 2026-09-20 - reason: demo"],
         chat + " override red flag accuracy-imbalanced."),
        ("final-test override mentioning 'model'", "DENY",
         [a_ok, f"- Override: final test - user {APP} a second evaluation 2026-09-21 - reason: model retrained"],
         chat + " override final test."),
        ("real B record with a note containing 'not'", "ALLOW",
         [a_ok, f"- {G} B: {APP} 2026-09-17 - split is not stratified by design"], chat),
    ]
    for i, (label, want, lines, said) in enumerate(ledgers):
        d = project(f"ledger{i}", ledger=lines)
        user_says(said, d)
        cases.append((f"ledger {label}", want, "Write", write(d / "train.py", TRAIN), d))

    wrong = 0
    print(f"{'case':46} {'should':7} {'does':6} {'':3}")
    for label, want, tool, tool_input, cwd in cases:
        got, msg = run(tool, tool_input, cwd)
        flag = "OK" if got == want else "XX"
        wrong += flag == "XX"
        print(f"{label[:46]:46} {want:7} {got:6} {flag}")
    print(f"\n{wrong} of {len(cases)} cases behave wrongly")


if __name__ == "__main__":
    main()
