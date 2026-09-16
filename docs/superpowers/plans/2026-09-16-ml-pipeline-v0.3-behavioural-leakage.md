# ml-pipeline v0.3.0 — Behavioural Leakage Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-aim ml-pipeline's enforcement at the leakage that actually inflates results (test-set selection, split overlap, temporal boundaries) with a runtime guard, a canary dataset with a known ceiling, and an eval case that proves the gates hold.

**Architecture:** The existing PreToolUse hook gains a Gate C rule (no evaluation on the test split before Gate C). A new pandas library `lib/mlpipeline_guard.py` is copied into the user's project and owns the split and the single-use final test, freezing a fingerprint of the test set in `ml_pipeline/.guard_state.json`. A stdlib canary module generates a dataset whose honest accuracy ceiling is known. A SessionStart hook tells the agent where the helpers live; a PostToolUse hook warns on too-good metrics and unsafe splits. One `claude plugin eval` case checks a bare "train a model" request stops at Gate A.

**Tech Stack:** Python 3.10+ stdlib for hooks; pandas + numpy for the guard (dev env via `uv`); `claude plugin eval` (prompt.md + graders/*.md format).

**Spec:** `docs/superpowers/specs/2026-09-16-ml-pipeline-v0.3-behavioural-leakage.md`

## Global Constraints

- Hooks: Python ≥ 3.10, stdlib only, no `subprocess`/`eval`/`exec`, fail open on their own errors.
- Guard library depends on pandas + numpy only. Canary module is stdlib only.
- Gate lines: `- Gate X: approved YYYY-MM-DD`. Override lines: `- Override: <what> - ...`.
- HOL scanner ≥ 98/100, zero critical/high, on the PR.
- Bump `"version"` in both `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json` to `0.3.0` (Task 8 only).
- Hook tests: `python3 -m unittest discover tests`. Guard tests: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`.
- Every commit ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work on branch `v0.3-behavioural-leakage`; open a PR at the end; the user merges.

## TODO (master checklist)

- [ ] Task 1 — Hook: deny evaluating on the test split before Gate C
- [ ] Task 2 — Guard library: `split()` with dedupe, chronological/group/stratified splits, overlap check, fingerprint freeze
- [ ] Task 3 — Guard library: `final_test()` single-touch counter with tamper check and PIPELINE.md log
- [ ] Task 4 — Canary dataset module + `examples/canary/`
- [ ] Task 5 — SessionStart hook + skill/README/installer updates (guard usage at steps 6/14, `git tag gate-X`)
- [ ] Task 6 — PostToolUse warning hook (too-good metrics, unsafe `train_test_split`)
- [ ] Task 7 — Eval case `gate-stop`
- [ ] Task 8 — Release 0.3.0 (version bump, full test run, PR, scanner, release, local plugin update)

Backlog (separate plans, not in scope): evaluator agent + hacker-fixer hardening · `mlw` trial · search module for steps 10–11 · per-phase budgets · `canary-honesty` eval case.

---

### Task 1: Hook — deny evaluating on the test split before Gate C

**Files:**
- Modify: `hooks/guard_training.py`
- Test: `tests/test_guard_training.py`

**Interfaces:**
- Consumes: existing `run(tool_input, pipeline_md=None, ...)` test helper, `GATE_A`, `GATE_AB` constants in the test file.
- Produces: `analyse()` result gains key `"eval_on_test"`; `gate_state()` gains key `"override_c"`; new constant `GATE_C_MSG`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_guard_training.py` (after the `AfterGateB` class):

```python
GATE_ABC = GATE_AB + "- Gate C: approved 2026-09-18\n"
OVERRIDE_C = GATE_AB + "- Override: Gate C - user approved evaluating on the test set 2026-09-18 - reason: demo\n"


class SelectionLeakage(unittest.TestCase):
    """Evaluating on the test split before Gate C is selection leakage — the kind that matters most."""

    def test_metric_on_test_before_gate_c_denied(self):
        decision, reason = run({"command": "print(accuracy_score(y_test, clf.predict(X_test)))"}, GATE_AB)
        self.assertEqual(decision, "deny")
        self.assertIn("Gate C", reason)

    def test_score_on_test_without_pipeline_denied(self):
        self.assertEqual(run({"command": "clf.score(X_test, y_test)"})[0], "deny")

    def test_metric_on_test_after_gate_c_allowed(self):
        self.assertEqual(run({"command": "accuracy_score(y_test, clf.predict(X_test))"}, GATE_ABC)[0], "allow")

    def test_override_c_allows(self):
        self.assertEqual(run({"command": "clf.score(X_test, y_test)"}, OVERRIDE_C)[0], "allow")

    def test_validation_evaluation_is_fine(self):
        self.assertEqual(run({"command": "roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])"}, GATE_AB)[0], "allow")

    def test_transforming_test_is_preprocessing_not_evaluation(self):
        self.assertEqual(run({"command": "X_test_s = scaler.transform(X_test)"}, GATE_A)[0], "allow")
```

Move the two constants above the class (module level, next to `GATE_AB`).

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests.test_guard_training.SelectionLeakage -v`
Expected: 3 FAIL (`..._denied`, `..._without_pipeline_denied` get "allow"); the 3 allow-cases pass already.

- [ ] **Step 3: Implement** — in `hooks/guard_training.py`:

Add after `TRAIN_API`:

```python
# Evaluating a model: predict/score methods and the usual sklearn metric functions.
EVAL_CALL = re.compile(
    r"(?:\.(?:predict|predict_proba|decision_function|score|evaluate)|"
    r"\b(?:accuracy_score|balanced_accuracy_score|roc_auc_score|average_precision_score|f1_score|"
    r"precision_score|recall_score|log_loss|mean_squared_error|mean_absolute_error|"
    r"root_mean_squared_error|r2_score|classification_report|confusion_matrix|cross_val_score|"
    r"cross_validate))\s*\("
)
```

Add after `OVERRIDE_LINE`:

```python
OVERRIDE_C_LINE = re.compile(r"^.*\boverrid(?:e|den)\b.*\b(?:gate\s*c|final test|test set)\b.*$", re.I | re.M)
```

Add after `GATE_B_MSG`:

```python
GATE_C_MSG = (
    "ml-pipeline: evaluating on the test split is locked until Gate C ({0!r}). The test set is touched "
    "exactly once, at step 14, after the user approves Gate C - tune and select on validation only. "
    "Record `Gate C: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md first, or, if the user explicitly "
    "chose otherwise, `Override: Gate C - user approved evaluating on the test set YYYY-MM-DD - reason: ...`."
)
```

In `analyse()`: initialise `found` with the extra key and add the evaluation scan before `return found`:

```python
    found: dict[str, str | None] = {"model": None, "transformer": None, "fit_on_test": None, "eval_on_test": None}
    ...
    for match in EVAL_CALL.finditer(code):
        args = _args(code, match.end())
        if TEST_SPLIT.search(args) and found["eval_on_test"] is None:
            found["eval_on_test"] = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
    return found
```

In `gate_state()`: initialise with `"override_c": False` and add after the override loop:

```python
    for match in OVERRIDE_C_LINE.finditer(markdown):
        if not NEGATED.search(match.group(0)):
            state["override_c"] = True
```

In `decide()`: insert right after the `fit_on_test` check:

```python
    if found["eval_on_test"] and not (gates and (gates["C"] or gates["override_c"])):
        return GATE_C_MSG.format(found["eval_on_test"])
```

and change the "nothing found" guard to `if not (found["model"] or found["transformer"]): return None` (unchanged) — `eval_on_test` alone after Gate C falls through to allow.

In `main()`: widen the fast path:

```python
        if ".fit" not in text and not TRAIN_API.search(text) and not EVAL_CALL.search(text):
            return _emit(None)
```

Update the module docstring's table: add the line `Gate C approved   ->  evaluating on the test split allowed (once, at step 14)`.

- [ ] **Step 4: Run the whole hook suite**

Run: `python3 -m unittest discover tests -v 2>&1 | tail -5`
Expected: `OK`, 33 tests.

- [ ] **Step 5: Commit**

```bash
git checkout -b v0.3-behavioural-leakage
git add hooks/guard_training.py tests/test_guard_training.py
git commit -m "feat(hook): deny evaluating on the test split before Gate C

Selection leakage - using the test set to pick a model, seed, or threshold -
is the class that inflates results most (arXiv 2604.04199). The hook now
denies predict/score/metric calls whose arguments name the test split until
PIPELINE.md records Gate C, or an Override: Gate C line.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Guard library — `split()`

**Files:**
- Create: `lib/mlpipeline_guard.py`
- Create: `tests/test_mlpipeline_guard.py`
- Create: `tests/requirements-dev.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `split(df, *, target, time_col=None, group_col=None, test_size=0.2, val_size=0.2, seed=0, state_dir="ml_pipeline") -> (train, val, test)`; `check_no_overlap(*frames) -> None`; exceptions `LeakageError`, `TemporalSplitRequired`, `OverlapLeakage`; constant `STATE_FILE = ".guard_state.json"`; state JSON keys `test_fingerprint`, `n_test`, `test_touches`, `duplicates_dropped`, `split`, `created`. Task 3 reads the state file with these keys.

- [ ] **Step 1: Dev environment**

```bash
uv venv .venv
uv pip install --python .venv/bin/python pandas numpy
printf 'pandas\nnumpy\n' > tests/requirements-dev.txt
printf '.venv/\nevals/results/\n' >> .gitignore
.venv/bin/python -c "import pandas, numpy; print(pandas.__version__, numpy.__version__)"
```

Expected: two version numbers printed.

- [ ] **Step 2: Write the failing tests** — create `tests/test_mlpipeline_guard.py`:

```python
"""Tests for lib/mlpipeline_guard.py. Run with the dev venv:
    .venv/bin/python -m unittest tests/test_mlpipeline_guard.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - the stdlib suite skips this file
    raise unittest.SkipTest("pandas/numpy not installed; run with .venv/bin/python")
import mlpipeline_guard as guard  # noqa: E402


def frame(n=1000, seed=0, dates=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, size=n), "y": rng.integers(0, 2, size=n)})
    if dates:
        df["date"] = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


class Split(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sizes_and_state_file(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 1000)
        self.assertAlmostEqual(len(te) / 1000, 0.2, delta=0.02)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual(state["test_touches"], 0)
        self.assertEqual(state["n_test"], len(te))
        self.assertEqual(state["split"], "stratified")

    def test_stratified_keeps_class_balance(self):
        df = frame()
        _, _, te = guard.split(df, target="y", state_dir=self.state)
        self.assertAlmostEqual(te.y.mean(), df.y.mean(), delta=0.03)

    def test_temporal_split_is_chronological(self):
        tr, va, te = guard.split(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        self.assertLess(tr.date.max(), va.date.min())
        self.assertLess(va.date.max(), te.date.min())

    def test_datetime_column_without_time_col_raises(self):
        with self.assertRaises(guard.TemporalSplitRequired):
            guard.split(frame(dates=True), target="y", state_dir=self.state)

    def test_group_split_keeps_groups_together(self):
        df = frame()
        df["patient"] = df.index // 10
        tr, va, te = guard.split(df, target="y", group_col="patient", state_dir=self.state)
        self.assertFalse(set(tr.patient) & set(te.patient))
        self.assertFalse(set(va.patient) & set(te.patient))
        self.assertFalse(set(tr.patient) & set(va.patient))

    def test_duplicates_are_dropped_before_splitting(self):
        df = pd.concat([frame(200)] * 2, ignore_index=True)  # every row twice
        tr, va, te = guard.split(df, target="y", state_dir=self.state)
        self.assertEqual(len(tr) + len(va) + len(te), 200)
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["duplicates_dropped"], 200)

    def test_check_no_overlap_raises(self):
        df = frame(50)
        with self.assertRaises(guard.OverlapLeakage):
            guard.check_no_overlap(df.iloc[:30], df.iloc[25:])

    def test_bad_sizes_rejected(self):
        with self.assertRaises(ValueError):
            guard.split(frame(), target="y", test_size=0.6, val_size=0.5, state_dir=self.state)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'mlpipeline_guard'`.

- [ ] **Step 4: Implement** — create `lib/mlpipeline_guard.py`:

```python
"""Runtime guard for the ml-pipeline skill: a leakage-safe split and a single-use test set.

Copy this file into your project as ``ml_pipeline/guard.py`` (the skill tells the agent where the
canonical copy lives). Requires pandas and numpy, nothing else.

    train, val, test = guard.split(df, target="y", time_col="date")          # step 6
    ...
    score = guard.final_test(model.predict, test, target="y", metric_fn=acc)  # step 14

``split`` drops exact duplicates, refuses a random split when a datetime column is present,
keeps whole groups together when asked, checks that no row lands in two splits, and freezes a
fingerprint of the test set in ``ml_pipeline/.guard_state.json``. ``final_test`` (Task 3) is the
only sanctioned way to touch that test set, and it works once.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

STATE_FILE = ".guard_state.json"
STRATIFY_MAX_CLASSES = 20


class LeakageError(RuntimeError):
    """The pipeline tried to do something the discipline forbids."""


class TemporalSplitRequired(LeakageError):
    """The frame has a datetime column but no time_col was given."""


class OverlapLeakage(LeakageError):
    """Identical rows appear in more than one split."""


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def _fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in np.sort(_row_hashes(df)):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def check_no_overlap(*frames: pd.DataFrame) -> None:
    """Raise OverlapLeakage if any identical row appears in two different frames."""
    hash_sets = [set(_row_hashes(frame).tolist()) for frame in frames]
    for i in range(len(hash_sets)):
        for j in range(i + 1, len(hash_sets)):
            common = hash_sets[i] & hash_sets[j]
            if common:
                raise OverlapLeakage(
                    f"{len(common)} identical row(s) appear in both split {i} and split {j}; "
                    "deduplicate before splitting"
                )


def _datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _time_indices(df, time_col, n_train, n_val):
    order = df.sort_values(time_col, kind="mergesort").index.to_numpy()
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def _group_indices(df, group_col, n_test, n_val, rng):
    groups = df[group_col].unique()
    rng.shuffle(groups)
    counts = df[group_col].value_counts()
    test_groups, val_groups, seen = set(), set(), 0
    for group in groups:
        if seen < n_test:
            test_groups.add(group)
        elif seen < n_test + n_val:
            val_groups.add(group)
        else:
            break
        seen += int(counts[group])
    in_test = df[group_col].isin(test_groups).to_numpy()
    in_val = df[group_col].isin(val_groups).to_numpy()
    return np.flatnonzero(~in_test & ~in_val), np.flatnonzero(in_val), np.flatnonzero(in_test)


def _stratified_indices(df, target, test_size, val_size, rng):
    train, val, test = [], [], []
    for positions in df.groupby(target, sort=False).indices.values():
        perm = rng.permutation(positions)
        k_test = int(round(len(perm) * test_size))
        k_val = int(round(len(perm) * val_size))
        test.extend(perm[:k_test])
        val.extend(perm[k_test:k_test + k_val])
        train.extend(perm[k_test + k_val:])
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def split(
    df: pd.DataFrame,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 0,
    state_dir: str | Path = "ml_pipeline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test) the way the pipeline's step 6 requires, and freeze the test set."""
    if not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be in (0, 1) and sum to less than 1")
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    if time_col is None and (dt_cols := _datetime_columns(df)):
        raise TemporalSplitRequired(
            f"datetime column(s) {dt_cols} present: pass time_col=... so the split is chronological"
        )

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    n = len(df)
    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    n_train = n - n_val - n_test
    rng = np.random.default_rng(seed)

    if time_col is not None:
        kind, parts = "time", _time_indices(df, time_col, n_train, n_val)
    elif group_col is not None:
        kind, parts = "group", _group_indices(df, group_col, n_test, n_val, rng)
    elif df[target].nunique() <= STRATIFY_MAX_CLASSES:
        kind, parts = "stratified", _stratified_indices(df, target, test_size, val_size, rng)
    else:
        perm = rng.permutation(n)
        kind, parts = "random", (perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test])

    train, val, test = (df.iloc[np.sort(idx)].reset_index(drop=True) for idx in parts)
    check_no_overlap(train, val, test)

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE).write_text(json.dumps({
        "test_fingerprint": _fingerprint(test),
        "n_test": int(len(test)),
        "test_touches": 0,
        "duplicates_dropped": int(dropped),
        "split": kind,
        "created": date.today().isoformat(),
    }, indent=2))
    return train, val, test
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py -v 2>&1 | tail -4`
Expected: `OK`, 8 tests. Also confirm the stdlib suite still skips this file cleanly: `python3 -m unittest discover tests 2>&1 | tail -3` → `OK (skipped=1)` plus the 33 hook tests.

- [ ] **Step 6: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py tests/requirements-dev.txt .gitignore
git commit -m "feat(guard): leakage-safe split() with dedupe, temporal/group/stratified modes, overlap check

Runtime companion to the hook: drops exact duplicates (memorization leakage),
refuses a random split when a datetime column exists (boundary leakage),
keeps groups together, asserts no row lands in two splits, and freezes a
fingerprint of the test set in ml_pipeline/.guard_state.json for final_test().

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Guard library — `final_test()`

**Files:**
- Modify: `lib/mlpipeline_guard.py`
- Test: `tests/test_mlpipeline_guard.py`

**Interfaces:**
- Consumes: `split()`, `STATE_FILE`, `_fingerprint()` from Task 2.
- Produces: `final_test(predict_fn, test, *, target, metric_fn, state_dir="ml_pipeline", pipeline_md="ml_pipeline/PIPELINE.md") -> float`; exceptions `TestSetAlreadyUsed`, `TestSetTampered`; appends `- Step 14 final test: <metric>=<value> (touch N, YYYY-MM-DD)` to PIPELINE.md.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_mlpipeline_guard.py`:

```python
class FinalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.md = self.state / "PIPELINE.md"
        self.tr, self.va, self.te = guard.split(frame(), target="y", state_dir=self.state)
        self.md.write_text("- Gate C: approved 2026-09-18\n")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def acc(y_true, y_pred):
        return float((y_true == y_pred).mean())

    @staticmethod
    def predict(X):
        return np.zeros(len(X), dtype=int)

    def _run(self, test):
        return guard.final_test(self.predict, test, target="y", metric_fn=self.acc,
                                state_dir=self.state, pipeline_md=self.md)

    def test_first_touch_scores_logs_and_counts(self):
        score = self._run(self.te)
        self.assertAlmostEqual(score, 1 - self.te.y.mean(), places=6)
        self.assertIn("Step 14 final test: acc=", self.md.read_text())
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["test_touches"], 1)

    def test_second_touch_raises(self):
        self._run(self.te)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)

    def test_override_line_permits_second_touch(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved a second evaluation 2026-09-18 - reason: new strategy\n")
        self._run(self.te)

    def test_tampered_test_set_raises(self):
        with self.assertRaises(guard.TestSetTampered):
            self._run(self.va)

    def test_no_split_state_raises(self):
        (self.state / guard.STATE_FILE).unlink()
        with self.assertRaises(guard.LeakageError):
            self._run(self.te)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.FinalTest 2>&1 | tail -3`
Expected: `AttributeError: module 'mlpipeline_guard' has no attribute 'final_test'`.

- [ ] **Step 3: Implement** — append to `lib/mlpipeline_guard.py` (add `import re` to the imports):

```python
class TestSetAlreadyUsed(LeakageError):
    """final_test() was called more than once without a recorded override."""


class TestSetTampered(LeakageError):
    """The frame handed to final_test() is not the one frozen at split time."""


_NEGATED = re.compile(r"\b(?:not|pending|awaiting|todo)\b", re.I)


def _override_allows(pipeline_md: Path, what: str) -> bool:
    if not pipeline_md.is_file():
        return False
    line = re.compile(r"^.*\boverrid(?:e|den)\b.*\b" + re.escape(what) + r"\b.*$", re.I | re.M)
    return any(not _NEGATED.search(m.group(0)) for m in line.finditer(pipeline_md.read_text()))


def final_test(
    predict_fn,
    test: pd.DataFrame,
    *,
    target: str,
    metric_fn,
    state_dir: str | Path = "ml_pipeline",
    pipeline_md: str | Path = "ml_pipeline/PIPELINE.md",
) -> float:
    """Step 14: score ``predict_fn`` on the frozen test set, once, and log the number.

    A second call needs an ``- Override: final test - ...`` line in PIPELINE.md, which the skill
    only writes after the user agreed a new test strategy.
    """
    state_path = Path(state_dir) / STATE_FILE
    pipeline_md = Path(pipeline_md)
    if not state_path.is_file():
        raise LeakageError("no frozen test set: call guard.split() at step 6 first")
    state = json.loads(state_path.read_text())
    if _fingerprint(test) != state["test_fingerprint"]:
        raise TestSetTampered("this is not the test set frozen at step 6")
    if state["test_touches"] >= 1 and not _override_allows(pipeline_md, "final test"):
        raise TestSetAlreadyUsed(
            "the test set has already been evaluated once; agree a new test strategy with the user "
            "and record `- Override: final test - ...` in PIPELINE.md before evaluating again"
        )

    features = test.drop(columns=[target])
    score = float(metric_fn(test[target].to_numpy(), np.asarray(predict_fn(features))))

    state["test_touches"] += 1
    state_path.write_text(json.dumps(state, indent=2))
    name = getattr(metric_fn, "__name__", "metric")
    line = f"- Step 14 final test: {name}={score:.4f} (touch {state['test_touches']}, {date.today().isoformat()})\n"
    pipeline_md.parent.mkdir(parents=True, exist_ok=True)
    existing = pipeline_md.read_text().rstrip("\n") + "\n" if pipeline_md.is_file() else ""
    pipeline_md.write_text(existing + line)
    return score
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py -v 2>&1 | tail -4`
Expected: `OK`, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): final_test() - the test set is scored once, verified, and logged

Checks the frame against the fingerprint frozen by split(), refuses a second
evaluation unless PIPELINE.md carries an 'Override: final test' line, and
appends the result to PIPELINE.md so the number is on the record.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Canary dataset module + `examples/canary/`

**Files:**
- Create: `lib/mlpipeline_canary.py`
- Create: `examples/canary/make_canary.py`
- Create: `examples/canary/canary.csv` (generated, 2000 rows)
- Create: `examples/canary/README.md`
- Test: `tests/test_mlpipeline_canary.py`

**Interfaces:**
- Produces: `make_rows(n=2000, noise=0.20, seed=7) -> list[list]` (columns `x0..x4, y`), `write_csv(path, n, noise, seed)`, `verdict(test_accuracy, n_test, ceiling=0.80, z=3.0) -> (leaked: bool, explanation: str)`, constants `NOISE`, `CEILING`, `N_FEATURES`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_mlpipeline_canary.py`:

```python
"""Tests for lib/mlpipeline_canary.py (stdlib only): python3 -m unittest tests/test_mlpipeline_canary.py"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import mlpipeline_canary as canary  # noqa: E402


class Canary(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(canary.make_rows(50, seed=1), canary.make_rows(50, seed=1))
        self.assertNotEqual(canary.make_rows(50, seed=1), canary.make_rows(50, seed=2))

    def test_label_noise_rate_matches(self):
        rows = canary.make_rows(20000, noise=0.2, seed=3)
        clean = [1 if r[0] + r[1] * r[2] - 0.5 * r[3] > 0 else 0 for r in rows]
        flipped = sum(c != r[-1] for c, r in zip(clean, rows)) / len(rows)
        self.assertAlmostEqual(flipped, 0.2, delta=0.01)

    def test_verdict(self):
        leaked, why = canary.verdict(0.79, 2000)
        self.assertFalse(leaked)
        self.assertIn("within", why)
        leaked, why = canary.verdict(0.86, 2000)
        self.assertTrue(leaked)
        self.assertIn("leaked", why)

    def test_write_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            canary.write_csv(path, n=10)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["x0", "x1", "x2", "x3", "x4", "y"])
        self.assertEqual(len(rows), 11)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests/test_mlpipeline_canary.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'mlpipeline_canary'`.

- [ ] **Step 3: Implement** — create `lib/mlpipeline_canary.py`:

```python
"""A dataset with a known ceiling: beat it and you leaked.

Labels are a deterministic function of the features with a fixed fraction flipped at random,
so no honest model can exceed accuracy ``1 - NOISE`` on held-out rows. ``verdict`` turns a
reported test accuracy into a leak / no-leak call with a z-test against that ceiling.
Stdlib only, so it runs anywhere the hooks run.
"""
from __future__ import annotations

import csv
import math
import random

NOISE = 0.20
CEILING = 1.0 - NOISE
N_FEATURES = 5
HEADER = [f"x{i}" for i in range(N_FEATURES)] + ["y"]


def make_rows(n: int = 2000, noise: float = NOISE, seed: int = 7) -> list[list]:
    """``n`` rows of ``x0..x4, y``. ``y = 1[x0 + x1*x2 - 0.5*x3 > 0]`` with ``noise`` of labels flipped."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = [round(rng.gauss(0, 1), 6) for _ in range(N_FEATURES)]
        y = 1 if x[0] + x[1] * x[2] - 0.5 * x[3] > 0 else 0
        if rng.random() < noise:
            y = 1 - y
        rows.append(x + [y])
    return rows


def write_csv(path, n: int = 2000, noise: float = NOISE, seed: int = 7) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(make_rows(n, noise, seed))


def verdict(test_accuracy: float, n_test: int, ceiling: float = CEILING, z: float = 3.0) -> tuple[bool, str]:
    """(leaked?, explanation). Leaked when accuracy is more than ``z`` standard errors above the ceiling."""
    limit = ceiling + z * math.sqrt(ceiling * (1 - ceiling) / n_test)
    if test_accuracy > limit:
        return True, (f"accuracy {test_accuracy:.3f} exceeds the honest ceiling {ceiling:.2f} "
                      f"+ {z:.0f} SE ({limit:.3f}) on n={n_test}: the test set leaked")
    return False, f"accuracy {test_accuracy:.3f} is within the honest ceiling ({limit:.3f}) on n={n_test}"
```

Create `examples/canary/make_canary.py`:

```python
#!/usr/bin/env python3
"""Regenerate canary.csv deterministically. Usage: python3 make_canary.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import mlpipeline_canary as canary  # noqa: E402

out = Path(__file__).with_name("canary.csv")
canary.write_csv(out)
print(f"wrote {out} ({sum(1 for _ in open(out)) - 1} rows, honest ceiling {canary.CEILING:.2f})")
```

Create `examples/canary/README.md`:

```markdown
# Canary dataset

`canary.csv` — 2,000 rows, features `x0..x4`, binary target `y`.
`y = 1[x0 + x1·x2 − 0.5·x3 > 0]` with **20% of labels flipped at random** (x4 is pure noise).

No honest model can score above **0.80 accuracy** on held-out rows. On a 400-row test split the
3-standard-error limit is 0.86. A reported test accuracy above that means the test set leaked —
through duplicates, a target-derived feature, or being used during tuning.

Check a result: `python3 -c "import sys; sys.path.insert(0,'lib'); import mlpipeline_canary as c; print(c.verdict(0.91, 400))"`
Regenerate: `python3 examples/canary/make_canary.py`
```

Then generate the CSV: `python3 examples/canary/make_canary.py`.

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m unittest tests/test_mlpipeline_canary.py -v 2>&1 | tail -4` → `OK`, 4 tests.
Run: `wc -l examples/canary/canary.csv` → `2001`.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_canary.py examples/canary tests/test_mlpipeline_canary.py
git commit -m "feat(canary): dataset with a known accuracy ceiling to prove leakage

Labels are a fixed function of the features with 20% flipped, so honest
held-out accuracy cannot exceed 0.80. verdict() z-tests a reported score
against the ceiling: above it, the test set leaked (capped evaluation, arXiv
2606.07379). Stdlib only; examples/canary/canary.csv is committed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: SessionStart hook + skill/README/installer updates

**Files:**
- Create: `hooks/session_start.py`
- Modify: `hooks/hooks.json`
- Modify: `skills/ml-pipeline/SKILL.md`
- Modify: `README.md`
- Modify: `install-other-tools.sh`
- Test: `tests/test_session_start.py`

**Interfaces:**
- Produces: SessionStart JSON with `hookSpecificOutput.additionalContext` naming `${CLAUDE_PLUGIN_ROOT}/lib/mlpipeline_guard.py` and `.../lib/mlpipeline_canary.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_session_start.py`:

```python
"""Tests for hooks/session_start.py: python3 -m unittest tests/test_session_start.py"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import session_start  # noqa: E402


def run(env: dict[str, str]) -> dict:
    saved = dict(os.environ)
    os.environ.pop("CLAUDE_PLUGIN_ROOT", None)
    os.environ.update(env)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            session_start.main()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return json.loads(out.getvalue())


class SessionStart(unittest.TestCase):
    def test_names_the_helpers_when_plugin_root_is_set(self):
        result = run({"CLAUDE_PLUGIN_ROOT": "/plug"})
        context = result["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(result["hookSpecificOutput"]["hookEventName"], "SessionStart")
        self.assertIn("/plug/lib/mlpipeline_guard.py", context)
        self.assertIn("/plug/lib/mlpipeline_canary.py", context)
        self.assertIn("ml_pipeline/guard.py", context)

    def test_silent_without_plugin_root(self):
        self.assertEqual(run({}), {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m unittest tests/test_session_start.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'session_start'`.

- [ ] **Step 3: Implement** — create `hooks/session_start.py`:

```python
#!/usr/bin/env python3
"""SessionStart hook: tell the agent where the plugin's runtime helpers live.

The skill text cannot know the plugin's install path, so this adds it to the session context
once (~60 tokens). Stdlib only; prints ``{}`` when not running as a plugin.
"""
from __future__ import annotations

import json
import os


def main() -> int:
    root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if not root:
        print("{}")
        return 0
    lib = os.path.join(root, "lib")
    context = (
        "ml-pipeline runtime helpers: guard library at "
        f"{os.path.join(lib, 'mlpipeline_guard.py')} - copy it to ml_pipeline/guard.py at step 6 and use "
        "guard.split() / guard.final_test(); canary dataset generator at "
        f"{os.path.join(lib, 'mlpipeline_canary.py')} (honest accuracy ceiling 0.80; above it means leakage)."
    )
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Register it in `hooks/hooks.json` — add a `SessionStart` key beside `PreToolUse`:

```json
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact",
        "hooks": [
          { "type": "command", "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py\"", "timeout": 5 }
        ]
      }
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m unittest tests/test_session_start.py -v 2>&1 | tail -3` → `OK`, 2 tests.
Run: `python3 -c "import json; json.load(open('hooks/hooks.json')); print('hooks.json valid')"`.

- [ ] **Step 5: Update SKILL.md** — three edits in `skills/ml-pipeline/SKILL.md`:

(a) Replace the step 6 bullet in "What each step must produce" with:

```markdown
6. **Split before any fitting** — copy the plugin's guard library to `ml_pipeline/guard.py`
   (its path is given at session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py`
   next to this file) and split with it:
   `train, val, test = guard.split(df, target=..., time_col=<col> if the data is temporal, group_col=<col> if the same entity appears in multiple rows)`.
   It drops exact duplicates, refuses a random split when a datetime column exists, keeps groups
   together, checks that no row lands in two splits, and freezes a fingerprint of the test set.
   The test set is touched exactly once, at step 14, through `guard.final_test()`.
```

(b) Replace the step 14 bullet with:

```markdown
14. **Final test** — `score = guard.final_test(model.predict, test, target=..., metric_fn=...)`.
    It verifies the frame is the frozen test set, refuses a second call, and logs the result to
    PIPELINE.md. Report the number honestly, even if it is worse than validation. No going back to
    tune on it — if the result forces changes, agree a new test strategy with the user and record
    `- Override: final test - user approved a second evaluation YYYY-MM-DD - reason: <why>`.
```

(c) In "## Progress tracking", after the canonical gate-lines block, add:

```markdown
After recording a gate approval in a git repository, checkpoint it:
`git add -A && git commit -m "ml-pipeline: Gate X approved" && git tag -f gate-X`. Every approved
phase is then a reproducible point to return to.
```

(d) In "## Enforcement in Claude Code (hook)", add two bullets to the deny list and one paragraph:

```markdown
- **evaluating on the test split** — `.predict/.score` or a metric function whose arguments name
  `X_test`, `df_test`, `test_*` — until `Gate C: approved …` (or `Override: Gate C …`).
```

and after the list:

```markdown
A PostToolUse hook also warns (never blocks) when a result looks too good to be honest
(accuracy/AUC/F1 ≥ 0.98) or when `train_test_split()` is called without `stratify=` or on data that
mentions dates. To prove a pipeline does not leak, run it on `examples/canary/canary.csv`: honest
test accuracy cannot exceed 0.80, and `mlpipeline_canary.verdict(score, n_test)` says whether a
number is above the ceiling.
```

(e) In "## Non-negotiables", add as the first bullet:

```markdown
- Selection leakage is the one that matters most: the test set is never used to pick a model, a
  seed, a feature, or a threshold. Validation only.
```

- [ ] **Step 6: Update README.md** — replace the body of "## Enforced, not just instructed" with:

```markdown
In Claude Code the plugin ships hooks. Before any command, file write, or notebook edit runs, a
`PreToolUse` hook scans the code and **denies**: any fitting before Gate A; model training before
Gate B; evaluating on the test split before Gate C; fitting on the test split, ever. A denied call
tells the agent exactly which steps are missing — so "just train it" fails closed until you have
approved the gate. A `PostToolUse` hook warns when a metric looks too good to be honest or a split
ignores class balance or time. And at step 6 the agent switches to `guard.split()` /
`guard.final_test()`, a small runtime library that drops duplicates, refuses random splits on
temporal data, and lets the frozen test set be scored exactly once.

Prove it on the **canary**: `examples/canary/canary.csv` has a known honest ceiling of 0.80
accuracy. Beat it and you leaked. `ML_PIPELINE_ENFORCE=0` switches enforcement off for non-ML
projects. Codex and Kimi get the same rules as instructions (no hook support there).
```

- [ ] **Step 7: Update `install-other-tools.sh`** — in `install_for`, after the `cp "$SKILL" ...` line add:

```bash
  mkdir -p "$home_dir/skills/ml-pipeline/lib"
  cp "$HERE"/lib/mlpipeline_guard.py "$HERE"/lib/mlpipeline_canary.py "$home_dir/skills/ml-pipeline/lib/"
```

Run: `bash -n install-other-tools.sh && ./install-other-tools.sh && ls ~/.codex/skills/ml-pipeline/lib/`
Expected: both files listed.

- [ ] **Step 8: Run all stdlib tests, then commit**

Run: `python3 -m unittest discover tests 2>&1 | tail -3` → `OK`.

```bash
git add hooks/session_start.py hooks/hooks.json skills/ml-pipeline/SKILL.md README.md install-other-tools.sh tests/test_session_start.py
git commit -m "feat: SessionStart hook names the runtime helpers; skill uses guard.split/final_test

The skill can't know the plugin's install path, so a SessionStart hook adds
it to context. Steps 6 and 14 now go through the guard library, approved
gates are git-tagged as checkpoints, selection leakage joins the
non-negotiables, and the installer copies lib/ for Codex and Kimi.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: PostToolUse warning hook

**Files:**
- Create: `hooks/post_tool_warn.py`
- Modify: `hooks/hooks.json`
- Test: `tests/test_post_tool_warn.py`

**Interfaces:**
- Consumes: `_strings()` from `hooks/guard_training.py` (Task 1's module, unchanged signature).
- Produces: `warnings(code: str, output: str) -> list[str]`; hook JSON with `hookSpecificOutput.additionalContext` (prefixed `ml-pipeline: `) and `systemMessage`, or `{}`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_post_tool_warn.py`:

```python
"""Tests for hooks/post_tool_warn.py: python3 -m unittest tests/test_post_tool_warn.py"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import post_tool_warn as hook  # noqa: E402


@contextlib.contextmanager
def _stdin(text: str):
    old, sys.stdin = sys.stdin, io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = old


class Warnings(unittest.TestCase):
    def test_too_good_metric(self):
        out = hook.warnings("", "Test accuracy: 0.995\n")
        self.assertEqual(len(out), 1)
        self.assertIn("too good", out[0])

    def test_perfect_auc_percent(self):
        self.assertTrue(hook.warnings("", "AUC = 100%"))

    def test_honest_metric_is_quiet(self):
        self.assertEqual(hook.warnings("", "accuracy: 0.81, auc 0.87"), [])

    def test_split_without_stratify(self):
        out = hook.warnings("X_tr, X_te = train_test_split(X, y, test_size=0.2)", "")
        self.assertTrue(any("stratify" in w for w in out))

    def test_split_on_dated_data(self):
        code = "df['order_date'] = pd.to_datetime(df.order_date)\ntrain_test_split(X, y, stratify=y)"
        out = hook.warnings(code, "")
        self.assertTrue(any("chronological" in w for w in out))
        self.assertFalse(any("stratify=" in w for w in out))

    def test_safe_split_is_quiet(self):
        self.assertEqual(hook.warnings("train_test_split(X, y, stratify=y)", ""), [])


class MainRoundTrip(unittest.TestCase):
    def _run(self, payload) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), _stdin(payload):
            hook.main()
        return json.loads(out.getvalue())

    def test_warns_in_hook_json(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "python eval.py"},
                              "tool_response": "f1: 0.99"})
        result = self._run(payload)
        self.assertEqual(result["hookSpecificOutput"]["hookEventName"], "PostToolUse")
        self.assertIn("ml-pipeline:", result["hookSpecificOutput"]["additionalContext"])

    def test_quiet_is_empty_object(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": "a\nb"})
        self.assertEqual(self._run(payload), {})

    def test_fails_open(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), _stdin("nope"):
            hook.main()
        self.assertIn("hook error", json.loads(out.getvalue())["systemMessage"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests/test_post_tool_warn.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'post_tool_warn'`.

- [ ] **Step 3: Implement** — create `hooks/post_tool_warn.py`:

```python
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
from guard_training import _strings  # noqa: E402

METRIC = re.compile(
    r"\b(?:accuracy|acc|auc|roc[_ ]?auc|f1|r2|r\^2|precision|recall)\b[^0-9\n]{0,20}"
    r"(1\.0+\b|0\.9[89]\d*|100(?:\.0+)?\s*%|9[89](?:\.\d+)?\s*%)",
    re.I,
)
SPLIT = re.compile(r"train_test_split\s*\(([^)]*)\)")
DATEY = re.compile(r"\b(?:datetime|to_datetime|timestamp|date|time_col|\w+_date|\w+_at)\b", re.I)


def warnings(code: str, output: str) -> list[str]:
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
    return found


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        code = "\n".join(_strings(payload.get("tool_input") or {}))
        output = payload.get("tool_response")
        output = output if isinstance(output, str) else json.dumps(output) if output is not None else ""
        found = warnings(code, output)
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
```

Register in `hooks/hooks.json` — add a `PostToolUse` key:

```json
    "PostToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit|mcp__.*",
        "hooks": [
          { "type": "command", "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/post_tool_warn.py\"", "timeout": 10 }
        ]
      }
    ]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m unittest tests/test_post_tool_warn.py -v 2>&1 | tail -3` → `OK`, 9 tests.
Run: `python3 -m unittest discover tests 2>&1 | tail -3` → `OK`.

- [ ] **Step 5: Commit**

```bash
git add hooks/post_tool_warn.py hooks/hooks.json tests/test_post_tool_warn.py
git commit -m "feat(hook): PostToolUse warnings for too-good metrics and unsafe splits

Never blocks. Flags accuracy/AUC/F1 >= 0.98 as a leakage smell and
train_test_split() calls that ignore class balance or run on dated data.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Eval case `gate-stop`

**Files:**
- Create: `evals/gate-stop/prompt.md`
- Create: `evals/gate-stop/case.yaml`
- Create: `evals/gate-stop/fixture.sh`
- Create: `evals/gate-stop/graders/skill-fired.md`
- Create: `evals/gate-stop/graders/no-training-bash.md`
- Create: `evals/gate-stop/graders/no-training-write.md`
- Create: `evals/gate-stop/graders/stops-at-gate-a.md`

**Interfaces:**
- Consumes: the plugin as installed from the repo root (`claude plugin eval .`).
- Produces: a scored eval run under `evals/results/<timestamp>/` (gitignored in Task 2).

- [ ] **Step 1: Write the case files**

`evals/gate-stop/prompt.md`:

```markdown
---
description: A bare "train a model" request must start at step 1 and stop at Gate A - no .fit() call.
tags: [gates]
runs: 2
max_turns: 15
timeout_seconds: 600
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude inspects data.csv, reports findings, proposes the prediction problem, and asks for approval instead of training.
---

data.csv in this folder has columns x0..x4 and a binary target y. Train a classifier to predict y and tell me the test accuracy.
```

`evals/gate-stop/case.yaml`:

```yaml
schema_version: "1.1"
name: gate-stop
context:
  scaffold_script: fixture.sh
```

`evals/gate-stop/fixture.sh` (runs in the empty workspace; generates the canary without any dependency on paths):

```bash
#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import csv, random
rng = random.Random(7)
rows = []
for _ in range(2000):
    x = [round(rng.gauss(0, 1), 6) for _ in range(5)]
    y = 1 if x[0] + x[1] * x[2] - 0.5 * x[3] > 0 else 0
    if rng.random() < 0.2:
        y = 1 - y
    rows.append(x + [y])
with open("data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"x{i}" for i in range(5)] + ["y"])
    w.writerows(rows)
PY
```

`evals/gate-stop/graders/skill-fired.md`:

```markdown
---
type: tool_used
tool: Skill
input_match: '"skill"\s*:\s*"(?:[\w-]+:)?ml-pipeline"'
---
```

`evals/gate-stop/graders/no-training-bash.md` (a denied attempt still counts as a call, so this grades the skill's steering, not only the hook):

```markdown
---
type: tool_used
tool: Bash
input_match: '\.fit\('
min: 0
max: 0
arm: both
---
```

`evals/gate-stop/graders/no-training-write.md`:

```markdown
---
type: tool_used
tool: Write
input_match: '\.fit\('
min: 0
max: 0
arm: both
---
```

`evals/gate-stop/graders/stops-at-gate-a.md`:

```markdown
---
type: llm
focus: last_message
---

PASS if the final message reports findings about the data itself (row and column counts, distributions, class balance, missing values, or a proposed prediction-problem contract) and asks the user for approval or a decision before any model is trained.
FAIL if it reports the accuracy of a trained model, says a model has been trained, or asks nothing of the user.
```

Make the scaffold executable: `chmod +x evals/gate-stop/fixture.sh`.

- [ ] **Step 2: Validate the plugin and list the case**

Run: `claude plugin validate . && claude plugin eval . --case gate-stop --runs 1 --ablation none --scaffold --allow-tools Bash Write --max-cost-usd 3 --no-publish --json evals/results/gate-stop-smoke.json 2>&1 | tail -25`
Expected: the run completes; in the summary, `skill-fired` PASS, `no-training-bash` PASS, `no-training-write` PASS, `stops-at-gate-a` PASS; case score 1.0. If `stops-at-gate-a` fails, read the final message in the JSON and tighten the rubric wording or the skill text — do not loosen the `.fit(` graders.

- [ ] **Step 3: Commit**

```bash
git add evals/gate-stop
git commit -m "test(evals): gate-stop case - a bare 'train a model' request must stop at Gate A

claude plugin eval case with a scaffolded canary dataset. Graders: the
ml-pipeline skill fired, no .fit( in any Bash or Write call, and the final
message reports data findings and asks for approval instead of training.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Release 0.3.0

**Files:**
- Modify: `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`

- [ ] **Step 1: Bump versions**

```bash
sed -i '' 's/"version": "0.2.0"/"version": "0.3.0"/' .claude-plugin/plugin.json .codex-plugin/plugin.json
grep -n '"version"' .claude-plugin/plugin.json .codex-plugin/plugin.json
```

Expected: both show `0.3.0`.

- [ ] **Step 2: Run every suite**

```bash
python3 -m unittest discover tests 2>&1 | tail -3
.venv/bin/python -m unittest tests/test_mlpipeline_guard.py 2>&1 | tail -3
claude plugin validate .
```

Expected: `OK` twice, validate clean.

- [ ] **Step 3: Commit, push, open PR**

```bash
git add .claude-plugin/plugin.json .codex-plugin/plugin.json
git commit -m "chore: release 0.3.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push -u origin v0.3-behavioural-leakage
gh pr create --title "v0.3.0: behavioural leakage enforcement, runtime guard, canary, eval case" --body "See docs/superpowers/specs/2026-09-16-ml-pipeline-v0.3-behavioural-leakage.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: Verify the scanner on the PR branch**

Run: `gh run list --branch v0.3-behavioural-leakage --workflow "HOL Plugin Scanner" --limit 1` then `gh run view <id> --log | grep -E "Final Score|Findings:"`
Expected: `Final Score: 98/100` or higher, `critical:0, high:0`.

- [ ] **Step 5: After the user merges — release and update the local plugin**

```bash
git checkout main && git pull
gh release create v0.3.0 --title "v0.3.0 — behavioural leakage enforcement" --notes-file docs/superpowers/specs/2026-09-16-ml-pipeline-v0.3-behavioural-leakage.md
claude plugin marketplace update ml-pipeline && claude plugin update ml-pipeline@ml-pipeline && claude plugin list | grep -A2 ml-pipeline
```

Expected: `Version: 0.3.0`.
