# ml-pipeline v0.5.0 — Rigorous Data Work Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the data work before training complete and provable: a full-data proof, a complete explained record for every column, train-only target statistics, and a full-data training receipt, each checked by the hook when a gate is recorded.

**Architecture:** The guard library (`lib/mlpipeline_guard.py`, one file that projects copy as `ml_pipeline/guard.py`) computes every number and writes one small record per artifact into `ml_pipeline/`; agents fill only word fields. A new stdlib module `hooks/gate_records.py` checks those records against the profile's real column list when a gate line is written; `hooks/guard_training.py` wires it in and gains the pilot rule and the `target-stats-outside-train` red flag.

**Tech Stack:** Python stdlib for hooks (3.9+); pandas + numpy + matplotlib for the guard (dev venv, Python 3.11); `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-22-ml-pipeline-v0.5-rigorous-data-work-design.md`

## Global Constraints

- Hooks: Python ≥ 3.9, stdlib only, no `subprocess` / `eval` / `exec`, fail open, 10-second timeout.
- Guard: pandas + numpy required, matplotlib imported lazily with the `Agg` backend; no new dependencies; stays **one file** (projects copy it as `ml_pipeline/guard.py`).
- Constants: `RARE_LEVEL_FRAC = 0.01`, `COLLINEAR_RHO = 0.9`, `MIN_NOTE_CHARS = 40`, `LIST_CAP = 10`, `PAGE_COLUMNS = 12`; unchanged `SMALL_DATA_ROWS = 5000`, `IMBALANCE_MINORITY = 0.10`, `HIGH_CARD_LEVELS = 50`, `LEAK_CORR = 0.95`, `LEAK_PURITY = 0.98`, `MIN_PNG_BYTES = 1024`; removed `SAMPLE_ROWS`.
- Ledger lines are strict and current-run scoped (after the newest `## Run:` heading): `- Override: partial data - …`, `- Override: pilot - …`, `- Override: red flag target-stats-outside-train - …`, `- Override: final test - …`, `- Cleaning: none needed - reason: …`.
- CSV records are read as `utf-8-sig` and written as plain UTF-8.
- Versions → `0.5.0` in `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`.
- Work on branch `v0.5-rigorous-data-work` (created from `main` in Task 1). Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Development sessions run with enforcement switched off by the user at launch (the installed hook denies edits to these test strings). Never work around the hook.
- Test commands: hooks `python3 -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py`; guard `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py tests/test_v05_end_to_end.py`; probe `python3 tests/probe_hook_audit.py` (must end `0 of 23 cases behave wrongly`).

## Review Focus

1. **A record edited in Excel or Numbers** (UTF-8 byte-order mark, CRLF line endings) must still merge in the guard and pass the hook. Tests: Task 5 `test_dictionary_keeps_words_and_survives_excel`, Task 9 `test_excel_saved_dictionary_passes`.
2. **Headerless data or integer / awkward column names** (`0`, `'weight, "kg"'`) must round-trip as the same strings through profile, CSVs, and `pages.json`. Test: Task 5 `test_awkward_column_names_round_trip`.
3. **An all-missing column** must get the common profile keys only, a figure panel, a missingness row, and no demand for kind-specific statistics at the gate. Tests: Task 3 `test_all_missing_column`, Task 6 `test_all_missing_column_gets_a_panel`, Task 9 `test_all_missing_column_needs_common_keys_only`.
4. **Wide data** (hundreds of columns) must split into pages and keep deny messages readable (at most 10 names, then `and N more`). Tests: Task 6 `test_wide_frame_pages`, Task 9 `test_long_lists_are_capped`.
5. **Ordinary code near the new rules** must not be denied: column selection (`X.loc[:, cols]`, `X[:, :5]`), a model fit after a train-only selector, and transformer fits on a slice. Tests: Task 11 `test_whole_split_and_column_selection_allowed`, `test_transformer_slices_are_not_pilots`, and the allowed cases in `test_target_stats_outside_the_training_split`.

## File structure

| File | Responsibility |
|---|---|
| `lib/mlpipeline_guard.py` (modify) | New sections: sources (`load`, `derive`), profile schema 2, the note rule, records (`data_dictionary`, `missingness`, `categorical_plan`, `label_audit`), `variable_figures`, `log_cleaning`, `train_associations`, training receipts (`fit`, `record_training`), `importance`; split state and final-test fixes |
| `hooks/gate_records.py` (create) | Stdlib gate-time checks for every v0.5 record and the note rule |
| `hooks/guard_training.py` (modify) | Wires `gate_records` into `gate_problems`; figure-explanation rule; pilot rule; `target-stats-outside-train` red flag |
| `hooks/session_start.py` (modify) | Names the new helpers |
| `tests/note_rule_cases.json` (create) | Shared note-rule vectors run against both implementations |
| `tests/v05_fixtures.py` (create) | Complete v0.5 records as plain files (stdlib) for the hook tests |
| `tests/test_gate_records.py` (create) | Stdlib tests for `hooks/gate_records.py` |
| `tests/test_v05_end_to_end.py` (create) | Honest path through the real guard and hook; each shortcut stopped |
| `tests/test_mlpipeline_guard.py`, `tests/test_guard_training.py`, `tests/test_session_start.py` (modify) | New and updated tests |
| `skills/ml-pipeline/SKILL.md`, `README.md`, `.github/workflows/tests.yml`, `evals/understand-first/graders/*`, manifests (modify/create) | Docs, CI, eval, release |

---

### Task 1: Split state, a single-use test set, strict final-test override

**Files:**
- Modify: `lib/mlpipeline_guard.py` (`split`, `_override_allows`, exception classes, new `_current_run`, `_split_state`)
- Test: `tests/test_mlpipeline_guard.py` (`Split`, `FinalTest`)

**Interfaces:**
- Produces: `guard._split_state(state_dir) -> dict` (raises `LeakageError` when absent or older than v0.5); state keys `n_train`, `n_val`, `train_fingerprint`, `val_fingerprint`; `guard.split(..., pipeline_md=None)`; `guard._current_run(markdown) -> str`.

- [ ] **Step 1: Create the branch**

```bash
cd ~/DataJP/ml-pipeline && git checkout -b v0.5-rigorous-data-work
```

- [ ] **Step 2: Write the failing tests**

Add to class `Split`:

```python
    def test_state_records_train_and_validation(self):
        tr, va, te = guard.split(frame(), target="y", state_dir=self.state)
        state = json.loads((self.state / guard.STATE_FILE).read_text())
        self.assertEqual((state["n_train"], state["n_val"]), (len(tr), len(va)))
        self.assertEqual((len(state["train_fingerprint"]), len(state["val_fingerprint"])), (64, 64))

    def test_resplitting_before_any_test_touch_is_fine(self):
        guard.split(frame(), target="y", state_dir=self.state)
        guard.split(frame(), target="y", state_dir=self.state)
```

Add to class `FinalTest`:

```python
    def test_resplit_after_final_test_raises(self):
        self._run(self.te)
        with self.assertRaises(guard.TestSetAlreadyUsed):
            guard.split(frame(), target="y", state_dir=self.state)

    def test_resplit_allowed_with_final_test_override(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved a new test strategy 2026-09-24 - reason: new data\n")
        guard.split(frame(), target="y", state_dir=self.state)
        self.assertEqual(json.loads((self.state / guard.STATE_FILE).read_text())["test_touches"], 0)

    def test_override_must_be_the_strict_line(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() + "### Override requested by the agent for the final test\n")
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)

    def test_override_from_an_older_run_does_not_count(self):
        self._run(self.te)
        self.md.write_text(self.md.read_text() +
                           "- Override: final test - user approved 2026-09-18 - reason: x\n## Run: retrain (2026-09-24)\n")
        with self.assertRaises(guard.TestSetAlreadyUsed):
            self._run(self.te)
```

- [ ] **Step 3: Run them and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Split tests.test_mlpipeline_guard.FinalTest -v`
Expected: 5 failures — `KeyError: 'n_train'`, two `TestSetAlreadyUsed not raised` (resplit, strict line), one `TestSetAlreadyUsed not raised` (older run); `test_resplit_allowed_with_final_test_override` and `test_resplitting_before_any_test_touch_is_fine` may pass already (existing behaviour, kept as guards).

- [ ] **Step 4: Implement**

In `lib/mlpipeline_guard.py`, move `class TestSetAlreadyUsed` and `class TestSetTampered` up to sit directly after `class OverlapLeakage`. Delete `_NEGATED` and the old `_override_allows`. After the exception classes add:

```python
_RUN_HEAD = re.compile(r"^[ \t]*##[ \t]+Run[ \t]*:", re.I | re.M)


def _current_run(markdown: str) -> str:
    """The newest run: from the last ``## Run:`` heading on, or the whole ledger (the hook's rule)."""
    heads = list(_RUN_HEAD.finditer(markdown))
    return markdown[heads[-1].start():] if heads else markdown


def _override_allows(pipeline_md: Path, what: str) -> bool:
    """True when the current run has the strict line ``- Override: <what> - ...``."""
    if not pipeline_md.is_file():
        return False
    line = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*" + re.escape(what) + r"[ \t]*-", re.I | re.M)
    return bool(line.search(_current_run(pipeline_md.read_text())))


def _split_state(state_dir) -> dict:
    """The frozen split's state; raises when there is none or it predates train/validation sizes."""
    path = Path(state_dir) / STATE_FILE
    if not path.is_file():
        raise LeakageError("no split state: call guard.split() at step 6 first")
    state = json.loads(path.read_text())
    if "n_train" not in state:
        raise LeakageError("split state is from an older guard: re-run guard.split()")
    return state
```

Change `split`'s signature to add `pipeline_md: str | Path | None = None` after `state_dir`, and insert right after the `TemporalSplitRequired` check:

```python
    state_dir = Path(state_dir)
    ledger = Path(pipeline_md) if pipeline_md is not None else state_dir / "PIPELINE.md"
    previous = state_dir / STATE_FILE
    if (previous.is_file() and json.loads(previous.read_text()).get("test_touches", 0) >= 1
            and not _override_allows(ledger, "final test")):
        raise TestSetAlreadyUsed(
            "the frozen test set was already evaluated; re-splitting would reset that. Agree a new test "
            "strategy with the user and record `- Override: final test - ...` in PIPELINE.md first")
```

Replace the state-writing block at the end of `split` with:

```python
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE).write_text(json.dumps({
        "test_fingerprint": _fingerprint(test),
        "n_test": int(len(test)),
        "test_touches": 0,
        "duplicates_dropped": int(dropped),
        "split": kind,
        "created": date.today().isoformat(),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "train_fingerprint": _fingerprint(train),
        "val_fingerprint": _fingerprint(val),
    }, indent=2))
    return train, val, test
```

- [ ] **Step 5: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK` (all old and new tests).

- [ ] **Step 6: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "fix(guard): a used test set cannot be re-split; strict run-scoped final-test override; split records train/val

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `guard.load` and `guard.derive`

**Files:**
- Modify: `lib/mlpipeline_guard.py` (new section `# ---- sources (step 1)` before the profile section)
- Test: `tests/test_mlpipeline_guard.py` (new class `Sources`; add `import hashlib`)

**Interfaces:**
- Produces: `guard.load(path, *, state_dir="ml_pipeline", **read_kwargs) -> pd.DataFrame`; `guard.derive(df, name, *, parents, how, state_dir="ml_pipeline") -> pd.DataFrame`; `guard.PartialData(LeakageError)`; `guard.SOURCE_FILE = "source.json"`; `guard._sources(state_dir: Path) -> dict` (`{"schema_version": "1", "sources": {...}}`).

- [ ] **Step 1: Write the failing tests**

```python
class Sources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"
        self.csv = self.root / "raw.csv"
        frame(50).to_csv(self.csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def sources(self):
        return json.loads((self.state / guard.SOURCE_FILE).read_text())["sources"]

    def test_load_records_hash_rows_and_columns(self):
        df = guard.load(self.csv, state_dir=self.state)
        entry = self.sources()[str(self.csv)]
        self.assertEqual(len(df), 50)
        self.assertEqual(entry["sha256"], hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual((entry["kind"], entry["n_rows"], entry["n_cols"]), ("file", 50, 3))
        self.assertEqual(entry["columns"], ["a", "b", "y"])

    def test_partial_loads_are_refused(self):
        for kwargs in ({"nrows": 10}, {"skiprows": 5}, {"skipfooter": 1}, {"chunksize": 10},
                       {"iterator": True}, {"usecols": ["a"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(guard.PartialData):
                guard.load(self.csv, state_dir=self.state, **kwargs)
        self.assertFalse((self.state / guard.SOURCE_FILE).exists())

    def test_other_read_options_pass_through(self):
        self.assertEqual(str(guard.load(self.csv, state_dir=self.state, dtype={"b": "float64"})["b"].dtype), "float64")

    def test_tab_separated(self):
        tsv = self.root / "raw.tsv"
        frame(20).to_csv(tsv, sep="\t", index=False)
        self.assertEqual(list(guard.load(tsv, state_dir=self.state).columns), ["a", "b", "y"])

    def test_unsupported_suffix(self):
        odd = self.root / "raw.dat"
        odd.write_text("a\n1\n")
        with self.assertRaises(ValueError):
            guard.load(odd, state_dir=self.state)

    def test_derive_records_the_chain(self):
        df = guard.load(self.csv, state_dir=self.state)
        guard.derive(df.assign(c=1), "raw_plus_c", parents=[str(self.csv)], how="added a constant column c",
                     state_dir=self.state)
        entry = self.sources()["raw_plus_c"]
        self.assertEqual((entry["kind"], entry["parents"], entry["n_rows"], entry["n_cols"]),
                         ("derived", [str(self.csv)], 50, 4))

    def test_derive_needs_known_parents_and_a_reason(self):
        df = guard.load(self.csv, state_dir=self.state)
        with self.assertRaises(KeyError):
            guard.derive(df, "x", parents=["never_loaded.csv"], how="join", state_dir=self.state)
        with self.assertRaises(ValueError):
            guard.derive(df, "x", parents=[str(self.csv)], how="  ", state_dir=self.state)
        with self.assertRaises(ValueError):
            guard.derive(df, "x", parents=[], how="join", state_dir=self.state)
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Sources -v`
Expected: every test errors with `AttributeError: module 'mlpipeline_guard' has no attribute 'load'` / `'SOURCE_FILE'` / `'PartialData'`.

- [ ] **Step 3: Implement**

```python
# ------------------------------------------------------------------ sources (step 1)
SOURCE_FILE = "source.json"
_READERS = {
    ".csv": ("read_csv", {}), ".txt": ("read_csv", {}), ".tsv": ("read_csv", {"sep": "\t"}),
    ".parquet": ("read_parquet", {}), ".pq": ("read_parquet", {}),
    ".xlsx": ("read_excel", {}), ".xls": ("read_excel", {}),
    ".json": ("read_json", {}), ".jsonl": ("read_json", {"lines": True}), ".ndjson": ("read_json", {"lines": True}),
    ".feather": ("read_feather", {}),
}
_PARTIAL_KWARGS = ("nrows", "skiprows", "skipfooter", "chunksize", "iterator", "usecols")


class PartialData(LeakageError):
    """A load or profile that does not cover the full data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(state_dir: Path) -> dict:
    path = Path(state_dir) / SOURCE_FILE
    return json.loads(path.read_text()) if path.is_file() else {"schema_version": "1", "sources": {}}


def _record_source(state_dir: Path, name: str, entry: dict) -> None:
    data = _sources(state_dir)
    data["sources"][name] = entry
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / SOURCE_FILE).write_text(json.dumps(data, indent=2))


def load(path, *, state_dir: str | Path = "ml_pipeline", **read_kwargs) -> pd.DataFrame:
    """Step 1: read a whole data file and record its hash, rows, and columns in source.json."""
    partial = sorted(key for key in read_kwargs if key in _PARTIAL_KWARGS)
    if partial:
        raise PartialData(f"{', '.join(partial)} would load part of the data: load the whole file, then give "
                          "unwanted columns role 'drop' in the data dictionary")
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _READERS:
        raise ValueError(f"unsupported file type {suffix!r}; supported: {', '.join(sorted(_READERS))}")
    reader, defaults = _READERS[suffix]
    df = getattr(pd, reader)(path, **{**defaults, **read_kwargs})
    _record_source(Path(state_dir), str(path), {
        "kind": "file", "sha256": _sha256(path), "bytes": path.stat().st_size,
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": [str(c) for c in df.columns],
        "loaded": date.today().isoformat()})
    return df


def derive(df: pd.DataFrame, name: str, *, parents, how: str, state_dir: str | Path = "ml_pipeline") -> pd.DataFrame:
    """Step 5: register a frame built from loaded sources (a join or an aggregation) as a source."""
    state_dir = Path(state_dir)
    parents = list(parents)
    if not parents:
        raise ValueError("a derived frame needs at least one parent source")
    known = _sources(state_dir)["sources"]
    unknown = [p for p in parents if p not in known]
    if unknown:
        raise KeyError(f"parents not loaded with guard.load() or guard.derive(): {unknown}")
    if not str(how).strip():
        raise ValueError("describe how the frame was derived")
    _record_source(state_dir, str(name), {
        "kind": "derived", "parents": parents, "how": str(how).strip(),
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": [str(c) for c in df.columns],
        "loaded": date.today().isoformat()})
    return df
```

- [ ] **Step 4: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): load() and derive() record every source's hash, rows, and columns

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Profile schema 2

**Files:**
- Modify: `lib/mlpipeline_guard.py` (`_column_stats`, `_target_info`, `profile`, constants; new `_source_match`)
- Test: `tests/test_mlpipeline_guard.py` (`Profile`: replace `test_sampling_flag`, add tests)

**Interfaces:**
- Consumes: `guard._sources(state_dir)` (Task 2).
- Produces: profile `schema_version "2"`; per-column keys (spec §1.2); `target.class_counts` (classification) or `target.mean/std/min/max` (regression); `profile["source"] = {"name", "n_rows", "full_data", ["reason"]}`; `guard.RARE_LEVEL_FRAC = 0.01`; `guard._column_stats(series, kind) -> dict`; `guard._target_info(y) -> dict`.

- [ ] **Step 1: Write the failing tests**

Delete `test_sampling_flag` and add to class `Profile`:

```python
    def test_every_row_counts(self):
        n = 250_000
        df = pd.DataFrame({"c": np.zeros(n), "y": np.arange(n) % 2})
        df.loc[n - 10:, "c"] = 5.0
        p = guard.profile(df, target="y", state_dir=self.state)
        self.assertEqual(p["schema_version"], "2")
        self.assertEqual(p["columns"]["c"]["zeros"], n - 10)
        self.assertNotIn("sampled", p["shape"])
        self.assertFalse(hasattr(guard, "SAMPLE_ROWS"))

    def test_kind_specific_statistics(self):
        df = self._df()
        c = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)["columns"]
        common = {"kind", "dtype", "non_null", "missing", "missing_frac", "n_unique", "unique_frac", "memory_bytes"}
        levels = {"n_levels", "top", "top_counts", "rare_levels"}
        expected = {
            "age": {"mean", "std", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "skew",
                    "kurtosis", "zeros", "negatives", "outliers", "outlier_frac"},
            "zip": levels, "flag": levels, "admitted_at": {"start", "end", "span_days"},
            "notes": {"length_min", "length_median", "length_max"}, "patient_id": set(),
        }
        for col, keys in expected.items():
            with self.subTest(col=col):
                self.assertLessEqual(common | keys, set(c[col]))
        self.assertEqual(c["zip"]["n_levels"], df["zip"].nunique())
        self.assertEqual(sum(c["flag"]["top_counts"].values()), 1000)
        self.assertEqual(c["admitted_at"]["span_days"], 41)

    def test_all_missing_column(self):
        df = self._df()
        df["empty"] = np.nan
        info = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)["columns"]["empty"]
        self.assertEqual((info["missing"], info["non_null"]), (1000, 0))
        self.assertNotIn("mean", info)

    def test_target_counts_and_regression_stats(self):
        df = self._df()
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        self.assertEqual(p["target"]["class_counts"], {str(k): int(v) for k, v in df["y"].value_counts().items()})
        df["amount"] = np.random.default_rng(1).normal(size=len(df))
        t = guard.profile(df, target="amount", time_col="admitted_at", state_dir=self.state)["target"]
        self.assertIsNone(t["class_counts"])
        self.assertAlmostEqual(t["mean"], round(float(df["amount"].mean()), 4))

    def _loaded(self, n=300):
        csv_path = Path(self.tmp.name) / "raw.csv"
        frame(n).to_csv(csv_path, index=False)
        return csv_path, guard.load(csv_path, state_dir=self.state)

    def test_full_data_when_the_frame_is_a_loaded_source(self):
        csv_path, df = self._loaded()
        self.assertEqual(guard.profile(df, target="y", state_dir=self.state)["source"],
                         {"name": str(csv_path), "n_rows": 300, "full_data": True})

    def test_a_sample_is_not_full_data(self):
        _, df = self._loaded()
        src = guard.profile(df.sample(frac=0.5, random_state=0), target="y", state_dir=self.state)["source"]
        self.assertFalse(src["full_data"])
        self.assertIn("matches no loaded source", src["reason"])

    def test_no_source_loaded(self):
        src = guard.profile(frame(), target="y", state_dir=self.state)["source"]
        self.assertFalse(src["full_data"])
        self.assertIn("guard.load()", src["reason"])

    def test_named_source_must_exist(self):
        self._loaded()
        src = guard.profile(frame(), target="y", source="nope.csv", state_dir=self.state)["source"]
        self.assertIn("not in source.json", src["reason"])
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Profile -v`
Expected: the new tests fail (`schema_version '1' != '2'`, `KeyError: 'zeros'`, `KeyError: 'class_counts'`, `KeyError: 'source'`); the old ones pass.

- [ ] **Step 3: Implement**

Replace the constant `SAMPLE_ROWS = 200_000` with `RARE_LEVEL_FRAC = 0.01`. Replace `_column_stats` and `_target_info`:

```python
def _column_stats(series: pd.Series, kind: str) -> dict:
    """The describe + info equivalent for one column, on every row."""
    n = len(series)
    missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    info = {"kind": kind, "dtype": str(series.dtype), "non_null": int(n - missing), "missing": missing,
            "missing_frac": round(missing / n, 4) if n else 0.0,
            "n_unique": n_unique, "unique_frac": round(n_unique / n, 4) if n else 0.0,
            "memory_bytes": int(series.memory_usage(deep=True, index=False))}
    values = series.dropna()
    if not len(values):
        return info
    if kind == "numeric":
        v = values.astype(float)
        q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = q[0.75] - q[0.25]
        outliers = int(((v < q[0.25] - 1.5 * iqr) | (v > q[0.75] + 1.5 * iqr)).sum()) if iqr > 0 else 0
        info.update(mean=_num(v.mean()), std=_num(v.std()), min=_num(v.min()), p01=_num(q[0.01]),
                    p05=_num(q[0.05]), p25=_num(q[0.25]), median=_num(q[0.5]), p75=_num(q[0.75]),
                    p95=_num(q[0.95]), p99=_num(q[0.99]), max=_num(v.max()),
                    skew=_num(v.skew()) if len(v) > 2 else 0.0, kurtosis=_num(v.kurt()) if len(v) > 3 else 0.0,
                    zeros=int((v == 0).sum()), negatives=int((v < 0).sum()),
                    outliers=outliers, outlier_frac=round(outliers / len(v), 4))
    elif kind in ("categorical", "bool"):
        counts = values.astype(str).value_counts()
        top = counts.head(10)
        info.update(n_levels=int(len(counts)),
                    top={k: round(float(c) / len(values), 4) for k, c in top.items()},
                    top_counts={k: int(c) for k, c in top.items()},
                    rare_levels=int((counts / len(values) < RARE_LEVEL_FRAC).sum()))
    elif kind == "datetime":
        start, end = values.min(), values.max()
        info.update(start=str(start)[:10], end=str(end)[:10], span_days=int((end - start).days))
    elif kind == "text":
        lengths = values.astype(str).str.len()
        info.update(length_min=int(lengths.min()), length_median=int(lengths.median()), length_max=int(lengths.max()))
    return info


def _target_info(y: pd.Series) -> dict:
    values = y.dropna()
    n_unique = int(values.nunique())
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and n_unique > 20:
        v = values.astype(float)
        return {"name": str(y.name), "task": "regression", "n_classes": None, "class_balance": None,
                "minority_frac": None, "class_counts": None,
                "mean": _num(v.mean()), "std": _num(v.std()), "min": _num(v.min()), "max": _num(v.max())}
    counts = values.astype(str).value_counts()
    balance = counts / counts.sum()
    return {"name": str(y.name), "task": "binary" if n_unique == 2 else "multiclass", "n_classes": n_unique,
            "class_balance": {k: round(float(v), 4) for k, v in balance.items()},
            "minority_frac": round(float(balance.min()), 4),
            "class_counts": {k: int(c) for k, c in counts.items()}}


def _source_match(df: pd.DataFrame, source: str | None, state_dir: Path) -> dict:
    """Whether ``df`` is a whole loaded source: same row count and the same column set."""
    sources = _sources(state_dir)["sources"]
    columns = {str(c) for c in df.columns}
    for name in ([source] if source is not None else list(sources)):
        entry = sources.get(name)
        if entry and entry["n_rows"] == len(df) and set(entry["columns"]) == columns:
            return {"name": name, "n_rows": int(len(df)), "full_data": True}
    if not sources:
        reason = "no source loaded with guard.load(); the profile cannot be tied to a full data file"
    elif source is not None and source not in sources:
        reason = f"source {source!r} is not in source.json"
    else:
        reason = f"{len(df):,} rows x {len(columns)} columns matches no loaded source (a sample or a slice?)"
    return {"name": source, "n_rows": int(len(df)), "full_data": False, "reason": reason}
```

In `profile`: add the keyword `source: str | None = None` after `group_col`; delete the `sampled` / `sample` lines; compute `columns = {str(col): _column_stats(df[col], kinds[col]) for col in df.columns}`; set `"schema_version": "2"`; set `"shape": {"n_rows": int(n), "n_cols": int(df.shape[1]), "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2)}`; add `"source": _source_match(df, source, Path(state_dir)),` after `"shape"`; call `_leakage_suspects(df, target, time_col, kinds, target_info["task"])` on the full frame. Update the docstring's last paragraph to "Every statistic uses every row."

- [ ] **Step 4: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): profile schema 2 - every row, full describe/info per column, class counts, full-data match

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The real-explanation rule in the guard; `fig()` and `eda_figures()` follow it

**Files:**
- Create: `tests/note_rule_cases.json`
- Modify: `lib/mlpipeline_guard.py` (new `# ---- the real-explanation rule` section before the figures section; `fig`; `eda_figures`)
- Test: `tests/test_mlpipeline_guard.py` (new class `NoteRule`; `Figures`; add `import re`)

**Interfaces:**
- Produces: `guard.note_problem(note: str, stats: dict | None = None) -> str | None` returning exactly `"shorter than 40 characters"`, `"contains no number"`, `"cites no statistic of this column"`, or `None`; `guard.MIN_NOTE_CHARS = 40`; `guard.COLLINEAR_RHO = 0.9`. `tests/note_rule_cases.json` is shared with Task 9.

- [ ] **Step 1: Write the shared vectors**

`tests/note_rule_cases.json`:

```json
{
  "stats": {"kind": "numeric", "non_null": 196, "missing": 4, "missing_frac": 0.0213, "n_unique": 71,
            "median": 54.0, "p99": 91.2, "top_counts": {}},
  "cases": [
    {"note": "age: shows the distribution", "column": true, "problem": "shorter than 40 characters"},
    {"note": "age: shows the distribution of the values across all patients", "column": true, "problem": "contains no number"},
    {"note": "age: looks fine overall, there are 12 things to check here", "column": true, "problem": "cites no statistic of this column"},
    {"note": "age: about 1 in 10 look odd and 0 are negative, worth a look", "column": true, "problem": "cites no statistic of this column"},
    {"note": "age: right-skewed, median 54, 2% missing; cap at the 99th percentile (91) before scaling", "column": true, "problem": null},
    {"note": "age: median 54.0 years, which is typical for this cohort", "column": true, "problem": null},
    {"note": "age: 2.13% of values are missing, spread evenly over time", "column": true, "problem": null},
    {"note": "age: 196 recorded values in total, which is nearly every row", "column": true, "problem": null},
    {"note": "Page 1 of 3: 12 columns, 4 with missing values and 2 skewed", "column": false, "problem": null},
    {"note": "Page one shows every column's distribution in a grid", "column": false, "problem": "contains no number"}
  ]
}
```

- [ ] **Step 2: Write the failing tests**

```python
NOTE_CASES = json.loads((Path(__file__).resolve().parent / "note_rule_cases.json").read_text())


class NoteRule(unittest.TestCase):
    def test_shared_cases(self):
        for case in NOTE_CASES["cases"]:
            with self.subTest(note=case["note"]):
                stats = NOTE_CASES["stats"] if case["column"] else None
                self.assertEqual(guard.note_problem(case["note"], stats), case["problem"])
```

In class `Figures`, change `test_fig_saves_and_ledgers`'s explanation to `"AUC 0.81 on validation against a dummy baseline of 0.50."` (its assertion on `"- Figure 12_roc_curve.png: AUC 0.81"` stays), and add:

```python
    def ledger(self):
        return dict(re.findall(r"^- Figure (\S+): (.*)$", self.md.read_text(), re.M))

    def test_fig_rejects_an_explanation_without_a_number(self):
        import matplotlib.pyplot as plt
        with self.assertRaises(ValueError):
            guard.fig(4, "x", plt.figure(), "shows the distribution of values nicely here", state_dir=self.state)

    def test_eda_explanations_are_real(self):
        guard.eda_figures(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        for name, explanation in self.ledger().items():
            with self.subTest(figure=name):
                self.assertIsNone(guard.note_problem(explanation))

    def test_correlations_leave_out_the_target(self):
        guard.eda_figures(frame(), target="y", state_dir=self.state)
        self.assertIn("target is left out", self.ledger()["02_correlations.png"])
```

- [ ] **Step 3: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.NoteRule tests.test_mlpipeline_guard.Figures -v`
Expected: `AttributeError: ... no attribute 'note_problem'`; `ValueError not raised`; `02_correlations` assertion fails.

- [ ] **Step 4: Implement the rule**

Add the section (after the profile section, before `# ---- figures`):

```python
# ------------------------------------------------------------------ the real-explanation rule
MIN_NOTE_CHARS = 40
COLLINEAR_RHO = 0.9
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _stat_values(obj) -> list[float]:
    """Every number inside a profile entry, recursively (booleans and NaN excluded)."""
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
    """True when ``note`` has a number other than 0 or 1 equal to one of ``stats`` at its written precision."""
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
```

In `fig`, replace the explanation check with:

```python
    problem = note_problem(explanation)
    if problem:
        raise ValueError(f"figure explanation {problem}: say what it shows, with a number from the data")
```

- [ ] **Step 5: Update `eda_figures`**

Replace the missingness explanation with:

```python
    explanation = (f"{n_missing} of {len(missing)} columns have missing values; worst is {missing.index[0]} "
                   f"at {missing.iloc[0]:.1%}." if n_missing else
                   f"0 of {len(missing)} columns have missing values, so step 4 needs no imputation.")
    paths.append(fig(1, "missingness", f, explanation, state_dir))
```

Replace the whole correlation block (from `with_target = ...` to its `paths.append`) with:

```python
    numeric = df.select_dtypes("number").drop(columns=[target], errors="ignore").iloc[:, :30]
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        f, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels([str(c) for c in corr.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels([str(c) for c in corr.columns], fontsize=7)
        f.colorbar(image)
        ax.set_title("Correlations between features (target left out until the split)")
        values = corr.to_numpy()
        pairs = [(abs(values[i, j]), corr.columns[i], corr.columns[j])
                 for i in range(len(corr)) for j in range(i + 1, len(corr)) if values[i, j] == values[i, j]]
        if pairs:
            r, a, b = max(pairs, key=lambda t: t[0])
            high = sum(1 for t in pairs if t[0] > COLLINEAR_RHO)
            explanation = (f"Strongest pair among {len(corr)} numeric features: {a} and {b} (|r| = {r:.2f}); "
                           f"{high} pairs exceed {COLLINEAR_RHO}. The target is left out until the split.")
        else:
            explanation = (f"Correlations between {len(corr)} numeric features could not be computed (constant "
                           "columns). The target is left out until the split.")
        paths.append(fig(2, "correlations", f, explanation, state_dir))
```

- [ ] **Step 6: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py tests/note_rule_cases.json
git commit -m "feat(guard): the real-explanation rule; fig() enforces it; step-2 heatmap leaves the target out

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Data dictionary, missingness, categorical plan

**Files:**
- Modify: `lib/mlpipeline_guard.py` (imports `csv`, `warnings`; new section `# ---- records (steps 1-8)` after the figures section)
- Test: `tests/test_mlpipeline_guard.py` (module helper `messy()`; new class `Records`; add `import csv`)

**Interfaces:**
- Consumes: `guard._column_kind`, `guard._column_stats` (Task 3), `guard.fig`, `guard._plt`.
- Produces: `guard.data_dictionary(df, *, state_dir) -> Path`; `guard.missingness(df, *, state_dir) -> Path`; `guard.categorical_plan(df, *, state_dir) -> Path`; constants `DICTIONARY_FILE`, `DICTIONARY_HEADER`, `MISSINGNESS_FILE`, `MISSINGNESS_HEADER`, `PLAN_FILE`, `PLAN_HEADER`; helpers `guard._merge_csv(path, header, rows, *, key, computed) -> list[str]`, `guard._load_profile(state_dir) -> dict`.

- [ ] **Step 1: Write the failing tests**

Module helper (after `frame`):

```python
def messy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(50, 12, n).round(1),
        "income": rng.lognormal(10, 1, n).round(0),
        "zip": rng.integers(0, 120, n).astype(str),          # ~119 levels -> high cardinality
        "plan": rng.choice(["basic", "plus", "pro"], n).astype(object),
        "smoker": rng.integers(0, 2, n).astype(bool),
        "patient_id": np.arange(n),
        "y": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 40, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "plan"] = np.nan
    return df
```

```python
class Records(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.df = messy()
        guard.profile(self.df, target="y", state_dir=self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, name, state=None):
        with ((state or self.state) / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_rows(self, name, rows, excel=False):
        with (self.state / name).open("w", newline="", encoding="utf-8-sig" if excel else "utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\r\n" if excel else "\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_dictionary_skeleton(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        rows = self.rows(guard.DICTIONARY_FILE)
        self.assertEqual([r["column"] for r in rows], list(self.df.columns))
        by = {r["column"]: r for r in rows}
        self.assertEqual(by["y"]["role"], "target")
        self.assertEqual(by["zip"]["kind"], "categorical")
        self.assertIn("median", by["age"]["observed"])
        self.assertEqual({by["age"][f] for f in ("meaning", "unit", "role", "figure_note")}, {""})

    def test_dictionary_keeps_words_and_survives_excel(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        rows = self.rows(guard.DICTIONARY_FILE)
        rows[0]["meaning"] = "age at admission"
        self.write_rows(guard.DICTIONARY_FILE, rows, excel=True)
        guard.data_dictionary(self.df, state_dir=self.state)
        self.assertEqual(self.rows(guard.DICTIONARY_FILE)[0]["meaning"], "age at admission")

    def test_dictionary_drops_stale_rows_with_a_warning(self):
        guard.data_dictionary(self.df, state_dir=self.state)
        with self.assertWarns(UserWarning) as ctx:
            guard.data_dictionary(self.df.drop(columns=["plan"]), state_dir=self.state)
        self.assertIn("plan", str(ctx.warning))
        self.assertNotIn("plan", [r["column"] for r in self.rows(guard.DICTIONARY_FILE)])

    def test_dictionary_needs_a_profile(self):
        (self.state / guard.PROFILE_FILE).unlink()
        with self.assertRaises(guard.LeakageError):
            guard.data_dictionary(self.df, state_dir=self.state)

    def test_awkward_column_names_round_trip(self):
        df = pd.DataFrame({0: [1.5, 2.5, 3.5, 4.5], 'weight, "kg"': [60.0, 70.0, 80.0, 90.0], "y": [0, 1, 0, 1]})
        guard.profile(df, target="y", state_dir=self.state)
        guard.data_dictionary(df, state_dir=self.state)
        self.assertEqual([r["column"] for r in self.rows(guard.DICTIONARY_FILE)], ["0", 'weight, "kg"', "y"])

    def test_missingness_rows_counts_and_pattern(self):
        guard.missingness(self.df, state_dir=self.state)
        rows = {r["column"]: r for r in self.rows(guard.MISSINGNESS_FILE)}
        self.assertEqual(set(rows), {"income", "plan"})
        self.assertEqual((rows["income"]["n_missing"], rows["income"]["pct_missing"]), ("40", "6.67"))
        self.assertGreater((self.state / "figures" / "01_missing_pattern.png").stat().st_size, 1024)
        self.assertIn("- Figure 01_missing_pattern.png: 2 columns have gaps", (self.state / "PIPELINE.md").read_text())

    def test_missingness_single_column_and_no_gaps(self):
        guard.missingness(self.df.drop(columns=["plan"]), state_dir=self.state)
        self.assertIn("Only income has gaps", (self.state / "PIPELINE.md").read_text())
        clean = Path(self.tmp.name) / "clean"
        guard.missingness(self.df.dropna(), state_dir=clean)
        self.assertEqual(self.rows(guard.MISSINGNESS_FILE, state=clean), [])
        self.assertFalse((clean / "figures" / "01_missing_pattern.png").exists())

    def test_categorical_plan_rows(self):
        guard.categorical_plan(self.df, state_dir=self.state)
        rows = {r["column"]: r for r in self.rows(guard.PLAN_FILE)}
        self.assertEqual(set(rows), {"zip", "plan", "smoker"})
        self.assertEqual((rows["zip"]["high_cardinality"], rows["plan"]["n_levels"], rows["smoker"]["high_cardinality"]),
                         ("yes", "3", "no"))
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Records -v`
Expected: `AttributeError: ... no attribute 'data_dictionary'` (and `missingness`, `categorical_plan`, `DICTIONARY_FILE`, ...).

- [ ] **Step 3: Implement**

Add `import csv` and `import warnings` to the imports. Add the section:

```python
# ------------------------------------------------------------------ records (steps 1-8)
DICTIONARY_FILE = "data_dictionary.csv"
DICTIONARY_HEADER = ["column", "kind", "dtype", "observed", "meaning", "unit", "allowed", "source",
                     "available_at_prediction", "role", "figure_note"]
MISSINGNESS_FILE = "missingness.csv"
MISSINGNESS_HEADER = ["column", "n_missing", "pct_missing", "mechanism", "evidence", "decision", "reason"]
PLAN_FILE = "categorical_plan.csv"
PLAN_HEADER = ["column", "n_levels", "rare_levels", "high_cardinality", "encoding", "reason"]


def _load_profile(state_dir) -> dict:
    path = Path(state_dir) / PROFILE_FILE
    if not path.is_file():
        raise LeakageError("run guard.profile() first")
    prof = json.loads(path.read_text())
    if prof.get("schema_version") != "2":
        raise LeakageError("data_profile.json is from an older guard: re-run guard.profile()")
    return prof


def _merge_csv(path: Path, header: list[str], rows: list[dict], *, key: str, computed: list[str]) -> list[str]:
    """Write ``rows``, keeping the agent's words for rows that still exist; return the keys dropped.

    Computed fields always come from ``rows``; word fields keep a non-empty earlier value, else take the
    default in ``rows``. Reads ``utf-8-sig`` so files saved by Excel or Numbers still merge.
    """
    old: dict[str, dict] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            old = {r.get(key) or "": r for r in csv.DictReader(handle)}
    merged = [{h: (row.get(h, "") if h in computed else (old.get(row[key], {}).get(h) or row.get(h, "")))
               for h in header} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    current = {row[key] for row in rows}
    return sorted(k for k in old if k and k not in current)


def _warn_removed(removed: list[str], path: Path) -> None:
    if removed:
        warnings.warn(f"{path.name}: removed rows for columns no longer in the frame: {', '.join(removed)}",
                      UserWarning, stacklevel=3)


def _observed(info: dict) -> str:
    kind = info["kind"]
    if not info.get("non_null"):
        return "all missing"
    if kind == "numeric":
        return f"{info['min']}..{info['max']}, median {info['median']}"
    if kind in ("categorical", "bool"):
        shown = list(info["top"])[:5]
        return f"{info['n_levels']} levels: {', '.join(shown)}{'…' if info['n_levels'] > len(shown) else ''}"
    if kind == "datetime":
        return f"{info['start']}..{info['end']}"
    if kind == "text":
        return f"length {info['length_min']}..{info['length_max']}"
    return f"{info['n_unique']} unique"


def data_dictionary(df: pd.DataFrame, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: one row per column; the guard fills kind, dtype, and observed values, the agent the words."""
    target = _load_profile(state_dir)["target"]["name"]
    rows = []
    for col in df.columns:
        info = _column_stats(df[col], _column_kind(df[col]))
        rows.append({"column": str(col), "kind": info["kind"], "dtype": info["dtype"], "observed": _observed(info),
                     "role": "target" if str(col) == str(target) else ""})
    path = Path(state_dir) / DICTIONARY_FILE
    _warn_removed(_merge_csv(path, DICTIONARY_HEADER, rows, key="column",
                             computed=["column", "kind", "dtype", "observed"]), path)
    return path


def missingness(df: pd.DataFrame, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: one row per column with gaps (counts computed) and a figure of which go missing together."""
    counts = df.isna().sum()
    gappy = [c for c in df.columns if counts[c] > 0]
    rows = [{"column": str(c), "n_missing": str(int(counts[c])), "pct_missing": f"{100 * counts[c] / len(df):.2f}"}
            for c in gappy]
    path = Path(state_dir) / MISSINGNESS_FILE
    _warn_removed(_merge_csv(path, MISSINGNESS_HEADER, rows, key="column",
                             computed=["column", "n_missing", "pct_missing"]), path)
    if not gappy:
        return path
    plt = _plt()
    indicators = df[gappy].isna()
    if len(gappy) >= 2:
        corr = indicators.astype(float).corr()
        size = 0.5 * len(gappy)
        f, ax = plt.subplots(figsize=(max(5, size + 3), max(4, size + 2)))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(gappy)))
        ax.set_xticklabels([str(c) for c in gappy], rotation=90, fontsize=7)
        ax.set_yticks(range(len(gappy)))
        ax.set_yticklabels([str(c) for c in gappy], fontsize=7)
        f.colorbar(image)
        ax.set_title("Which columns go missing together")
        a, b, share = max(((a, b, float((indicators[a] & indicators[b]).sum() / indicators[a].sum()))
                           for a in gappy for b in gappy if a != b), key=lambda t: t[2])
        explanation = (f"{len(gappy)} columns have gaps; {b} is also missing in {share:.0%} of {a}'s "
                       f"{int(indicators[a].sum()):,} gap rows.")
    else:
        col = gappy[0]
        f, ax = plt.subplots(figsize=(6, 3))
        ax.bar([str(col)], [int(counts[col])])
        ax.set_ylabel("missing rows")
        ax.set_title("Missing values")
        explanation = (f"Only {col} has gaps: {int(counts[col]):,} of {len(df):,} rows "
                       f"({counts[col] / len(df):.1%}); every other column is complete.")
    fig(1, "missing_pattern", f, explanation, state_dir)
    return path


def categorical_plan(df: pd.DataFrame, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 7: one row per categorical or bool column (levels computed); the agent picks the encoding."""
    target = str(_load_profile(state_dir)["target"]["name"])
    rows = []
    for col in df.columns:
        if str(col) == target or _column_kind(df[col]) not in ("categorical", "bool"):
            continue
        values = df[col].dropna().astype(str)
        counts = values.value_counts()
        rows.append({"column": str(col), "n_levels": str(len(counts)),
                     "rare_levels": str(int((counts / max(len(values), 1) < RARE_LEVEL_FRAC).sum())),
                     "high_cardinality": "yes" if len(counts) > HIGH_CARD_LEVELS else "no"})
    path = Path(state_dir) / PLAN_FILE
    _warn_removed(_merge_csv(path, PLAN_HEADER, rows, key="column",
                             computed=["column", "n_levels", "rare_levels", "high_cardinality"]), path)
    return path
```

- [ ] **Step 4: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): data dictionary, missingness, and categorical plan records with merge-safe CSVs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Variable figure pages and the label audit

**Files:**
- Modify: `lib/mlpipeline_guard.py` (figures section: `variable_figures`, `_draw_variable`; records section: `label_audit`)
- Test: `tests/test_mlpipeline_guard.py` (new class `VariablePagesAndLabels`)

**Interfaces:**
- Consumes: `guard.fig`, `guard._column_kind`, `guard._target_info` (Task 3).
- Produces: `guard.variable_figures(df, *, target, per_page=PAGE_COLUMNS, state_dir) -> list[Path]` writing `figures/02_variables_pNN.png` and `figures/pages.json`; `guard.label_audit(df, *, target, state_dir) -> Path`; constants `PAGE_COLUMNS = 12`, `PAGES_FILE = "pages.json"`, `LABEL_AUDIT_FILE = "label_audit.md"`, `LABEL_SECTIONS` (ordered dict name → hint), `_COMPUTED_START`, `_COMPUTED_END`.

- [ ] **Step 1: Write the failing tests**

```python
class VariablePagesAndLabels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.df = messy()

    def tearDown(self):
        self.tmp.cleanup()

    def pages(self):
        return json.loads((self.state / "figures" / guard.PAGES_FILE).read_text())

    def test_pages_cover_every_column(self):
        paths = guard.variable_figures(self.df, target="y", per_page=3, state_dir=self.state)
        self.assertEqual([p.name for p in paths], ["02_variables_p01.png", "02_variables_p02.png", "02_variables_p03.png"])
        self.assertEqual(sorted(c for cols in self.pages().values() for c in cols), sorted(self.df.columns))
        ledger = dict(re.findall(r"^- Figure (\S+): (.*)$", (self.state / "PIPELINE.md").read_text(), re.M))
        for p in paths:
            self.assertGreater(p.stat().st_size, 1024)
            self.assertIsNone(guard.note_problem(ledger[p.name]))

    def test_wide_frame_pages(self):
        wide = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 30)), columns=[f"f{i}" for i in range(30)])
        wide["y"] = np.arange(50) % 2
        self.assertEqual(len(guard.variable_figures(wide, target="y", state_dir=self.state)), 3)  # 31 columns / 12

    def test_all_missing_column_gets_a_panel(self):
        guard.variable_figures(self.df.assign(empty=np.nan), target="y", state_dir=self.state)
        self.assertIn("empty", [c for cols in self.pages().values() for c in cols])

    def test_label_audit_template_and_counts(self):
        flipped = self.df.iloc[:5].assign(y=1 - self.df["y"].iloc[:5])
        df = pd.concat([self.df, flipped], ignore_index=True)
        guard.label_audit(df, target="y", state_dir=self.state)
        text = (self.state / guard.LABEL_AUDIT_FILE).read_text()
        counts = df["y"].value_counts()
        self.assertIn(f"0 = {counts[0]:,} (", text)
        self.assertIn(f"1 = {counts[1]:,} (", text)
        self.assertIn("- identical feature rows with different labels: 10 rows in 5 groups", text)
        for section in guard.LABEL_SECTIONS:
            self.assertIn(f"## {section}\n", text)

    def test_label_audit_refresh_keeps_the_writing(self):
        guard.label_audit(self.df, target="y", state_dir=self.state)
        path = self.state / guard.LABEL_AUDIT_FILE
        path.write_text(path.read_text().replace("## Source\n", "## Source\nClaims data from the hospital billing team.\n"))
        guard.label_audit(self.df.iloc[:300], target="y", state_dir=self.state)
        text = path.read_text()
        self.assertIn("Claims data from the hospital billing team.", text)
        self.assertIn("- rows: 300", text)
        self.assertEqual(text.count(guard._COMPUTED_START), 1)

    def test_label_audit_regression_line(self):
        guard.label_audit(self.df.assign(y=np.random.default_rng(3).normal(size=len(self.df))), target="y",
                          state_dir=self.state)
        self.assertIn("- target: min ", (self.state / guard.LABEL_AUDIT_FILE).read_text())
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.VariablePagesAndLabels -v`
Expected: `AttributeError: ... no attribute 'variable_figures'` / `'label_audit'` / `'LABEL_SECTIONS'`.

- [ ] **Step 3: Implement the pages (figures section, after `eda_figures`)**

```python
PAGE_COLUMNS = 12
PAGES_FILE = "pages.json"


def _draw_variable(ax, series: pd.Series, kind: str) -> None:
    values = series.dropna()
    if not len(values):
        ax.text(0.5, 0.5, "all missing", ha="center", va="center")
        ax.set_axis_off()
    elif kind == "numeric":
        ax.hist(values.astype(float), bins=30)
        ax.axvline(float(values.median()), color="black", linestyle="--", linewidth=1)
    elif kind in ("categorical", "bool"):
        counts = values.astype(str).value_counts().head(10)
        ax.barh(counts.index[::-1], counts.to_numpy()[::-1])
        ax.tick_params(axis="y", labelsize=7)
    elif kind == "datetime":
        counts = values.dt.to_period("M").value_counts().sort_index()
        ax.plot([str(p) for p in counts.index], counts.to_numpy())
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    elif kind == "text":
        ax.hist(values.astype(str).str.len(), bins=30)
        ax.set_xlabel("length", fontsize=7)
    else:
        ax.text(0.5, 0.5, f"{values.nunique():,} unique ids", ha="center", va="center")
        ax.set_axis_off()


def variable_figures(df: pd.DataFrame, *, target: str, per_page: int = PAGE_COLUMNS,
                     state_dir: str | Path = "ml_pipeline") -> list[Path]:
    """Step 2: every column's distribution on grouped pages, plus figures/pages.json (page -> columns)."""
    plt = _plt()
    columns = list(df.columns)
    pages = [columns[i:i + per_page] for i in range(0, len(columns), per_page)]
    mapping: dict[str, list[str]] = {}
    paths: list[Path] = []
    for number, cols in enumerate(pages, start=1):
        rows = int(np.ceil(len(cols) / 4))
        f, axes = plt.subplots(rows, 4, figsize=(14, 3 * rows))
        axes = np.atleast_1d(axes).ravel()
        gappy, skewed = [], []
        for ax, col in zip(axes, cols):
            series = df[col]
            kind = _column_kind(series)
            _draw_variable(ax, series, kind)
            ax.set_title(f"{col} ({'target' if str(col) == str(target) else kind})", fontsize=9)
            if series.isna().any():
                gappy.append(str(col))
            if kind == "numeric" and series.dropna().size > 2 and abs(float(series.skew())) > 1:
                skewed.append(str(col))
        for ax in axes[len(cols):]:
            ax.axis("off")
        detail = f" ({', '.join(skewed[:5])})" if skewed else ""
        explanation = (f"Page {number} of {len(pages)}: {len(cols)} columns ({cols[0]} to {cols[-1]}); "
                       f"{len(gappy)} have missing values, {len(skewed)} are strongly skewed{detail}. "
                       "Per-column notes are in data_dictionary.csv.")
        path = fig(2, f"variables_p{number:02d}", f, explanation, state_dir)
        mapping[path.name] = [str(c) for c in cols]
        paths.append(path)
    out = Path(state_dir) / FIGURES_DIR / PAGES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2))
    return paths
```

- [ ] **Step 4: Implement the label audit (records section)**

```python
LABEL_AUDIT_FILE = "label_audit.md"
_COMPUTED_START = "<!-- computed by guard.label_audit - do not edit -->"
_COMPUTED_END = "<!-- end computed -->"
LABEL_SECTIONS = {
    "Definition": "the exact rule that produces the label, with units and time window",
    "Source": "who or what produced the label, and how",
    "When labeled": "when the label becomes known, relative to the prediction time",
    "Leakage check": "which columns could encode the label; every leakage suspect kept as a feature is named here",
    "Noise and consistency": "how reliable the label is; what the conflicting-label count above means",
}


def _label_conflicts(df: pd.DataFrame, target: str) -> tuple[int, int]:
    """(rows, groups) of rows identical in every non-target, non-id column but with different labels."""
    features = [c for c in df.columns if c != target and _column_kind(df[c]) != "id"]
    if not features:
        return 0, 0
    keys = pd.Series(pd.util.hash_pandas_object(df[features], index=False).to_numpy(), index=df.index)
    per_key = df[target].groupby(keys).nunique(dropna=True)
    conflicted = per_key[per_key > 1].index
    return int(keys.isin(conflicted).sum()), int(len(conflicted))


def _label_block(df: pd.DataFrame, target: str) -> str:
    info = _target_info(df[target])
    lines = [_COMPUTED_START, f"- rows: {len(df):,}", f"- missing labels: {int(df[target].isna().sum()):,}"]
    if info["task"] == "regression":
        lines.append(f"- target: min {info['min']}, max {info['max']}, mean {info['mean']}, std {info['std']}")
    else:
        total = sum(info["class_counts"].values())
        lines.append("- classes: " + "; ".join(f"{k} = {c:,} ({c / total:.1%})" for k, c in info["class_counts"].items()))
    rows, groups = _label_conflicts(df, target)
    lines += [f"- identical feature rows with different labels: {rows:,} rows in {groups:,} groups", _COMPUTED_END]
    return "\n".join(lines)


def label_audit(df: pd.DataFrame, *, target: str, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: the label's computed facts plus five sections the agent writes; later calls refresh the facts only."""
    path = Path(state_dir) / LABEL_AUDIT_FILE
    block = _label_block(df, target)
    if path.is_file():
        text = path.read_text()
        start, end = text.find(_COMPUTED_START), text.find(_COMPUTED_END)
        if start != -1 and end != -1:
            text = text[:start] + block + text[end + len(_COMPUTED_END):]
        else:
            head, _, rest = text.partition("\n")
            text = f"{head}\n\n{block}\n{rest}"
    else:
        sections = "\n\n".join(f"## {name}\n<!-- {hint} -->" for name, hint in LABEL_SECTIONS.items())
        text = f"# Label audit: {target}\n\n{block}\n\n{sections}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
```

- [ ] **Step 5: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): variable figure pages for every column and the label audit

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Cleaning log and train-only associations

**Files:**
- Modify: `lib/mlpipeline_guard.py` (new section `# ---- preparation (steps 4-8)` after the records section)
- Test: `tests/test_mlpipeline_guard.py` (new class `CleaningAndAssociations`)

**Interfaces:**
- Consumes: `guard._split_state` (Task 1), `guard._target_info`, `guard._column_kind`, `guard.fig`, `guard.COLLINEAR_RHO` (Task 4).
- Produces: `guard.log_cleaning(rule, before, after, *, reason, state_dir) -> pd.DataFrame`; `guard.train_associations(train, *, target, state_dir) -> dict`; `guard.NotTrainingSplit(LeakageError)`; constants `CLEANING_FILE`, `CLEANING_HEADER`, `ASSOCIATIONS_FILE`; figures `07_correlations_train.png`, `07_feature_target_train.png`.

- [ ] **Step 1: Write the failing tests**

```python
class CleaningAndAssociations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_cleaning_counts(self):
        before = messy()
        after = before.drop(index=range(10)).assign(age=before["age"].clip(upper=60))
        returned = guard.log_cleaning("clip age at 60 and drop 10 rows", before, after,
                                      reason="entry errors above 60", state_dir=self.state)
        self.assertIs(returned, after)
        with (self.state / guard.CLEANING_FILE).open(encoding="utf-8") as handle:
            row = list(csv.DictReader(handle))[0]
        changed = int((before["age"].drop(index=range(10)) > 60).sum())
        self.assertEqual((row["rows_before"], row["rows_after"], row["cells_changed"], row["columns_changed"]),
                         ("600", "590", str(changed), "age"))

    def test_log_cleaning_needs_a_reason(self):
        with self.assertRaises(ValueError):
            guard.log_cleaning("drop rows", messy(), messy(), reason=" ", state_dir=self.state)

    @staticmethod
    def signal_frame(n=800):
        rng = np.random.default_rng(5)
        signal = rng.normal(size=n)
        weight = rng.normal(70, 10, size=n)
        return pd.DataFrame({"signal": signal, "noise": rng.normal(size=n), "weight": weight,
                             "bmi": weight * 0.3 + rng.normal(0, 0.1, size=n), "plan": rng.choice(["a", "b"], n),
                             "y": (signal + rng.normal(0, 0.3, size=n) > 0).astype(int)})

    def test_train_associations(self):
        tr, _, _ = guard.split(self.signal_frame(), target="y", state_dir=self.state)
        result = guard.train_associations(tr, target="y", state_dir=self.state)
        self.assertEqual(result["feature_target"][0]["column"], "signal")
        self.assertEqual((result["computed_on"]["split"], result["computed_on"]["n_rows"]), ("train", len(tr)))
        self.assertEqual([(p["a"], p["b"]) for p in result["collinear_pairs"]], [("weight", "bmi")])
        measures = {e["column"]: e["measure"] for e in result["feature_target"]}
        self.assertEqual((measures["signal"], measures["plan"]), ("eta", "cramers_v"))
        for name in ("07_correlations_train.png", "07_feature_target_train.png"):
            self.assertGreater((self.state / "figures" / name).stat().st_size, 1024)
        saved = json.loads((self.state / guard.ASSOCIATIONS_FILE).read_text())
        self.assertEqual(saved["computed_on"]["n_rows"], len(tr))

    def test_regression_target_measures(self):
        df = self.signal_frame()
        df["y"] = df["signal"] * 2 + np.random.default_rng(1).normal(0, 0.1, len(df))
        tr, _, _ = guard.split(df, target="y", state_dir=self.state)
        result = guard.train_associations(tr, target="y", state_dir=self.state)
        measures = {e["column"]: e["measure"] for e in result["feature_target"]}
        self.assertEqual((measures["signal"], measures["plan"]), ("spearman", "eta"))
        self.assertEqual(result["feature_target"][0]["column"], "signal")

    def test_associations_refuse_the_full_frame(self):
        df = self.signal_frame()
        guard.split(df, target="y", state_dir=self.state)
        with self.assertRaises(guard.NotTrainingSplit):
            guard.train_associations(df, target="y", state_dir=self.state)

    def test_associations_need_a_split(self):
        with self.assertRaises(guard.LeakageError):
            guard.train_associations(self.signal_frame(), target="y", state_dir=self.state)
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.CleaningAndAssociations -v`
Expected: `AttributeError: ... no attribute 'log_cleaning'` / `'train_associations'` / `'NotTrainingSplit'`.

- [ ] **Step 3: Implement**

```python
# ------------------------------------------------------------------ preparation (steps 4-8)
CLEANING_FILE = "cleaning_log.csv"
CLEANING_HEADER = ["rule", "reason", "rows_before", "rows_after", "cells_changed", "columns_changed", "date"]
ASSOCIATIONS_FILE = "associations.json"


class NotTrainingSplit(LeakageError):
    """Target statistics were requested on a frame that is not the frozen training split."""


def log_cleaning(rule: str, before: pd.DataFrame, after: pd.DataFrame, *, reason: str,
                 state_dir: str | Path = "ml_pipeline") -> pd.DataFrame:
    """Step 4: record one cleaning rule with its before/after counts; returns ``after``.

    Rows are matched by index label and cells compared on the shared columns, NaN equal to NaN.
    """
    if not str(rule).strip() or not str(reason).strip():
        raise ValueError("every cleaning rule needs a name and a reason")
    rows = before.index.intersection(after.index)
    cols = [c for c in before.columns if c in after.columns]
    a = before.loc[rows, cols].astype(object)
    b = after.loc[rows, cols].astype(object)
    changed = (~((a == b) | (a.isna() & b.isna()))).sum()
    path = Path(state_dir) / CLEANING_FILE
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_HEADER, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({"rule": str(rule).strip(), "reason": str(reason).strip(), "rows_before": len(before),
                         "rows_after": len(after), "cells_changed": int(changed.sum()),
                         "columns_changed": ";".join(str(c) for c in changed[changed > 0].index),
                         "date": date.today().isoformat()})
    return after


def _finite(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _eta(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio: the root of the share of spread in ``values`` explained by ``groups``, in [0, 1]."""
    overall = values.mean()
    total = float(((values - overall) ** 2).sum())
    if total == 0:
        return 0.0
    between = sum(len(v) * (v.mean() - overall) ** 2 for v in (values[groups == g] for g in np.unique(groups)))
    return float(np.sqrt(between / total))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    if min(table.shape) < 2:
        return 0.0
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def _feature_target(x: pd.Series, y: pd.Series, x_kind: str, task: str) -> tuple[str, float]:
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 3:
        return "none", 0.0
    xs, ys = both.iloc[:, 0], both.iloc[:, 1]
    categorical = x_kind in ("categorical", "bool")
    if task == "regression":
        if categorical:
            return "eta", _eta(ys.astype(float).to_numpy(), xs.astype(str).to_numpy())
        return "spearman", abs(_finite(xs.astype(float).rank().corr(ys.astype(float).rank())))
    if categorical:
        return "cramers_v", _cramers_v(xs.astype(str), ys.astype(str))
    return "eta", _eta(xs.astype(float).to_numpy(), ys.astype(str).to_numpy())


def train_associations(train: pd.DataFrame, *, target: str, state_dir: str | Path = "ml_pipeline") -> dict:
    """Step 7: correlations and feature-target strength on the frozen training split only."""
    state = _split_state(state_dir)
    if len(train) != state["n_train"]:
        raise NotTrainingSplit(f"{len(train):,} rows, but the frozen training split has {state['n_train']:,}: "
                               "compute target statistics on the training split only")
    task = _target_info(train[target])["task"]
    kinds = {c: _column_kind(train[c]) for c in train.columns if c != target}
    usable = [c for c, k in kinds.items() if k in ("numeric", "bool", "categorical")]
    strengths = []
    for col in usable:
        measure, value = _feature_target(train[col], train[target], kinds[col], task)
        entry = {"column": str(col), "measure": measure, "value": round(_finite(value), 4)}
        if kinds[col] == "categorical" and train[col].nunique() > HIGH_CARD_LEVELS:
            entry["high_cardinality"] = True
        strengths.append(entry)
    strengths.sort(key=lambda e: e["value"], reverse=True)
    numeric = [c for c in usable if kinds[c] in ("numeric", "bool")]
    corr = train[numeric].astype(float).rank().corr() if len(numeric) >= 2 else pd.DataFrame()
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            rho = float(corr.loc[a, b])
            if rho == rho and abs(rho) > COLLINEAR_RHO:
                pairs.append({"a": str(a), "b": str(b), "rho": round(rho, 4)})
    result = {"schema_version": "1",
              "computed_on": {"split": "train", "n_rows": int(len(train)), "date": date.today().isoformat()},
              "target": str(target), "task": task, "feature_target": strengths, "collinear_pairs": pairs}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / ASSOCIATIONS_FILE).write_text(json.dumps(result, indent=2))
    _association_figures(corr, strengths, pairs, len(train), state_dir)
    return result


def _association_figures(corr: pd.DataFrame, strengths: list[dict], pairs: list[dict], n_rows: int,
                         state_dir) -> None:
    plt = _plt()
    shown = corr.iloc[:40, :40]
    f, ax = plt.subplots(figsize=(8, 7))
    if len(shown) >= 2:
        image = ax.imshow(shown.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([str(c) for c in shown.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([str(c) for c in shown.columns], fontsize=7)
        f.colorbar(image)
    else:
        ax.text(0.5, 0.5, "fewer than 2 numeric features", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Spearman correlations, training split")
    if pairs:
        top = max(pairs, key=lambda p: abs(p["rho"]))
        explanation = (f"{len(pairs)} feature pairs have |rho| above {COLLINEAR_RHO} on the training split; "
                       f"strongest {top['a']} and {top['b']} ({top['rho']:.2f}).")
    else:
        explanation = (f"No pair of the {len(shown)} numeric features exceeds |rho| {COLLINEAR_RHO} on the "
                       f"{n_rows:,} training rows.")
    fig(7, "correlations_train", f, explanation, state_dir)

    f, ax = plt.subplots(figsize=(8, max(3, 0.3 * min(len(strengths), 25) + 1)))
    top25 = strengths[:25]
    ax.barh([e["column"] for e in top25][::-1], [e["value"] for e in top25][::-1])
    ax.set_xlabel("strength of link to the target (0 to 1)")
    ax.set_title("Feature-target strength, training split")
    if strengths:
        best = strengths[0]
        weak = sum(1 for e in strengths if e["value"] < 0.05)
        explanation = (f"Strongest link to the target on the training split: {best['column']} "
                       f"({best['measure']} {best['value']:.2f}); {weak} of {len(strengths)} features score below 0.05.")
    else:
        explanation = f"No numeric or categorical feature to compare with the target among {n_rows:,} training rows."
    fig(7, "feature_target_train", f, explanation, state_dir)
```

- [ ] **Step 4: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`, no warnings printed.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): cleaning log and train-only correlations and feature-target strength

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Training receipts and permutation importance

**Files:**
- Modify: `lib/mlpipeline_guard.py` (new section `# ---- training and importance (steps 9-12)`)
- Test: `tests/test_mlpipeline_guard.py` (module class `Stub`; new class `TrainingAndImportance`)

**Interfaces:**
- Consumes: `guard._split_state` (Task 1), `guard.fig`.
- Produces: `guard.fit(model, X, y=None, *, name=None, state_dir, **fit_kwargs) -> model`; `guard.record_training(data, *, name, state_dir) -> None`; `guard.importance(predict_fn, val, *, target, metric_fn, greater_is_better=True, n_repeats=5, seed=0, state_dir) -> dict`; `guard.NotValidationSplit(LeakageError)`; constants `TRAINING_FILE = "training_runs.jsonl"`, `IMPORTANCE_FILE = "importance.json"`.

- [ ] **Step 1: Write the failing tests**

```python
class Stub:
    """A model that remembers how many rows it was fit on."""
    def fit(self, X, y=None):
        self.n_rows = len(X)
        return self


class TrainingAndImportance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        rng = np.random.default_rng(9)
        signal = rng.normal(size=600)
        df = pd.DataFrame({"signal": signal, "noise": rng.normal(size=600), "y": (signal > 0).astype(int)})
        self.tr, self.va, self.te = guard.split(df, target="y", state_dir=self.state)

    def tearDown(self):
        self.tmp.cleanup()

    def runs(self):
        return [json.loads(line) for line in (self.state / guard.TRAINING_FILE).read_text().splitlines()]

    @staticmethod
    def predict(X):
        return (X["signal"] > 0).astype(int).to_numpy()

    @staticmethod
    def accuracy(y_true, y_pred):
        return float((y_true == y_pred).mean())

    def test_fit_records_a_full_run(self):
        model = guard.fit(Stub(), self.tr.drop(columns="y"), self.tr["y"], state_dir=self.state)
        self.assertEqual(model.n_rows, len(self.tr))
        run = self.runs()[0]
        self.assertEqual((run["model"], run["n_rows"], run["n_train"], run["full"]),
                         ("Stub", len(self.tr), len(self.tr), True))

    def test_a_slice_is_recorded_as_not_full(self):
        guard.fit(Stub(), self.tr.drop(columns="y").iloc[:50], self.tr["y"].iloc[:50], name="pilot",
                  state_dir=self.state)
        self.assertEqual((self.runs()[0]["model"], self.runs()[0]["full"]), ("pilot", False))

    def test_fit_needs_a_split_before_training(self):
        model = Stub()
        with self.assertRaises(guard.LeakageError):
            guard.fit(model, self.tr, state_dir=Path(self.tmp.name) / "nosplit")
        self.assertFalse(hasattr(model, "n_rows"))

    def test_record_training_measures_len(self):
        guard.record_training(self.tr, name="mlp", state_dir=self.state)
        self.assertTrue(self.runs()[0]["full"])
        with self.assertRaises(TypeError):
            guard.record_training(5, name="typed", state_dir=self.state)

    def test_importance_ranks_the_signal_first(self):
        result = guard.importance(self.predict, self.va, target="y", metric_fn=self.accuracy, state_dir=self.state)
        self.assertEqual([f["column"] for f in result["features"]], ["signal", "noise"])
        self.assertGreater(result["features"][0]["importance_mean"], 0.3)
        self.assertEqual(result["features"][1]["importance_mean"], 0.0)
        self.assertEqual((result["computed_on"]["split"], result["computed_on"]["n_rows"]), ("validation", len(self.va)))
        self.assertGreater((self.state / "figures" / "12_permutation_importance.png").stat().st_size, 1024)

    def test_lower_is_better_metrics(self):
        def mse(y_true, y_pred):
            return float(((y_true - y_pred) ** 2).mean())
        result = guard.importance(self.predict, self.va, target="y", metric_fn=mse, greater_is_better=False,
                                  state_dir=self.state)
        self.assertGreater(result["features"][0]["importance_mean"], 0.3)

    def test_importance_refuses_other_frames(self):
        with self.assertRaises(guard.NotValidationSplit):
            guard.importance(self.predict, self.tr, target="y", metric_fn=self.accuracy, state_dir=self.state)
```

- [ ] **Step 2: Run and watch them fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.TrainingAndImportance -v`
Expected: `AttributeError: ... no attribute 'fit'` / `'TRAINING_FILE'` / `'importance'`.

- [ ] **Step 3: Implement**

```python
# ------------------------------------------------------------------ training and importance (steps 9-12)
TRAINING_FILE = "training_runs.jsonl"
IMPORTANCE_FILE = "importance.json"


class NotValidationSplit(LeakageError):
    """Importance was requested on a frame that is not the frozen validation split."""


def _append_run(state_dir, name: str, n_rows: int) -> None:
    state = _split_state(state_dir)
    line = {"model": str(name), "n_rows": int(n_rows), "n_train": int(state["n_train"]),
            "full": int(n_rows) == int(state["n_train"]), "date": date.today().isoformat()}
    with (Path(state_dir) / TRAINING_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def fit(model, X, y=None, *, name: str | None = None, state_dir: str | Path = "ml_pipeline", **fit_kwargs):
    """Steps 9-10: train ``model`` and record how many rows it saw, so Gate C can prove a full-data run."""
    _split_state(state_dir)  # no split, no training - checked before the model is touched
    if y is None:
        model.fit(X, **fit_kwargs)
    else:
        model.fit(X, y, **fit_kwargs)
    _append_run(state_dir, name or type(model).__name__, len(X))
    return model


def record_training(data, *, name: str, state_dir: str | Path = "ml_pipeline") -> None:
    """Custom training loops: record the rows ``data`` holds. It must support len(), so the count is measured."""
    _append_run(state_dir, name, len(data))


def importance(predict_fn, val: pd.DataFrame, *, target: str, metric_fn, greater_is_better: bool = True,
               n_repeats: int = 5, seed: int = 0, state_dir: str | Path = "ml_pipeline") -> dict:
    """Step 12: permutation importance on the frozen validation split, with a figure."""
    state = _split_state(state_dir)
    if len(val) != state["n_val"]:
        raise NotValidationSplit(f"{len(val):,} rows, but the frozen validation split has {state['n_val']:,}")
    X = val.drop(columns=[target])
    y = val[target].to_numpy()
    sign = 1.0 if greater_is_better else -1.0
    baseline = float(metric_fn(y, np.asarray(predict_fn(X))))
    rng = np.random.default_rng(seed)
    features = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(sign * (baseline - float(metric_fn(y, np.asarray(predict_fn(shuffled))))))
        features.append({"column": str(col), "importance_mean": round(float(np.mean(drops)), 6),
                         "importance_std": round(float(np.std(drops)), 6)})
    features.sort(key=lambda e: e["importance_mean"], reverse=True)
    metric = getattr(metric_fn, "__name__", "metric")
    result = {"schema_version": "1",
              "computed_on": {"split": "validation", "n_rows": int(len(val)), "date": date.today().isoformat()},
              "metric": metric, "baseline_score": round(baseline, 6), "n_repeats": int(n_repeats),
              "features": features}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / IMPORTANCE_FILE).write_text(json.dumps(result, indent=2))

    plt = _plt()
    top = features[:25]
    f, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top) + 1)))
    ax.barh([e["column"] for e in top][::-1], [e["importance_mean"] for e in top][::-1],
            xerr=[e["importance_std"] for e in top][::-1])
    ax.set_xlabel(f"drop in {metric} when shuffled")
    ax.set_title("Permutation importance, validation split")
    cheap = sum(1 for e in features if e["importance_mean"] < 0.001)
    explanation = (f"Shuffling {features[0]['column']} costs {features[0]['importance_mean']:.3f} of {metric} "
                   f"(validation baseline {baseline:.3f}); {cheap} of {len(features)} features cost less than 0.001 "
                   "and are candidates to drop.")
    fig(12, "permutation_importance", f, explanation, state_dir)
    return result
```

- [ ] **Step 4: Run the guard suite**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): training receipts prove full-data runs; permutation importance on validation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `hooks/gate_records.py` — the note rule, shared helpers, Gate A checks

**Files:**
- Create: `hooks/gate_records.py`, `tests/v05_fixtures.py`, `tests/test_gate_records.py`
- Test: `tests/test_gate_records.py`

**Interfaces:**
- Consumes: `tests/note_rule_cases.json` (Task 4).
- Produces: `gate_records.note_problem(note, stats=None) -> str | None` (same results as the guard's); `gate_records.gate_a_problems(state_dir: Path, run_text: str) -> list[str]`; helpers `_listing`, `_json`, `_csv_rows`, `_valid_combo`, `_int`, `_missingness_problems(state_dir, columns, stage)`; constants listed below. Fixtures: `fx.records(upto) -> dict[str, str | bytes]`, `fx.ledger(upto) -> str`, `fx.write(root, files)`, `fx.profile()`, `fx.dictionary_rows()`, `fx.dictionary_csv(rows)`, `fx.missingness_rows()`, `fx.missingness_csv(rows)`, `fx.plan_rows()`, `fx.plan_csv(rows)`, `fx.associations()`, `fx.importance()`, path constants `fx.D`, `fx.PROFILE`, `fx.SOURCE`, `fx.DICTIONARY`, `fx.MISSINGNESS`, `fx.LABEL_AUDIT`, `fx.PAGES`, `fx.CLEANING`, `fx.PLAN`, `fx.ASSOCIATIONS`, `fx.STATE`, `fx.TRAINING`, `fx.IMPORTANCE`, `fx.FIG(name)`, `fx.PNG`, `fx.RATIONALE`, `fx.LABEL_AUDIT_TEXT`.

- [ ] **Step 1: Write the fixtures**

`tests/v05_fixtures.py`:

```python
"""Complete v0.5 gate records as plain files (stdlib only), for the hook tests.

records(upto) maps relative paths to contents for everything Gates A..upto need; ledger(upto) is the
matching PIPELINE.md text. Tests change one piece of a fresh copy to create exactly one problem.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2000
D = "ml_pipeline/"
PROFILE, SOURCE, DICTIONARY = D + "data_profile.json", D + "source.json", D + "data_dictionary.csv"
MISSINGNESS, LABEL_AUDIT, PAGES = D + "missingness.csv", D + "label_audit.md", D + "figures/pages.json"
CLEANING, PLAN, ASSOCIATIONS = D + "cleaning_log.csv", D + "categorical_plan.csv", D + "associations.json"
STATE, TRAINING, IMPORTANCE = D + ".guard_state.json", D + "training_runs.jsonl", D + "importance.json"


def FIG(name: str) -> str:
    return D + "figures/" + name


COLUMNS = {
    "age": {"kind": "numeric", "dtype": "float64", "non_null": 196, "missing": 4, "missing_frac": 0.02,
            "n_unique": 71, "unique_frac": 0.355, "memory_bytes": 1600, "mean": 53.8, "std": 12.1, "min": 18.0,
            "p01": 21.0, "p05": 33.0, "p25": 45.0, "median": 54.0, "p75": 62.0, "p95": 74.0, "p99": 91.2,
            "max": 97.0, "skew": 0.3, "kurtosis": -0.2, "zeros": 0, "negatives": 0, "outliers": 3,
            "outlier_frac": 0.0153},
    "plan": {"kind": "categorical", "dtype": "object", "non_null": 200, "missing": 0, "missing_frac": 0.0,
             "n_unique": 3, "unique_frac": 0.015, "memory_bytes": 12000, "n_levels": 3,
             "top": {"basic": 0.5, "plus": 0.3, "pro": 0.2}, "top_counts": {"basic": 100, "plus": 60, "pro": 40},
             "rare_levels": 0},
    "patient_id": {"kind": "id", "dtype": "int64", "non_null": 200, "missing": 0, "missing_frac": 0.0,
                   "n_unique": 200, "unique_frac": 1.0, "memory_bytes": 1600},
    "y": {"kind": "numeric", "dtype": "int64", "non_null": 200, "missing": 0, "missing_frac": 0.0, "n_unique": 2,
          "unique_frac": 0.01, "memory_bytes": 1600, "mean": 0.3, "std": 0.46, "min": 0.0, "p01": 0.0,
          "p05": 0.0, "p25": 0.0, "median": 0.0, "p75": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0, "skew": 0.87,
          "kurtosis": -1.2, "zeros": 140, "negatives": 0, "outliers": 0, "outlier_frac": 0.0},
}


def profile() -> dict:
    return {"schema_version": "2", "created": "2026-09-23",
            "shape": {"n_rows": 200, "n_cols": 4, "memory_mb": 0.02},
            "source": {"name": "data.csv", "n_rows": 200, "full_data": True},
            "target": {"name": "y", "task": "binary", "n_classes": 2, "class_balance": {"0": 0.7, "1": 0.3},
                       "minority_frac": 0.3, "class_counts": {"0": 140, "1": 60}},
            "columns": copy.deepcopy(COLUMNS), "duplicates": {"exact_row_frac": 0.0}, "constant_columns": [],
            "high_cardinality_categoricals": [], "datetime_columns": [], "time_col": None, "group_col": None,
            "entity_candidates": ["patient_id"], "leakage_suspects": [],
            "traits": {"n_rows": 200, "n_features": 3, "task": "binary", "minority_frac": 0.3, "has_datetime": False,
                       "has_groups": False, "small_data": True, "imbalanced": False, "high_card_categoricals": []}}


SOURCE_JSON = {"schema_version": "1", "sources": {"data.csv": {
    "kind": "file", "sha256": "0" * 64, "bytes": 4000, "n_rows": 200, "n_cols": 4,
    "columns": ["age", "plan", "patient_id", "y"], "loaded": "2026-09-23"}}}

DICTIONARY_HEADER = ["column", "kind", "dtype", "observed", "meaning", "unit", "allowed", "source",
                     "available_at_prediction", "role", "figure_note"]
MISSINGNESS_HEADER = ["column", "n_missing", "pct_missing", "mechanism", "evidence", "decision", "reason"]
PLAN_HEADER = ["column", "n_levels", "rare_levels", "high_cardinality", "encoding", "reason"]
CLEANING_HEADER = ["rule", "reason", "rows_before", "rows_after", "cells_changed", "columns_changed", "date"]


def dictionary_rows() -> list[dict]:
    return [
        {"column": "age", "kind": "numeric", "dtype": "float64", "observed": "18.0..97.0, median 54.0",
         "meaning": "age at admission", "unit": "years", "allowed": "18..110", "source": "admissions form",
         "available_at_prediction": "yes", "role": "feature",
         "figure_note": "age: slightly right-skewed, median 54, 2% missing; cap at the 99th percentile (91)"},
        {"column": "plan", "kind": "categorical", "dtype": "object", "observed": "3 levels: basic, plus, pro",
         "meaning": "insurance plan at admission", "unit": "none", "allowed": "basic, plus, pro",
         "source": "billing system", "available_at_prediction": "yes", "role": "feature",
         "figure_note": "plan: three levels and basic dominates with 100 of 200 rows; nothing rare to group"},
        {"column": "patient_id", "kind": "id", "dtype": "int64", "observed": "200 unique",
         "meaning": "hospital patient number", "unit": "none", "allowed": "positive integers",
         "source": "admissions form", "available_at_prediction": "yes", "role": "id",
         "figure_note": "patient_id: 200 unique ids, one per row, so no patient repeats across rows"},
        {"column": "y", "kind": "numeric", "dtype": "int64", "observed": "0.0..1.0, median 0.0",
         "meaning": "readmitted within 30 days", "unit": "none", "allowed": "0 or 1", "source": "claims data",
         "available_at_prediction": "no", "role": "target",
         "figure_note": "y: readmitted within 30 days for 30% of patients, a moderate imbalance"},
    ]


def missingness_rows() -> list[dict]:
    return [{"column": "age", "n_missing": "4", "pct_missing": "2.00", "mechanism": "random",
             "evidence": "gaps spread evenly across admission months", "decision": "impute-median",
             "reason": "four gaps; the median resists the right skew"}]


def plan_rows() -> list[dict]:
    return [{"column": "plan", "n_levels": "3", "rare_levels": "0", "high_cardinality": "no", "encoding": "one-hot",
             "reason": "three levels and none rare; one-hot keeps them separate"}]


def _csv(header: list[str], rows: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def dictionary_csv(rows: list[dict]) -> str:
    return _csv(DICTIONARY_HEADER, rows)


def missingness_csv(rows: list[dict]) -> str:
    return _csv(MISSINGNESS_HEADER, rows)


def plan_csv(rows: list[dict]) -> str:
    return _csv(PLAN_HEADER, rows)


LABEL_AUDIT_TEXT = """# Label audit: y

<!-- computed by guard.label_audit - do not edit -->
- rows: 200
- missing labels: 0
- classes: 0 = 140 (70.0%); 1 = 60 (30.0%)
- identical feature rows with different labels: 0 rows in 0 groups
<!-- end computed -->

## Definition
Readmission to any ward within 30 days of discharge, counted once per admission.

## Source
Claims data joined on patient number by the hospital data team.

## When labeled
Known 30 days after discharge, which is after the prediction time at discharge.

## Leakage check
No column is recorded after discharge; the profile flagged no leakage suspects.

## Noise and consistency
Claims can arrive late, so a few readmissions may be missing; no conflicting rows.
"""

PAGES_JSON = {"02_variables_p01.png": ["age", "plan", "patient_id", "y"]}
STATE_JSON = {"test_fingerprint": "f" * 64, "n_test": 40, "test_touches": 0, "duplicates_dropped": 0,
              "split": "stratified", "created": "2026-09-23", "n_train": 120, "n_val": 40,
              "train_fingerprint": "t" * 64, "val_fingerprint": "v" * 64}


def associations() -> dict:
    return {"schema_version": "1", "computed_on": {"split": "train", "n_rows": 120, "date": "2026-09-23"},
            "target": "y", "task": "binary",
            "feature_target": [{"column": "age", "measure": "eta", "value": 0.21},
                               {"column": "plan", "measure": "cramers_v", "value": 0.03}],
            "collinear_pairs": []}


def importance() -> dict:
    return {"schema_version": "1", "computed_on": {"split": "validation", "n_rows": 40, "date": "2026-09-24"},
            "metric": "accuracy", "baseline_score": 0.725, "n_repeats": 5,
            "features": [{"column": "age", "importance_mean": 0.041, "importance_std": 0.01},
                         {"column": "plan", "importance_mean": 0.0, "importance_std": 0.0}]}


RATIONALE = ("Model rationale:\n- traits: binary, 200 rows (small), minority 30%\n"
             "- baseline: majority-class dummy + logistic regression\n- candidates: gradient boosting with small trees\n"
             "- ruled out: neural nets (small data)\n- metric: balanced accuracy\n")
LEDGER_A = (
    "- Figure 01_missingness.png: 1 of 4 columns has missing values; worst is age at 2.0%.\n"
    "- Figure 01_missing_pattern.png: Only age has gaps: 4 of 200 rows (2.0%); every other column is complete.\n"
    "- Figure 02_target_balance.png: y=0 is 70.0% of 200 rows, so a majority-class dummy scores 70.0%.\n"
    "- Figure 02_variables_p01.png: Page 1 of 1: 4 columns (age to y); 1 has missing values, 0 are strongly skewed.\n"
)
LEDGER_B = (
    "- Figure 04_cleaning.png: 2 ages above 110 were clipped; all 200 rows kept before the split.\n"
    "- Figure 07_correlations_train.png: No pair of the 2 numeric features exceeds |rho| 0.9 on the 120 training rows.\n"
    "- Figure 07_feature_target_train.png: Strongest link to the target on the training split: age (eta 0.21); "
    "1 of 2 features score below 0.05.\n"
)
LEDGER_C = (
    "- Figure 12_permutation_importance.png: Shuffling age costs 0.041 of accuracy (validation baseline 0.725); "
    "1 of 2 features cost less than 0.001.\n"
    "- Figure 12_attribution_coefficients.png: Each extra 10 years of age raises the readmission log-odds by 0.18.\n"
)
FIGURES = {"A": ["01_missingness.png", "01_missing_pattern.png", "02_target_balance.png", "02_variables_p01.png"],
           "B": ["04_cleaning.png", "07_correlations_train.png", "07_feature_target_train.png"],
           "C": ["12_permutation_importance.png", "12_attribution_coefficients.png"]}


def records(upto: str = "A") -> dict:
    """Every file Gates A..``upto`` need, as {relative path: str or bytes}."""
    files: dict = {SOURCE: json.dumps(SOURCE_JSON), PROFILE: json.dumps(profile()),
                   DICTIONARY: dictionary_csv(dictionary_rows()), MISSINGNESS: missingness_csv(missingness_rows()),
                   LABEL_AUDIT: LABEL_AUDIT_TEXT, PAGES: json.dumps(PAGES_JSON)}
    gates = "ABC"[: "ABC".index(upto) + 1]
    for gate in gates:
        files.update({FIG(name): PNG for name in FIGURES[gate]})
    if "B" in gates:
        files.update({STATE: json.dumps(STATE_JSON), PLAN: plan_csv(plan_rows()),
                      ASSOCIATIONS: json.dumps(associations()),
                      CLEANING: _csv(CLEANING_HEADER, [{"rule": "clip age to 18..110",
                                                        "reason": "two ages above 110 are entry errors",
                                                        "rows_before": "200", "rows_after": "200",
                                                        "cells_changed": "2", "columns_changed": "age",
                                                        "date": "2026-09-23"}])})
    if "C" in gates:
        files.update({TRAINING: json.dumps({"model": "LogisticRegression", "n_rows": 120, "n_train": 120,
                                            "full": True, "date": "2026-09-24"}) + "\n",
                      IMPORTANCE: json.dumps(importance())})
    return files


def ledger(upto: str = "A") -> str:
    """PIPELINE.md text with every figure line (and, from Gate B, the model rationale) Gates A..``upto`` need."""
    text = LEDGER_A
    if upto in ("B", "C"):
        text += LEDGER_B + RATIONALE
    if upto == "C":
        text += LEDGER_C
    return text


def write(root: Path, files: dict) -> None:
    for rel, data in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_gate_records.py`:

```python
"""Tests for hooks/gate_records.py (stdlib): python3 -m unittest tests/test_gate_records.py"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hooks"))
sys.path.insert(0, str(HERE))
import gate_records as gr  # noqa: E402
import v05_fixtures as fx  # noqa: E402

NOTE_CASES = json.loads((HERE / "note_rule_cases.json").read_text())
PARTIAL = "- Override: partial data - user approved 2026-09-23 - reason: demo on a sample\n"


def problems(gate, files, run_text=""):
    with tempfile.TemporaryDirectory() as tmp:
        fx.write(Path(tmp), files)
        check = {"A": gr.gate_a_problems, "B": gr.gate_b_problems, "C": gr.gate_c_problems}[gate]
        return check(Path(tmp) / "ml_pipeline", run_text)


def joined(gate, files, run_text=""):
    return " | ".join(problems(gate, files, run_text))


def with_(key, value, upto="A"):
    files = fx.records(upto)
    if value is None:
        del files[key]
    else:
        files[key] = value
    return files


class NoteRule(unittest.TestCase):
    def test_shared_cases(self):
        for case in NOTE_CASES["cases"]:
            with self.subTest(note=case["note"]):
                stats = NOTE_CASES["stats"] if case["column"] else None
                self.assertEqual(gr.note_problem(case["note"], stats), case["problem"])


class GateA(unittest.TestCase):
    def test_complete_records_pass(self):
        self.assertEqual(problems("A", fx.records("A")), [])

    def test_source_receipt_required(self):
        files = with_(fx.SOURCE, None)
        self.assertIn("source.json missing", joined("A", files))
        self.assertEqual(problems("A", files, PARTIAL), [])
        self.assertIn("source.json missing", joined("A", files, "### Override partial data agreed\n"))

    def test_partial_profile_needs_an_override(self):
        p = fx.profile()
        p["source"] = {"name": None, "n_rows": 100, "full_data": False,
                       "reason": "100 rows x 4 columns matches no loaded source (a sample or a slice?)"}
        files = with_(fx.PROFILE, json.dumps(p))
        self.assertIn("does not cover the full data (100 rows x 4 columns", joined("A", files))
        self.assertEqual(problems("A", files, PARTIAL), [])

    def test_old_profile_schema(self):
        p = fx.profile()
        p["schema_version"] = "1"
        self.assertIn("from an older guard", joined("A", with_(fx.PROFILE, json.dumps(p))))

    def test_profile_statistics_complete(self):
        p = fx.profile()
        del p["columns"]["age"]["p99"]
        self.assertIn("lacks describe/info statistics for age", joined("A", with_(fx.PROFILE, json.dumps(p))))

    def test_all_missing_column_needs_common_keys_only(self):
        p = fx.profile()
        p["columns"]["empty"] = {"kind": "numeric", "dtype": "float64", "non_null": 0, "missing": 200,
                                 "missing_frac": 1.0, "n_unique": 0, "unique_frac": 0.0, "memory_bytes": 1600}
        p["shape"]["n_cols"] = 5
        rows = fx.dictionary_rows() + [dict(fx.dictionary_rows()[0], column="empty", observed="all missing",
                                            role="drop", figure_note="empty: all 200 rows are missing, so the column is dropped")]
        missing = fx.missingness_rows() + [{"column": "empty", "n_missing": "200", "pct_missing": "100.00",
                                            "mechanism": "structural", "evidence": "never collected at this site",
                                            "decision": "drop-column", "reason": "no values at all"}]
        files = fx.records("A")
        files.update({fx.PROFILE: json.dumps(p), fx.DICTIONARY: fx.dictionary_csv(rows),
                      fx.MISSINGNESS: fx.missingness_csv(missing),
                      fx.PAGES: json.dumps({"02_variables_p01.png": ["age", "plan", "patient_id", "y", "empty"]})})
        self.assertEqual(problems("A", files), [])

    def test_dictionary_must_match_the_profile(self):
        base = fx.dictionary_rows()
        cases = {
            "data_dictionary.csv has no row for age": base[1:],
            "rows for columns not in the profile: ghost": base + [dict(base[0], column="ghost")],
            "more than one row for plan": base + [dict(base[1])],
        }
        for expected, rows in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, joined("A", with_(fx.DICTIONARY, fx.dictionary_csv(rows))))
        self.assertIn("data_dictionary.csv missing", joined("A", with_(fx.DICTIONARY, None)))

    def test_dictionary_word_fields(self):
        cases = {
            "blank fields: age (unit, figure_note)": lambda r: r[0].update(unit="", figure_note=""),
            "available_at_prediction must be yes or no: plan": lambda r: r[1].update(available_at_prediction="sometimes"),
            "role must be one of feature, target, id, leak, drop: plan": lambda r: r[1].update(role="predictor"),
            "exactly one target row, y (found: none)": lambda r: r[3].update(role="feature"),
            "features not available at prediction time: age": lambda r: r[0].update(available_at_prediction="no"),
            "age (cites no statistic of this column)":
                lambda r: r[0].update(figure_note="age: looks fine overall, there are 12 things to check here"),
            "columns share the same figure note: age = plan": lambda r: r[1].update(figure_note=r[0]["figure_note"]),
        }
        for expected, change in cases.items():
            with self.subTest(expected=expected):
                rows = fx.dictionary_rows()
                change(rows)
                self.assertIn(expected, joined("A", with_(fx.DICTIONARY, fx.dictionary_csv(rows))))

    def test_excel_saved_dictionary_passes(self):
        excel = "﻿" + fx.dictionary_csv(fx.dictionary_rows()).replace("\n", "\r\n")
        self.assertEqual(problems("A", with_(fx.DICTIONARY, excel)), [])

    def test_long_lists_are_capped(self):
        p = fx.profile()
        for i in range(15):
            p["columns"][f"f{i:02d}"] = fx.profile()["columns"]["plan"]
        message = joined("A", with_(fx.PROFILE, json.dumps(p)))
        self.assertIn("has no row for f00, f01, f02, f03, f04, f05, f06, f07, f08, f09 and 5 more", message)

    def test_every_column_on_a_page(self):
        self.assertIn("figures/pages.json missing", joined("A", with_(fx.PAGES, None)))
        absent = json.dumps({"02_variables_p01.png": ["age", "plan", "y"]})
        self.assertIn("columns not on any figure page: patient_id", joined("A", with_(fx.PAGES, absent)))
        extra = json.dumps({"02_variables_p01.png": ["age", "plan", "y"], "02_variables_p02.png": ["patient_id"]})
        self.assertIn("pages listed in pages.json but not on disk: 02_variables_p02.png",
                      joined("A", with_(fx.PAGES, extra)))

    def test_missingness(self):
        self.assertIn("missingness.csv missing", joined("A", with_(fx.MISSINGNESS, None)))
        self.assertIn("list each column with gaps once: age", joined("A", with_(fx.MISSINGNESS, fx.missingness_csv([]))))
        cases = {
            "counts differ from the profile for age": {"n_missing": "5"},
            "needs a likely cause": {"mechanism": "unclear"},
        }
        for expected, change in cases.items():
            with self.subTest(expected=expected):
                rows = fx.missingness_rows()
                rows[0].update(change)
                self.assertIn(expected, joined("A", with_(fx.MISSINGNESS, fx.missingness_csv(rows))))
        rows = fx.missingness_rows()
        rows[0].update(evidence="")
        self.assertIn("needs a likely cause", joined("A", with_(fx.MISSINGNESS, fx.missingness_csv(rows))))
        rows = fx.missingness_rows()
        rows[0].update(mechanism="MAR")
        self.assertEqual(problems("A", with_(fx.MISSINGNESS, fx.missingness_csv(rows))), [])
        self.assertIn("01_missing_pattern.png missing", joined("A", with_(fx.FIG("01_missing_pattern.png"), None)))

    def test_label_audit(self):
        text = fx.LABEL_AUDIT_TEXT
        self.assertIn("label_audit.md missing", joined("A", with_(fx.LABEL_AUDIT, None)))
        no_block = text.split("<!-- end computed -->")[1]
        self.assertIn("lost its computed block", joined("A", with_(fx.LABEL_AUDIT, no_block)))
        self.assertIn("class counts differ from the profile",
                      joined("A", with_(fx.LABEL_AUDIT, text.replace("0 = 140", "0 = 139"))))
        thin = text.replace("Claims data joined on patient number by the hospital data team.", "Claims.")
        self.assertIn("at least 40 characters of writing: Source", joined("A", with_(fx.LABEL_AUDIT, thin)))
        template = text.replace("Claims data joined on patient number by the hospital data team.",
                                "<!-- who or what produced the label, and how -->")
        self.assertIn("at least 40 characters of writing: Source", joined("A", with_(fx.LABEL_AUDIT, template)))

    def test_leakage_suspects_kept_must_be_discussed(self):
        p = fx.profile()
        p["leakage_suspects"] = [{"column": "plan", "reason": "predicts target with 99% purity"}]
        files = with_(fx.PROFILE, json.dumps(p))
        self.assertIn("must discuss each leakage suspect kept as a feature or id: plan", joined("A", files))
        rows = fx.dictionary_rows()
        rows[1]["role"] = "leak"
        files[fx.DICTIONARY] = fx.dictionary_csv(rows)
        self.assertEqual(problems("A", files), [])
        files = with_(fx.PROFILE, json.dumps(p))
        files[fx.LABEL_AUDIT] = fx.LABEL_AUDIT_TEXT.replace(
            "No column is recorded after discharge;", "Plan is chosen at admission, before the outcome;")
        self.assertEqual(problems("A", files), [])

    def test_regression_label_line(self):
        p = fx.profile()
        p["target"] = {"name": "y", "task": "regression", "n_classes": None, "class_balance": None,
                       "minority_frac": None, "class_counts": None, "mean": 0.3, "std": 0.46, "min": 0.0, "max": 1.0}
        text = fx.LABEL_AUDIT_TEXT.replace("- classes: 0 = 140 (70.0%); 1 = 60 (30.0%)",
                                           "- target: min 0.0, max 1.0, mean 0.3, std 0.46")
        files = fx.records("A")
        files.update({fx.PROFILE: json.dumps(p), fx.LABEL_AUDIT: text})
        self.assertEqual(problems("A", files), [])


if __name__ == "__main__":
    unittest.main()
```

(The `GateB` / `GateC` classes are added in Task 10.)

- [ ] **Step 3: Run and watch them fail**

Run: `python3 -m unittest tests/test_gate_records.py -v`
Expected: `ModuleNotFoundError: No module named 'gate_records'`.

- [ ] **Step 4: Implement**

`hooks/gate_records.py`:

```python
"""Gate-time checks for the v0.5 records: every profiled column has its complete, explained record.

Imported by guard_training.py. Each check reads the small files the guard library writes into
ml_pipeline/ and returns problems in the words the agent will read. Stdlib only.
"""
from __future__ import annotations

import csv
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


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
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


def _csv_rows(path: Path) -> list[dict] | None:
    """Rows of a CSV record with stripped values; None when the file is absent. Excel's BOM and CRLF are fine."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{k: (v or "").strip() for k, v in row.items() if k is not None} for row in csv.DictReader(handle)]


def _valid_combo(value: str, allowed) -> bool:
    parts = [p.strip() for p in value.lower().split("+")]
    return bool(value.strip()) and all(p in allowed for p in parts)


def _int(value) -> int | None:
    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
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
    text = path.read_text(encoding="utf-8-sig", errors="replace")
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
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m unittest tests/test_gate_records.py -v`
Expected: `OK` (NoteRule and GateA).

- [ ] **Step 6: Commit**

```bash
git add hooks/gate_records.py tests/v05_fixtures.py tests/test_gate_records.py
git commit -m "feat(hook): gate_records - Gate A checks every profiled column's records and notes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Gate B and Gate C checks; wire the records and the explanation rule into the hook

**Files:**
- Modify: `hooks/gate_records.py` (add Gate B/C), `hooks/guard_training.py` (`LEDGER`, `gate_problems`, import)
- Test: `tests/test_gate_records.py` (classes `GateB`, `GateC`), `tests/test_guard_training.py` (rewrite `GateRecording` on the fixtures)

**Interfaces:**
- Consumes: `gate_records.gate_a_problems`, `note_problem`, helpers (Task 9); `guard_training.current_run` (existing).
- Produces: `gate_records.gate_b_problems(state_dir, run_text)`, `gate_records.gate_c_problems(state_dir, run_text)`; `guard_training.gate_problems` now also returns the records' problems and `"<file>.png: explanation <why>"` for vague figure lines.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gate_records.py` (before the `__main__` guard):

```python
class GateB(unittest.TestCase):
    def test_complete_records_pass(self):
        self.assertEqual(problems("B", fx.records("B")), [])

    def test_cleaning(self):
        files = with_(fx.CLEANING, None, "B")
        self.assertIn("no cleaning recorded", joined("B", files))
        self.assertEqual(problems("B", files, "- Cleaning: none needed - reason: validated upstream by the registry\n"), [])

    def test_missingness_decisions(self):
        rows = fx.missingness_rows()
        rows[0]["decision"] = "guess"
        self.assertIn("needs a decision", joined("B", with_(fx.MISSINGNESS, fx.missingness_csv(rows), "B")))
        rows[0]["decision"] = "impute-median+indicator"
        self.assertEqual(problems("B", with_(fx.MISSINGNESS, fx.missingness_csv(rows), "B")), [])

    def test_categorical_plan(self):
        self.assertIn("categorical_plan.csv missing", joined("B", with_(fx.PLAN, None, "B")))
        rows = fx.plan_rows()
        rows[0]["encoding"] = ""
        self.assertIn("needs an encoding", joined("B", with_(fx.PLAN, fx.plan_csv(rows), "B")))

    def test_plan_ignores_non_features(self):
        rows = fx.dictionary_rows()
        rows[1]["role"] = "drop"
        files = with_(fx.PLAN, None, "B")
        files[fx.DICTIONARY] = fx.dictionary_csv(rows)
        self.assertEqual(problems("B", files), [])

    def test_associations_on_the_training_split(self):
        self.assertIn("associations.json missing", joined("B", with_(fx.ASSOCIATIONS, None, "B")))
        a = fx.associations()
        a["computed_on"]["n_rows"] = 200
        self.assertIn("was not computed on the train split (200 rows, the train split has 120)",
                      joined("B", with_(fx.ASSOCIATIONS, json.dumps(a), "B")))
        self.assertIn("figures/07_feature_target_train.png missing",
                      joined("B", with_(fx.FIG("07_feature_target_train.png"), None, "B")))


class GateC(unittest.TestCase):
    def test_complete_records_pass(self):
        self.assertEqual(problems("C", fx.records("C")), [])

    def test_full_training_run(self):
        self.assertIn("no full-data training run", joined("C", with_(fx.TRAINING, None, "C")))
        pilot = json.dumps({"model": "pilot", "n_rows": 50, "n_train": 120, "full": False, "date": "2026-09-24"}) + "\n"
        self.assertIn("no full-data training run", joined("C", with_(fx.TRAINING, pilot, "C")))

    def test_importance_on_validation(self):
        imp = fx.importance()
        imp["computed_on"]["split"] = "train"
        self.assertIn("importance.json was not computed on the validation split",
                      joined("C", with_(fx.IMPORTANCE, json.dumps(imp), "C")))

    def test_attribution_figure(self):
        self.assertIn("no figures/12_attribution_*.png",
                      joined("C", with_(fx.FIG("12_attribution_coefficients.png"), None, "C")))
```

In `tests/test_guard_training.py`: add after the `sys.path.insert` for hooks:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v05_fixtures as fx  # noqa: E402
```

Delete the constants `PNG`, `LEDGER_A`, `ARTIFACTS_A`, `RATIONALE` (keep `TINY` and `write_md`), and replace class `GateRecording` with:

```python
class GateRecording(unittest.TestCase):
    """A gate line is denied until that gate's records exist and explain every column."""

    def record(self, gate, files, ledger=None, existing=""):
        text = (fx.ledger(gate if gate in "ABC" else "C") if ledger is None else ledger)
        return run(write_md(text + f"- Gate {gate}: approved 2026-09-23\n"), pipeline_md=existing, tool="Write",
                   files=files)

    def test_gate_a_denied_without_anything(self):
        decision, reason = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write")
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertIn("figures/01_*.png 0 found (need 1)", reason)

    def test_gate_a_allowed_when_complete(self):
        self.assertEqual(self.record("A", fx.records("A")), ("allow", ""))

    def test_missing_records_named_in_the_denial(self):
        files = fx.records("A")
        del files[fx.DICTIONARY]
        decision, reason = self.record("A", files)
        self.assertEqual(decision, "deny")
        self.assertIn("data_dictionary.csv missing", reason)

    def test_png_without_ledger_line_denied(self):
        decision, reason = self.record("A", {**fx.records("A"), fx.FIG("02_extra.png"): fx.PNG})
        self.assertEqual(decision, "deny")
        self.assertIn("02_extra.png has no '- Figure 02_extra.png:' line", reason)

    def test_tiny_png_denied(self):
        decision, reason = self.record("A", {**fx.records("A"), fx.FIG("01_missingness.png"): TINY})
        self.assertEqual(decision, "deny")
        self.assertIn("(empty?)", reason)

    def test_vague_figure_explanation_denied(self):
        ledger = fx.ledger("A").replace(
            "y=0 is 70.0% of 200 rows, so a majority-class dummy scores 70.0%.",
            "shows the class balance of the target variable nicely")
        decision, reason = self.record("A", fx.records("A"), ledger=ledger)
        self.assertEqual(decision, "deny")
        self.assertIn("02_target_balance.png: explanation contains no number", reason)

    def test_ledger_in_existing_file_counts(self):
        decision, _ = run(write_md("- Gate A: approved 2026-09-23\n"), pipeline_md=fx.ledger("A"), tool="Write",
                          files=fx.records("A"))
        self.assertEqual(decision, "allow")

    def test_bash_append_is_checked_too(self):
        cmd = {"command": "echo '- Gate A: approved 2026-09-16' >> ml_pipeline/PIPELINE.md"}
        self.assertEqual(run(cmd, pipeline_md="")[0], "deny")

    def test_negated_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate A: not yet approved\n"), pipeline_md="", tool="Write")[0], "allow")

    def test_gate_b_needs_rationale_and_records(self):
        decision, reason = self.record("B", fx.records("B"), ledger=fx.ledger("B").replace(fx.RATIONALE, ""))
        self.assertEqual(decision, "deny")
        self.assertIn("Model rationale block missing lines: traits, baseline, candidates, ruled out, metric", reason)
        files = fx.records("B")
        del files[fx.ASSOCIATIONS]
        decision, reason = self.record("B", files)
        self.assertIn("associations.json missing", reason)
        self.assertEqual(self.record("B", fx.records("B")), ("allow", ""))

    def test_gate_c_and_d(self):
        self.assertEqual(self.record("C", fx.records("C")), ("allow", ""))
        files = {**fx.records("C"), fx.FIG("13_errors.png"): fx.PNG}
        ledger = fx.ledger("C") + "- Figure 13_errors.png: the worst slice is plan=pro with 12 of 40 validation rows wrong.\n"
        decision, reason = self.record("D", files, ledger=ledger)
        self.assertEqual(decision, "deny")
        self.assertIn("no '- Step 14 final test:' line", reason)
        ledger += "- Step 14 final test: accuracy=0.7900 (touch 1, 2026-09-25)\n"
        self.assertEqual(self.record("D", files, ledger=ledger)[0], "allow")
```

- [ ] **Step 2: Run and watch them fail**

Run: `python3 -m unittest tests/test_gate_records.py tests/test_guard_training.py`
Expected: `AttributeError: module 'gate_records' has no attribute 'gate_b_problems'`; `GateRecording` failures (`test_missing_records_named_in_the_denial`, `test_vague_figure_explanation_denied`, `test_gate_b_needs_rationale_and_records` expect v0.5 checks that are not wired yet).

- [ ] **Step 3: Implement Gate B and C in `hooks/gate_records.py`**

```python
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
    cleaning = _csv_rows(state_dir / CLEANING_FILE) or []
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
```

- [ ] **Step 4: Wire into `hooks/guard_training.py`**

After `from pathlib import Path` add:

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_records  # noqa: E402
```

Change `LEDGER` to capture the explanation:

```python
LEDGER = re.compile(r"^\s*-\s*Figure\s+(\S+\.png):[ \t]*(\S[^\n]*)$", re.I | re.M)
```

In `gate_problems`, replace `ledgered = {m.group(1) for m in LEDGER.finditer(ledger_text)}` and the PNG loop with:

```python
    explained = {m.group(1): m.group(2) for m in LEDGER.finditer(ledger_text)}
    for png in figures:
        size = png.stat().st_size
        if size < MIN_PNG_BYTES:
            problems.append(f"{FIGURES_DIR}/{png.name} is {size} bytes (empty?)")
        if png.name not in explained:
            problems.append(f"{png.name} has no '- Figure {png.name}:' line")
        else:
            why = gate_records.note_problem(explained[png.name])
            if why:
                problems.append(f"{png.name}: explanation {why}")
```

and before `return problems` add:

```python
    checks = {"A": gate_records.gate_a_problems, "B": gate_records.gate_b_problems, "C": gate_records.gate_c_problems}
    if gate in checks:
        problems += checks[gate](state_dir, current_run(ledger_text))
```

- [ ] **Step 5: Run the hook suites and the probe**

Run: `python3 -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py && python3 tests/probe_hook_audit.py | tail -1`
Expected: `OK`; `0 of 23 cases behave wrongly`.

- [ ] **Step 6: Commit**

```bash
git add hooks/gate_records.py hooks/guard_training.py tests/test_gate_records.py tests/test_guard_training.py
git commit -m "feat(hook): Gates A-C check the v0.5 records; every figure line needs a real explanation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: The pilot rule and the `target-stats-outside-train` red flag

**Files:**
- Modify: `hooks/guard_training.py` (constants, `analyse`, `gate_state`, `decide`, `red_flag`, new `_target_stats_outside_train`)
- Test: `tests/test_guard_training.py` (new class `PilotRule`; `RedFlags`; `RealLedgerRegression` expected dict gains `"pilot": False`)

**Interfaces:**
- Produces: `analyse()` result gains key `"pilot"`; `gate_state()` result gains key `"pilot"` (True when the current run has `- Override: pilot - …`); `PILOT_MSG`.

- [ ] **Step 1: Write the failing tests**

```python
class PilotRule(unittest.TestCase):
    """After Gate B, training on a slice needs the user's pilot override; Gate C still needs a full run."""
    PILOT = "- Override: pilot - 5,000 rows - user approved 2026-09-24 - reason: smoke test\n"

    def test_slices_in_a_fit_call_denied(self):
        for code in ("model.fit(X_train[:5000], y_train[:5000])", "model.fit(X_train.iloc[:500], y_train.iloc[:500])",
                     "model.fit(X_train.sample(1000), y)", "guard.fit(clf, X_train.head(500), y_train.head(500))",
                     "df = pd.read_csv('d.csv', nrows=1000)\nmodel.fit(X_train, y_train)"):
            with self.subTest(code=code):
                decision, reason = run({"command": code}, GATE_AB)
                self.assertEqual(decision, "deny")
                self.assertIn("slice of the data", reason)

    def test_whole_split_and_column_selection_allowed(self):
        for code in ("model.fit(X_train, y_train)", "model.fit(X_train.loc[:, features], y_train)",
                     "model.fit(X_train[:, :5], y_train)"):
            with self.subTest(code=code):
                self.assertEqual(run({"command": code}, GATE_AB)[0], "allow")

    def test_override_allows_a_pilot(self):
        self.assertEqual(run({"command": "model.fit(X_train[:5000], y_train[:5000])"}, GATE_AB + self.PILOT)[0], "allow")

    def test_override_from_an_older_run_does_not_count(self):
        md = GATE_AB + self.PILOT + "## Run: next (2026-09-25)\n- Gate A: approved 2026-09-25\n- Gate B: approved 2026-09-25\n"
        self.assertEqual(run({"command": "model.fit(X_train[:5000], y_train[:5000])"}, md)[0], "deny")

    def test_transformer_slices_are_not_pilots(self):
        self.assertEqual(run({"command": "scaler.fit(X_train[:1000])"}, GATE_AB)[0], "allow")
```

Add to class `RedFlags`:

```python
    def test_target_stats_outside_the_training_split(self):
        for code in ("scores = mutual_info_classif(X, y)", "f_classif(X_all, y_all)",
                     "sel = SelectKBest(k=5)\nsel.fit(X, y)", "SelectKBest(chi2, k=5).fit_transform(X, y)"):
            with self.subTest(code=code):
                decision, reason = run({"command": code}, GATE_A, files=profile_with())
                self.assertEqual(decision, "deny")
                self.assertIn("red flag target-stats-outside-train", reason)
        for code in ("mutual_info_classif(X_train, y_train)", "SelectKBest(k=5).fit(X_train, y_train)",
                     "sel = SelectKBest(k=5).fit(X_train, y_train)\nmodel.fit(X_tr_sel, y_tr)"):
            with self.subTest(code=code):
                self.assertEqual(run({"command": code}, GATE_AB, files=profile_with())[0], "allow")

    def test_target_stats_override(self):
        md = GATE_A + "- Override: red flag target-stats-outside-train - user approved 2026-09-24 - reason: inside CV\n"
        self.assertEqual(run({"command": "mutual_info_classif(X, y)"}, md, files=profile_with())[0], "allow")
```

In `RealLedgerRegression.test_parses_as_gate_b_only`, change the expected dict to
`{"A": False, "B": True, "C": False, "D": False, "override": False, "override_c": False, "pilot": False}`.

- [ ] **Step 2: Run and watch them fail**

Run: `python3 -m unittest tests.test_guard_training.PilotRule tests.test_guard_training.RedFlags tests.test_guard_training.RealLedgerRegression -v`
Expected: the slice cases, the older-run case, the target-stats deny cases, and the real-ledger dict fail; the allow cases pass (kept as guards).

- [ ] **Step 3: Implement**

Constants (after `OPT_STEP`):

```python
# A slice of the rows inside a model-training call, or nrows= in code that trains: a pilot.
SLICE = re.compile(r"\[\s*-?\d*\s*:\s*-?\d+\s*\]|\.\s*i?loc\s*\[\s*-?\d*\s*:\s*-?\d+|\.\s*(?:head|sample)\s*\(")
NROWS = re.compile(r"\bnrows\s*=")
PILOT_OVERRIDE = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*pilot[ \t]*-", re.I | re.M)
SELECTOR_NAMES = r"SelectKBest|SelectPercentile|GenericUnivariateSelect|SelectFpr|SelectFdr|SelectFwe"
SELECTOR = re.compile(r"\b(?:" + SELECTOR_NAMES + r")\b")
SELECTION_SCORE = re.compile(
    r"\b(?:mutual_info_classif|mutual_info_regression|f_classif|f_regression|chi2|r_regression)\s*\(")
```

Message (after `SELF_DISABLE_MSG`):

```python
PILOT_MSG = (
    "ml-pipeline: this trains on a slice of the data ({0!r}). Train on the full training split. If the user "
    "explicitly agreed to a quick pilot, record `- Override: pilot - <size> - user approved YYYY-MM-DD - reason: ...` "
    "first; Gate C still needs a full-data training run."
)
TARGET_STATS_WHY = ("target-based feature selection must use the training split only ({0}) - select features inside "
                    "the training data or a CV fold")
```

In `analyse`, initialise `found` with `"pilot": None`, and inside the `FIT_CALL` loop after `kind = _classify(...)` add:

```python
        if kind == "model" and found["pilot"] is None and SLICE.search(args):
            found["pilot"] = snippet
```

and after the `EVAL_CALL` loop:

```python
    nrows = NROWS.search(code)
    if found["model"] and found["pilot"] is None and nrows:
        found["pilot"] = code[max(0, nrows.start() - 30): nrows.end() + 10].strip()
```

In `gate_state`, compute `run = current_run(markdown)` once, iterate `run.splitlines()`, and before `return state` add `state["pilot"] = bool(PILOT_OVERRIDE.search(run))`.

In `decide`, replace `if gates["B"] or gates["override"]: return None` with:

```python
    if gates["B"] or gates["override"]:
        if found["pilot"] and not gates.get("pilot"):
            return PILOT_MSG.format(found["pilot"])
        return None
```

Add before `red_flag`:

```python
def _target_stats_outside_train(code: str) -> str | None:
    """The first feature-selection call whose target statistics are not computed on the training split."""
    for match in SELECTION_SCORE.finditer(code):
        args = _args(code, match.end())
        if "train" not in args.lower():
            return code[match.start(): match.end() + len(args) + 1][:60]
    for match in FIT_CALL.finditer(code):
        receiver = match.group(1) or ""
        args = _args(code, match.end())
        assigned = bool(receiver) and re.search(
            r"\b" + re.escape(receiver) + r"\s*=\s*[\w.]*\b(?:" + SELECTOR_NAMES + r")\s*\(", code)
        if (SELECTOR.search(receiver) or assigned) and "train" not in args.lower():
            return code[match.start(): match.end() + len(args) + 1].strip()[:60]
    return None
```

At the end of `red_flag`, before `return None`:

```python
    if "target-stats-outside-train" not in overrides:
        snippet = _target_stats_outside_train(code)
        if snippet:
            return RED_FLAG_MSG.format("target-stats-outside-train", TARGET_STATS_WHY.format(snippet))
```

- [ ] **Step 4: Run the hook suites and the probe**

Run: `python3 -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py && python3 tests/probe_hook_audit.py | tail -1`
Expected: `OK`; `0 of 23 cases behave wrongly`.

- [ ] **Step 5: Commit**

```bash
git add hooks/guard_training.py tests/test_guard_training.py
git commit -m "feat(hook): pilot rule after Gate B and the target-stats-outside-train red flag

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: End to end through the real guard and hook

**Files:**
- Create: `tests/test_v05_end_to_end.py`

**Interfaces:**
- Consumes: every guard helper (Tasks 1–8) and `guard_training.main` (Tasks 9–11).

- [ ] **Step 1: Write the test**

```python
"""End to end: the honest v0.5 path records Gates A-C through the real hook; each shortcut is stopped.

Run with the dev venv: .venv/bin/python -m unittest tests/test_v05_end_to_end.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "hooks"))
try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - the stdlib suite skips this file
    raise unittest.SkipTest("pandas/numpy not installed; run with .venv/bin/python")
import guard_training as hook  # noqa: E402
import mlpipeline_guard as guard  # noqa: E402

RATIONALE = ("Model rationale:\n- traits: binary, 600 rows (small), has_datetime, no groups\n"
             "- baseline: majority-class dummy + logistic regression\n- candidates: gradient boosting with small trees\n"
             "- ruled out: neural nets (small data); random split (temporal)\n- metric: balanced accuracy\n")
SECTIONS = {
    "Definition": "Readmission within 30 days of discharge, counted once per admission.",
    "Source": "Claims data joined on patient number by the hospital data team.",
    "When labeled": "Known 30 days after discharge, after the prediction time at discharge.",
    "Noise and consistency": "Late claims can hide a few readmissions; conflicting rows are listed above.",
}


def admissions(n=600, seed=11):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    df = pd.DataFrame({
        "admitted_at": pd.date_range("2024-01-01", periods=n, freq="12h"),
        "age": rng.normal(55, 14, n).round(1),
        "income": rng.lognormal(10, 0.8, n).round(0),
        "zip": rng.integers(0, 90, n).astype(str),
        "plan": rng.choice(["basic", "plus", "pro"], n).astype(object),
        "smoker": rng.integers(0, 2, n).astype(bool),
        "prior_visits": np.clip(np.round(signal * 2 + 3), 0, None),
        "patient_id": np.arange(n),
        "y": (signal + rng.normal(0, 0.8, n) > 0).astype(int),
    })
    df.loc[rng.choice(n, 30, replace=False), "income"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "plan"] = np.nan
    return df


class Threshold:
    """Predicts readmission when prior visits exceed 3."""
    def fit(self, X, y=None):
        return self

    @staticmethod
    def predict_frame(X):
        return (X["prior_visits"] > 3).astype(int).to_numpy()


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "ml_pipeline"
        self.csv = self.root / "admissions.csv"
        admissions().to_csv(self.csv, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def hook(self, tool_input, tool="Write"):
        """The real PreToolUse hook, in-process, exactly as Claude Code calls it."""
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(self.root)})
        out = io.StringIO()
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(out), mock.patch("sys.stdin", io.StringIO(payload)):
            os.environ.pop("ML_PIPELINE_ENFORCE", None)
            os.environ.pop("ML_PIPELINE_FILE", None)
            hook.main()
        result = json.loads(out.getvalue() or "{}").get("hookSpecificOutput", {})
        return result.get("permissionDecision", "allow"), result.get("permissionDecisionReason", "")

    def record(self, gate, extra=""):
        """Ask the hook to append a gate line to PIPELINE.md; write it only when allowed."""
        md = self.state / "PIPELINE.md"
        content = (md.read_text() if md.is_file() else "") + extra + f"- Gate {gate}: approved 2026-09-24\n"
        decision, reason = self.hook({"file_path": str(md), "content": content})
        if decision == "allow":
            md.write_text(content)
        return decision, reason

    def edit_csv(self, name, fill):
        path = self.state / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header, rows = reader.fieldnames, list(reader)
        for row in rows:
            fill(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    def phase1(self, profiled=None):
        df = guard.load(self.csv, state_dir=self.state, parse_dates=["admitted_at"])
        prof = guard.profile(df if profiled is None else profiled(df), target="y", time_col="admitted_at",
                             state_dir=self.state)
        guard.eda_figures(df, target="y", time_col="admitted_at", state_dir=self.state)
        guard.variable_figures(df, target="y", state_dir=self.state)
        guard.data_dictionary(df, state_dir=self.state)
        roles = {"y": "target", "patient_id": "id"}

        def fill(row):
            info = prof["columns"][row["column"]]
            row.update(meaning=f"{row['column']} as recorded at admission", unit="none", allowed="as observed",
                       source="admissions.csv", available_at_prediction="no" if row["column"] == "y" else "yes",
                       role=roles.get(row["column"], "feature"),
                       figure_note=(f"{row['column']}: {info['non_null']:,} recorded values and "
                                    f"{info['n_unique']:,} distinct ones in the full data"))
        self.edit_csv(guard.DICTIONARY_FILE, fill)
        guard.missingness(df, state_dir=self.state)
        self.edit_csv(guard.MISSINGNESS_FILE, lambda r: r.update(
            mechanism="random", evidence="gaps are spread evenly across admission months",
            decision="impute-median" if r["column"] == "income" else "impute-mode",
            reason="few gaps and no pattern over time"))
        guard.label_audit(df, target="y", state_dir=self.state)
        suspects = ", ".join(s["column"] for s in prof["leakage_suspects"]) or "none"
        bodies = dict(SECTIONS, **{"Leakage check": f"Profile leakage suspects: {suspects}; none is recorded after discharge."})
        audit = self.state / guard.LABEL_AUDIT_FILE
        text = audit.read_text()
        for name, body in bodies.items():
            text = re.sub(rf"(## {name}\n)<!--.*?-->", lambda m, body=body: m.group(1) + body, text, flags=re.S)
        audit.write_text(text)
        return df

    def phase2(self, df):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        clipped = df.assign(age=df["age"].clip(18, 100))
        cleaned = guard.log_cleaning("clip age to 18..100", df, clipped, reason="ages outside 18..100 are entry errors",
                                     state_dir=self.state)
        train, val, test = guard.split(cleaned, target="y", time_col="admitted_at", state_dir=self.state)
        guard.categorical_plan(cleaned, state_dir=self.state)
        self.edit_csv(guard.PLAN_FILE, lambda r: r.update(
            encoding="rare-grouping+one-hot" if r["high_cardinality"] == "yes" else "one-hot",
            reason="levels kept separate; rare zips grouped first"))
        guard.train_associations(train, target="y", state_dir=self.state)
        moved = int(((df["age"] < 18) | (df["age"] > 100)).sum())
        f, ax = plt.subplots()
        ax.bar(["before", "after"], [len(df), len(cleaned)])
        guard.fig(4, "cleaning", f, f"Clipping age kept all {len(cleaned):,} rows; {moved} values moved into 18..100.",
                  state_dir=self.state)
        return train, val, test

    def phase3(self, train, val, rows=None):
        import matplotlib.pyplot as plt
        X, y = train.drop(columns=["y"]), train["y"]
        if rows is not None:
            X, y = X.iloc[:rows], y.iloc[:rows]
        guard.fit(Threshold(), X, y, state_dir=self.state)
        guard.importance(Threshold.predict_frame, val, target="y", metric_fn=accuracy, state_dir=self.state)
        f, ax = plt.subplots()
        ax.bar(["prior_visits"], [1.0])
        guard.fig(12, "attribution_rule", f, "The model predicts readmission when prior_visits exceeds 3, its only input.",
                  state_dir=self.state)

    def test_honest_path_records_every_gate(self):
        df = self.phase1()
        self.assertEqual(self.record("A"), ("allow", ""))
        train, val, _ = self.phase2(df)
        self.assertEqual(self.record("B", extra=RATIONALE), ("allow", ""))
        self.phase3(train, val)
        self.assertEqual(self.record("C"), ("allow", ""))

    def test_a_sampled_profile_is_stopped(self):
        self.phase1(profiled=lambda df: df.sample(frac=0.5, random_state=0))
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("does not cover the full data", reason)

    def test_a_missing_column_note_is_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE, lambda r: r.update(figure_note="") if r["column"] == "income" else None)
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("blank fields: income (figure_note)", reason)

    def test_copy_pasted_notes_are_stopped(self):
        self.phase1()
        self.edit_csv(guard.DICTIONARY_FILE,
                      lambda r: r.update(figure_note="numbers look fine, 600 rows, nothing unusual in this one"))
        decision, reason = self.record("A")
        self.assertEqual(decision, "deny")
        self.assertIn("columns share the same figure note", reason)

    def test_associations_on_the_full_frame_are_refused(self):
        df = self.phase1()
        guard.split(df, target="y", time_col="admitted_at", state_dir=self.state)
        with self.assertRaises(guard.NotTrainingSplit):
            guard.train_associations(df, target="y", state_dir=self.state)

    def test_pilot_training_is_stopped(self):
        df = self.phase1()
        self.assertEqual(self.record("A")[0], "allow")
        train, val, _ = self.phase2(df)
        self.assertEqual(self.record("B", extra=RATIONALE)[0], "allow")
        decision, reason = self.hook({"command": "python -c 'model.fit(X_train[:100], y_train[:100])'"}, tool="Bash")
        self.assertEqual(decision, "deny")
        self.assertIn("slice of the data", reason)
        self.phase3(train, val, rows=100)
        decision, reason = self.record("C")
        self.assertEqual(decision, "deny")
        self.assertIn("no full-data training run", reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m unittest tests/test_v05_end_to_end.py -v`
Expected: all six pass. A test here is the first time every record is produced by the real helpers and read by the real hook; a failure means Tasks 1–11 disagree on a format. Debug with superpowers:systematic-debugging, fix the disagreeing side (spec wins), and add the missing unit test to that side's task file before the fix.

- [ ] **Step 3: Commit**

```bash
git add tests/test_v05_end_to_end.py
git commit -m "test: end to end - the honest path records Gates A-C; each shortcut is stopped

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Skill, SessionStart context, eval graders, CI

**Files:**
- Modify: `skills/ml-pipeline/SKILL.md`, `hooks/session_start.py`, `.github/workflows/tests.yml`
- Create: `evals/understand-first/graders/source-exists.md`, `evals/understand-first/graders/dictionary-exists.md`
- Test: `tests/test_session_start.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_session_start.py`, `test_names_the_helpers_when_plugin_root_is_set`, add:

```python
        for helper in ("guard.load()", "guard.data_dictionary()", "guard.missingness()", "guard.label_audit()",
                       "guard.variable_figures()", "guard.train_associations()", "guard.fit()", "guard.importance()"):
            self.assertIn(helper, context)
```

- [ ] **Step 2: Run and watch it fail**

Run: `python3 -m unittest tests/test_session_start.py -v`
Expected: `AssertionError: 'guard.load()' not found in ...`.

- [ ] **Step 3: Implement**

In `hooks/session_start.py`, replace
`"guard.profile() / guard.eda_figures() / guard.fig() / guard.split() / guard.final_test(); model-selection "`
with
`"guard.load() / guard.profile() / guard.data_dictionary() / guard.missingness() / guard.label_audit() / guard.eda_figures() / guard.variable_figures() / guard.fig() / guard.split() / guard.train_associations() / guard.fit() / guard.importance() / guard.final_test(); model-selection "`.

`evals/understand-first/graders/source-exists.md`:

```markdown
---
type: file_exists
path: ml_pipeline/source.json
---
```

`evals/understand-first/graders/dictionary-exists.md`:

```markdown
---
type: file_exists
path: ml_pipeline/data_dictionary.csv
---
```

In `.github/workflows/tests.yml`: the hooks job's test step becomes
`run: python -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py`;
the guard job's step becomes
`run: python -m unittest tests/test_mlpipeline_guard.py tests/test_v05_end_to_end.py`.

In `skills/ml-pipeline/SKILL.md` make these replacements (exact old text → new text):

Step 1 — replace the whole item `1. **Data inspection** — ... No modification yet.` with:

```markdown
1. **Data inspection** — copy the plugin's guard library to `ml_pipeline/guard.py` with `cp` (its path is
   given at session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py` next to this
   file). Load every raw file whole with `df = guard.load(path)` — never `nrows`, `usecols`, or a sample —
   which records the file's hash and row count; register a frame joined from several files with
   `guard.derive(df, name, parents=[...], how=...)`. Run `guard.profile(df, target=..., time_col=...,
   group_col=...)` on the whole frame and read `ml_pipeline/data_profile.json` before anything else:
   `source.full_data` must be true, and its `leakage_suspects` and `traits` decide the split, the metric,
   and the model family. Create the per-column records: `guard.data_dictionary(df)`,
   `guard.missingness(df)`, `guard.label_audit(df, target=...)`. For images or text, build a manifest table
   first (path, label, size or length) and profile that. No modification yet.
```

Step 2 — replace the whole item `2. **EDA** — ... problems spotted.` with:

```markdown
2. **EDA** — `guard.eda_figures(df, target=..., time_col=...)` writes the required figures
   (`01_missingness`, `02_target_balance`, `02_distributions`, `02_correlations` between features only,
   `02_temporal_coverage` when a date column exists) and `guard.variable_figures(df, target=...)` draws every
   column on grouped pages. Then fill every word field: in `data_dictionary.csv` the meaning, unit, allowed
   values, source, available-at-prediction (yes/no), role (feature/target/id/leak/drop), and a `figure_note`
   for every column; in `missingness.csv` the likely cause and evidence; every section of `label_audit.md`,
   naming each leakage suspect kept as a feature. A note is at least 40 characters and cites a real statistic
   of its column from the profile ("median 54, 2% missing"); no two columns share a note. Add other figures
   with `guard.fig(step, name, figure, explanation)`; every explanation has a number. If matplotlib is
   missing, install it — text descriptions are not figures. The Gate A report shows the row chain (file rows
   → profiled rows), the dictionary, the missingness table, and the label audit.
```

Step 4 — replace `Report before/after counts for every rule. Document every
   rule in PIPELINE.md.` with `Apply every rule as
   `cleaned = guard.log_cleaning(rule, before, after, reason=...)`, which records before/after counts in
   `cleaning_log.csv`; if nothing needs cleaning, record `- Cleaning: none needed - reason: <why>`. Fill
   `decision` and `reason` for every row of `missingness.csv`.`

Step 7 — replace the item with:

```markdown
7. **Feature engineering** — run `guard.categorical_plan(df)` and fill an encoding and a reason for every
   categorical feature. Run `guard.train_associations(train, target=...)` on the training split only for the
   correlation matrix and each feature's strength with the target — never on the full frame. Design features
   on the training set's statistics only, then apply the same transformations to validation/test.
```

Step 9 — append to the item: `Train through `guard.fit(model, X_train, y_train)` (custom loops:
`guard.record_training(train_data, name=...)`) so each run's row count is recorded; Gate C needs one run on
the whole training split, and a pilot on a slice needs the user's `- Override: pilot - ...` first.`

Step 12 — append to the item: `Run `guard.importance(model.predict, val, target=..., metric_fn=...)` for
permutation importance on the validation split, and save one attribution figure (SHAP, partial dependence,
or coefficients) with `guard.fig(12, 'attribution_<name>', ...)`.`

Enforcement section — replace the bullet `- **recording a gate before its artifacts exist** — ... The denial lists exactly what is missing.` with:

```markdown
- **recording a gate before its records are complete** — the denial lists exactly what is missing:
  - `Gate A`: `source.json` and a full-data profile (or `- Override: partial data - ...`); the schema-2
    profile; `data_dictionary.csv` with every column and every word field; every column on a
    `02_variables_pNN.png` page; `missingness.csv` with a cause for every column with gaps; a complete
    `label_audit.md`; one `01_*.png` and two `02_*.png`.
  - `Gate B`: a cleaning log (or `- Cleaning: none needed - reason: ...`); a decision for every column with
    gaps; `categorical_plan.csv` for every categorical feature; `associations.json` from the training split
    with its two `07_*_train.png` figures; a `04_*.png`; the `Model rationale:` block.
  - `Gate C`: a full-data training run in `training_runs.jsonl`; `importance.json` from the validation
    split; `12_permutation_importance.png` and one `12_attribution_*.png`.
  - `Gate D`: a `13_*.png` and the `Step 14 final test:` line.
  - Every PNG is larger than 1 KB and has a `- Figure <file>: <explanation>` line of at least 40
    characters with a number.
- **training on a slice** after Gate B — `X_train[:5000]`, `.sample(...)`, `.head(...)` in the fit call, or
  `nrows=` — unless the current run has `- Override: pilot - <size> - user approved YYYY-MM-DD - reason: ...`.
```

and in the red-flags bullet change `` `accuracy-imbalanced`, `resample-before-split` `` to `` `accuracy-imbalanced`, `resample-before-split`, `target-stats-outside-train` ``.

- [ ] **Step 4: Run everything**

Run: `python3 -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py && .venv/bin/python -m unittest tests/test_mlpipeline_guard.py tests/test_v05_end_to_end.py && python3 tests/probe_hook_audit.py | tail -1`
Expected: `OK`, `OK`, `0 of 23 cases behave wrongly`.

- [ ] **Step 5: Commit**

```bash
git add skills/ml-pipeline/SKILL.md hooks/session_start.py tests/test_session_start.py .github/workflows/tests.yml evals/understand-first/graders
git commit -m "docs(skill): v0.5 steps, gate records, pilot rule; CI runs the new suites; eval graders

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Release 0.5.0 (local)

**Files:**
- Modify: `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json` (and `.claude-plugin/marketplace.json` if it carries a version), `README.md`

- [ ] **Step 1: Bump the versions**

Set `"version": "0.5.0"` in both manifests. Run `grep -n '"version"' .claude-plugin/marketplace.json .agents/marketplace.json`; bump any `0.4.0` found there too.

- [ ] **Step 2: README**

After the paragraph that begins "It also refuses to let understanding be skipped.", add:

```markdown
v0.5 makes the data work complete and provable. `guard.load()` records each file's hash and row count, and
Gate A needs a profile of the whole file — a sample or an `nrows=` slice is stopped. Every column gets a
data-dictionary row, a place on a figure page, a written note that cites a real number from that column, and
— if it has gaps — a likely cause and a decision; the label gets an audit. Correlations and feature–target
strength are computed on the training split only, training runs are recorded so Gate C can prove one used
the whole training split, and permutation importance comes from the validation split after the baseline.
The harness checks completeness against the real columns and rows; you judge the quality at each gate.
Approvals never carry over between experiments: each starts with a `## Run: <name> (date)` heading.
```

- [ ] **Step 3: Verify everything**

Run: `python3 -m unittest tests/test_guard_training.py tests/test_post_tool_warn.py tests/test_session_start.py tests/test_mlpipeline_canary.py tests/test_gate_records.py && /usr/bin/python3 -m unittest tests/test_guard_training.py tests/test_gate_records.py && .venv/bin/python -m unittest tests/test_mlpipeline_guard.py tests/test_v05_end_to_end.py && python3 tests/probe_hook_audit.py | tail -1 && python3 -c "import json; [json.load(open(p)) for p in ('.claude-plugin/plugin.json', '.codex-plugin/plugin.json')]"`
Expected: `OK` ×3 (3.14, 3.9, 3.11), `0 of 23 cases behave wrongly`, manifests parse.

- [ ] **Step 4: Commit**

```bash
git add .claude-plugin .codex-plugin .agents README.md
git commit -m "chore: release 0.5.0

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Pushing, merging to `main`, and updating the installed plugin happen after the owner confirms (outside this plan).
