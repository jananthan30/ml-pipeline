# ml-pipeline v0.4.0 — Understand First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make data understanding mechanical — a data profile and figures before Gate A, a written model rationale before Gate B, and five provably-wrong model applications denied by the hook.

**Architecture:** `lib/mlpipeline_guard.py` gains `profile()` (writes `ml_pipeline/data_profile.json` with per-column facts, leakage suspects and a `traits` block), `eda_figures()` and `fig()` (PNGs plus a ledger line in `PIPELINE.md`). `hooks/guard_training.py` gains a gate-recording check (a `Gate X: approved` line is denied until that gate's artifacts exist) and red flags driven by the profile's traits. `hooks/post_tool_warn.py` gains three trait-aware warnings. A playbook `skills/ml-pipeline/references/model-selection.md` is read by the skill; the Gate B report must include a `Model rationale:` block.

**Tech Stack:** Python 3.10+ stdlib for hooks; pandas + numpy + matplotlib (lazy, `Agg`) for the guard; `claude plugin eval`.

**Spec:** `docs/superpowers/specs/2026-09-16-ml-pipeline-v0.4-understand-first-design.md`

## Global Constraints

- Hooks: Python ≥ 3.10, stdlib only, no `subprocess`/`eval`/`exec`, fail open.
- Guard: pandas + numpy required; matplotlib imported lazily with the `Agg` backend; `RuntimeError("matplotlib is required for figures: pip install matplotlib")` when absent. No text-figure fallback. Canary stays stdlib.
- Constants (exact): `SMALL_DATA_ROWS = 5000`, `IMBALANCE_MINORITY = 0.10`, `HIGH_CARD_LEVELS = 50`, `LEAK_CORR = 0.95`, `LEAK_PURITY = 0.98`, `MIN_PNG_BYTES = 1024`, `SAMPLE_ROWS = 200_000`.
- Gate lines `- Gate X: approved YYYY-MM-DD`; override lines `- Override: red flag <name> - …`; ledger lines `- Figure <file>.png: <explanation>`.
- Gate requirements: A = profile + ≥1 `01_*.png` + ≥2 `02_*.png`; B = ≥1 `04_*.png` + rationale block; C = ≥1 `12_*.png`; D = ≥1 `13_*.png` + `- Step 14 final test:` line. Every PNG > 1,024 bytes with a ledger line.
- HOL scanner ≥ 98/100, zero critical/high. Versions → `0.4.0` in both manifests (Task 8 only).
- Branch `v0.4-understand-first` (exists). Hook tests: `python3 -m unittest discover tests`. Guard tests: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py`.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe`.

## TODO (master checklist)

- [x] Task 1 — Guard: `profile()` → `data_profile.json`
- [x] Task 2 — Guard: `fig()` and `eda_figures()` with the ledger
- [x] Task 3 — Hook: gate-recording check (artifacts, ledger, rationale block)
- [x] Task 4 — Hook: red flags from profile traits
- [x] Task 5 — PostToolUse: trait-aware warnings
- [x] Task 6 — Playbook + skill / README / SessionStart / installer
- [x] Task 7 — Eval case `understand-first`
- [ ] Task 8 — Release 0.4.0

---

### Task 1: Guard — `profile()`

**Files:**
- Modify: `lib/mlpipeline_guard.py` (append)
- Test: `tests/test_mlpipeline_guard.py` (append)

**Interfaces:**
- Produces: `profile(df, *, target, time_col=None, group_col=None, state_dir="ml_pipeline") -> dict` writing `<state_dir>/data_profile.json`; constants `PROFILE_FILE = "data_profile.json"`, `SMALL_DATA_ROWS`, `IMBALANCE_MINORITY`, `HIGH_CARD_LEVELS`, `LEAK_CORR`, `LEAK_PURITY`, `SAMPLE_ROWS`. The dict has keys `schema_version, created, shape, target, columns, duplicates, constant_columns, high_cardinality_categoricals, datetime_columns, time_col, group_col, entity_candidates, leakage_suspects, traits` exactly as in spec §1. Tasks 4–5 read `traits`.

- [x] **Step 1: Write the failing tests** — append to `tests/test_mlpipeline_guard.py` before the `if __name__` block:

```python
class Profile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"

    def tearDown(self):
        self.tmp.cleanup()

    def _df(self, n=1000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        return pd.DataFrame({
            "age": rng.normal(50, 12, size=n),
            "zip": rng.integers(90000, 90400, size=n).astype(str),          # 400 levels -> high cardinality
            "flag": rng.integers(0, 2, size=n).astype(bool),
            "notes": ["x" * 40 + str(i) for i in range(n)],                 # long strings -> text
            "patient_id": np.arange(n),                                     # unique ints -> id
            "admitted_at": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y": y,
        })

    def test_column_kinds(self):
        p = guard.profile(self._df(), target="y", time_col="admitted_at", state_dir=self.state)
        kinds = {c: p["columns"][c]["kind"] for c in p["columns"]}
        self.assertEqual(kinds["age"], "numeric")
        self.assertEqual(kinds["zip"], "categorical")
        self.assertEqual(kinds["flag"], "bool")
        self.assertEqual(kinds["notes"], "text")
        self.assertEqual(kinds["patient_id"], "id")
        self.assertEqual(kinds["admitted_at"], "datetime")

    def test_shape_target_and_file(self):
        p = guard.profile(self._df(), target="y", time_col="admitted_at", state_dir=self.state)
        self.assertEqual(p["shape"]["n_rows"], 1000)
        self.assertEqual(p["target"]["task"], "binary")
        self.assertAlmostEqual(sum(p["target"]["class_balance"].values()), 1.0, places=3)
        self.assertIn("zip", p["high_cardinality_categoricals"])
        self.assertIn("patient_id", p["entity_candidates"])
        self.assertEqual(json.loads((self.state / guard.PROFILE_FILE).read_text())["shape"]["n_rows"], 1000)

    def test_missing_and_duplicates(self):
        df = self._df(200)
        df.loc[:19, "age"] = np.nan
        df = pd.concat([df, df.iloc[:50]], ignore_index=True)
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        self.assertAlmostEqual(p["columns"]["age"]["missing_frac"], 20 / 250, places=3)
        self.assertAlmostEqual(p["duplicates"]["exact_row_frac"], 50 / 250, places=3)

    def test_regression_and_multiclass_targets(self):
        df = self._df()
        df["amount"] = np.random.default_rng(1).normal(size=len(df))
        self.assertEqual(guard.profile(df, target="amount", time_col="admitted_at", state_dir=self.state)["target"]["task"], "regression")
        df["cls"] = np.random.default_rng(2).integers(0, 4, size=len(df))
        self.assertEqual(guard.profile(df, target="cls", time_col="admitted_at", state_dir=self.state)["target"]["task"], "multiclass")

    def test_leakage_suspects(self):
        df = self._df()
        df["risk_final"] = df["y"] + np.random.default_rng(3).normal(0, 0.01, size=len(df))   # |corr| > 0.95
        df["discharge_outcome"] = "a"                                                            # name rule
        df["y_copy_cat"] = df["y"].astype(str)                                                   # purity rule
        df["recorded_at"] = df["admitted_at"] + pd.Timedelta(days=3)                              # later than time_col
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        flagged = {s["column"] for s in p["leakage_suspects"]}
        self.assertTrue({"risk_final", "discharge_outcome", "y_copy_cat", "recorded_at"} <= flagged, flagged)
        self.assertNotIn("age", flagged)

    def test_traits(self):
        df = self._df()
        df["y"] = (np.arange(len(df)) % 20 == 0).astype(int)   # 5% minority
        p = guard.profile(df, target="y", time_col="admitted_at", state_dir=self.state)
        t = p["traits"]
        self.assertTrue(t["small_data"])
        self.assertTrue(t["imbalanced"])
        self.assertTrue(t["has_datetime"])
        self.assertFalse(t["has_groups"])              # entity candidates never set has_groups
        self.assertEqual(t["n_features"], 6)
        p2 = guard.profile(df, target="y", time_col="admitted_at", group_col="patient_id", state_dir=self.state)
        self.assertTrue(p2["traits"]["has_groups"])

    def test_sampling_flag(self):
        saved = guard.SAMPLE_ROWS
        guard.SAMPLE_ROWS = 300
        try:
            p = guard.profile(self._df(1000), target="y", time_col="admitted_at", state_dir=self.state)
        finally:
            guard.SAMPLE_ROWS = saved
        self.assertTrue(p["shape"]["sampled"])
        self.assertEqual(p["shape"]["sample_rows"], 300)
        self.assertEqual(p["shape"]["n_rows"], 1000)
```

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Profile 2>&1 | tail -3`
Expected: `AttributeError: module 'mlpipeline_guard' has no attribute 'profile'`.

- [x] **Step 3: Implement** — append to `lib/mlpipeline_guard.py`:

```python
# ------------------------------------------------------------------ profile (step 1)
PROFILE_FILE = "data_profile.json"
SMALL_DATA_ROWS = 5000
IMBALANCE_MINORITY = 0.10
HIGH_CARD_LEVELS = 50
LEAK_CORR = 0.95
LEAK_PURITY = 0.98
SAMPLE_ROWS = 200_000
_ID_NAME = re.compile(r"(^|_)id$", re.I)
_LEAK_NAME = re.compile(r"(outcome|result|after|label|target|final|actual)|_y$", re.I)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, 4)


def _integer_like(values: pd.Series) -> bool:
    values = values.dropna()
    if not len(values):
        return False
    if pd.api.types.is_integer_dtype(values):
        return True
    if pd.api.types.is_float_dtype(values):
        return bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))
    return False


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    n = len(series)
    unique_frac = series.nunique(dropna=True) / n if n else 0.0
    if pd.api.types.is_numeric_dtype(series):
        return "id" if unique_frac > 0.95 and _integer_like(series) else "numeric"
    if unique_frac > 0.95:
        return "id"
    lengths = series.dropna().astype(str).str.len()
    if len(lengths) and lengths.median() > 30:
        return "text"
    return "categorical"


def _column_stats(full: pd.Series, sample: pd.Series, kind: str) -> dict:
    n = len(full)
    n_unique = int(full.nunique(dropna=True))
    info = {"kind": kind, "missing_frac": round(float(full.isna().mean()), 4),
            "n_unique": n_unique, "unique_frac": round(n_unique / n, 4) if n else 0.0}
    values = sample.dropna()
    if not len(values):
        return info
    if kind == "numeric":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
        info.update(min=_num(values.min()), median=_num(values.median()), max=_num(values.max()),
                    skew=_num(values.skew()) if len(values) > 2 else 0.0, outlier_frac=round(float(outliers), 4))
    elif kind in ("categorical", "bool"):
        top = values.astype(str).value_counts(normalize=True).head(5)
        info.update(n_levels=int(values.nunique()), top={k: round(float(v), 4) for k, v in top.items()})
    elif kind == "datetime":
        info.update(start=str(values.min())[:10], end=str(values.max())[:10])
    elif kind == "text":
        info.update(median_length=int(values.astype(str).str.len().median()))
    return info


def _target_info(y: pd.Series) -> dict:
    values = y.dropna()
    n_unique = int(values.nunique())
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and n_unique > 20:
        return {"name": y.name, "task": "regression", "n_classes": None, "class_balance": None, "minority_frac": None}
    balance = values.astype(str).value_counts(normalize=True)
    return {"name": y.name, "task": "binary" if n_unique == 2 else "multiclass", "n_classes": n_unique,
            "class_balance": {k: round(float(v), 4) for k, v in balance.items()},
            "minority_frac": round(float(balance.min()), 4)}


def _leakage_suspects(df: pd.DataFrame, target: str, time_col, kinds: dict, task: str) -> list[dict]:
    y = df[target]
    y_num = y.astype(float) if task == "regression" else pd.Series(pd.factorize(y)[0], index=y.index).astype(float)
    suspects = []
    for col, kind in kinds.items():
        if col == target:
            continue
        if kind == "numeric":
            x = df[col].astype(float)
            if x.notna().sum() > 2 and x.std() > 0 and y_num.std() > 0:
                corr = abs(np.corrcoef(x.fillna(x.median()), y_num)[0, 1])
                if corr > LEAK_CORR:
                    suspects.append({"column": col, "reason": f"|corr| with target = {corr:.2f}"})
        elif kind in ("categorical", "bool") and task != "regression" and df[col].nunique() <= HIGH_CARD_LEVELS:
            table = pd.crosstab(df[col].astype(str), y.astype(str))
            purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
            if purity > LEAK_PURITY:
                suspects.append({"column": col, "reason": f"predicts target with {purity:.0%} purity"})
        elif kind == "datetime" and time_col and col != time_col:
            later = float((df[col] > df[time_col]).mean())
            if later > 0.5:
                suspects.append({"column": col, "reason": f"datetime later than {time_col} in {later:.0%} of rows"})
        if _LEAK_NAME.search(str(col)):
            suspects.append({"column": col, "reason": "name suggests post-outcome information"})
    return suspects


def _entity_candidates(df: pd.DataFrame, kinds: dict) -> list[str]:
    out = []
    for col, kind in kinds.items():
        if kind == "datetime":
            continue
        name = str(col)
        unique_frac = df[col].nunique(dropna=True) / len(df) if len(df) else 0.0
        if _ID_NAME.search(name) or (name.lower().endswith("id") and 0.01 < unique_frac < 0.95):
            out.append(col)
    return out


def profile(
    df: pd.DataFrame,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict:
    """Step 1: describe the data and write ml_pipeline/data_profile.json.

    Everything later in the pipeline — figures, the model playbook, the hook's red flags — reads
    this file instead of re-deriving facts from the data.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    n = len(df)
    sampled = n > SAMPLE_ROWS
    sample = df.sample(SAMPLE_ROWS, random_state=0) if sampled else df
    kinds = {col: _column_kind(df[col]) for col in df.columns}
    target_info = _target_info(df[target])
    columns = {str(col): _column_stats(df[col], sample[col], kinds[col]) for col in df.columns}
    high_card = [str(c) for c, k in kinds.items() if k == "categorical" and columns[str(c)].get("n_levels", 0) > HIGH_CARD_LEVELS]
    datetime_cols = [str(c) for c, k in kinds.items() if k == "datetime"]
    minority = target_info["minority_frac"]
    result = {
        "schema_version": "1",
        "created": date.today().isoformat(),
        "shape": {"n_rows": int(n), "n_cols": int(df.shape[1]),
                  "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
                  "sampled": bool(sampled), "sample_rows": SAMPLE_ROWS if sampled else None},
        "target": target_info,
        "columns": columns,
        "duplicates": {"exact_row_frac": round(float(df.duplicated().mean()), 4) if n else 0.0},
        "constant_columns": [str(c) for c in df.columns if columns[str(c)]["n_unique"] <= 1],
        "high_cardinality_categoricals": high_card,
        "datetime_columns": datetime_cols,
        "time_col": time_col,
        "group_col": group_col,
        "entity_candidates": [str(c) for c in _entity_candidates(df, kinds)],
        "leakage_suspects": _leakage_suspects(sample, target, time_col, kinds, target_info["task"]),
        "traits": {
            "n_rows": int(n), "n_features": int(df.shape[1] - 1), "task": target_info["task"],
            "minority_frac": minority,
            "has_datetime": bool(datetime_cols) or time_col is not None,
            "has_groups": group_col is not None,
            "small_data": n < SMALL_DATA_ROWS,
            "imbalanced": target_info["task"] != "regression" and (minority if minority is not None else 1.0) < IMBALANCE_MINORITY,
            "high_card_categoricals": high_card,
        },
    }
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / PROFILE_FILE).write_text(json.dumps(result, indent=2, default=str))
    return result
```

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py -v 2>&1 | tail -4`
Expected: `OK`, 20 tests.

- [x] **Step 5: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py
git commit -m "feat(guard): profile() writes data_profile.json with leakage suspects and traits

Column kinds, missingness, duplicates, target task and balance, high-
cardinality categoricals, entity candidates, and four leakage-suspect
rules (correlation, purity, post-outcome datetime, name). The traits block
is what the playbook and the hook's red flags read.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 2: Guard — `fig()` and `eda_figures()`

**Files:**
- Modify: `lib/mlpipeline_guard.py` (append)
- Modify: `tests/requirements-dev.txt` (add `matplotlib`)
- Test: `tests/test_mlpipeline_guard.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1 (independent of the profile).
- Produces: `fig(step: int, name: str, figure, explanation: str, state_dir="ml_pipeline") -> Path`; `eda_figures(df, *, target, time_col=None, state_dir="ml_pipeline") -> list[Path]`; `FIGURES_DIR = "figures"`; ledger lines `- Figure <file>: <explanation>` appended to `<state_dir>/PIPELINE.md`. File names `NN_<slug>.png`.

- [x] **Step 1: Dev env**

```bash
uv pip install --python .venv/bin/python matplotlib
printf 'pandas\nnumpy\nmatplotlib\n' > tests/requirements-dev.txt
```

- [x] **Step 2: Write the failing tests** — append to `tests/test_mlpipeline_guard.py`:

```python
class Figures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "ml_pipeline"
        self.md = self.state / "PIPELINE.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_eda_figures_writes_required_set_with_ledger(self):
        paths = guard.eda_figures(frame(dates=True), target="y", time_col="date", state_dir=self.state)
        names = sorted(p.name for p in paths)
        self.assertEqual(names, ["01_missingness.png", "02_correlations.png", "02_distributions.png",
                                 "02_target_balance.png", "02_temporal_coverage.png"])
        for p in paths:
            self.assertGreater(p.stat().st_size, 1024, p.name)
            self.assertIn(f"- Figure {p.name}: ", self.md.read_text())

    def test_no_temporal_figure_without_dates(self):
        names = {p.name for p in guard.eda_figures(frame(), target="y", state_dir=self.state)}
        self.assertNotIn("02_temporal_coverage.png", names)

    def test_fig_saves_and_ledgers(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = guard.fig(12, "ROC curve", f, "AUC 0.81 on validation, baseline 0.50.", state_dir=self.state)
        self.assertEqual(path.name, "12_roc_curve.png")
        self.assertGreater(path.stat().st_size, 1024)
        self.assertIn("- Figure 12_roc_curve.png: AUC 0.81", self.md.read_text())

    def test_fig_requires_explanation(self):
        import matplotlib.pyplot as plt
        with self.assertRaises(ValueError):
            guard.fig(4, "x", plt.figure(), "   ", state_dir=self.state)

    def test_missing_matplotlib_is_a_clear_error(self):
        import sys
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "matplotlib" or k.startswith("matplotlib.")}
        sys.modules["matplotlib"] = None  # makes `import matplotlib` raise ImportError
        try:
            with self.assertRaises(RuntimeError) as ctx:
                guard.eda_figures(frame(), target="y", state_dir=self.state)
            self.assertIn("pip install matplotlib", str(ctx.exception))
        finally:
            del sys.modules["matplotlib"]
            sys.modules.update(saved)
```

- [x] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_mlpipeline_guard.Figures 2>&1 | tail -3`
Expected: `AttributeError: module 'mlpipeline_guard' has no attribute 'eda_figures'`.

- [x] **Step 4: Implement** — append to `lib/mlpipeline_guard.py`:

```python
# ------------------------------------------------------------------ figures (steps 1-2 and any step)
FIGURES_DIR = "figures"


def _plt():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for figures: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ledger(pipeline_md: Path, filename: str, explanation: str) -> None:
    line = f"- Figure {filename}: {explanation.strip()}\n"
    pipeline_md.parent.mkdir(parents=True, exist_ok=True)
    existing = pipeline_md.read_text().rstrip("\n") + "\n" if pipeline_md.is_file() else ""
    if line not in existing:
        pipeline_md.write_text(existing + line)


def fig(step: int, name: str, figure, explanation: str, state_dir: str | Path = "ml_pipeline") -> Path:
    """Save a matplotlib figure as ml_pipeline/figures/NN_<slug>.png and record its explanation."""
    if not explanation or not explanation.strip():
        raise ValueError("every figure needs an explanation: what it shows and why it matters")
    plt = _plt()
    figure = getattr(figure, "figure", figure)  # accept an Axes too
    state_dir = Path(state_dir)
    out_dir = state_dir / FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    path = out_dir / f"{int(step):02d}_{slug}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    _ledger(state_dir / "PIPELINE.md", path.name, explanation)
    return path


def eda_figures(df: pd.DataFrame, *, target: str, time_col: str | None = None,
                state_dir: str | Path = "ml_pipeline") -> list[Path]:
    """The required step 1-2 figures, each with an explanation computed from the data."""
    plt = _plt()
    paths: list[Path] = []
    n = len(df)

    missing = df.isna().mean().sort_values(ascending=False)
    f, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(missing))))
    ax.barh(missing.index.astype(str)[::-1], missing.to_numpy()[::-1])
    ax.set_xlabel("missing fraction")
    ax.set_title("Missing values per column")
    n_missing = int((missing > 0).sum())
    paths.append(fig(1, "missingness", f,
                     f"{n_missing} of {len(missing)} columns have missing values; worst is {missing.index[0]} "
                     f"at {missing.iloc[0]:.1%}." if n_missing else "No column has missing values.", state_dir))

    y = df[target]
    f, ax = plt.subplots(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 20:
        ax.hist(y.dropna(), bins=30)
        ax.set_title(f"Target {target} distribution")
        explanation = (f"{target} ranges {y.min():.3g} to {y.max():.3g} with median {y.median():.3g}; "
                       "a mean-predictor baseline is the number to beat.")
    else:
        balance = y.astype(str).value_counts(normalize=True)
        ax.bar(balance.index, balance.to_numpy())
        ax.set_title(f"Target {target} balance")
        explanation = (f"{target}={balance.index[0]} is {balance.iloc[0]:.1%} of {n:,} rows, so a majority-class "
                       f"dummy scores {balance.iloc[0]:.1%}; the minority share is {balance.min():.1%}.")
    paths.append(fig(2, "target_balance", f, explanation, state_dir))

    numeric = df.select_dtypes("number").drop(columns=[target], errors="ignore").iloc[:, :12]
    if numeric.shape[1]:
        cols = numeric.shape[1]
        rows = int(np.ceil(cols / 4))
        f, axes = plt.subplots(rows, 4, figsize=(12, 2.6 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, numeric.columns):
            ax.hist(numeric[col].dropna(), bins=30)
            ax.set_title(str(col), fontsize=9)
        for ax in axes[cols:]:
            ax.axis("off")
        skewed = [str(c) for c in numeric.columns if abs(float(numeric[c].skew())) > 1]
        paths.append(fig(2, "distributions", f,
                         f"Histograms of {cols} numeric features; {len(skewed)} are strongly skewed "
                         f"({', '.join(skewed[:5]) or 'none'}), which matters for scaling and outliers.", state_dir))

    with_target = df.select_dtypes("number").iloc[:, :30]
    if target in df.columns and target not in with_target.columns:
        with_target = with_target.assign(**{target: pd.factorize(df[target])[0]})
    if with_target.shape[1] >= 2:
        corr = with_target.corr()
        f, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels([str(c) for c in corr.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels([str(c) for c in corr.columns], fontsize=7)
        f.colorbar(image)
        ax.set_title("Correlations")
        ranked = corr[target].drop(target).abs().sort_values(ascending=False) if target in corr.columns else pd.Series(dtype=float)
        explanation = (f"Strongest correlation with {target}: {ranked.index[0]} ({ranked.iloc[0]:.2f}); anything "
                       "above 0.95 is a leakage suspect." if len(ranked) else "Correlations between numeric features.")
        paths.append(fig(2, "correlations", f, explanation, state_dir))

    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    tcol = time_col or (datetime_cols[0] if datetime_cols else None)
    if tcol is not None:
        counts = df[tcol].dt.to_period("M").value_counts().sort_index()
        f, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot([str(p) for p in counts.index], counts.to_numpy())
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.set_title(f"Rows per month over {tcol}")
        paths.append(fig(2, "temporal_coverage", f,
                         f"{tcol} spans {str(df[tcol].min())[:10]} to {str(df[tcol].max())[:10]} over {len(counts)} months; "
                         "a chronological split must hold out the latest period.", state_dir))
    return paths
```

- [x] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m unittest tests/test_mlpipeline_guard.py -v 2>&1 | tail -4`
Expected: `OK`, 25 tests.

- [x] **Step 6: Commit**

```bash
git add lib/mlpipeline_guard.py tests/test_mlpipeline_guard.py tests/requirements-dev.txt
git commit -m "feat(guard): eda_figures() and fig() write PNGs and a ledger line every time

The required step 1-2 figures are generated from the data with factual
explanations; fig() saves any other figure. Both append a '- Figure <file>:'
line to PIPELINE.md. matplotlib is imported lazily (Agg); when it is missing
the error says to install it - there is no text-figure fallback.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 3: Hook — gate-recording check

**Files:**
- Modify: `hooks/guard_training.py`
- Test: `tests/test_guard_training.py`

**Interfaces:**
- Consumes: existing `GATE_LINE`, `APPROVED`, `NEGATED`, `_strings`, `find_pipeline`, `_emit`, `DOC_SUFFIXES`.
- Produces: `approved_gates_in(text) -> list[str]`; `rationale_missing(text) -> list[str]`; `gate_problems(gate, state_dir: Path, ledger_text) -> list[str]`; `_state_dir_for(tool_input, cwd) -> Path`; constants `PROFILE_FILE`, `FIGURES_DIR`, `MIN_PNG_BYTES`, `GATE_REQUIREMENTS`, `GATE_RECORD_MSG`. Task 4 and Task 5 reuse `_state_dir_for`.

- [x] **Step 1: Extend the test helper and write the failing tests** — in `tests/test_guard_training.py` replace the `run` function with this version (adds `files`, a mapping of relative path → bytes or str, created under the temp project):

```python
def run(tool_input, pipeline_md=None, tool="Bash", env=None, cwd_sub="", files=None):
    """Run the hook against a temp project. Returns (decision, reason)."""
    with tempfile.TemporaryDirectory() as tmp:
        if pipeline_md is not None:
            (Path(tmp) / "ml_pipeline").mkdir(exist_ok=True)
            (Path(tmp) / "ml_pipeline" / "PIPELINE.md").write_text(pipeline_md)
        for rel, data in (files or {}).items():
            path = Path(tmp) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
        cwd = Path(tmp) / cwd_sub
        cwd.mkdir(parents=True, exist_ok=True)
        if "file_path" in tool_input and str(tool_input["file_path"]).startswith("./"):
            tool_input = {**tool_input, "file_path": str(Path(tmp) / tool_input["file_path"][2:])}
        payload = json.dumps({"tool_name": tool, "tool_input": tool_input, "cwd": str(cwd)})
        saved = dict(os.environ)
        os.environ.pop("ML_PIPELINE_ENFORCE", None)
        os.environ.pop("ML_PIPELINE_FILE", None)
        os.environ.update(env or {})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), _stdin(payload):
                hook.main()
        finally:
            os.environ.clear()
            os.environ.update(saved)
    result = json.loads(out.getvalue() or "{}")
    specific = result.get("hookSpecificOutput", {})
    return specific.get("permissionDecision", "allow"), specific.get("permissionDecisionReason", "")
```

Then append these tests (module level constants first):

```python
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2000          # > 1,024 bytes, fine for the hook's size check
TINY = b"\x89PNG\r\n\x1a\n" + b"\0" * 50
LEDGER_A = ("- Figure 01_missingness.png: none missing.\n"
            "- Figure 02_target_balance.png: 51% vs 49%.\n"
            "- Figure 02_distributions.png: two skewed.\n")
ARTIFACTS_A = {"ml_pipeline/data_profile.json": "{}", "ml_pipeline/figures/01_missingness.png": PNG,
               "ml_pipeline/figures/02_target_balance.png": PNG, "ml_pipeline/figures/02_distributions.png": PNG}
RATIONALE = ("Model rationale:\n- traits: binary, 1,000 rows (small)\n- baseline: dummy + logistic\n"
             "- candidates: boosting\n- ruled out: neural nets (small data)\n- metric: PR-AUC\n")


def write_md(content):
    return {"file_path": "./ml_pipeline/PIPELINE.md", "content": content}


class GateRecording(unittest.TestCase):
    """A gate line is denied until that gate's artifacts exist on disk."""

    def test_gate_a_denied_without_anything(self):
        decision, reason = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write")
        self.assertEqual(decision, "deny")
        self.assertIn("data_profile.json missing", reason)
        self.assertIn("figures/01_*.png 0 found (need 1)", reason)

    def test_gate_a_allowed_when_complete(self):
        decision, _ = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=ARTIFACTS_A)
        self.assertEqual(decision, "allow")

    def test_png_without_ledger_line_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/02_extra.png": PNG}
        decision, reason = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("02_extra.png has no '- Figure 02_extra.png:' line", reason)

    def test_tiny_png_denied(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/01_missingness.png": TINY}
        decision, reason = run(write_md(LEDGER_A + "- Gate A: approved 2026-09-16\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("(empty?)", reason)

    def test_ledger_in_existing_file_counts(self):
        decision, _ = run(write_md("- Gate A: approved 2026-09-16\n"), pipeline_md=LEDGER_A, tool="Write", files=ARTIFACTS_A)
        self.assertEqual(decision, "allow")

    def test_bash_append_is_checked_too(self):
        cmd = {"command": "echo '- Gate A: approved 2026-09-16' >> ml_pipeline/PIPELINE.md"}
        self.assertEqual(run(cmd, pipeline_md="")[0], "deny")

    def test_negated_line_is_not_a_recording(self):
        self.assertEqual(run(write_md("- Gate A: not yet approved\n"), pipeline_md="", tool="Write")[0], "allow")

    def test_gate_b_needs_rationale_and_figure(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/04_cleaning.png": PNG}
        ledger = LEDGER_A + "- Figure 04_cleaning.png: 312 rows dropped.\n"
        decision, reason = run(write_md(ledger + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("Model rationale block missing lines: traits, baseline, candidates, ruled out, metric", reason)
        partial = "Model rationale:\n- traits: x\n- baseline: y\n"
        decision, reason = run(write_md(ledger + partial + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertIn("missing lines: candidates, ruled out, metric", reason)
        decision, _ = run(write_md(ledger + RATIONALE + "- Gate B: approved 2026-09-17\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "allow")

    def test_gate_c_and_d(self):
        files = {**ARTIFACTS_A, "ml_pipeline/figures/12_roc.png": PNG, "ml_pipeline/figures/13_errors.png": PNG}
        ledger = LEDGER_A + "- Figure 12_roc.png: AUC 0.8.\n- Figure 13_errors.png: worst slice.\n"
        self.assertEqual(run(write_md(ledger + "- Gate C: approved 2026-09-18\n"), pipeline_md="", tool="Write", files=files)[0], "allow")
        decision, reason = run(write_md(ledger + "- Gate D: approved 2026-09-19\n"), pipeline_md="", tool="Write", files=files)
        self.assertEqual(decision, "deny")
        self.assertIn("no '- Step 14 final test:' line", reason)
        ledger += "- Step 14 final test: acc=0.79 (touch 1, 2026-09-19)\n"
        self.assertEqual(run(write_md(ledger + "- Gate D: approved 2026-09-19\n"), pipeline_md="", tool="Write", files=files)[0], "allow")
```

- [x] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests.test_guard_training.GateRecording 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `FAILED` — the deny cases currently return "allow" (`.md` files are skipped).

- [x] **Step 3: Implement** — in `hooks/guard_training.py` add after `TEST_MSG`:

```python
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
    """Gate letters that ``text`` records as approved (non-negated lines only)."""
    gates = []
    for match in GATE_LINE.finditer(text):
        line = match.group(0)
        if APPROVED.search(line) and not NEGATED.search(line):
            gates.append(match.group(1).upper())
    return gates


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
```

Then replace the body of `main()` between `payload = json.load(sys.stdin)` and `found = analyse(text)` with:

```python
        payload = json.load(sys.stdin)
        tool_input = payload.get("tool_input") or {}
        cwd = payload.get("cwd") or os.getcwd()
        text = "\n".join(_strings(tool_input))
        # 1. Recording a gate: that gate's artifacts must already exist.
        for gate in approved_gates_in(text):
            state_dir = _state_dir_for(tool_input, cwd)
            pipeline_md = state_dir / "PIPELINE.md"
            existing = pipeline_md.read_text(encoding="utf-8", errors="replace") if pipeline_md.is_file() else ""
            problems = gate_problems(gate, state_dir, existing + "\n" + text)
            if problems:
                return _emit(GATE_RECORD_MSG.format(gate, "; ".join(problems)))
        # 2. Prose files are not code.
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if target and Path(target).suffix.lower() in DOC_SUFFIXES:
            return _emit(None)
        # 3. Fast path: nothing to look at.
        if ".fit" not in text and not TRAIN_API.search(text) and not EVAL_CALL.search(text):
            return _emit(None)
```

(The old lines that computed `target`, skipped docs, then computed `text` are replaced by this block; `found = analyse(text)` and everything after stay as they are.)

- [x] **Step 4: Run the whole hook suite**

Run: `python3 -m unittest discover tests 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `OK`, 58 tests (49 + 9), 1 skipped.

- [x] **Step 5: Commit**

```bash
git add hooks/guard_training.py tests/test_guard_training.py
git commit -m "feat(hook): a gate can't be recorded until its artifacts exist

Any tool input that adds a 'Gate X: approved' line is checked against that
gate's requirements: the data profile and step 1-2 figures for A, a cleaning
figure and the Model rationale block for B, an evaluation figure for C, an
error-analysis figure and the final-test line for D. Every PNG must be
larger than 1 KB and have a '- Figure <file>:' ledger line. The denial
lists exactly what is missing.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 4: Hook — red flags from profile traits

**Files:**
- Modify: `hooks/guard_training.py`
- Test: `tests/test_guard_training.py`

**Interfaces:**
- Consumes: `_state_dir_for`, `PROFILE_FILE`, `NEGATED`, `TRAIN_API`, `COMMENT` from earlier tasks/this module.
- Produces: `_load_profile(state_dir: Path) -> dict | None`; `overridden_flags(markdown) -> set[str]`; `red_flag(code, traits, overrides) -> str | None`; `RED_FLAGS`, `RED_FLAG_MSG`. Task 5 reuses `_load_profile`.

- [x] **Step 1: Write the failing tests** — append to `tests/test_guard_training.py`:

```python
def profile_with(**traits):
    base = {"n_rows": 1000, "n_features": 5, "task": "binary", "minority_frac": 0.4, "has_datetime": False,
            "has_groups": False, "small_data": True, "imbalanced": False, "high_card_categoricals": []}
    base.update(traits)
    return {"ml_pipeline/data_profile.json": json.dumps({"traits": base})}


class RedFlags(unittest.TestCase):
    """Provably wrong model applications are denied; each has an override line."""

    def test_neural_net_on_small_data(self):
        decision, reason = run({"command": "clf = MLPClassifier()\nclf.fit(X_train, y_train)"}, GATE_AB, files=profile_with(small_data=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag neural-net-small-data", reason)

    def test_neural_net_without_training_call_is_fine(self):
        self.assertEqual(run({"command": "from sklearn.neural_network import MLPClassifier"}, GATE_AB, files=profile_with())[0], "allow")

    def test_random_split_on_temporal_data(self):
        decision, reason = run({"command": "train_test_split(X, y, test_size=0.2)"}, GATE_A, files=profile_with(has_datetime=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag random-split-temporal", reason)
        self.assertEqual(run({"command": "TimeSeriesSplit(n_splits=5)"}, GATE_A, files=profile_with(has_datetime=True))[0], "allow")

    def test_group_split(self):
        decision, reason = run({"command": "KFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag group-split", reason)
        self.assertEqual(run({"command": "GroupKFold(n_splits=5)"}, GATE_A, files=profile_with(has_groups=True))[0], "allow")

    def test_accuracy_selection_on_imbalanced(self):
        decision, reason = run({"command": "GridSearchCV(clf, grid, scoring='accuracy')"}, GATE_AB, files=profile_with(imbalanced=True, minority_frac=0.05))
        self.assertEqual(decision, "deny")
        self.assertIn("red flag accuracy-imbalanced", reason)
        self.assertEqual(run({"command": "GridSearchCV(clf, grid, scoring='average_precision')"}, GATE_AB, files=profile_with(imbalanced=True))[0], "allow")

    def test_resample_before_split(self):
        decision, reason = run({"command": "X_res, y_res = SMOTE().fit_resample(X, y)"}, GATE_AB, files=profile_with())
        self.assertEqual(decision, "deny")
        self.assertIn("red flag resample-before-split", reason)
        self.assertEqual(run({"command": "SMOTE().fit_resample(X_train, y_train)"}, GATE_AB, files=profile_with())[0], "allow")

    def test_override_lifts_one_flag(self):
        md = GATE_A + "- Override: red flag random-split-temporal - user approved a random split 2026-09-16 - reason: signup date only\n"
        self.assertEqual(run({"command": "train_test_split(X, y)"}, md, files=profile_with(has_datetime=True))[0], "allow")
        self.assertEqual(run({"command": "KFold(5)"}, md, files=profile_with(has_datetime=True, has_groups=True))[0], "deny")

    def test_no_profile_no_flags(self):
        self.assertEqual(run({"command": "train_test_split(X, y)"}, GATE_A)[0], "allow")
```

- [x] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests.test_guard_training.RedFlags 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `FAILED` (deny cases return allow).

- [x] **Step 3: Implement** — add to `hooks/guard_training.py` after `_state_dir_for`:

```python
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
OVERRIDE_FLAG = re.compile(r"^.*\boverrid(?:e|den)\b.*\bred flag\s+([a-z-]+)\b.*$", re.I | re.M)
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
    return {m.group(1).lower() for m in OVERRIDE_FLAG.finditer(markdown) if not NEGATED.search(m.group(0))}


def red_flag(code: str, traits: dict, overrides: set[str]) -> str | None:
    """The first red flag the code trips given the profile's traits, or None."""
    values = {"n_rows": int(traits.get("n_rows") or 0),
              "minority_pct": f"{float(traits.get('minority_frac') or 0):.1%}"}
    trains = ".fit" in code or bool(TRAIN_API.search(code))
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
```

Then in `main()`, insert between step 2 (the prose skip) and step 3 (the fast path):

```python
        # 2b. Red flags: wrong applications the data profile can prove.
        state_dir = _state_dir_for(tool_input, cwd)
        profile = _load_profile(state_dir)
        if profile is not None:
            pipeline_md = state_dir / "PIPELINE.md"
            markdown = pipeline_md.read_text(encoding="utf-8", errors="replace") if pipeline_md.is_file() else ""
            reason = red_flag(COMMENT.sub("", text), profile.get("traits") or {}, overridden_flags(markdown))
            if reason:
                return _emit(reason)
```

- [x] **Step 4: Run the whole hook suite**

Run: `python3 -m unittest discover tests 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `OK`, 66 tests, 1 skipped.

- [x] **Step 5: Commit**

```bash
git add hooks/guard_training.py tests/test_guard_training.py
git commit -m "feat(hook): deny provably wrong model applications from the data profile

Five red flags read data_profile.json traits: neural nets on small tables,
random splits or k-fold on temporal data, non-group splits on grouped data,
accuracy as the selection metric on an imbalanced target, and resampling
before the split. Each names an 'Override: red flag <name>' line.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 5: PostToolUse — trait-aware warnings

**Files:**
- Modify: `hooks/post_tool_warn.py`
- Test: `tests/test_post_tool_warn.py`

**Interfaces:**
- Consumes: `_strings`, `_state_dir_for`, `_load_profile` from `hooks/guard_training.py`.
- Produces: `trait_warnings(code, traits) -> list[str]`; `warnings(code, output, traits=None)` (extra optional parameter, existing behaviour unchanged).

- [x] **Step 1: Write the failing tests** — append to `tests/test_post_tool_warn.py`:

```python
class TraitWarnings(unittest.TestCase):
    def test_slow_models_on_large_n(self):
        out = hook.trait_warnings("KNeighborsClassifier().fit(X_train, y_train)", {"n_rows": 120000})
        self.assertTrue(any("scales badly" in w for w in out))
        self.assertEqual(hook.trait_warnings("KNeighborsClassifier()", {"n_rows": 1000}), [])

    def test_one_hot_on_high_cardinality(self):
        out = hook.trait_warnings("pd.get_dummies(df)", {"high_card_categoricals": ["zip"]})
        self.assertTrue(any("zip" in w for w in out))

    def test_accuracy_reported_on_imbalanced(self):
        out = hook.trait_warnings("print(accuracy_score(y_val, pred))", {"imbalanced": True})
        self.assertTrue(any("imbalanced" in w for w in out))
        self.assertEqual(hook.trait_warnings("print(accuracy_score(y_val, pred))", {"imbalanced": False}), [])

    def test_main_reads_profile_from_cwd(self):
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ml_pipeline").mkdir()
            (Path(tmp) / "ml_pipeline" / "data_profile.json").write_text(json.dumps({"traits": {"imbalanced": True}}))
            payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "accuracy_score(y_val, p)"},
                                  "tool_response": "0.93", "cwd": tmp})
            out = io.StringIO()
            with contextlib.redirect_stdout(out), _stdin(payload):
                hook.main()
        self.assertIn("imbalanced", json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"])
```

- [x] **Step 2: Run to verify they fail**

Run: `python3 -m unittest tests.test_post_tool_warn.TraitWarnings 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `FAILED` with `AttributeError: ... has no attribute 'trait_warnings'`.

- [x] **Step 3: Implement** — in `hooks/post_tool_warn.py` change the import line to
`from guard_training import _load_profile, _state_dir_for, _strings  # noqa: E402`, add after `DATEY`:

```python
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
```

change the signature of `warnings` to `def warnings(code: str, output: str, traits: dict | None = None) -> list[str]:` and add as its last line before `return found`: `found.extend(trait_warnings(code, traits or {}))`; and in `main()` replace `found = warnings(code, output)` with:

```python
        profile = _load_profile(_state_dir_for(payload.get("tool_input") or {}, payload.get("cwd") or os.getcwd()))
        found = warnings(code, output, (profile or {}).get("traits"))
```

- [x] **Step 4: Run**

Run: `python3 -m unittest discover tests 2>&1 | grep -E "^(Ran|OK|FAILED)"`
Expected: `OK`, 70 tests, 1 skipped.

- [x] **Step 5: Commit**

```bash
git add hooks/post_tool_warn.py tests/test_post_tool_warn.py
git commit -m "feat(hook): PostToolUse warnings that read the data profile

k-NN / kernel SVM above 50k rows, one-hot on high-cardinality columns, and
accuracy reported on an imbalanced target.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 6: Playbook + skill / README / SessionStart / installer

**Files:**
- Create: `skills/ml-pipeline/references/model-selection.md`
- Modify: `skills/ml-pipeline/SKILL.md`
- Modify: `README.md`
- Modify: `hooks/session_start.py`
- Modify: `install-other-tools.sh`
- Test: `tests/test_session_start.py`

**Interfaces:**
- Produces: the playbook path `skills/ml-pipeline/references/model-selection.md`; SessionStart context now names it.

- [x] **Step 1: Write the failing test** — in `tests/test_session_start.py`, add to `test_names_the_helpers_when_plugin_root_is_set`:

```python
        self.assertIn("/plug/skills/ml-pipeline/references/model-selection.md", context)
```

Run: `python3 -m unittest tests/test_session_start.py 2>&1 | grep -E "^(Ran|OK|FAILED)"` → `FAILED`.

- [x] **Step 2: Update `hooks/session_start.py`** — replace the `context = (...)` expression with:

```python
    context = (
        "ml-pipeline runtime helpers: guard library at "
        f"{os.path.join(lib, 'mlpipeline_guard.py')} - copy it to ml_pipeline/guard.py at step 1 (profile) and use "
        "guard.profile() / guard.eda_figures() / guard.fig() / guard.split() / guard.final_test(); model-selection "
        f"playbook at {os.path.join(root, 'skills', 'ml-pipeline', 'references', 'model-selection.md')} (read at step 3 "
        f"and before Gate B); canary dataset generator at {os.path.join(lib, 'mlpipeline_canary.py')} "
        "(honest accuracy ceiling 0.80; above it means leakage)."
    )
```

Run the test again → `OK`.

- [x] **Step 3: Write the playbook** — create `skills/ml-pipeline/references/model-selection.md`:

```markdown
# Model selection playbook

Read at step 3 (when writing the prediction contract) and again before the Gate B report. Key on
`ml_pipeline/data_profile.json → traits`. Always start with a baseline (a dummy predictor and one
simple model), judged on validation with the contract's metric. "Avoid" means: not without a
written reason in the `Model rationale:` block.

## Tabular, small (`small_data`: fewer than 5,000 rows)
- Baseline: majority-class / mean dummy, then logistic or linear regression with L2.
- Strong default: gradient boosting with shallow trees (depth 2–4, strong regularization, early
  stopping) or the regularized linear model itself.
- Worth trying: random forest (few knobs); GAM / EBM when the model must be read by a person.
- Avoid: neural nets (overfit and hide it — the hook denies this); large hyperparameter searches
  (use repeated k-fold on train and report the spread); anything you cannot explain to the user.

## Tabular, medium (5,000 – 500,000 rows)
- Baseline as above. Strong default: LightGBM / XGBoost / CatBoost with early stopping on validation.
- Worth trying: a linear model with feature crosses; an MLP only if boosting plateaus and rows > 50,000.
- Avoid: k-NN and RBF-SVM above ~50,000 rows (quadratic time); unregularized trees.

## Tabular, large (more than 500,000 rows)
- Strong default: histogram gradient boosting (`HistGradientBoosting*`, LightGBM) with subsampling;
  linear models on hashed or sparse features when the table is very wide.
- Avoid: anything O(n²); exact k-NN; kernel SVMs.

## Imbalanced target (`imbalanced`: minority class under 10%)
- Metric: average precision (PR-AUC) primary; recall at a fixed precision or F1 at a chosen
  threshold; never accuracy for model selection (the hook denies `scoring="accuracy"`).
- First try `class_weight="balanced"` / `scale_pos_weight`; if you resample, `fit_resample` on the
  training split only (the hook denies anything else).
- The majority-class dummy baseline shows exactly what accuracy hides.

## High-cardinality categoricals (`high_card_categoricals`: more than 50 levels)
- Prefer CatBoost, or target / ordinal encoding fit on train inside a pipeline object.
- Avoid one-hot encoding (width explosion, and leakage if fit on all rows).

## Temporal data (`has_datetime`)
- Split: chronological — `guard.split(time_col=...)`; validation is the period after train, test the
  latest period; `TimeSeriesSplit` for CV. Random splits and k-fold are denied by the hook.
- Features: lags and rolling statistics computed only from the past; calendar features.
- Baseline: last value / seasonal naive for forecasting; logistic on lag features for temporal
  classification. Strong default: gradient boosting on lag features; ETS / ARIMA for univariate series.
- Avoid: tree models extrapolating a trend (detrend first); LSTMs before boosting has been tried.

## Repeated entities (`has_groups`: patients, users, devices, sites)
- Split by group — `guard.split(group_col=...)`, `GroupKFold` for CV. Row-level splits are denied.
- Avoid: features that identify the entity; the model memorizes it instead of learning.

## Text
- Baseline: TF-IDF (1–2 grams) + logistic regression or linear SVM.
- Strong default: a fine-tuned small transformer only when the baseline is clearly insufficient and
  there are more than ~5,000 labelled examples. Avoid training embeddings from scratch on small corpora.

## Images
- Baseline: features from a pretrained CNN + logistic regression. Strong default: fine-tune a
  pretrained backbone with augmentation applied after the split.
- Avoid: training from scratch below ~100,000 images; augmenting before splitting; random splits when
  images share a source (patient, device, session → group split).

## Interpretability required
- Logistic / linear with monotone constraints, GAM / EBM, shallow trees; gradient boosting with
  monotone constraints plus SHAP for explanation. Avoid black-box ensembles where a clinician or
  regulator must read the model.

## Wrong applications the hook denies (red flags) and their override names

| red flag | what fires it | do instead |
|---|---|---|
| `neural-net-small-data` | MLP / keras / torch training with fewer than 5,000 rows | regularized linear, gradient boosting |
| `random-split-temporal` | `train_test_split` / `KFold` with a datetime column | `guard.split(time_col=)`, `TimeSeriesSplit` |
| `group-split` | any non-Group split with `group_col` set | `guard.split(group_col=)`, `GroupKFold` |
| `accuracy-imbalanced` | `scoring="accuracy"` with minority under 10% | `average_precision`, `f1`, `balanced_accuracy` |
| `resample-before-split` | `fit_resample` on anything but the train split | split first, then resample train |

Override one with `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>` in
`PIPELINE.md`, only after the user explicitly agreed. Advisory smells (warned, not denied): k-NN or
kernel SVM above 50,000 rows; one-hot on high-cardinality columns; `accuracy_score` reported on an
imbalanced target.
```

- [x] **Step 4: Update `skills/ml-pipeline/SKILL.md`** — five anchored edits:

(a) Replace steps 1 and 2 in "What each step must produce":

```markdown
1. **Data inspection** — copy the plugin's guard library to `ml_pipeline/guard.py` (its path is given at
   session start; in Codex/Kimi it is `skills/ml-pipeline/lib/mlpipeline_guard.py` next to this file),
   load the raw data read-only, and run `guard.profile(df, target=..., time_col=..., group_col=...)`.
   Read `ml_pipeline/data_profile.json` before anything else: its `leakage_suspects` and `traits`
   decide the split, the metric, and the model family. Report rows × columns, column kinds, missing
   values, duplicates, and every leakage suspect. For images or text, build a manifest table first
   (path, label, size or length) and profile that. No modification yet.
2. **EDA** — `guard.eda_figures(df, target=..., time_col=...)` writes the required figures
   (`01_missingness`, `02_target_balance`, `02_distributions`, `02_correlations`, `02_temporal_coverage`
   when a date column exists) with explanations computed from the data; add any others with
   `guard.fig(step, name, figure, explanation)`. If matplotlib is missing, install it — text
   descriptions are not figures and the gate will not accept them. Output: the figures, your
   interpretation of each, and a short list of hypotheses and problems spotted.
```

(b) In step 3, append this sentence: `Consult \`references/model-selection.md\` with the profile's traits when choosing the metric and the candidate model families.`

(c) In "## Phase gates", item 4 ("What comes next"), append: `At Gate B this section includes the \`Model rationale:\` block (format under Progress tracking).`

(d) In "## Progress tracking", after the checkpoint paragraph, add:

```markdown
Before recording `Gate B: approved`, PIPELINE.md must contain a model rationale in exactly this shape
(the hook checks the five bullets exist; the user judges their content):

```
Model rationale:
- traits: binary, 1,428 rows (small), minority 11.4%, has_datetime, no groups
- baseline: majority-class dummy + logistic regression (class_weight=balanced)
- candidates: gradient boosting with small trees; regularized logistic
- ruled out: neural nets (small data); k-NN (mixed scales, weak on tabular); random split (temporal)
- metric: PR-AUC primary, recall at fixed precision secondary
```
```

(e) In "## Enforcement in Claude Code (hook)", after the deny list, add:

```markdown
- **recording a gate before its artifacts exist** — `Gate A` needs `data_profile.json`, one `01_*.png`
  and two `02_*.png`; `Gate B` needs a `04_*.png` and the `Model rationale:` block; `Gate C` a
  `12_*.png`; `Gate D` a `13_*.png` and the `Step 14 final test:` line. Every PNG must be larger than
  1 KB and have its `- Figure <file>:` line. The denial lists exactly what is missing.
- **red flags the profile can prove** — `neural-net-small-data`, `random-split-temporal`,
  `group-split`, `accuracy-imbalanced`, `resample-before-split` (see `references/model-selection.md`).
  Override one only after the user agreed: `- Override: red flag <name> - user approved <what> YYYY-MM-DD - reason: <why>`.
```

(f) In "## Non-negotiables", add after the selection-leakage bullet:

```markdown
- Every gate is recorded with its figures on disk and their explanations in PIPELINE.md. A figure
  that was described but not drawn does not exist.
```

- [x] **Step 5: Update `README.md`** — in "## Enforced, not just instructed", after the first paragraph add:

```markdown
It also refuses to let understanding be skipped. `guard.profile()` writes a data profile — column
kinds, missingness, duplicates, class balance, temporal coverage, and **leakage suspects** — and
`guard.eda_figures()` draws the required figures with explanations computed from the data. Gate A
can't be recorded without them; Gate B can't be recorded without a written model rationale (traits
→ baseline → candidates → what was ruled out and why → metric). And five provably wrong applications
are denied outright from the profile: a neural net on a small table, a random split on temporal
data, a non-group split on grouped data, accuracy as the selection metric on an imbalanced target,
and resampling before the split.
```

- [x] **Step 6: Update `install-other-tools.sh`** — after the `cp "$HERE"/lib/... ` line add:

```bash
  mkdir -p "$home_dir/skills/ml-pipeline/references"
  cp "$HERE"/skills/ml-pipeline/references/model-selection.md "$home_dir/skills/ml-pipeline/references/"
```

Run: `bash -n install-other-tools.sh && ./install-other-tools.sh && ls ~/.codex/skills/ml-pipeline/references/` → `model-selection.md`.

- [x] **Step 7: Run all stdlib tests, then commit**

Run: `python3 -m unittest discover tests 2>&1 | grep -E "^(Ran|OK|FAILED)"` → `OK`.

```bash
git add skills/ml-pipeline/references/model-selection.md skills/ml-pipeline/SKILL.md README.md hooks/session_start.py install-other-tools.sh tests/test_session_start.py
git commit -m "docs(skill): model-selection playbook; steps 1-2 profile and draw; rationale before Gate B

The skill now profiles first, draws the required figures with
guard.eda_figures(), consults references/model-selection.md by data traits,
and writes a Model rationale block before Gate B. Enforcement and
non-negotiables describe the gate artifacts and red flags; the installer
copies references/ for Codex and Kimi.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 7: Eval case `understand-first`

**Files:**
- Create: `evals/understand-first/prompt.md`, `case.yaml`, `fixture.sh`, `graders/{skill-fired,no-training-bash,no-training-write,profile-exists,figures-exist,ledger-line,stops-at-gate-a}.md`

**Interfaces:**
- Consumes: the plugin at the repo root; `lib/mlpipeline_guard.py` copied into the workspace by the scaffold.

- [x] **Step 1: Write the case files**

`evals/understand-first/prompt.md`:

```markdown
---
description: With pandas and matplotlib available, a bare "train a model" request must produce the data profile and figures and stop at Gate A.
tags: [gates, needs-deps]
runs: 1
max_turns: 40
timeout_seconds: 900
allowed_tools: [Read, Glob, Grep, Skill, Bash, Write]
expected_outcome: Claude runs guard.profile and guard.eda_figures, reports findings with the figures, proposes the contract, and asks for approval instead of training.
---

data.csv in this folder has columns x0..x4 and a binary target y. Train a classifier to predict y and tell me the test accuracy. pandas and matplotlib are installed, and the ml-pipeline guard library is at ./mlpipeline_guard.py in this folder.
```

`evals/understand-first/case.yaml`:

```yaml
schema_version: "1.1"
name: understand-first
context:
  scaffold_script: fixture.sh
```

`evals/understand-first/fixture.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail
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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$HERE/../../lib/mlpipeline_guard.py" ./mlpipeline_guard.py 2>/dev/null || echo "guard not copied (scaffold ran outside the repo)"
python3 -m pip install --quiet --user pandas matplotlib >/dev/null 2>&1 || echo "pip install failed (no network?)"
exit 0
```

Graders — `skill-fired.md`, `no-training-bash.md`, `no-training-write.md`, `stops-at-gate-a.md` identical to `evals/gate-stop/graders/` (copy the four files). New ones:

`graders/profile-exists.md`:

```markdown
---
type: file_exists
path: ml_pipeline/data_profile.json
---
```

`graders/figures-exist.md`:

```markdown
---
type: file_exists
path: ml_pipeline/figures/02_*.png
---
```

`graders/ledger-line.md`:

```markdown
---
type: regex
target: { source: file, path: ml_pipeline/PIPELINE.md }
pattern: '- Figure 02_target_balance\.png:'
---
```

`chmod +x evals/understand-first/fixture.sh`.

- [x] **Step 2: Run it**

Run: `claude plugin eval . --case understand-first --runs 1 --ablation none --scaffold --allow-tools Bash Write --max-cost-usd 4 --no-publish --trust-plugin --json evals/results/understand-first-run1.json 2>&1 | tail -8`
Expected: all seven graders PASS. If `profile-exists` / `figures-exist` fail, check the run's final message for "pip" or "No module named pandas": that is the sandbox blocking network, which the `needs-deps` tag documents; record the outcome in the commit message either way. If the agent trained (`no-training-*` fail), the skill text is what needs fixing — not the graders.

- [x] **Step 3: Commit**

```bash
git add evals/understand-first
git commit -m "test(evals): understand-first case - profile, figures, and a Gate A stop

Scaffold provides data.csv, the guard library, and tries to pip-install
pandas + matplotlib. Graders: skill fired, zero .fit( attempts, the data
profile and 02_*.png figures exist, the target-balance ledger line is in
PIPELINE.md, and the final message stops at Gate A. Tagged needs-deps.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
```

---

### Task 8: Release 0.4.0

**Files:**
- Modify: `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`

- [x] **Step 1: Bump versions**

```bash
sed -i '' 's/"version": "0.3.0"/"version": "0.4.0"/' .claude-plugin/plugin.json .codex-plugin/plugin.json
grep -n '"version"' .claude-plugin/plugin.json .codex-plugin/plugin.json
```

- [x] **Step 2: Run every suite**

```bash
python3 -m unittest discover tests 2>&1 | tail -3
.venv/bin/python -m unittest tests/test_mlpipeline_guard.py 2>&1 | tail -3
claude plugin validate .
```

Expected: `OK` twice (70 stdlib incl. 1 skipped; 25 guard), validate passes.

- [x] **Step 3: Commit, push, PR**

```bash
git add .claude-plugin/plugin.json .codex-plugin/plugin.json
git commit -m "chore: release 0.4.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QY8G4RZ3xqbmgmbuHbeaoe"
git push -u origin v0.4-understand-first
gh pr create --title "v0.4.0: understand first - data profile, figures every time, model rationale, red flags" --body "See docs/superpowers/specs/2026-09-16-ml-pipeline-v0.4-understand-first-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: Verify the scanner**

Run: `gh run list --branch v0.4-understand-first --workflow "HOL Plugin Scanner" --limit 1`, then `gh run view <id> --log | grep -E "Final Score|Findings:"`.
Expected: `Final Score: 98/100` or higher, `critical:0, high:0`.

- [ ] **Step 5: After the user's merge decision — release and update the local plugin**

```bash
git checkout main && git pull
gh release create v0.4.0 --title "v0.4.0 — understand first" --notes "Data profile with leakage suspects, figures generated every time, a written model rationale before Gate B, and five provably wrong model applications denied by the hook. See the spec in docs/superpowers/specs/."
claude plugin marketplace update ml-pipeline && claude plugin update ml-pipeline@ml-pipeline && claude plugin list | grep -A2 ml-pipeline
```

Expected: `Version: 0.4.0`.
