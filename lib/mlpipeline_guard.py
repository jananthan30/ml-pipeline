"""Runtime guard for the ml-pipeline skill: a leakage-safe split and a single-use test set.

Copy this file into your project as ``ml_pipeline/guard.py`` (the skill tells the agent where the
canonical copy lives). Requires pandas and numpy, nothing else.

    train, val, test = guard.split(df, target="y", time_col="date")          # step 6
    ...
    score = guard.final_test(model.predict, test, target="y", metric_fn=acc)  # step 14

``split`` drops exact duplicates, refuses a random split when a datetime column is present,
keeps whole groups together when asked, checks that no row lands in two splits, and freezes a
fingerprint of the test set in ``ml_pipeline/.guard_state.json``. ``final_test`` is the only
sanctioned way to touch that test set, and it works once.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import warnings
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


class TestSetAlreadyUsed(LeakageError):
    """final_test() was called more than once without a recorded override."""


class TestSetTampered(LeakageError):
    """The frame handed to final_test() is not the one frozen at split time."""


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
    pipeline_md: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test) the way the pipeline's step 6 requires, and freeze the test set.

    Refuses to re-split once the frozen test set has been evaluated (that would reset the single
    use), unless the current run records ``- Override: final test - ...``.
    """
    if not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be in (0, 1) and sum to less than 1")
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    if time_col is None and (dt_cols := _datetime_columns(df)):
        raise TemporalSplitRequired(
            f"datetime column(s) {dt_cols} present: pass time_col=... so the split is chronological"
        )
    state_dir = Path(state_dir)
    ledger = Path(pipeline_md) if pipeline_md is not None else state_dir / "PIPELINE.md"
    previous = state_dir / STATE_FILE
    if (previous.is_file() and json.loads(previous.read_text()).get("test_touches", 0) >= 1
            and not _override_allows(ledger, "final test")):
        raise TestSetAlreadyUsed(
            "the frozen test set was already evaluated; re-splitting would reset that. Agree a new test "
            "strategy with the user and record `- Override: final test - ...` in PIPELINE.md first")

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


# ------------------------------------------------------------------ profile (step 1)
PROFILE_FILE = "data_profile.json"
SMALL_DATA_ROWS = 5000
IMBALANCE_MINORITY = 0.10
HIGH_CARD_LEVELS = 50
LEAK_CORR = 0.95
LEAK_PURITY = 0.98
RARE_LEVEL_FRAC = 0.01
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
    lengths = series.dropna().astype(str).str.len()
    if len(lengths) and lengths.median() > 30:
        return "text"  # free text is almost always unique, so check length before uniqueness
    if unique_frac > 0.95:
        return "id"
    return "categorical"


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
        info.update(length_min=int(lengths.min()), length_median=int(lengths.median()),
                    length_max=int(lengths.max()), median_length=int(lengths.median()))
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
    source: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict:
    """Step 1: describe the data and write ml_pipeline/data_profile.json.

    Everything later in the pipeline — figures, the model playbook, the hook's red flags — reads
    this file instead of re-deriving facts from the data. Every statistic uses every row, and
    ``source`` records whether the frame is a whole loaded source (``guard.load`` / ``guard.derive``).
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    n = len(df)
    kinds = {col: _column_kind(df[col]) for col in df.columns}
    target_info = _target_info(df[target])
    columns = {str(col): _column_stats(df[col], kinds[col]) for col in df.columns}
    high_card = [str(c) for c, k in kinds.items() if k == "categorical" and columns[str(c)].get("n_levels", 0) > HIGH_CARD_LEVELS]
    datetime_cols = [str(c) for c, k in kinds.items() if k == "datetime"]
    minority = target_info["minority_frac"]
    result = {
        "schema_version": "2",
        "created": date.today().isoformat(),
        "shape": {"n_rows": int(n), "n_cols": int(df.shape[1]),
                  "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 2)},
        "source": _source_match(df, source, Path(state_dir)),
        "target": target_info,
        "columns": columns,
        "duplicates": {"exact_row_frac": round(float(df.duplicated().mean()), 4) if n else 0.0},
        "constant_columns": [str(c) for c in df.columns if columns[str(c)]["n_unique"] <= 1],
        "high_cardinality_categoricals": high_card,
        "datetime_columns": datetime_cols,
        "time_col": time_col,
        "group_col": group_col,
        "entity_candidates": [str(c) for c in _entity_candidates(df, kinds)],
        "leakage_suspects": _leakage_suspects(df, target, time_col, kinds, target_info["task"]),
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
    problem = note_problem(explanation)
    if problem:
        raise ValueError(f"figure explanation {problem}: say what it shows, with a number from the data")
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
    explanation = (f"{n_missing} of {len(missing)} columns have missing values; worst is {missing.index[0]} "
                   f"at {missing.iloc[0]:.1%}." if n_missing else
                   f"0 of {len(missing)} columns have missing values, so step 4 needs no imputation.")
    paths.append(fig(1, "missingness", f, explanation, state_dir))

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
