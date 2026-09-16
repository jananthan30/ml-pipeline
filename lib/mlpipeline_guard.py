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

import hashlib
import json
import re
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
    lengths = series.dropna().astype(str).str.len()
    if len(lengths) and lengths.median() > 30:
        return "text"  # free text is almost always unique, so check length before uniqueness
    if unique_frac > 0.95:
        return "id"
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
