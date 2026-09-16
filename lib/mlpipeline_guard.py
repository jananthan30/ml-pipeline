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
