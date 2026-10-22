"""Runtime guard for the ml-pipeline skill: data profile, figures, leakage-safe splits, a single-use
test set, cross-validation folds, a run log, and manifests for image/text data.

Copy this file into your project as ``ml_pipeline/guard.py`` (the skill tells the agent where the
canonical copy lives). Requires pandas and numpy; pyarrow is used when present for Parquet/Arrow
input and output; matplotlib is needed for figures.

    p = guard.profile("data.parquet", target="y", time_col="date")     # step 1 (file or DataFrame)
    guard.eda_figures("data.parquet", target="y", time_col="date")      # step 2
    train, val, test = guard.split(df, target="y", time_col="date")     # step 6, fits in memory
    paths = guard.split_to_files("big.parquet", target="y", time_col="date")  # step 6, out of core
    folds = guard.cv_splits(train, target="y", time_col="date")         # steps 10-11
    guard.log_run("baseline", kind="baseline", metrics={"ap": 0.31})    # steps 9-12
    score = guard.final_test(model.predict, test, target="y", metric_fn=acc)  # step 14, once

Every function works on a pandas DataFrame and, where it matters for size, on a path (Parquet,
CSV/TSV, JSON lines, Feather), a pyarrow Table, or a polars DataFrame, read in batches so the data
never has to fit in memory at once. Statistics that need all values (missingness, distinct counts,
duplicates, min/max) are computed over every row; shape-of-distribution statistics and figures
use a uniform random sample of ``SAMPLE_ROWS`` rows, and say so.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATE_FILE = ".guard_state.json"
APPROVALS_FILE = ".approvals.json"
PROFILE_FILE = "data_profile.json"
SUMMARY_FILE = "data_profile_summary.md"
RUNS_FILE = "runs.jsonl"
FIGURES_DIR = "figures"

STRATIFY_MAX_CLASSES = 20
SAMPLE_ROWS = 200_000          # rows kept for distribution statistics and figures
SIGNAL_ROWS = 50_000           # rows used to rank feature-target signal on wide data
BATCH_ROWS = 100_000           # rows per batch when streaming
EXACT_ROWS = 10_000_000        # exact duplicate counting up to this many rows (~80 MB per hash stream), sketch above
KMV_K = 4096                   # distinct-count sketch size (exact below this many distinct values)
SMALL_DATA_ROWS = 5000
IMBALANCE_MINORITY = 0.10
HIGH_CARD_LEVELS = 50
LEAK_CORR = 0.95
LEAK_ETA2 = 0.90
LEAK_ERROR_REDUCTION = 0.90
SUMMARY_TOP = 20


class LeakageError(RuntimeError):
    """The pipeline tried to do something the discipline forbids."""


class TemporalSplitRequired(LeakageError):
    """The frame has a datetime column (native, text, or epoch) but no time_col was given."""


class GroupSplitRequired(LeakageError):
    """The frame has a repeated entity column (patient, user, device, ...) but no group_col was given."""


class OverlapLeakage(LeakageError):
    """Identical rows appear in more than one split."""


class TestSetAlreadyUsed(LeakageError):
    """final_test() was called more than once without a user-approved override."""


class TestSetTampered(LeakageError):
    """The data handed to final_test() is not the test set frozen at split time."""


# ====================================================================== input: frames, files, batches
def _is_path(source) -> bool:
    return isinstance(source, (str, Path))


_CSV_TYPES: dict = {}


def _csv_dtypes(path: Path, sep: str, columns) -> tuple[dict, list]:
    """(read dtypes, integer columns) from one pass over the WHOLE file, so a late "A12" in a numeric-looking
    column can't break a later chunk, and every chunk hashes identically. Cached per file version."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, sep, tuple(columns) if columns else None)
    if key in _CSV_TYPES:
        return _CSV_TYPES[key]
    numeric: dict = {}
    integral: dict = {}
    for chunk in pd.read_csv(path, sep=sep, chunksize=BATCH_ROWS, usecols=columns, dtype=str, keep_default_na=True):
        for col in chunk.columns:
            raw = chunk[col]
            present = raw.notna() & (raw.str.strip() != "")
            values = pd.to_numeric(raw.where(present), errors="coerce")
            ok = not bool((present & values.isna()).any())
            numeric[col] = numeric.get(col, True) and ok
            if numeric[col]:
                finite = values.dropna()
                integral[col] = integral.get(col, True) and bool(np.all(np.mod(finite.to_numpy(dtype=float), 1) == 0)) \
                    and not bool(raw.isna().any() | (~present).any())
    dtypes = {c: ("float64" if numeric[c] else "object") for c in numeric}
    ints = [c for c in numeric if numeric[c] and integral.get(c)]
    _CSV_TYPES[key] = (dtypes, ints)
    return dtypes, ints


def scan(source, *, columns=None, batch_rows: int | None = None):
    """Yield pandas DataFrames of at most ``batch_rows`` rows from any supported source.

    Supported: pandas DataFrame (sliced, no copy), a path to Parquet (file or directory), CSV, TSV,
    JSON lines, Feather/Arrow IPC, a pyarrow Table, or a polars DataFrame/LazyFrame.
    """
    batch_rows = int(batch_rows or BATCH_ROWS)
    if isinstance(source, pd.DataFrame):
        frame = source if columns is None else source[list(columns)]
        for start in range(0, max(len(frame), 1), batch_rows):
            yield frame.iloc[start:start + batch_rows]
        return
    mod = type(source).__module__
    if mod.startswith("pyarrow"):
        for batch in source.to_batches(max_chunksize=batch_rows):
            out = batch.to_pandas()
            yield out if columns is None else out[list(columns)]
        return
    if mod.startswith("polars"):
        frame = source.collect() if hasattr(source, "collect") else source
        if columns is not None:
            frame = frame.select(list(columns))
        for chunk in frame.iter_slices(n_rows=batch_rows):
            yield chunk.to_pandas()
        return
    if not _is_path(source):
        raise TypeError(f"unsupported data source {type(source).__name__}: pass a DataFrame or a file path")
    path = Path(source)
    suffix = path.suffix.lower()
    if path.is_dir() or suffix in (".parquet", ".pq"):
        try:
            import pyarrow.dataset as ds
        except ImportError:
            yield from scan(pd.read_parquet(path, columns=columns), batch_rows=batch_rows)
            return
        # pre-buffering and readahead off: by default pyarrow holds whole column chunks and ~16 batches
        # in memory ahead of the reader, which defeats streaming on files larger than RAM
        fmt = ds.ParquetFileFormat(default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=False))
        dataset = ds.dataset(str(path), format=fmt)
        for batch in dataset.to_batches(columns=list(columns) if columns else None, batch_size=batch_rows,
                                        batch_readahead=0, fragment_readahead=0):
            yield batch.to_pandas()
        return
    if suffix in (".feather", ".arrow", ".ipc"):
        import pyarrow.feather as feather
        yield from scan(feather.read_table(str(path), columns=columns), batch_rows=batch_rows)
        return
    if suffix in (".jsonl", ".ndjson"):
        for chunk in pd.read_json(path, lines=True, chunksize=batch_rows):
            yield chunk if columns is None else chunk[list(columns)]
        return
    if suffix in (".csv", ".tsv", ".txt", ".gz", ".bz2", ".zip", ".xz"):
        sep = "\t" if ".tsv" in path.name.lower() else ","
        dtypes, ints = _csv_dtypes(path, sep, columns)
        for chunk in pd.read_csv(path, sep=sep, chunksize=batch_rows, usecols=columns, dtype=dtypes, low_memory=False):
            yield chunk.astype({c: "int64" for c in ints if c in chunk.columns}) if ints else chunk
        return
    raise ValueError(f"don't know how to read {path.name}: use Parquet, CSV/TSV, JSON lines, or Feather")


def _trim_sample(frames: list, key_parts: list, n: int):
    kept = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)
    keys = np.concatenate(key_parts)
    if len(kept) > n:
        idx = np.argpartition(keys, n - 1)[:n]
        kept, keys = kept.iloc[idx].reset_index(drop=True), keys[idx]
    return [kept], [keys], float(keys.max()) if len(keys) else np.inf


def load_sample(source, n: int | None = None, *, seed: int = 0, columns=None) -> tuple[pd.DataFrame, int]:
    """(uniform random sample of at most ``n`` rows, total row count), in one streaming pass.

    Bottom-k sampling: every row gets a random key and the ``n`` smallest keys win, so the sample
    is uniform over the whole source whatever its size or order.
    """
    n = int(n or SAMPLE_ROWS)
    if isinstance(source, pd.DataFrame) and columns is None:
        total = len(source)
        return (source if total <= n else source.sample(n, random_state=seed)), total
    rng = np.random.default_rng(seed)
    frames, key_parts, held, total, threshold = [], [], 0, 0, np.inf
    for batch in scan(source, columns=columns):
        total += len(batch)
        keys = rng.random(len(batch))
        mask = keys < threshold  # once the sample is full, only rows that beat the worst kept key matter
        if mask.any():
            frames.append(batch[mask] if not mask.all() else batch)
            key_parts.append(keys[mask])
            held += int(mask.sum())
        if held > 2 * n:
            frames, key_parts, threshold = _trim_sample(frames, key_parts, n)
            held = n
    if not frames:
        raise ValueError("the data source has no rows")
    frames, key_parts, _ = _trim_sample(frames, key_parts, n)
    kept = frames[0]
    return kept.reset_index(drop=True), total


# ====================================================================== hashing and fingerprints
def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)


class _Fingerprint:
    """Order-independent fingerprint of a multiset of rows, built batch by batch."""

    def __init__(self):
        self.n, self.total, self.xor, self.mix = 0, np.uint64(0), np.uint64(0), np.uint64(0)

    def add(self, hashes: np.ndarray) -> None:
        if not len(hashes):
            return
        with np.errstate(over="ignore"):
            self.n += len(hashes)
            self.total = np.uint64(self.total + hashes.sum(dtype=np.uint64))
            self.xor = np.uint64(self.xor ^ np.bitwise_xor.reduce(hashes))
            mixed = (hashes ^ (hashes >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)
            self.mix = np.uint64(self.mix + mixed.sum(dtype=np.uint64))

    def hexdigest(self) -> str:
        return hashlib.sha256(f"v2:{self.n}:{int(self.total)}:{int(self.xor)}:{int(self.mix)}".encode()).hexdigest()


def _fingerprint(df: pd.DataFrame) -> str:
    fp = _Fingerprint()
    fp.add(_row_hashes(df))
    return fp.hexdigest()


def _fingerprint_v1(df: pd.DataFrame) -> str:
    """The v0.4 fingerprint, kept so state files written by v0.4 still verify."""
    digest = hashlib.sha256()
    for value in np.sort(_row_hashes(df)):
        digest.update(int(value).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _fingerprint_source(source, columns=None) -> tuple[str, int]:
    fp = _Fingerprint()
    for batch in scan(source, columns=columns):
        fp.add(_row_hashes(batch))
    return fp.hexdigest(), fp.n


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    """np.unique for uint64 hashes via sort + diff (much faster than numpy's hash-based unique)."""
    if not len(values):
        return values
    ordered = np.sort(values, kind="stable") if values.dtype != np.uint64 else np.sort(values)
    keep = np.empty(len(ordered), dtype=bool)
    keep[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=keep[1:])
    return ordered[keep]


class _KMV:
    """K-minimum-values distinct-count sketch over 64-bit hashes; exact below ``k`` distinct values."""

    def __init__(self, k: int = KMV_K):
        self.k, self.values, self.exact = k, np.empty(0, dtype=np.uint64), True

    def add(self, hashes: np.ndarray) -> None:
        if not len(hashes):
            return
        if len(self.values) >= self.k:  # full: only hashes below the current k-th smallest can matter
            hashes = hashes[hashes < self.values[-1]]
            if not len(hashes):
                return
        merged = _sorted_unique(np.concatenate([self.values, hashes]))
        if len(merged) > self.k:
            merged, self.exact = merged[:self.k], False
        self.values = merged

    def estimate(self) -> int:
        if self.exact:
            return int(len(self.values))
        return int((self.k - 1) / (float(self.values[-1]) / 2.0 ** 64))


class _Distinct:
    """Exact distinct count of row hashes up to EXACT_ROWS rows, a KMV estimate beyond."""

    def __init__(self):
        self.parts, self.n, self.sketch = [], 0, _KMV()

    def add(self, hashes: np.ndarray) -> None:
        self.n += len(hashes)
        self.sketch.add(hashes)
        if self.parts is not None:
            self.parts.append(hashes)
            if self.n > EXACT_ROWS:
                self.parts = None

    def count(self) -> tuple[int, bool]:
        if self.parts is not None:
            return (int(len(_sorted_unique(np.concatenate(self.parts)))) if self.parts else 0), True
        return self.sketch.estimate(), False


def check_no_overlap(*frames: pd.DataFrame) -> None:
    """Raise OverlapLeakage if any identical row appears in two different frames."""
    hashes = [_sorted_unique(_row_hashes(frame)) for frame in frames]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            common = np.intersect1d(hashes[i], hashes[j], assume_unique=True)
            if len(common):
                raise OverlapLeakage(f"{len(common)} identical row(s) appear in both split {i} and split {j}; "
                                     "deduplicate before splitting")


# ====================================================================== dates stored as text or numbers
_DATE_TEXT = re.compile(
    r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{2,4}|"
    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}|\d{4}-\d{2})"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\s*(?:Z|UTC|[+-]\d{2}:?\d{2})?\s*$"
)
_DATE_NAME = re.compile(r"(date|time|timestamp|datetime|epoch|_at$|_ts$|^ts$|_dt$|^dt$)", re.I)
_EPOCH_RANGES = (("epoch_s", 1e8, 4.2e9, "s"), ("epoch_ms", 1e11, 4.2e12, "ms"), ("epoch_ns", 1e17, 4.2e18, "ns"))


def _datetime_source(series: pd.Series) -> str | None:
    """'native', 'text', 'epoch_s' / 'epoch_ms' / 'epoch_ns', or None when the column is not a date."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "native"
    values = series.dropna()
    if not len(values):
        return None
    head = values.iloc[:500]
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        text = head.astype(str)
        if text.str.match(_DATE_TEXT).mean() < 0.9:
            return None
        return "text" if _parse_text_dates(text).notna().mean() >= 0.9 else None
    if (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            and _DATE_NAME.search(str(series.name or ""))):
        med, low = float(head.median()), float(head.min())
        for name, lo, hi, _unit in _EPOCH_RANGES:
            if lo <= med <= hi and low > 0:
                return name
    return None


def _naive_utc(parsed: pd.Series) -> pd.Series:
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def _parse_text_dates(text: pd.Series) -> pd.Series:
    """Parse date strings; mixed UTC offsets (any data crossing a DST change) are normalized to UTC."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in ({"format": "mixed"}, {}, {"format": "mixed", "utc": True}, {"utc": True}):
            try:
                parsed = pd.to_datetime(text, errors="coerce", **kwargs)
            except (TypeError, ValueError):
                continue
            # pandas 2.x returns plain objects (not an error) for mixed offsets without utc=True
            if pd.api.types.is_datetime64_any_dtype(parsed):
                return _naive_utc(parsed)
    return pd.Series(pd.NaT, index=text.index)


def as_datetime(series: pd.Series, source: str | None = None) -> pd.Series:
    """Convert a native, text, or epoch date column to timezone-naive datetime64 (UTC for zoned input).

    Unparseable values become NaT.
    """
    source = source or _datetime_source(series) or "text"
    if source == "native":
        return _naive_utc(series)
    if source.startswith("epoch"):
        unit = dict((n, u) for n, _, _, u in _EPOCH_RANGES)[source]
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return _parse_text_dates(series)


def datetime_columns(df: pd.DataFrame) -> dict[str, str]:
    """Every column that holds dates, however stored: {column: 'native' | 'text' | 'epoch_*'}."""
    sample = df if len(df) <= 2000 else df.sample(2000, random_state=0)
    found = {}
    for col in df.columns:
        source = _datetime_source(sample[col])
        if source:
            found[str(col)] = source
    return found


def parse_datetimes(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """A copy of ``df`` with text and epoch date columns converted to datetime64."""
    found = datetime_columns(df)
    targets = {c: s for c, s in found.items() if s != "native" and (columns is None or c in columns)}
    if not targets:
        return df
    return df.assign(**{c: as_datetime(df[c], s) for c, s in targets.items()})


# ====================================================================== column kinds and entities
_ID_NAME = re.compile(r"(^|_)(id|key|uuid|guid)$|id$", re.I)
_ENTITY_NAME = re.compile(r"(^|_)(id|key|uuid|guid|patient|user|customer|client|account|device|session|subject|"
                          r"member|site|store|hospital|household|md5)(_|$)", re.I)
_CAMEL_ID = re.compile(r"[a-z0-9](?:ID|Id)$")
_LEAK_NAME = re.compile(r"(outcome|result|after|label|target|final|actual|post_|_post|future|next_)|_y$", re.I)


def _integer_like(values: pd.Series) -> bool:
    values = values.dropna()
    if not len(values):
        return False
    if pd.api.types.is_integer_dtype(values):
        return True
    if pd.api.types.is_float_dtype(values):
        with np.errstate(invalid="ignore"):
            return bool(np.all(np.mod(values.to_numpy(dtype=float), 1) == 0))
    return False


def _column_kind(series: pd.Series, date_source: str | None = None) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if date_source or pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    n = len(series)
    n_unique = series.nunique(dropna=True)
    unique_frac = n_unique / n if n else 0.0
    if pd.api.types.is_numeric_dtype(series):
        if unique_frac > 0.95 and _integer_like(series):
            values = series.dropna()
            span = float(values.max() - values.min() + 1) if len(values) else 0.0
            consecutive = n_unique > 0 and span / n_unique < 1.2
            if _ID_NAME.search(str(series.name or "")) or consecutive:
                return "id"
        return "numeric"
    lengths = series.dropna().astype(str).str.len()
    if len(lengths) and lengths.median() > 30:
        return "text"  # free text is almost always unique, so check length before uniqueness
    if unique_frac > 0.95:
        return "id"
    return "categorical"


def _entity_candidates(df: pd.DataFrame, kinds: dict, exclude=()) -> list[dict]:
    """Columns that name an entity (patient, user, device...). ``repeated`` means rows share it."""
    out = []
    n = len(df)
    for col, kind in kinds.items():
        if kind in ("datetime", "text") or col in exclude:
            continue
        name = str(col)
        if not (_ENTITY_NAME.search(name) or _CAMEL_ID.search(name) or kind == "id" or name in ("group", "md5")):
            continue
        n_unique = int(df[col].nunique(dropna=True))
        rows_per = n / n_unique if n_unique else 0.0
        out.append({"column": name, "n_entities": n_unique, "rows_per_entity": round(rows_per, 2),
                    "repeated": bool(n_unique >= 3 and rows_per >= 1.1)})
    return out


# ====================================================================== signal and leakage
def _target_matrix(y: pd.Series, task: str) -> np.ndarray | None:
    """Target as one or more numeric columns: the value (regression) or class indicators (top 5)."""
    if task == "regression":
        return y.astype(float).to_numpy()[:, None]
    labels = y.astype(str)
    classes = labels.value_counts().index[: (1 if labels.nunique() == 2 else 5)]
    return np.column_stack([(labels == c).to_numpy(dtype=float) for c in classes]) if len(classes) else None


def _codes(series: pd.Series, kind: str) -> np.ndarray:
    if kind == "numeric":
        x = series.astype(float)
        if x.nunique() <= HIGH_CARD_LEVELS:  # few distinct values: each value is its own group
            return pd.factorize(x)[0]
        try:  # rank first, so a point mass (many identical values) can't merge with its neighbours
            codes = pd.qcut(x.rank(method="first"), 20, labels=False, duplicates="drop")
        except (ValueError, IndexError):
            codes = pd.Series(np.zeros(len(x)), index=x.index)
        return codes.fillna(-1).to_numpy(dtype=np.int64)
    values = series.astype(str).where(series.notna(), "<missing>")
    top = values.value_counts().index[:HIGH_CARD_LEVELS]
    return pd.factorize(values.where(values.isin(top), "<other>"))[0]


def _eta2(codes: np.ndarray, Y: np.ndarray) -> float:
    """Share of target variance explained by the groups in ``codes`` (correlation ratio, 0..1)."""
    _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    best = 0.0
    for j in range(Y.shape[1]):
        y = Y[:, j]
        total = float(((y - y.mean()) ** 2).sum())
        if total <= 0:
            continue
        means = np.bincount(inverse, weights=y) / counts
        between = float((counts * (means - y.mean()) ** 2).sum())
        best = max(best, between / total)
    return best


def _feature_signal(df: pd.DataFrame, target: str, kinds: dict, task: str) -> dict[str, float]:
    """Correlation ratio of each feature with the target on a sample: a model-free ranking for EDA."""
    data = df[df[target].notna()]
    if len(data) > SIGNAL_ROWS:
        data = data.sample(SIGNAL_ROWS, random_state=0)
    Y = _target_matrix(data[target], task)
    if Y is None or not len(data):
        return {}
    signal = {}
    for col, kind in kinds.items():
        if col == target or kind not in ("numeric", "categorical", "bool"):
            continue
        signal[str(col)] = round(_eta2(_codes(data[col], kind), Y), 4)
    return dict(sorted(signal.items(), key=lambda kv: -kv[1]))


def _leakage_suspects(df: pd.DataFrame, target: str, time_col, kinds: dict, task: str,
                      signal: dict, date_sources: dict) -> list[dict]:
    y = df[target]
    y_num = y.astype(float) if task == "regression" else pd.Series(pd.factorize(y)[0], index=y.index).astype(float)
    majority = float(y.astype(str).value_counts(normalize=True).iloc[0]) if task != "regression" and len(y) else 0.0
    target_name = str(target).lower()
    suspects = []
    for col, kind in kinds.items():
        if col == target:
            continue
        name = str(col)
        if kind == "numeric":
            x = df[col].astype(float)
            if x.notna().sum() > 2 and x.std() > 0 and y_num.std() > 0:
                corr = abs(np.corrcoef(x.fillna(x.median()), y_num)[0, 1])
                if corr > LEAK_CORR:
                    suspects.append({"column": name, "reason": f"|corr| with target = {corr:.2f}"})
                elif signal.get(name, 0.0) > LEAK_ETA2:
                    suspects.append({"column": name, "reason": f"explains {signal[name]:.0%} of target variance "
                                                               "through a non-linear pattern (correlation ratio)"})
        elif kind in ("categorical", "bool", "id") and task != "regression":
            levels = df[col].astype(str)
            counts = levels.value_counts()
            repeated = levels.isin(counts.index[counts >= 2])
            coverage = float(repeated.mean())
            if coverage >= 0.5 and repeated.sum() > 0:
                table = pd.crosstab(levels[repeated], y[repeated].astype(str))
                purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
                reduction = (purity - majority) / (1 - majority) if majority < 1 else 0.0
                if purity > 0.98 and reduction > LEAK_ERROR_REDUCTION:
                    what = "identifies the target (memorizable)" if counts.size > HIGH_CARD_LEVELS else "predicts target"
                    suspects.append({"column": name, "reason": f"{what} with {purity:.0%} purity"})
        elif kind == "datetime" and time_col and col != time_col:
            mine = as_datetime(df[col], date_sources.get(name))
            ref = as_datetime(df[time_col], date_sources.get(str(time_col)))
            later = float((mine > ref).mean())
            if later > 0.5:
                suspects.append({"column": name, "reason": f"datetime later than {time_col} in {later:.0%} of rows"})
        if kind == "id" and task != "regression" and pd.api.types.is_numeric_dtype(df[col]):
            rank = df[col].rank().corr(y_num.rank())
            if rank is not None and not math.isnan(rank) and abs(rank) > 0.3:
                suspects.append({"column": name, "reason": f"id order correlates with target (rank corr {rank:.2f}) - "
                                                           "rows were sorted or ids assigned after the outcome"})
        if _LEAK_NAME.search(name):
            suspects.append({"column": name, "reason": "name suggests post-outcome information"})
        elif len(target_name) >= 3 and target_name in name.lower():
            suspects.append({"column": name, "reason": f"name contains the target name '{target}' (derived from it?)"})
    return suspects


# ====================================================================== profile (step 1)
def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


def _column_stats(sample: pd.Series, kind: str) -> dict:
    info: dict = {"kind": kind}
    values = sample.dropna()
    if kind == "numeric" and len(values):
        finite = np.isfinite(values.astype(float))
        if not finite.all():
            info["inf_frac"] = round(float((~finite).mean()), 4)
            values = values[finite]
    if not len(values):
        return info
    if kind == "numeric":
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
        info.update(median=_num(values.median()), mean=_num(values.mean()), std=_num(values.std()),
                    skew=_num(values.skew()) if len(values) > 2 else 0.0, outlier_frac=round(float(outliers), 4))
    elif kind in ("categorical", "bool"):
        top = values.astype(str).value_counts(normalize=True).head(5)
        info.update(n_levels_in_sample=int(values.nunique()), top={k: round(float(v), 4) for k, v in top.items()})
    elif kind == "text":
        info.update(median_length=int(values.astype(str).str.len().median()))
    return info


def _target_info(y: pd.Series) -> dict:
    values = y.dropna()
    n_unique = int(values.nunique())
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and n_unique > 20:
        return {"name": str(y.name), "task": "regression", "n_classes": None, "class_balance": None,
                "minority_frac": None, "missing_frac": round(float(y.isna().mean()), 4)}
    balance = values.astype(str).value_counts(normalize=True)
    return {"name": str(y.name), "task": "binary" if n_unique == 2 else "multiclass", "n_classes": n_unique,
            "class_balance": {k: round(float(v), 4) for k, v in balance.head(50).items()},
            "minority_frac": round(float(balance.min()), 4) if len(balance) else None,
            "missing_frac": round(float(y.isna().mean()), 4)}


def _full_pass(source, kinds: dict, target: str, id_cols: list[str]) -> dict:
    """Every-row statistics in one streaming pass: missingness, distinct counts, min/max, duplicates."""
    cols = list(kinds)
    missing = dict.fromkeys(cols, 0)
    distinct = {c: _KMV() for c in cols}
    lo, hi = {}, {}
    rows, no_ids, features, with_target = _Distinct(), _Distinct(), _Distinct(), _Distinct()
    feature_cols = [c for c in cols if c != target and c not in id_cols]
    n = 0
    for batch in scan(source):
        n += len(batch)
        for c in cols:
            s = batch[c]
            nulls = s.isna()
            missing[c] += int(nulls.sum())
            present = s[~nulls]
            if len(present):
                distinct[c].add(pd.util.hash_pandas_object(present, index=False).to_numpy(dtype=np.uint64))
                if kinds[c] == "numeric":
                    vals = present.astype(float)
                    lo[c] = min(lo.get(c, np.inf), float(vals.min()))
                    hi[c] = max(hi.get(c, -np.inf), float(vals.max()))
        rows.add(_row_hashes(batch))
        if id_cols:
            no_ids.add(_row_hashes(batch[[c for c in cols if c not in id_cols]]))
        if feature_cols:
            features.add(_row_hashes(batch[feature_cols]))
            with_target.add(_row_hashes(batch[feature_cols + [target]]))
    distinct_rows, exact = rows.count()
    out = {"n": n, "missing": missing, "distinct": {c: (d.estimate(), d.exact) for c, d in distinct.items()},
           "min": lo, "max": hi, "exact": exact,
           "exact_row_frac": round(1 - distinct_rows / n, 4) if n else 0.0}
    if id_cols:
        d, _ = no_ids.count()
        out["duplicate_ignoring_ids_frac"] = round(1 - d / n, 4) if n else 0.0
    if feature_cols:
        f, _ = features.count()
        ft, _ = with_target.count()
        out["label_conflict_frac"] = round(max(ft - f, 0) / n, 4) if n else 0.0
    return out


def _summary_markdown(result: dict) -> str:
    shape, target, traits = result["shape"], result["target"], result["traits"]
    lines = [
        "# Data profile summary",
        "",
        f"- rows x columns: {shape['n_rows']:,} x {shape['n_cols']:,}"
        + (f" (distribution stats from a {shape['sample_rows']:,}-row uniform sample)" if shape["sampled"] else ""),
        f"- target `{target['name']}`: {target['task']}"
        + (f", minority {target['minority_frac']:.1%}" if target.get("minority_frac") is not None else "")
        + (f", {target['missing_frac']:.1%} missing" if target.get("missing_frac") else ""),
        f"- duplicates: {result['duplicates']['exact_row_frac']:.1%} exact rows"
        + (" (estimated)" if not result["duplicates"]["exact"] else "")
        + (f"; {result['duplicates']['duplicate_ignoring_ids_frac']:.1%} ignoring id columns"
           if "duplicate_ignoring_ids_frac" in result["duplicates"] else "")
        + (f"; {result['duplicates']['label_conflict_frac']:.1%} same features with a different target"
           if result["duplicates"].get("label_conflict_frac") else ""),
        f"- traits: {', '.join(k for k, v in traits.items() if v is True) or 'none'}",
        f"- date columns: {', '.join(f'{c} ({s})' for c, s in result['datetime_columns'].items()) or 'none'}"
        + ("  <- stored as text/numbers: split with time_col and parse with guard.parse_datetimes()"
           if any(s != "native" for s in result["datetime_columns"].values()) else ""),
        f"- entity columns that repeat (need a group split): "
        f"{', '.join(c['column'] for c in result['entity_candidates'] if c['repeated']) or 'none'}",
    ]
    suspects = result["leakage_suspects"]
    lines += ["", f"## Leakage suspects ({len(suspects)})"]
    lines += [f"- {s['column']}: {s['reason']}" for s in suspects[:SUMMARY_TOP]] or ["- none"]
    if len(suspects) > SUMMARY_TOP:
        lines.append(f"- ... {len(suspects) - SUMMARY_TOP} more in data_profile.json -> leakage_suspects")
    lines += ["", f"## Strongest target signal (top {SUMMARY_TOP} of {len(result['feature_signal'])})"]
    lines += [f"- {c}: {v:.3f}" for c, v in list(result["feature_signal"].items())[:SUMMARY_TOP]] or ["- none"]
    worst = sorted(((c, i["missing_frac"]) for c, i in result["columns"].items() if i["missing_frac"] > 0),
                   key=lambda kv: -kv[1])
    lines += ["", f"## Missing values ({len(worst)} columns)"]
    lines += [f"- {c}: {m:.1%}" for c, m in worst[:SUMMARY_TOP]] or ["- none"]
    kinds: dict = {}
    for col, info in result["columns"].items():
        kinds.setdefault(info["kind"], []).append(col)
    lines += ["", "## Column kinds"]
    lines += [f"- {k} ({len(v)}): {', '.join(v[:12])}{' ...' if len(v) > 12 else ''}" for k, v in sorted(kinds.items())]
    for label, key in (("Constant columns", "constant_columns"), ("High-cardinality categoricals", "high_cardinality_categoricals")):
        values = result[key]
        lines.append(f"- {label} ({len(values)}): {', '.join(values[:12]) or 'none'}{' ...' if len(values) > 12 else ''}")
    lines += ["", "Full per-column detail: data_profile.json -> columns.<name>. Read single columns, not the whole file."]
    return "\n".join(lines) + "\n"


def profile(
    source,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict:
    """Step 1: describe the data, write ml_pipeline/data_profile.json and a short data_profile_summary.md.

    ``source`` is a DataFrame or anything ``scan`` reads (Parquet, CSV, ...), so data larger than
    memory is profiled in batches. Read the summary first; the JSON is the per-column reference.
    """
    sample, n_total = load_sample(source)
    if not n_total:
        raise ValueError("the data has no rows")
    if target not in sample.columns:
        raise KeyError(f"target column {target!r} not in data")
    if time_col is not None and time_col not in sample.columns:
        raise KeyError(f"time_col {time_col!r} not in data")
    if group_col is not None and group_col not in sample.columns:
        raise KeyError(f"group_col {group_col!r} not in data")
    sampled = n_total > len(sample)
    date_sources = datetime_columns(sample)
    if time_col is not None:
        date_sources.setdefault(str(time_col), _datetime_source(sample[time_col]) or "text")
    kinds = {col: _column_kind(sample[col], date_sources.get(str(col)) if col != target else None) for col in sample.columns}
    target_info = _target_info(sample[target])
    id_cols = [c for c, k in kinds.items() if k == "id" and c != target]
    full = _full_pass(source, kinds, target, id_cols)
    n = full["n"]

    columns = {}
    for col in sample.columns:
        n_unique, exact = full["distinct"][col]
        info = {"kind": kinds[col], "missing_frac": round(full["missing"][col] / n, 4) if n else 0.0,
                "n_unique": n_unique, "n_unique_exact": exact, "unique_frac": round(n_unique / n, 4) if n else 0.0}
        if kinds[col] == "numeric" and col in full["min"]:
            info.update(min=_num(full["min"][col]), max=_num(full["max"][col]))
        if kinds[col] == "datetime":
            parsed = as_datetime(sample[col], date_sources.get(str(col)))
            info.update(datetime_source=date_sources.get(str(col), "native"),
                        start=str(parsed.min())[:10], end=str(parsed.max())[:10])
        info.update({k: v for k, v in _column_stats(sample[col], kinds[col]).items() if k != "kind"})
        if kinds[col] == "categorical":
            info["n_levels"] = n_unique
        columns[str(col)] = info

    signal = _feature_signal(sample, target, kinds, target_info["task"])
    entities = _entity_candidates(sample, kinds, exclude=(target,))
    if group_col is not None and not any(e["column"] == str(group_col) for e in entities):
        nu = int(sample[group_col].nunique())
        entities.append({"column": str(group_col), "n_entities": nu, "rows_per_entity": round(len(sample) / max(nu, 1), 2),
                         "repeated": True})
    repeated = [e["column"] for e in entities if e["repeated"]]
    high_card = [str(c) for c, k in kinds.items() if k == "categorical" and columns[str(c)]["n_unique"] > HIGH_CARD_LEVELS]
    minority = target_info["minority_frac"]
    duplicates = {"exact_row_frac": full["exact_row_frac"], "exact": full["exact"]}
    for key in ("duplicate_ignoring_ids_frac", "label_conflict_frac"):
        if key in full:
            duplicates[key] = full[key]
    result = {
        "schema_version": "2",
        "created": date.today().isoformat(),
        "shape": {"n_rows": int(n), "n_cols": int(sample.shape[1]),
                  "memory_mb_estimate": round(float(sample.memory_usage(deep=True).sum()) / max(len(sample), 1) * n / 1e6, 2),
                  "sampled": bool(sampled), "sample_rows": int(len(sample)) if sampled else None},
        "target": target_info,
        "columns": columns,
        "duplicates": duplicates,
        "constant_columns": [str(c) for c in sample.columns if columns[str(c)]["n_unique"] <= 1],
        "high_cardinality_categoricals": high_card,
        "datetime_columns": {c: s for c, s in date_sources.items() if c != str(target)},
        "time_col": time_col,
        "group_col": group_col,
        "entity_candidates": entities,
        "feature_signal": signal,
        "leakage_suspects": _leakage_suspects(sample, target, time_col, kinds, target_info["task"], signal, date_sources),
        "traits": {
            "n_rows": int(n), "n_features": int(sample.shape[1] - 1), "task": target_info["task"],
            "minority_frac": minority,
            "has_datetime": bool(date_sources) or time_col is not None,
            "datetime_as_text": any(s != "native" for s in date_sources.values()),
            # a time split already orders repeated entities; without dates they need a group split
            "has_groups": group_col is not None or (bool(repeated) and time_col is None and not date_sources),
            "group_candidates": repeated,
            "small_data": n < SMALL_DATA_ROWS,
            "large_data": n > 500_000,
            "wide_data": sample.shape[1] > 200,
            "imbalanced": target_info["task"] != "regression" and (minority if minority is not None else 1.0) < IMBALANCE_MINORITY,
            "high_card_categoricals": high_card,
        },
    }
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / PROFILE_FILE).write_text(json.dumps(result, indent=2, default=str))
    (state_dir / SUMMARY_FILE).write_text(_summary_markdown(result))
    return result


def summary(state_dir: str | Path = "ml_pipeline") -> str:
    """The short profile summary (what the agent should read instead of the full JSON)."""
    return (Path(state_dir) / SUMMARY_FILE).read_text()


# ====================================================================== figures (steps 1-2 and any step)
def _plt():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for figures: pip install matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ledger(pipeline_md: Path, filename: str, explanation: str) -> None:
    line = f"- Figure {filename}: {' '.join(explanation.split())}\n"
    pipeline_md.parent.mkdir(parents=True, exist_ok=True)
    existing = pipeline_md.read_text().rstrip("\n") + "\n" if pipeline_md.is_file() else ""
    if not re.search(r"^- Figure " + re.escape(filename) + ":", existing, re.M):
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
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    _ledger(state_dir / "PIPELINE.md", path.name, explanation)
    return path


def _pages(items: list, per_page: int, max_pages: int) -> list[list]:
    return [items[i:i + per_page] for i in range(0, min(len(items), per_page * max_pages), per_page)]


def _grid(plt, k: int, ncols: int = 4, height: float = 2.6):
    nrows = max(1, int(np.ceil(k / ncols)))
    f, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, height * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[k:]:
        ax.axis("off")
    return f, axes


def _target_view(y: pd.Series, task: str) -> tuple[pd.Series, str]:
    """A numeric view of the target for plots: the value, the positive rate, or the majority-class rate."""
    if task == "regression":
        return y.astype(float), f"mean {y.name}"
    labels = y.astype(str)
    counts = labels.value_counts()
    positive = counts.index[-1] if task == "binary" else counts.index[0]
    return (labels == positive).astype(float), f"share {y.name}={positive}"


def eda_figures(source, *, target: str, time_col: str | None = None, state_dir: str | Path = "ml_pipeline",
                per_page: int = 16, max_pages: int = 8) -> list[Path]:
    """The step 1-2 figures, with explanations computed from the data. Works on files of any size.

    Plots use a uniform sample of SAMPLE_ROWS rows. Every column is covered: the feature overview
    shows all of them at once, and the paged figures go through them ranked by target signal (up to
    ``max_pages`` pages each); the explanations say exactly how much each figure covers.
    """
    plt = _plt()
    state_dir = Path(state_dir)
    df, n_total = load_sample(source)
    df = df.rename(columns=str)  # a private copy: dates are converted in place below
    target = str(target)
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in data")
    date_sources = datetime_columns(df.drop(columns=[target]))
    if time_col is not None:
        date_sources.setdefault(str(time_col), _datetime_source(df[time_col]) or "text")
    for col, src in date_sources.items():
        df[col] = as_datetime(df[col], src)
    for col in df.select_dtypes("number").columns:  # inf breaks histograms and binning; it is reported in the profile
        if col != target and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
    sampled_note = f" (uniform sample of {len(df):,} of {n_total:,} rows)" if n_total > len(df) else ""
    profile_path = state_dir / PROFILE_FILE
    prof = json.loads(profile_path.read_text()) if profile_path.is_file() else None
    kinds = ({c: i["kind"] for c, i in prof["columns"].items() if c in df.columns} if prof
             else {str(c): _column_kind(df[c], date_sources.get(str(c))) for c in df.columns if c != target})
    kinds.pop(str(target), None)
    task = prof["target"]["task"] if prof else _target_info(df[target])["task"]
    signal = prof["feature_signal"] if prof and prof.get("feature_signal") else _feature_signal(df, target, {**kinds, target: "target"}, task)
    ranked = lambda cols: sorted(cols, key=lambda c: -signal.get(c, -1.0))  # noqa: E731
    paths: list[Path] = []

    # 01 missingness ------------------------------------------------------------------
    if prof:
        missing = pd.Series({c: i["missing_frac"] for c, i in prof["columns"].items()}).sort_values(ascending=False)
    else:
        missing = df.isna().mean().sort_values(ascending=False)
    shown = missing.head(40)
    f, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(shown))))
    ax.barh(shown.index.astype(str)[::-1], shown.to_numpy()[::-1])
    ax.set_xlabel("missing fraction")
    ax.set_title("Missing values per column" + (f" (worst 40 of {len(missing)})" if len(missing) > 40 else ""))
    n_missing = int((missing > 0).sum())
    text = (f"{n_missing} of {len(missing)} columns have missing values; worst is {missing.index[0]} at {missing.iloc[0]:.1%}."
            if n_missing else f"None of the {len(missing)} columns has missing values.")
    if len(missing) > 40 and n_missing > 40:
        text += " The plot shows the worst 40; every column's rate is in data_profile.json."
    paths.append(fig(1, "missingness", f, text, state_dir))

    # 02 target -----------------------------------------------------------------------
    y = df[target]
    f, ax = plt.subplots(figsize=(6, 4))
    if task == "regression":
        ax.hist(y.dropna().astype(float), bins=40)
        ax.set_title(f"Target {target} distribution")
        explanation = (f"{target} ranges {y.min():.3g} to {y.max():.3g} with median {y.median():.3g}{sampled_note}; "
                       "a mean-predictor baseline is the number to beat.")
    else:
        balance = y.astype(str).value_counts(normalize=True).head(30)
        ax.bar(balance.index, balance.to_numpy())
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"Target {target} balance")
        explanation = (f"{target}={balance.index[0]} is {balance.iloc[0]:.1%} of rows{sampled_note}, so a majority-class "
                       f"dummy scores {balance.iloc[0]:.1%}; the smallest class shown is {balance.min():.1%}.")
    paths.append(fig(2, "target_balance", f, explanation, state_dir))

    # 02 feature overview: every column in one picture ----------------------------------
    cols_all = [c for c in kinds if c != target]
    if cols_all:
        f, ax = plt.subplots(figsize=(8, 5))
        palette = {"numeric": "tab:blue", "categorical": "tab:orange", "bool": "tab:green", "datetime": "tab:purple",
                   "id": "tab:gray", "text": "tab:brown"}
        for kind, color in palette.items():
            members = [c for c in cols_all if kinds[c] == kind]
            if members:
                ax.scatter([float(missing.get(c, 0.0)) for c in members], [signal.get(c, 0.0) for c in members],
                           s=14, alpha=0.7, color=color, label=f"{kind} ({len(members)})")
        for c in ranked(cols_all)[:10]:
            ax.annotate(str(c)[:18], (float(missing.get(c, 0.0)), signal.get(c, 0.0)), fontsize=7)
        ax.set_xlabel("missing fraction")
        ax.set_ylabel("target signal (correlation ratio)")
        ax.set_title(f"All {len(cols_all)} columns: missingness vs. target signal")
        ax.legend(fontsize=7)
        top = ranked(cols_all)[:3]
        paths.append(fig(2, "feature_overview", f,
                         f"Every one of the {len(cols_all)} columns as a dot; the strongest target signal is in "
                         f"{', '.join(map(str, top))}. Dots near the top are worth a look and, if near 1.0, a leakage check.",
                         state_dir))

    # 02 distributions, paged by signal -------------------------------------------------
    numeric = ranked([c for c in cols_all if kinds[c] == "numeric"])
    pages = _pages(numeric, per_page, max_pages)
    for i, page in enumerate(pages, 1):
        f, axes = _grid(plt, len(page))
        for ax, col in zip(axes, page):
            ax.hist(df[col].dropna().astype(float), bins=30)
            ax.set_title(str(col)[:28], fontsize=8)
        skewed = [str(c) for c in page if abs(float(df[c].astype(float).skew() or 0)) > 1]
        name = "distributions" if len(pages) == 1 else f"distributions_p{i}"
        covered = f"columns {(i - 1) * per_page + 1}-{(i - 1) * per_page + len(page)} of {len(numeric)} numeric, ranked by target signal"
        rest = (f" The remaining {len(numeric) - per_page * max_pages} are summarized in data_profile.json (median, skew, outliers)."
                if i == len(pages) and len(numeric) > per_page * max_pages else "")
        paths.append(fig(2, name, f, f"Histograms of {covered}{sampled_note}; {len(skewed)} are strongly skewed "
                                     f"({', '.join(skewed[:5]) or 'none'}), which matters for scaling and outliers.{rest}", state_dir))

    # 02 feature vs target, paged ------------------------------------------------------
    yv, ylabel = _target_view(y, task)
    pages = _pages(numeric, per_page, max(1, max_pages // 2))
    for i, page in enumerate(pages, 1):
        f, axes = _grid(plt, len(page))
        for ax, col in zip(axes, page):
            x = df[col].astype(float)
            ok = x.notna() & yv.notna()
            try:
                bins = pd.qcut(x[ok], 10, duplicates="drop")
                means = yv[ok].groupby(bins, observed=True).mean()
                ax.plot([iv.mid for iv in means.index], means.to_numpy(), marker="o")
            except (ValueError, IndexError, TypeError):
                ax.text(0.5, 0.5, "constant", ha="center", transform=ax.transAxes)
            ax.set_title(f"{str(col)[:22]} ({signal.get(col, 0):.2f})", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=7)
        name = "feature_vs_target" if len(pages) == 1 else f"feature_vs_target_p{i}"
        paths.append(fig(2, name, f,
                         f"{ylabel} across deciles of {len(page)} numeric features (page {i} of {len(pages)}, strongest signal first)"
                         f"{sampled_note}; a steep or step-shaped line is a strong feature, a perfect step is a leakage suspect.",
                         state_dir))

    # 02 categorical vs target, paged -------------------------------------------------
    categorical = ranked([c for c in cols_all if kinds[c] in ("categorical", "bool")])
    pages = _pages(categorical, 9, max(1, max_pages // 2))
    for i, page in enumerate(pages, 1):
        f, axes = _grid(plt, len(page), ncols=3, height=3.0)
        for ax, col in zip(axes, page):
            levels = df[col].astype(str).where(df[col].notna(), "<missing>")
            top = levels.value_counts().head(12)
            rates = yv.groupby(levels).mean().reindex(top.index)
            ax.barh([str(v)[:16] for v in top.index[::-1]], rates.to_numpy()[::-1])
            ax.set_title(f"{str(col)[:22]} ({signal.get(col, 0):.2f})", fontsize=8)
            ax.tick_params(labelsize=7)
        name = "categorical_vs_target" if len(pages) == 1 else f"categorical_vs_target_p{i}"
        paths.append(fig(2, name, f,
                         f"{ylabel} for the 12 most common levels of {len(page)} categorical columns (page {i} of {len(pages)}, "
                         f"{len(categorical)} categorical in total){sampled_note}; very different bars mean the column "
                         "carries signal, a bar at exactly 0 or 1 for a common level is a leakage suspect.", state_dir))

    # 02 outliers -----------------------------------------------------------------------
    if numeric:
        if prof:
            frac = pd.Series({c: prof["columns"][c].get("outlier_frac", 0.0) for c in numeric if c in prof["columns"]})
        else:
            def _out(s):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
            frac = pd.Series({c: _out(df[c].astype(float).dropna()) for c in numeric})
        frac = frac.sort_values(ascending=False)
        shown = frac.head(40)
        f, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(shown))))
        ax.barh(shown.index.astype(str)[::-1], shown.to_numpy()[::-1])
        ax.set_xlabel("share of rows beyond 1.5 IQR")
        ax.set_title("Outliers per numeric column" + (f" (top 40 of {len(frac)})" if len(frac) > 40 else ""))
        heavy = frac[frac > 0.05]
        paths.append(fig(2, "outliers", f,
                         f"{len(heavy)} of {len(frac)} numeric columns have more than 5% outlying rows "
                         f"({', '.join(map(str, heavy.index[:5])) or 'none'}); decide at step 4 whether they are errors or real extremes.",
                         state_dir))

    # 02 correlations: top features, full matrix to CSV ----------------------------------------
    corr_cols = ranked(numeric)[:40]
    if corr_cols:
        with_target = df[corr_cols].astype(float).assign(**{"__target__": yv.to_numpy()})
        corr = with_target.corr().rename(index={"__target__": str(target)}, columns={"__target__": str(target)})
        f, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels([str(c)[:18] for c in corr.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels([str(c)[:18] for c in corr.columns], fontsize=6)
        f.colorbar(image)
        ax.set_title("Correlations" + (f" (top {len(corr_cols)} of {len(numeric)} by signal)" if len(numeric) > 40 else ""))
        ranked_corr = corr[str(target)].drop(str(target)).abs().sort_values(ascending=False)
        full_cols = ranked(numeric)[:500]
        rows = df[full_cols] if len(df) <= 20_000 else df[full_cols].sample(20_000, random_state=0)
        rows.astype(float).corr().round(3).to_csv(state_dir / "correlations.csv")
        redundant = int(((corr.drop(index=str(target), columns=str(target)).abs() > 0.9).to_numpy().sum() - len(corr_cols)) // 2)
        paths.append(fig(2, "correlations", f,
                         f"Strongest linear correlation with {target}: {ranked_corr.index[0]} ({ranked_corr.iloc[0]:.2f}); anything "
                         f"above 0.95 is a leakage suspect. {redundant} feature pairs here correlate above 0.9 (redundant). "
                         f"The matrix for {len(full_cols)} numeric columns is in correlations.csv.", state_dir))

    # 02 temporal coverage with target drift -------------------------------------------------
    tcol = time_col or next(iter(date_sources), None)
    if tcol is not None:
        t = df[tcol]
        months = t.dt.to_period("M")
        counts = months.value_counts().sort_index()
        rate = yv.groupby(months).mean().reindex(counts.index)
        f, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot([str(p) for p in counts.index], counts.to_numpy(), label="rows")
        ax.set_ylabel("rows per month")
        ax2 = ax.twinx()
        ax2.plot([str(p) for p in counts.index], rate.to_numpy(), color="tab:red", label=ylabel)
        ax2.set_ylabel(ylabel, color="tab:red")
        step = max(1, len(counts) // 24)
        ax.set_xticks(range(0, len(counts), step))
        ax.set_xticklabels([str(p) for p in counts.index][::step], rotation=90, fontsize=7)
        ax.set_title(f"Rows and {ylabel} per month over {tcol}")
        drift = float(rate.max() - rate.min()) if len(rate) > 1 else 0.0
        paths.append(fig(2, "temporal_coverage", f,
                         f"{tcol} spans {str(t.min())[:10]} to {str(t.max())[:10]} over {len(counts)} months; {ylabel} moves "
                         f"by {drift:.3g} across months{sampled_note}. A chronological split must hold out the latest period.",
                         state_dir))
    return paths


# ====================================================================== split (step 6)
def _time_indices(t: pd.Series, n_train: int, n_val: int, gap) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Chronological cut that never splits identical timestamps and leaves ``gap`` between parts."""
    order = np.argsort(t.to_numpy(), kind="mergesort")
    ts = t.to_numpy()[order]
    n = len(ts)

    def boundary(i: int) -> int:
        i = min(max(i, 0), n)
        while 0 < i < n and ts[i] == ts[i - 1]:  # move past ties so one timestamp stays on one side
            i += 1
        return i

    b1 = boundary(n_train)
    b2 = boundary(max(b1, n_train + n_val))
    train, val, test = order[:b1], order[b1:b2], order[b2:]
    report = {"ties_moved": int((b1 - n_train) + (b2 - n_train - n_val))}
    if gap:
        if isinstance(gap, (int, np.integer)):
            report["gap_rows_dropped"] = int(min(gap, len(val)) + min(gap, len(test)))
            val, test = val[int(gap):], test[int(gap):]
        else:
            delta = pd.Timedelta(gap)
            delta = delta.to_timedelta64()
            if len(train) and len(val):
                keep = t.to_numpy()[val] >= ts[b1 - 1] + delta
                report["gap_rows_dropped"] = int((~keep).sum())
                val = val[keep]
            if len(test) and b2 > 0:
                keep = t.to_numpy()[test] >= ts[b2 - 1] + delta
                report["gap_rows_dropped"] = report.get("gap_rows_dropped", 0) + int((~keep).sum())
                test = test[keep]
    return train, val, test, report


def _group_indices(groups: pd.Series, n_test: int, n_val: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Assign whole groups so each part lands near its target size, skipping groups that would overshoot."""
    counts = groups.value_counts()
    if len(counts) < 3:
        raise ValueError(f"group split needs at least 3 groups, found {len(counts)}")
    names = counts.index.to_numpy().copy()
    rng.shuffle(names)
    test_g, val_g, n_t, n_v = set(), set(), 0, 0
    for g in names:
        c = int(counts[g])
        if n_t < n_test and (n_t + c <= n_test * 1.25 or not test_g):
            test_g.add(g)
            n_t += c
        elif n_val and n_v < n_val and (n_v + c <= n_val * 1.25 or not val_g):
            val_g.add(g)
            n_v += c
    in_test = groups.isin(test_g).to_numpy()
    in_val = groups.isin(val_g).to_numpy()
    report = {"n_groups": int(len(counts)), "largest_group_rows": int(counts.iloc[0])}
    return np.flatnonzero(~in_test & ~in_val), np.flatnonzero(in_val), np.flatnonzero(in_test), report


def _stratified_indices(y: pd.Series, test_size: float, val_size: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train, val, test, rare = [], [], [], []
    labels = y.astype(str).to_numpy()
    for label, positions in pd.Series(np.arange(len(labels))).groupby(labels, sort=False).indices.items():
        perm = rng.permutation(positions)
        if len(perm) < 3:
            rare.append(str(label))
            train.extend(perm)
            continue
        k_test = max(1, int(round(len(perm) * test_size)))
        k_val = max(1 if val_size > 0 else 0, int(round(len(perm) * val_size)))
        test.extend(perm[:k_test])
        val.extend(perm[k_test:k_test + k_val])
        train.extend(perm[k_test + k_val:])
    report = {"rare_classes_kept_in_train": rare} if rare else {}
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int), report


def _guard_rails(df: pd.DataFrame, target: str, time_col, group_col, not_temporal_reason, not_grouped_reason) -> dict:
    """Refuse a split that would leak through time or through repeated entities, unless a reason is given."""
    dates = {c: s for c, s in datetime_columns(df.drop(columns=[target])).items()}
    if time_col is None and dates and not not_temporal_reason:
        raise TemporalSplitRequired(
            f"date column(s) {list(dates)} present (stored as: {', '.join(set(dates.values()))}): pass time_col=... so the "
            "split is chronological, or not_temporal_reason='...' after the user agreed the dates don't matter")
    if group_col is None and time_col is None and not not_grouped_reason:
        kinds = {c: _column_kind(df[c], dates.get(str(c))) for c in df.columns if c != target}
        repeated = [e for e in _entity_candidates(df, kinds, exclude=(target, time_col)) if e["repeated"]]
        if repeated:
            names = ", ".join(f"{e['column']} ({e['rows_per_entity']} rows each)" for e in repeated[:5])
            raise GroupSplitRequired(
                f"rows repeat an entity: {names}. Pass group_col=... so one entity never sits in two splits, "
                "or not_grouped_reason='...' after the user agreed these rows are independent")
    return dates


def split(
    df: pd.DataFrame,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    gap=None,
    seed: int = 0,
    not_temporal_reason: str | None = None,
    not_grouped_reason: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an in-memory frame into (train, val, test) and freeze the test set. Step 6.

    Refuses a random split when any date column exists (native, text, or epoch numbers) and a
    row-level split when an entity column repeats, unless the user-agreed reason is passed.
    ``gap`` (rows, or a time span like "7D" for time splits) leaves an embargo between parts.
    For data that does not fit in memory use ``split_to_files``.
    """
    if not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be in (0, 1) and sum to less than 1")
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    _no_missing_target(df[target])
    dates = _guard_rails(df, target, time_col, group_col, not_temporal_reason, not_grouped_reason)

    hashes = _row_hashes(df)
    keep = ~pd.Series(hashes).duplicated().to_numpy()
    dropped = int((~keep).sum())
    if dropped:
        df, hashes = df.iloc[np.flatnonzero(keep)].reset_index(drop=True), hashes[keep]
    n = len(df)
    n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
    n_train = n - n_val - n_test
    rng = np.random.default_rng(seed)

    if time_col is not None:
        t = as_datetime(df[time_col], dates.get(str(time_col)) or _datetime_source(df[time_col]))
        if t.isna().any():
            raise ValueError(f"time_col {time_col!r} has {int(t.isna().sum())} missing/unparseable values; clean them at step 4")
        kind, (*parts, report) = "time", _time_indices(t, n_train, n_val, gap)
    elif group_col is not None:
        kind, (*parts, report) = "group", _group_indices(df[group_col], n_test, n_val, rng)
    elif df[target].nunique() <= STRATIFY_MAX_CLASSES:
        kind, (*parts, report) = "stratified", _stratified_indices(df[target], test_size, val_size, rng)
    else:
        perm = rng.permutation(n)
        kind, parts, report = "random", [perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]], {}

    parts = [np.sort(p) for p in parts]
    for i in range(3):
        for j in range(i + 1, 3):
            if len(np.intersect1d(_sorted_unique(hashes[parts[i]]), _sorted_unique(hashes[parts[j]]), assume_unique=True)):
                raise OverlapLeakage(f"identical rows in split {i} and split {j}")
    train, val, test = (df.iloc[p].reset_index(drop=True) for p in parts)
    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    warnings_out = []
    if n and abs(len(test) / n - test_size) > max(0.05, test_size * 0.5):
        warnings_out.append(f"test is {len(test) / n:.0%} of rows, asked {test_size:.0%} (large groups or ties)")
    _write_state(state_dir, {
        "fingerprint_version": 2,
        "test_fingerprint": _fingerprint(test),
        "train_fingerprint": _fingerprint(train),
        "val_fingerprint": _fingerprint(val),
        "n_test": int(len(test)),
        "test_touches": 0,
        "duplicates_dropped": dropped,
        "split": kind,
        "sizes": sizes,
        "params": {"target": target, "time_col": time_col, "group_col": group_col, "test_size": test_size,
                   "val_size": val_size, "gap": str(gap) if gap is not None else None, "seed": seed,
                   "not_temporal_reason": not_temporal_reason, "not_grouped_reason": not_grouped_reason},
        "report": report,
        "warnings": warnings_out,
        "created": date.today().isoformat(),
    })
    return train, val, test


def _no_missing_target(y: pd.Series) -> None:
    missing = int(y.isna().sum())
    if missing:
        raise ValueError(f"{missing} row(s) have no target value; drop or label them at step 4 (and record the rule) "
                         "before splitting - they can't be stratified, scored, or silently dropped")


def _write_state(state_dir, state: dict) -> None:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE).write_text(json.dumps(state, indent=2, default=str))


def _u01(hashes: np.ndarray) -> np.ndarray:
    return (hashes >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def split_to_files(
    source,
    *,
    target: str,
    out_dir: str | Path = "ml_pipeline/splits",
    time_col: str | None = None,
    group_col: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    gap=None,
    seed: int = 0,
    not_temporal_reason: str | None = None,
    not_grouped_reason: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict[str, Path]:
    """Out-of-core step 6: stream ``source`` into train/val/test Parquet files and freeze the test file.

    Rows are assigned by a hash, so identical rows always land in the same split (no overlap by
    construction), whole groups stay together, and time splits cut at exact timestamp boundaries
    computed from every row. Needs pyarrow.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not (0 < test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be in (0, 1) and sum to less than 1")
    sample, n_total = load_sample(source)
    if target not in sample.columns:
        raise KeyError(f"target column {target!r} not in data")
    dates = _guard_rails(sample, target, time_col, group_col, not_temporal_reason, not_grouped_reason)
    salt = np.uint64(seed * 0x9E3779B97F4A7C15 % (1 << 64))

    cut_train = cut_val = None
    delta = pd.Timedelta(gap) if gap is not None and not isinstance(gap, (int, np.integer)) else None
    t_src = dates.get(str(time_col)) or (_datetime_source(sample[time_col]) if time_col else None)
    if time_col is not None:
        times = np.sort(np.concatenate([as_datetime(b[time_col], t_src).to_numpy(dtype="datetime64[ns]")
                                        for b in scan(source, columns=[time_col])]))
        if np.isnat(times).any():
            raise ValueError(f"time_col {time_col!r} has missing/unparseable values; clean them at step 4")
        n = len(times)
        cut_train = times[min(max(int(round(n * (1 - test_size - val_size))) - 1, 0), n - 1)]
        cut_val = times[min(max(int(round(n * (1 - test_size))) - 1, 0), n - 1)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")
    paths = {name: out_dir / f"{name}.parquet" for name in names}
    parts_dir = out_dir / ".parts"
    parts: dict = {name: [] for name in names}
    counts = dict.fromkeys(names, 0)
    dropped_gap = missing_target = 0
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    try:
        for i, batch in enumerate(scan(source)):
            missing_target += int(batch[target].isna().sum())
            if time_col is not None:
                t = as_datetime(batch[time_col], t_src).to_numpy(dtype="datetime64[ns]")
                part = np.where(t <= cut_train, 0, np.where(t <= cut_val, 1, 2))
                if delta is not None:
                    in_gap = ((part == 1) & (t < cut_train + delta.to_timedelta64())) | \
                             ((part == 2) & (t < cut_val + delta.to_timedelta64()))
                    dropped_gap += int(in_gap.sum())
                    part = np.where(in_gap, -1, part)
            else:
                key = (pd.util.hash_pandas_object(batch[group_col], index=False) if group_col is not None
                       else pd.util.hash_pandas_object(batch, index=False)).to_numpy(dtype=np.uint64) ^ salt
                u = _u01(key)
                part = np.where(u < test_size, 2, np.where(u < test_size + val_size, 1, 0))
            for code, name in enumerate(names):
                mask = part == code
                if mask.any():
                    piece = parts_dir / f"{name}-{i:06d}.parquet"
                    pq.write_table(_null_typed(pa.Table.from_pandas(batch[mask], preserve_index=False)), str(piece))
                    parts[name].append(piece)
                    counts[name] += int(mask.sum())
        if missing_target:
            raise ValueError(f"{missing_target} row(s) have no target value; drop or label them at step 4 before splitting")
        # one schema for every part: a column that is all-null early and text later is text, ints mixed with floats are floats
        schemas = [pq.read_schema(str(p)) for name in names for p in parts[name]]
        if not schemas:
            raise ValueError("the data source has no rows")
        schema = _unified_schema(schemas)
        for name in names:
            with pq.ParquetWriter(str(paths[name]), schema) as writer:
                for piece in parts[name]:
                    writer.write_table(pq.read_table(str(piece)).cast(schema))
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
    test_fp, n_test = _fingerprint_source(paths["test"])
    kind = "time" if time_col is not None else "group" if group_col is not None else "hashed"
    _write_state(state_dir, {
        "fingerprint_version": 2,
        "test_fingerprint": test_fp,
        "n_test": int(n_test),
        "test_touches": 0,
        "duplicates_dropped": 0,
        "split": kind,
        "files": {k: str(v) for k, v in paths.items()},
        "sizes": counts,
        "params": {"target": target, "time_col": time_col, "group_col": group_col, "test_size": test_size,
                   "val_size": val_size, "gap": str(gap) if gap is not None else None, "seed": seed,
                   "not_temporal_reason": not_temporal_reason, "not_grouped_reason": not_grouped_reason},
        "report": {"gap_rows_dropped": dropped_gap, "note": "identical rows share a split by construction"},
        "created": date.today().isoformat(),
    })
    return paths


# ====================================================================== cross-validation folds (steps 10-11)
def _null_typed(table):
    """Columns that are entirely null in this batch get the null type, so a later batch decides their real type."""
    import pyarrow as pa
    for i, field in enumerate(table.schema):
        if table.num_rows and table.column(i).null_count == table.num_rows and not pa.types.is_null(field.type):
            table = table.set_column(i, pa.field(field.name, pa.null()), pa.nulls(table.num_rows))
    return table


def _unified_schema(schemas: list):
    """One schema for all parts: permissive promotion, and text for a column that is numbers in some parts and text in others."""
    import pyarrow as pa
    try:
        return pa.unify_schemas(schemas, promote_options="permissive")
    except (pa.ArrowTypeError, pa.ArrowInvalid):
        fields = {}
        for schema in schemas:
            for field in schema:
                fields.setdefault(field.name, []).append(field.type)
        out = []
        for name, types in fields.items():
            try:
                merged = pa.unify_schemas([pa.schema([pa.field(name, t)]) for t in types], promote_options="permissive")
                out.append(merged.field(name))
            except (pa.ArrowTypeError, pa.ArrowInvalid):
                out.append(pa.field(name, pa.large_string()))
        return pa.schema(out)


def cv_splits(
    train: pd.DataFrame,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    gap: int = 0,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leakage-safe CV folds over the TRAIN split, as positional (fit_idx, score_idx) pairs.

    Time: expanding window with ``gap`` rows of embargo. Groups: balanced group k-fold. Otherwise
    stratified k-fold (classification) or k-fold, repeated ``n_repeats`` times. Pass the list
    straight to scikit-learn as ``cv=folds``.
    """
    n = len(train)
    _no_missing_target(train[target])
    if n_splits < 2 or n < n_splits + 1:
        raise ValueError("need n_splits >= 2 and more rows than folds")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if time_col is not None:
        order = np.argsort(as_datetime(train[time_col]).to_numpy(), kind="mergesort")
        size = n // (n_splits + 1)
        for k in range(1, n_splits + 1):
            fit_end = size * k
            folds.append((np.sort(order[:max(fit_end - gap, 1)]), np.sort(order[fit_end:fit_end + size])))
        return folds
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        if group_col is not None:
            counts = train[group_col].value_counts()
            names = counts.index.to_numpy().copy()
            rng.shuffle(names)
            load = np.zeros(n_splits)
            assign = {}
            for g in sorted(names, key=lambda g: -counts[g]):  # largest first into the lightest fold
                k = int(np.argmin(load))
                assign[g] = k
                load[k] += counts[g]
            fold_of = train[group_col].map(assign).to_numpy()
        elif train[target].nunique() <= STRATIFY_MAX_CLASSES:
            fold_of = np.empty(n, dtype=int)
            for positions in train[target].astype(str).groupby(train[target].astype(str)).indices.values():
                perm = rng.permutation(positions)
                fold_of[perm] = np.arange(len(perm)) % n_splits
        else:
            fold_of = rng.permutation(n) % n_splits
        for k in range(n_splits):
            folds.append((np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)))
    return folds


# ====================================================================== approvals and overrides
_NEGATED = re.compile(r"\b(?:not|pending|awaiting|todo)\b", re.I)


def _approvals(state_dir: Path) -> dict | None:
    path = Path(state_dir) / APPROVALS_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def _overrides_granted(state_dir: Path, pipeline_md: Path, what: str) -> int:
    """How many user-approved overrides of ``what`` exist (from the hook's record, else the ledger)."""
    record = _approvals(state_dir)
    if record is not None:
        return sum(1 for o in record.get("overrides", []) if str(o.get("what", "")).lower() == what)
    if not pipeline_md.is_file():
        return 0
    line = re.compile(r"^.*\boverrid(?:e|den)\b.*\b" + re.escape(what) + r"\b.*$", re.I | re.M)
    return sum(1 for m in line.finditer(pipeline_md.read_text()) if not _NEGATED.search(m.group(0)))


# ====================================================================== final test (step 14)
_FINAL_LINE = re.compile(r"^\s*-\s*Step 14 final test:", re.I | re.M)


def final_test(
    predict_fn,
    test,
    *,
    target: str,
    metric_fn,
    state_dir: str | Path = "ml_pipeline",
    pipeline_md: str | Path = "ml_pipeline/PIPELINE.md",
    batch_rows: int | None = None,
) -> float:
    """Step 14: score ``predict_fn`` on the frozen test set, once, and log the number.

    ``test`` is the frame ``split`` returned or the test file ``split_to_files`` wrote; it is
    predicted in batches of ``batch_rows`` so a large test set never has to be passed at once.
    Every further call needs one more user-approved ``final test`` override.
    """
    state_dir = Path(state_dir)
    state_path = state_dir / STATE_FILE
    pipeline_md = Path(pipeline_md)
    if not state_path.is_file():
        raise LeakageError("no frozen test set: call guard.split() or guard.split_to_files() at step 6 first")
    state = json.loads(state_path.read_text())
    if state.get("fingerprint_version", 1) >= 2:
        fingerprint, _ = _fingerprint_source(test)
    else:
        frame = test if isinstance(test, pd.DataFrame) else pd.concat(list(scan(test)), ignore_index=True)
        fingerprint = _fingerprint_v1(frame)
    if fingerprint != state["test_fingerprint"]:
        raise TestSetTampered("this is not the test set frozen at step 6")
    record = _approvals(state_dir)
    if record is not None and "C" not in (record.get("gates") or {}) and not any(
            str(o.get("what", "")).lower() == "gate c" for o in record.get("overrides", [])):
        raise LeakageError("the user has not approved Gate C yet; the test set is scored only after that approval")
    ledgers = {pipeline_md.resolve(), (state_dir / "PIPELINE.md").resolve()}
    logged = max((len(_FINAL_LINE.findall(p.read_text())) for p in ledgers if p.is_file()), default=0)
    touches = max(int(state.get("test_touches", 0)), logged)
    if touches >= 1 + _overrides_granted(state_dir, pipeline_md, "final test"):
        raise TestSetAlreadyUsed(
            "the test set has already been evaluated; agree a new test strategy with the user, who records the "
            "override by saying 'override final test' in chat, before evaluating again")

    y_parts, p_parts = [], []
    for batch in scan(test, batch_rows=batch_rows):
        y_parts.append(batch[target].to_numpy())
        p_parts.append(np.asarray(predict_fn(batch.drop(columns=[target]))))
    score = float(metric_fn(np.concatenate(y_parts), np.concatenate(p_parts)))

    state["test_touches"] = touches + 1
    state_path.write_text(json.dumps(state, indent=2, default=str))
    name = getattr(metric_fn, "__name__", "metric")
    line = f"- Step 14 final test: {name}={score:.4f} (touch {state['test_touches']}, {date.today().isoformat()})\n"
    pipeline_md.parent.mkdir(parents=True, exist_ok=True)
    existing = pipeline_md.read_text().rstrip("\n") + "\n" if pipeline_md.is_file() else ""
    pipeline_md.write_text(existing + line)
    return score


def status(state_dir: str | Path = "ml_pipeline") -> dict:
    """Read-only view of the split state, approvals, and run count (the state files are not to be edited)."""
    state_dir = Path(state_dir)
    out: dict = {}
    if (state_dir / STATE_FILE).is_file():
        state = json.loads((state_dir / STATE_FILE).read_text())
        out["split"] = {k: state.get(k) for k in ("split", "sizes", "test_touches", "duplicates_dropped", "warnings", "report")}
    out["approvals"] = _approvals(state_dir)
    runs = state_dir / RUNS_FILE
    out["runs"] = sum(1 for _ in runs.open()) if runs.is_file() else 0
    return out


# ====================================================================== run log and environment (steps 9-12)
_PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "torch", "tensorflow", "pyarrow",
             "polars", "statsmodels")


def environment() -> dict:
    """Python, platform, and installed versions of the usual ML packages (no imports, just metadata)."""
    from importlib import metadata
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def _git_commit(start: Path) -> str | None:
    here = start.resolve()
    for _ in range(6):
        head = here / ".git" / "HEAD"
        if head.is_file():
            ref = head.read_text().strip()
            if ref.startswith("ref: "):
                target = here / ".git" / ref[5:]
                return target.read_text().strip()[:12] if target.is_file() else None
            return ref[:12]
        if here.parent == here:
            break
        here = here.parent
    return None


def log_run(
    name: str,
    *,
    kind: str = "candidate",
    metrics: dict,
    params: dict | None = None,
    model=None,
    notes: str = "",
    seed: int | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict:
    """Append one training/tuning run to ml_pipeline/runs.jsonl with what is needed to reproduce it.

    ``kind`` is 'baseline' (step 9), 'candidate' (step 10), 'tuned' (step 11) or 'final'. The
    hook requires a baseline run before step 9 is recorded and a candidate before step 10.
    """
    if kind not in ("baseline", "candidate", "tuned", "final"):
        raise ValueError("kind must be baseline, candidate, tuned, or final")
    if not metrics:
        raise ValueError("log the validation metrics the run was judged on")
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if params is None and model is not None and hasattr(model, "get_params"):
        try:
            params = {k: repr(v)[:200] for k, v in model.get_params().items()}
        except Exception:  # noqa: BLE001 - params are best effort
            params = None
    state = json.loads((state_dir / STATE_FILE).read_text()) if (state_dir / STATE_FILE).is_file() else {}
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name, "kind": kind,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in metrics.items()},
        "params": params, "seed": seed, "notes": notes,
        "model_class": type(model).__name__ if model is not None else None,
        "data": {k: state.get(k) for k in ("train_fingerprint", "val_fingerprint", "split", "sizes")},
        "git_commit": _git_commit(state_dir.parent),
        "environment": environment(),
    }
    with (state_dir / RUNS_FILE).open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def runs(state_dir: str | Path = "ml_pipeline") -> pd.DataFrame:
    """The run log as a table: one row per run with its metrics flattened."""
    path = Path(state_dir) / RUNS_FILE
    if not path.is_file():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.json_normalize(rows, sep=".").drop(columns=["environment.packages"], errors="ignore")


# ====================================================================== manifests for images and text (step 1)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".dcm"}
TEXT_EXT = {".txt", ".md", ".json", ".html", ".xml", ".csv"}


def manifest(
    root,
    *,
    kind: str = "auto",
    label_from="parent",
    group_pattern: str | None = None,
    extensions=None,
    state_dir: str | Path = "ml_pipeline",
) -> pd.DataFrame:
    """Step 1 for image/text data: one row per file (path, label, size, content hash, group) -> manifest.csv.

    ``label_from`` is 'parent' (folder name), None, or a function path -> label. ``group_pattern`` is a
    regex whose first group names the source (patient, device, session) in the file path; files
    without a match are grouped by content hash, so exact duplicates always share a split.
    Profile and split the manifest like any table, with group_col='group'.
    """
    root = Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    if extensions is None:
        extensions = IMAGE_EXT if kind == "image" else TEXT_EXT if kind == "text" else IMAGE_EXT | TEXT_EXT
    extensions = {e.lower() for e in extensions}
    pattern = re.compile(group_pattern) if group_pattern else None
    try:
        from PIL import Image  # noqa: N811 - optional
    except ImportError:
        Image = None
    rows = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in extensions:
            continue
        data = path.read_bytes()
        digest = hashlib.md5(data).hexdigest()  # noqa: S324 - content identity, not security
        rel = str(path.relative_to(root)) if root.is_dir() else path.name
        label = path.parent.name if label_from == "parent" else label_from(path) if callable(label_from) else None
        match = pattern.search(rel) if pattern else None
        row = {"path": rel, "label": label, "bytes": len(data), "md5": digest,
               "group": match.group(1) if match else digest}
        if suffix in IMAGE_EXT:
            row["width"] = row["height"] = None
            if Image is not None and suffix != ".dcm":
                try:
                    with Image.open(path) as im:
                        row["width"], row["height"] = im.size
                except Exception:  # noqa: BLE001 - unreadable image stays in the manifest, flagged by size
                    pass
        else:
            text = data.decode("utf-8", errors="replace")
            row["n_chars"], row["n_words"] = len(text), len(text.split())
        rows.append(row)
    out = pd.DataFrame(rows)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(state_dir / "manifest.csv", index=False)
    return out
