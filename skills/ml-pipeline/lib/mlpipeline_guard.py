"""Runtime guard for the ml-pipeline skill: data profile, figures, leakage-safe splits, a single-use
test set, cross-validation folds, a run log, and manifests for image/text data. Also the per-column
records the gates check (source receipt, data dictionary, missingness, label audit, cleaning log,
categorical plan, train-only associations, training receipts, validation importance).

Copy this file into your project as ``ml_pipeline/guard.py`` (the skill tells the agent where the
canonical copy lives). Requires pandas and numpy; pyarrow is used when present for Parquet/Arrow
input and output; matplotlib is needed for figures.

    df = guard.load("data.csv")                                          # step 1: whole file, hash + rows recorded
    p = guard.profile("data.parquet", target="y", time_col="date")     # step 1 (file or DataFrame)
    guard.data_dictionary(); guard.missingness("data.parquet"); guard.label_audit("data.parquet", target="y")
    guard.eda_figures("data.parquet", target="y", time_col="date")      # step 2 (every column, on pages)
    train, val, test = guard.split(df, target="y", time_col="date")     # step 6, fits in memory
    paths = guard.split_to_files("big.parquet", target="y", time_col="date")  # step 6, out of core
    folds = guard.cv_splits(train, target="y", time_col="date")         # steps 10-11
    guard.train_associations(train, target="y")                        # step 7, training split only
    guard.fit(model, X_train, y_train)                                  # steps 9-10, receipt of a full-data run
    guard.log_run("baseline", kind="baseline", metrics={"ap": 0.31})    # steps 9-12
    guard.importance(model.predict, val, target="y", metric_fn=f1)      # step 12, validation split
    score = guard.final_test(model.predict, test, target="y", metric_fn=acc)  # step 14, once

Every function works on a pandas DataFrame and, where it matters for size, on a path (Parquet,
CSV/TSV, JSON lines, Feather), a pyarrow Table, or a polars DataFrame, read in batches so the data
never has to fit in memory at once. Counts and moments are computed over every row (missingness,
distinct counts, duplicates, min/max, mean/std/skew/kurtosis, zeros, negatives, outliers, level and
class counts); percentiles and figures use a uniform random sample of ``SAMPLE_ROWS`` rows when the
data is larger, and say so. Target statistics are computed on the training split only (step 7);
before the split the target is used only to look for leaks.
"""
from __future__ import annotations

import csv
import hashlib
import io
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
SOURCE_FILE = "source.json"

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
RARE_LEVEL_FRAC = 0.01         # a level under 1% of non-null rows is rare
LEVEL_COUNT_CAP = 10_000       # exact level counts per column up to this many levels, sample above
COLLINEAR_RHO = 0.9
MIN_NOTE_CHARS = 40
PAGE_COLUMNS = 12


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


class PartialData(LeakageError):
    """A load or profile that does not cover the full data."""


class NotTrainingSplit(LeakageError):
    """Target statistics were requested on data that is not the frozen training split."""


class NotValidationSplit(LeakageError):
    """Importance was requested on data that is not the frozen validation split."""


# ====================================================================== the ledger (PIPELINE.md)
_RUN_HEAD = re.compile(r"^[ \t]*##[ \t]+Run[ \t]*:", re.I | re.M)


def _current_run(markdown: str) -> str:
    """The newest run: from the last ``## Run:`` heading on, or the whole ledger (the hook's rule)."""
    heads = list(_RUN_HEAD.finditer(markdown))
    return markdown[heads[-1].start():] if heads else markdown


def _override_lines(pipeline_md: Path, what: str, *, current_run: bool) -> int:
    """How many strict ``- Override: <what> - ...`` lines the ledger (or only its newest run) holds."""
    if not Path(pipeline_md).is_file():
        return 0
    text = Path(pipeline_md).read_text()
    line = re.compile(r"^[ \t]*-[ \t]*Override[ \t]*:[ \t]*" + re.escape(what) + r"[ \t]*(?:-|$)", re.I | re.M)
    return len(line.findall(_current_run(text) if current_run else text))


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


# ====================================================================== sources (step 1): whole files, proven
_READERS = {
    ".csv": ("read_csv", {}), ".txt": ("read_csv", {}), ".tsv": ("read_csv", {"sep": "\t"}),
    ".parquet": ("read_parquet", {}), ".pq": ("read_parquet", {}),
    ".xlsx": ("read_excel", {}), ".xls": ("read_excel", {}),
    ".json": ("read_json", {}), ".jsonl": ("read_json", {"lines": True}), ".ndjson": ("read_json", {"lines": True}),
    ".feather": ("read_feather", {}),
}
_PARTIAL_KWARGS = ("nrows", "skiprows", "skipfooter", "chunksize", "iterator", "usecols", "columns", "filters")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():  # a Parquet dataset directory: hash its files in name order
        for part in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(part.relative_to(path)).encode())
            digest.update(_sha256(part).encode())
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sources(state_dir: Path) -> dict:
    path = Path(state_dir) / SOURCE_FILE
    return json.loads(path.read_text()) if path.is_file() else {"schema_version": "1", "sources": {}}


def _record_source(state_dir: Path, name: str, entry: dict) -> None:
    state_dir = Path(state_dir)
    data = _sources(state_dir)
    data["sources"][name] = entry
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / SOURCE_FILE).write_text(json.dumps(data, indent=2))


def _file_entry(path: Path, n_rows: int, columns) -> dict:
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
    return {"kind": "file", "sha256": _sha256(path), "bytes": int(size), "n_rows": int(n_rows),
            "n_cols": len(list(columns)), "columns": [str(c) for c in columns], "loaded": date.today().isoformat()}


def load(path, *, state_dir: str | Path = "ml_pipeline", **read_kwargs) -> pd.DataFrame:
    """Step 1: read a whole data file and record its hash, rows, and columns in source.json.

    A slice of rows or columns is a pilot, and a column dropped before inspection is a column never
    inspected: nrows, skiprows, usecols, columns, filters, chunks are refused (give unwanted columns role
    'drop' in the data dictionary instead). For files larger than memory pass the path to guard.profile().
    """
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
    _record_source(Path(state_dir), str(path), _file_entry(path, len(df), df.columns))
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


def _source_entry(source, columns, n_rows: int, name: str | None, state_dir: Path) -> dict:
    """Whether the profiled data is a whole source. A file path profiled here is recorded and is the full data."""
    if _is_path(source):
        path = Path(source)
        _record_source(state_dir, str(path), _file_entry(path, n_rows, columns))
        return {"name": str(path), "n_rows": int(n_rows), "full_data": True}
    sources = _sources(state_dir)["sources"]
    wanted = {str(c) for c in columns}
    for key in ([name] if name is not None else list(sources)):
        entry = sources.get(key)
        if entry and entry["n_rows"] == n_rows and set(entry["columns"]) == wanted:
            return {"name": key, "n_rows": int(n_rows), "full_data": True}
    if not sources:
        reason = "no source loaded with guard.load() or profiled from a file path; the profile cannot be tied to the full data"
    elif name is not None and name not in sources:
        reason = f"source {name!r} is not in source.json"
    else:
        reason = f"{n_rows:,} rows x {len(wanted)} columns matches no loaded source (a sample or a slice?)"
    return {"name": name, "n_rows": int(n_rows), "full_data": False, "reason": reason}


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
    """Shape statistics from ``sample`` (every row when the data fits in SAMPLE_ROWS): percentiles, levels, lengths.

    The full pass replaces every count and moment here with its exact value over all rows.
    """
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
        v = values.astype(float)
        q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = q[0.75] - q[0.25]
        outliers = ((v < q[0.25] - 1.5 * iqr) | (v > q[0.75] + 1.5 * iqr)).mean() if iqr > 0 else 0.0
        info.update(median=_num(q[0.5]), mean=_num(v.mean()), std=_num(v.std()),
                    skew=_num(v.skew()) if len(v) > 2 else 0.0, kurtosis=_num(v.kurt()) if len(v) > 3 else 0.0,
                    outlier_frac=round(float(outliers), 4), p01=_num(q[0.01]), p05=_num(q[0.05]), p25=_num(q[0.25]),
                    p75=_num(q[0.75]), p95=_num(q[0.95]), p99=_num(q[0.99]))
    elif kind in ("categorical", "bool"):
        counts = values.astype(str).value_counts()
        info.update(n_levels_in_sample=int(len(counts)), **_level_stats(counts, len(values)))
    elif kind == "text":
        lengths = values.astype(str).str.len()
        info.update(median_length=int(lengths.median()), length_median=int(lengths.median()),
                    length_min=int(lengths.min()), length_max=int(lengths.max()))
    return info


def _level_stats(counts: pd.Series, n_values: int) -> dict:
    """n_levels, the top 10 levels as fractions and counts, and how many levels are rare."""
    counts = counts.sort_values(ascending=False, kind="mergesort")
    top = counts.head(10)
    return {"n_levels": int(len(counts)), "top": {str(k): round(float(c) / max(n_values, 1), 4) for k, c in top.items()},
            "top_counts": {str(k): int(c) for k, c in top.items()},
            "rare_levels": int((counts / max(n_values, 1) < RARE_LEVEL_FRAC).sum())}


def _target_info(y: pd.Series, counts: pd.Series | None = None) -> dict:
    """Task, class balance and exact class counts (``counts`` from the full pass when given)."""
    values = y.dropna()
    n_unique = int(values.nunique())
    missing_frac = round(float(y.isna().mean()), 4)
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and n_unique > 20:
        v = values.astype(float)
        return {"name": str(y.name), "task": "regression", "n_classes": None, "class_balance": None,
                "minority_frac": None, "class_counts": None, "missing_frac": missing_frac,
                "mean": _num(v.mean()), "std": _num(v.std()), "min": _num(v.min()), "max": _num(v.max())}
    if counts is None:
        counts = values.astype(str).value_counts()
    counts = counts.sort_values(ascending=False, kind="mergesort")
    balance = counts / max(int(counts.sum()), 1)
    return {"name": str(y.name), "task": "binary" if len(counts) == 2 else "multiclass", "n_classes": int(len(counts)),
            "class_balance": {str(k): round(float(v), 4) for k, v in balance.head(50).items()},
            "minority_frac": round(float(balance.min()), 4) if len(balance) else None,
            "class_counts": {str(k): int(c) for k, c in counts.items()},
            "missing_frac": missing_frac}


class _Moments:
    """Exact count, sum and central moments of a numeric column, batch by batch (shifted for stability)."""

    def __init__(self, shift: float):
        self.shift, self.n, self.s1, self.s2, self.s3, self.s4 = shift, 0, 0.0, 0.0, 0.0, 0.0

    def add(self, x: np.ndarray) -> None:
        d = x - self.shift
        self.n += len(d)
        self.s1 += float(d.sum())
        self.s2 += float((d ** 2).sum())
        self.s3 += float((d ** 3).sum())
        self.s4 += float((d ** 4).sum())

    def stats(self) -> dict:
        n = self.n
        if not n:
            return {}
        m = self.s1 / n
        m2 = max(self.s2 / n - m ** 2, 0.0)
        m3 = self.s3 / n - 3 * m * self.s2 / n + 2 * m ** 3
        m4 = self.s4 / n - 4 * m * self.s3 / n + 6 * m ** 2 * self.s2 / n - 3 * m ** 4
        out = {"mean": _num(m + self.shift), "std": _num(math.sqrt(m2 * n / (n - 1))) if n > 1 else 0.0}
        if n > 2 and m2 > 0:  # the bias-adjusted estimators pandas uses for skew() and kurt()
            g1 = m3 / m2 ** 1.5
            out["skew"] = _num(g1 * math.sqrt(n * (n - 1)) / (n - 2))
            if n > 3:
                g2 = m4 / m2 ** 2 - 3
                out["kurtosis"] = _num((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6))
            else:
                out["kurtosis"] = 0.0
        else:
            out.update(skew=0.0, kurtosis=0.0)
        return out


def _full_pass(source, kinds: dict, target: str, id_cols: list[str], sample_stats: dict, date_sources: dict) -> dict:
    """Every-row statistics in one streaming pass.

    Missingness, distinct counts, min/max, duplicates and label conflicts, plus per column: exact moments,
    zeros, negatives and outliers (against the sample's quartile fences) for numbers; exact level counts up
    to LEVEL_COUNT_CAP levels; text lengths; date range; and the target's exact class counts.
    """
    cols = list(kinds)
    missing = dict.fromkeys(cols, 0)
    distinct = {c: _KMV() for c in cols}
    lo, hi = {}, {}
    moments: dict = {}
    counts_num: dict = {}
    levels: dict = {}
    lengths: dict = {}
    dates: dict = {}
    fences = {}
    for c, k in kinds.items():
        info = sample_stats.get(c, {})
        if k == "numeric":
            moments[c] = _Moments(float(info.get("median") or 0.0))
            counts_num[c] = {"zeros": 0, "negatives": 0, "outliers": 0, "finite": 0}
            q1, q3 = info.get("p25"), info.get("p75")
            if q1 is not None and q3 is not None and q3 > q1:
                fences[c] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        if k in ("categorical", "bool") or c == target:
            levels[c] = pd.Series(dtype="int64")
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
            if not len(present):
                continue
            distinct[c].add(pd.util.hash_pandas_object(present, index=False).to_numpy(dtype=np.uint64))
            kind = kinds[c]
            if kind == "numeric":
                vals = present.astype(float).to_numpy()
                lo[c] = min(lo.get(c, np.inf), float(np.nanmin(vals)))
                hi[c] = max(hi.get(c, -np.inf), float(np.nanmax(vals)))
                finite = vals[np.isfinite(vals)]
                moments[c].add(finite)
                tally = counts_num[c]
                tally["finite"] += len(finite)
                tally["zeros"] += int((finite == 0).sum())
                tally["negatives"] += int((finite < 0).sum())
                if c in fences:
                    tally["outliers"] += int(((finite < fences[c][0]) | (finite > fences[c][1])).sum())
            elif kind == "text":
                ln = present.astype(str).str.len()
                old = lengths.get(c, (np.inf, -np.inf))
                lengths[c] = (min(old[0], int(ln.min())), max(old[1], int(ln.max())))
            elif kind == "datetime":
                parsed = as_datetime(present, date_sources.get(str(c))).dropna()
                if len(parsed):
                    old = dates.get(c)
                    lo_d, hi_d = parsed.min(), parsed.max()
                    dates[c] = (min(old[0], lo_d), max(old[1], hi_d)) if old else (lo_d, hi_d)
            if c in levels and levels[c] is not None:
                merged = levels[c].add(present.astype(str).value_counts(), fill_value=0)
                levels[c] = merged if len(merged) <= LEVEL_COUNT_CAP else None
        rows.add(_row_hashes(batch))
        if id_cols:
            no_ids.add(_row_hashes(batch[[c for c in cols if c not in id_cols]]))
        if feature_cols:
            features.add(_row_hashes(batch[feature_cols]))
            with_target.add(_row_hashes(batch[feature_cols + [target]]))
    distinct_rows, exact = rows.count()
    out = {"n": n, "missing": missing, "distinct": {c: (d.estimate(), d.exact) for c, d in distinct.items()},
           "min": lo, "max": hi, "exact": exact,
           "exact_row_frac": round(1 - distinct_rows / n, 4) if n else 0.0,
           "moments": {c: m.stats() for c, m in moments.items()}, "numeric_counts": counts_num,
           "levels": {c: v.astype("int64") for c, v in levels.items() if v is not None},
           "lengths": lengths, "dates": dates}
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
    source = result.get("source") or {}
    lines += ["", "## Source",
              f"- {source.get('name')}: {source.get('n_rows', 0):,} rows profiled, "
              + ("the full data" if source.get("full_data") else f"NOT the full data ({source.get('reason')})")]
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
    lines += ["", "Feature-target strength is computed on the training split at step 7 (guard.train_associations), "
              "never before the split.",
              "Full per-column detail: data_profile.json -> columns.<name>. Read single columns, not the whole file."]
    return "\n".join(lines) + "\n"


def profile(
    source,
    *,
    target: str,
    time_col: str | None = None,
    group_col: str | None = None,
    source_name: str | None = None,
    state_dir: str | Path = "ml_pipeline",
) -> dict:
    """Step 1: describe the data, write ml_pipeline/data_profile.json and a short data_profile_summary.md.

    ``source`` is a DataFrame or anything ``scan`` reads (Parquet, CSV, ...), so data larger than
    memory is profiled in batches. A file path is recorded in source.json (hash, rows, columns) and the
    profile is tied to it; a DataFrame must be a whole source loaded with ``guard.load`` / ``guard.derive``
    (``source_name`` picks which), else the profile says it does not cover the full data and Gate A asks
    for the user's override. Read the summary first; the JSON is the per-column reference.
    """
    state_dir = Path(state_dir)
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
    id_cols = [c for c, k in kinds.items() if k == "id" and c != target]
    sample_stats = {col: _column_stats(sample[col], kinds[col]) for col in sample.columns}
    full = _full_pass(source, kinds, target, id_cols, sample_stats, date_sources)
    n = full["n"]
    per_row_bytes = float(sample.memory_usage(deep=True, index=False).sum()) / max(len(sample), 1)

    columns = {}
    for col in sample.columns:
        n_unique, exact = full["distinct"][col]
        n_missing = int(full["missing"][col])
        info = {"kind": kinds[col], "dtype": str(sample[col].dtype), "non_null": int(n - n_missing),
                "missing": n_missing, "missing_frac": round(n_missing / n, 4) if n else 0.0,
                "n_unique": n_unique, "n_unique_exact": exact, "unique_frac": round(n_unique / n, 4) if n else 0.0,
                "memory_bytes": int(round(float(sample[col].memory_usage(deep=True, index=False)) / max(len(sample), 1) * n))}
        info.update({k: v for k, v in sample_stats[col].items() if k != "kind"})
        if kinds[col] == "numeric" and col in full["min"]:
            info.update(min=_num(full["min"][col]), max=_num(full["max"][col]), **full["moments"].get(col, {}))
            tally = full["numeric_counts"][col]
            info.update(zeros=tally["zeros"], negatives=tally["negatives"], outliers=tally["outliers"],
                        outlier_frac=round(tally["outliers"] / tally["finite"], 4) if tally["finite"] else 0.0)
        if kinds[col] in ("categorical", "bool") and col in full["levels"]:
            info.update(_level_stats(full["levels"][col], n - n_missing))
            info["levels_exact"] = True
        elif kinds[col] in ("categorical", "bool"):
            info["n_levels"] = n_unique
            info["levels_exact"] = False  # more than LEVEL_COUNT_CAP levels: top levels from the sample
        if kinds[col] == "text" and col in full["lengths"]:
            info.update(length_min=int(full["lengths"][col][0]), length_max=int(full["lengths"][col][1]))
        if kinds[col] == "datetime":
            info["datetime_source"] = date_sources.get(str(col), "native")
            if col in full["dates"]:
                start, end = full["dates"][col]
                info.update(start=str(start)[:10], end=str(end)[:10], span_days=int((end - start).days))
        if info["non_null"]:  # values exist, but the sample may have missed them all: say so instead of omitting keys
            info.update({k: None for k in _KIND_STATS.get(kinds[col], ()) if k not in info})
        columns[str(col)] = info

    target_info = _target_info(sample[target], full["levels"].get(target))
    target_info["missing_frac"] = columns[str(target)]["missing_frac"]
    if target_info["task"] == "regression":
        tstats = columns[str(target)]
        target_info.update(mean=tstats.get("mean"), std=tstats.get("std"), min=tstats.get("min"), max=tstats.get("max"))
    # The target's link to each feature is used only to look for leaks here; strength of signal is a
    # training-split statistic (guard.train_associations at step 7), never ranked before the split.
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
                  "memory_mb_estimate": round(per_row_bytes * n / 1e6, 2),
                  "sampled": bool(sampled), "sample_rows": int(len(sample)) if sampled else None},
        "source": _source_entry(source, sample.columns, n, source_name, state_dir),
        "target": target_info,
        "columns": columns,
        "duplicates": duplicates,
        "constant_columns": [str(c) for c in sample.columns if columns[str(c)]["n_unique"] <= 1],
        "high_cardinality_categoricals": high_card,
        "datetime_columns": {c: s for c, s in date_sources.items() if c != str(target)},
        "time_col": time_col,
        "group_col": group_col,
        "entity_candidates": entities,
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
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / PROFILE_FILE).write_text(json.dumps(result, indent=2, default=str))
    (state_dir / SUMMARY_FILE).write_text(_summary_markdown(result))
    return result


def summary(state_dir: str | Path = "ml_pipeline") -> str:
    """The short profile summary (what the agent should read instead of the full JSON)."""
    return (Path(state_dir) / SUMMARY_FILE).read_text()


# ====================================================================== the real-explanation rule
# Same rule, same results as note_problem() in hooks/gate_records.py (tests/note_rule_cases.json).
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_KIND_STATS = {
    "numeric": ("mean", "std", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "skew",
                "kurtosis", "zeros", "negatives", "outliers", "outlier_frac"),
    "categorical": ("n_levels", "top", "top_counts", "rare_levels"),
    "bool": ("n_levels", "top", "top_counts", "rare_levels"),
    "datetime": ("start", "end", "span_days"),
    "text": ("length_min", "length_median", "length_max"),
}


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


def _citable(stats: dict) -> dict:
    """The statistics a column note may cite: the kind's own figures, plus the missing count when there are gaps.

    Counts every column has (rows, non-null values, distinct values, memory) do not prove the column was
    looked at, so they only count for columns without figures of their own: ids and all-missing columns.
    """
    own = {k: stats[k] for k in _KIND_STATS.get(stats.get("kind"), ()) if k in stats}
    if not own:
        return {k: stats[k] for k in ("n_unique", "unique_frac", "missing", "missing_frac") if k in stats}
    if (stats.get("missing") or 0) > 0:
        own.update(missing=stats["missing"], missing_frac=stats.get("missing_frac"))
    return own


def _cites(note: str, stats: dict) -> bool:
    """True when ``note`` has a number other than 0 or 1 equal to one of ``stats`` at its written precision."""
    facts = _stat_values(_citable(stats))
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
    """Save a matplotlib figure as ml_pipeline/figures/NN_<slug>.png and record its explanation.

    The explanation is a real one: at least MIN_NOTE_CHARS characters, with a number from the data.
    """
    if not explanation or not explanation.strip():
        raise ValueError("every figure needs an explanation: what it shows and why it matters")
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
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    _ledger(state_dir / "PIPELINE.md", path.name, explanation)
    return path


def _prepared_sample(source, target: str, time_col: str | None, prof: dict | None) -> tuple[pd.DataFrame, int, dict, dict]:
    """(sample with dates parsed and inf blanked, total rows, kinds, date sources) for drawing figures."""
    df, n_total = load_sample(source)
    df = df.rename(columns=str)  # a private copy: dates are converted in place below
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in data")
    if prof:
        date_sources = {c: i.get("datetime_source", "native") for c, i in prof["columns"].items()
                        if i.get("kind") == "datetime" and c in df.columns and c != target}
    else:
        date_sources = datetime_columns(df.drop(columns=[target]))
    if time_col is not None:
        date_sources.setdefault(str(time_col), _datetime_source(df[time_col]) or "text")
    for col, src in date_sources.items():
        df[col] = as_datetime(df[col], src)
    for col in df.select_dtypes("number").columns:  # inf breaks histograms and binning; it is reported in the profile
        if col != target and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
    if prof:
        kinds = {c: i["kind"] for c, i in prof["columns"].items() if c in df.columns}
    else:
        kinds = {str(c): _column_kind(df[c], date_sources.get(str(c))) for c in df.columns}
    return df, n_total, kinds, date_sources


def _read_profile(state_dir: Path) -> dict | None:
    path = Path(state_dir) / PROFILE_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def eda_figures(source, *, target: str, time_col: str | None = None, state_dir: str | Path = "ml_pipeline",
                per_page: int = PAGE_COLUMNS) -> list[Path]:
    """The step 1-2 figures, with explanations computed from the data. Works on files of any size.

    Plots use a uniform sample of SAMPLE_ROWS rows when the data is larger (the explanations say so);
    counts come from the profile, which covers every row. Every column is drawn: the overview shows all
    of them at once, and ``variable_figures`` puts each one on a page (pages.json lists which). Nothing
    here relates a feature to the target - that is a training-split statistic (step 7).
    """
    plt = _plt()
    state_dir = Path(state_dir)
    target = str(target)
    prof = _read_profile(state_dir)
    df, n_total, kinds, date_sources = _prepared_sample(source, target, time_col, prof)
    kinds = {c: k for c, k in kinds.items() if c != target}
    sampled_note = f" (uniform sample of {len(df):,} of {n_total:,} rows)" if n_total > len(df) else ""
    task = prof["target"]["task"] if prof else _target_info(df[target])["task"]
    paths: list[Path] = []

    # 01 missingness ------------------------------------------------------------------
    if prof:
        missing = pd.Series({c: i["missing_frac"] for c, i in prof["columns"].items()}).sort_values(ascending=False)
        unique = pd.Series({c: i["unique_frac"] for c, i in prof["columns"].items()})
    else:
        missing = df.isna().mean().sort_values(ascending=False)
        unique = df.nunique() / max(len(df), 1)
    shown = missing.head(40)
    f, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(shown))))
    ax.barh(shown.index.astype(str)[::-1], shown.to_numpy()[::-1])
    ax.set_xlabel("missing fraction")
    ax.set_title("Missing values per column" + (f" (worst 40 of {len(missing)})" if len(missing) > 40 else ""))
    n_missing = int((missing > 0).sum())
    text = (f"{n_missing} of {len(missing)} columns have missing values; worst is {missing.index[0]} at {missing.iloc[0]:.1%}."
            if n_missing else f"0 of {len(missing)} columns have missing values, so step 4 needs no imputation.")
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
        balance = (pd.Series(prof["target"]["class_balance"]) if prof and prof["target"].get("class_balance")
                   else y.astype(str).value_counts(normalize=True)).head(30)
        ax.bar(balance.index.astype(str), balance.to_numpy())
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"Target {target} balance")
        explanation = (f"{target}={balance.index[0]} is {balance.iloc[0]:.1%} of rows, so a majority-class "
                       f"dummy scores {balance.iloc[0]:.1%}; the smallest class shown is {balance.min():.1%}.")
    paths.append(fig(2, "target_balance", f, explanation, state_dir))

    # 02 overview: every column in one picture (no target relationship) ----------------
    cols_all = list(kinds)
    if cols_all:
        f, ax = plt.subplots(figsize=(8, 5))
        palette = {"numeric": "tab:blue", "categorical": "tab:orange", "bool": "tab:green", "datetime": "tab:purple",
                   "id": "tab:gray", "text": "tab:brown"}
        for kind, color in palette.items():
            members = [c for c in cols_all if kinds[c] == kind]
            if members:
                ax.scatter([float(missing.get(c, 0.0)) for c in members], [float(unique.get(c, 0.0)) for c in members],
                           s=14, alpha=0.7, color=color, label=f"{kind} ({len(members)})")
        for c in sorted(cols_all, key=lambda c: -float(missing.get(c, 0.0)))[:10]:
            ax.annotate(str(c)[:18], (float(missing.get(c, 0.0)), float(unique.get(c, 0.0))), fontsize=7)
        ax.set_xlabel("missing fraction")
        ax.set_ylabel("distinct values / rows")
        ax.set_title(f"All {len(cols_all)} feature columns: missingness vs. distinctness")
        ax.legend(fontsize=7)
        gappy = sum(1 for c in cols_all if float(missing.get(c, 0.0)) > 0)
        near_unique = sum(1 for c in cols_all if float(unique.get(c, 0.0)) > 0.95)
        paths.append(fig(2, "feature_overview", f,
                         f"Every one of the {len(cols_all)} feature columns as a dot: {gappy} have gaps and {near_unique} "
                         "are more than 95% distinct (ids or free text). The target is left out until the split.",
                         state_dir))

    # 02 outliers -----------------------------------------------------------------------
    numeric = [c for c in cols_all if kinds[c] == "numeric"]
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

    # 02 correlations between features (the target is left out until the split) ---------------
    if len(numeric) >= 2:
        corr_cols = numeric[:40]
        corr = df[corr_cols].astype(float).corr()
        f, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels([str(c)[:18] for c in corr.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels([str(c)[:18] for c in corr.columns], fontsize=6)
        f.colorbar(image)
        ax.set_title("Correlations between features (target left out until the split)"
                     + (f", first {len(corr_cols)} of {len(numeric)}" if len(numeric) > 40 else ""))
        full_cols = numeric[:500]
        rows = df[full_cols] if len(df) <= 20_000 else df[full_cols].sample(20_000, random_state=0)
        rows.astype(float).corr().round(3).to_csv(state_dir / "correlations.csv")
        values = corr.to_numpy()
        pairs = [(abs(values[i, j]), corr.columns[i], corr.columns[j])
                 for i in range(len(corr)) for j in range(i + 1, len(corr)) if values[i, j] == values[i, j]]
        if pairs:
            r, a, b = max(pairs, key=lambda t: t[0])
            high = sum(1 for t in pairs if t[0] > COLLINEAR_RHO)
            explanation = (f"Strongest pair among {len(corr)} numeric features: {a} and {b} (|r| = {r:.2f}); {high} pairs "
                           f"exceed {COLLINEAR_RHO}. The matrix for {len(full_cols)} numeric columns is in correlations.csv; "
                           "the target is left out until the split.")
        else:
            explanation = (f"Correlations between {len(corr)} numeric features could not be computed (constant "
                           "columns). The target is left out until the split.")
        paths.append(fig(2, "correlations", f, explanation, state_dir))

    # 02 temporal coverage ---------------------------------------------------------------------
    tcol = time_col or next(iter(date_sources), None)
    if tcol is not None:
        t = df[tcol]
        counts = t.dt.to_period("M").value_counts().sort_index()
        f, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot([str(p) for p in counts.index], counts.to_numpy())
        ax.set_ylabel("rows per month")
        step = max(1, len(counts) // 24)
        ax.set_xticks(range(0, len(counts), step))
        ax.set_xticklabels([str(p) for p in counts.index][::step], rotation=90, fontsize=7)
        ax.set_title(f"Rows per month over {tcol}")
        paths.append(fig(2, "temporal_coverage", f,
                         f"{tcol} spans {str(t.min())[:10]} to {str(t.max())[:10]} over {len(counts)} months{sampled_note}; "
                         "a chronological split must hold out the latest period.", state_dir))

    # 02 every column on a page ----------------------------------------------------------------
    paths += variable_figures(df, target=target, per_page=per_page, state_dir=state_dir,
                              _kinds={**kinds, target: prof["columns"][target]["kind"] if prof else "target"},
                              _n_total=n_total)
    return paths


PAGES_FILE = "pages.json"


def _draw_variable(ax, series: pd.Series, kind: str) -> None:
    values = series.dropna()
    if not len(values):
        ax.text(0.5, 0.5, "all missing", ha="center", va="center")
        ax.set_axis_off()
    elif kind == "numeric":
        ax.hist(values.astype(float), bins=30)
        ax.axvline(float(values.median()), color="black", linestyle="--", linewidth=1)
    elif kind in ("categorical", "bool", "target"):
        counts = values.astype(str).value_counts().head(10)
        ax.barh(counts.index[::-1], counts.to_numpy()[::-1])
        ax.tick_params(axis="y", labelsize=7)
    elif kind == "datetime" and pd.api.types.is_datetime64_any_dtype(values):
        counts = values.dt.to_period("M").value_counts().sort_index()
        ax.plot([str(p) for p in counts.index], counts.to_numpy())
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    elif kind == "text":
        ax.hist(values.astype(str).str.len(), bins=30)
        ax.set_xlabel("length", fontsize=7)
    else:
        ax.text(0.5, 0.5, f"{values.nunique():,} unique values", ha="center", va="center")
        ax.set_axis_off()


def variable_figures(source, *, target: str, per_page: int = PAGE_COLUMNS, state_dir: str | Path = "ml_pipeline",
                     _kinds: dict | None = None, _n_total: int | None = None) -> list[Path]:
    """Step 2: every column's distribution on grouped pages, plus figures/pages.json (page -> columns).

    Each column's written note goes in data_dictionary.csv (``figure_note``) and must cite one of its
    own figures from the profile. ``guard.eda_figures`` calls this; call it again after adding columns.
    """
    plt = _plt()
    state_dir = Path(state_dir)
    prof = _read_profile(state_dir)
    if _kinds is None:
        df, n_total, kinds, _ = _prepared_sample(source, str(target), None, prof)
    else:
        df, n_total, kinds = source, int(_n_total or len(source)), _kinds
    sampled_note = f" Drawn from a uniform sample of {len(df):,} of {n_total:,} rows." if n_total > len(df) else ""
    stats = prof["columns"] if prof else {}
    columns = [str(c) for c in df.columns]
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
            kind = kinds.get(col) or _column_kind(series)
            _draw_variable(ax, series, "categorical" if col == str(target) and kind != "numeric" else kind)
            ax.set_title(f"{str(col)[:28]} ({'target' if col == str(target) else kind})", fontsize=9)
            info = stats.get(col, {})
            if (info.get("missing") if info else int(series.isna().sum())):
                gappy.append(col)
            skew = info.get("skew") if info else (float(series.skew()) if kind == "numeric" and series.dropna().size > 2 else 0)
            if kind == "numeric" and skew is not None and abs(float(skew)) > 1:
                skewed.append(col)
        for ax in axes[len(cols):]:
            ax.axis("off")
        detail = f" ({', '.join(skewed[:5])})" if skewed else ""
        explanation = (f"Page {number} of {len(pages)}: {len(cols)} columns ({cols[0]} to {cols[-1]}); "
                       f"{len(gappy)} have missing values, {len(skewed)} are strongly skewed{detail}. "
                       f"Per-column notes are in data_dictionary.csv.{sampled_note}")
        path = fig(2, f"variables_p{number:02d}", f, explanation, state_dir)
        mapping[path.name] = cols
        paths.append(path)
    out = state_dir / FIGURES_DIR / PAGES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2))
    return paths


# ====================================================================== records (steps 1-8)
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


def _columns_of(data) -> list[str]:
    """Column names of a frame or any source ``scan`` reads, without reading the rows."""
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        return [str(c) for c in data.columns]
    if _is_path(data) and Path(data).suffix.lower() in (".parquet", ".pq") or (_is_path(data) and Path(data).is_dir()):
        try:
            import pyarrow.dataset as ds
            return [str(n) for n in ds.dataset(str(data)).schema.names]
        except ImportError:
            pass
    return [str(c) for c in next(iter(scan(data, batch_rows=1))).columns]


def _merge_csv(path: Path, header: list[str], rows: list[dict], *, key: str, computed: list[str]) -> list[str]:
    """Write ``rows``, keeping the agent's words for rows that still exist; return the keys dropped.

    Computed fields always come from ``rows``; word fields keep a non-empty earlier value, else take the
    default in ``rows``. Files saved by Excel or Numbers still merge (see _read_record).
    """
    old = _read_record(path, key) if path.is_file() else {}
    merged = [{h: (row.get(h, "") if h in computed else (old.get(row[key], {}).get(h) or row.get(h, "")))
               for h in header} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    current = {row[key] for row in rows}
    return sorted(k for k in old if k and k not in current)


def _read_record(path: Path, key: str) -> dict[str, dict]:
    """An existing record's rows by ``key``, as Excel or Numbers may have saved it (BOM, cp1252, ``;`` or tab).

    Raises instead of guessing, so a record that can't be read is never overwritten and its words never lost.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    for delimiter in (",", ";", "\t"):
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
        if reader.fieldnames and key in [f.strip() for f in reader.fieldnames]:
            return {(r.get(key) or "").strip(): {(k or "").strip(): v for k, v in r.items()} for r in reader}
    raise ValueError(f"can't read {path.name}: no '{key}' column. Save it as 'CSV UTF-8 (comma delimited)' and "
                     "re-run; the file was left unchanged")


def _warn_removed(removed: list[str], path: Path) -> None:
    if removed:
        warnings.warn(f"{path.name}: removed rows for columns no longer in the data: {', '.join(removed)}",
                      UserWarning, stacklevel=3)


def _observed(info: dict) -> str:
    kind = info["kind"]
    if not info.get("non_null"):
        return "all missing"
    if kind == "numeric" and "min" in info:
        return f"{info['min']}..{info['max']}, median {info.get('median')}"
    if kind in ("categorical", "bool") and "top" in info:
        shown = list(info["top"])[:5]
        return f"{info['n_levels']} levels: {', '.join(shown)}{'…' if info['n_levels'] > len(shown) else ''}"
    if kind == "datetime" and "start" in info:
        return f"{info['start']}..{info['end']}"
    if kind == "text" and "length_min" in info:
        return f"length {info['length_min']}..{info['length_max']}"
    return f"{info['n_unique']} unique"


def data_dictionary(data=None, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: one row per column; the guard fills kind, dtype, and observed values from the profile, the agent
    the words (meaning, unit, allowed, source, available_at_prediction, role, figure_note).

    ``data`` (a frame or file) gives the column list when columns were added or dropped since the profile;
    by default the profile's columns are used.
    """
    prof = _load_profile(state_dir)
    target = str(prof["target"]["name"])
    names = _columns_of(data) or list(prof["columns"])
    missing = [c for c in names if c not in prof["columns"]]
    if missing:
        raise LeakageError(f"columns not in the profile: {missing[:10]} - re-run guard.profile() on this data first")
    rows = []
    for col in names:
        info = prof["columns"][col]
        rows.append({"column": col, "kind": info["kind"], "dtype": info.get("dtype", ""), "observed": _observed(info),
                     "role": "target" if col == target else ""})
    path = Path(state_dir) / DICTIONARY_FILE
    _warn_removed(_merge_csv(path, DICTIONARY_HEADER, rows, key="column",
                             computed=["column", "kind", "dtype", "observed"]), path)
    return path


def missingness(data, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: one row per column with gaps (counts over every row) and a figure of which go missing together.

    The agent fills ``mechanism`` and ``evidence`` before Gate A, ``decision`` and ``reason`` before Gate B.
    """
    counts: pd.Series | None = None
    n = 0
    for batch in scan(data):
        part = batch.isna().sum()
        counts = part if counts is None else counts.add(part, fill_value=0)
        n += len(batch)
    counts = (counts if counts is not None else pd.Series(dtype=float)).astype(int)
    counts.index = [str(c) for c in counts.index]
    gappy = [c for c in counts.index if counts[c] > 0]
    rows = [{"column": c, "n_missing": str(int(counts[c])), "pct_missing": f"{100 * counts[c] / max(n, 1):.2f}"}
            for c in gappy]
    path = Path(state_dir) / MISSINGNESS_FILE
    _warn_removed(_merge_csv(path, MISSINGNESS_HEADER, rows, key="column",
                             computed=["column", "n_missing", "pct_missing"]), path)
    if not gappy:
        return path
    plt = _plt()
    sample, n_total = load_sample(data)
    sample = sample.rename(columns=str)
    note = f" Pattern from a uniform sample of {len(sample):,} of {n_total:,} rows." if n_total > len(sample) else ""
    if len(gappy) >= 2:
        indicators = sample[gappy].isna()
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
        a, b, share = max(((a, b, float((indicators[a] & indicators[b]).sum() / max(indicators[a].sum(), 1)))
                           for a in gappy for b in gappy if a != b), key=lambda t: t[2])
        explanation = (f"{len(gappy)} columns have gaps; {b} is also missing in {share:.0%} of {a}'s "
                       f"{int(indicators[a].sum()):,} gap rows.{note}")
    else:
        col = gappy[0]
        f, ax = plt.subplots(figsize=(6, 3))
        ax.bar([str(col)], [int(counts[col])])
        ax.set_ylabel("missing rows")
        ax.set_title("Missing values")
        explanation = (f"Only {col} has gaps: {int(counts[col]):,} of {n:,} rows "
                       f"({counts[col] / max(n, 1):.1%}); every other column is complete.")
    fig(1, "missing_pattern", f, explanation, state_dir)
    return path


def categorical_plan(data=None, *, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 7: one row per categorical or bool column (levels computed); the agent picks the encoding.

    Kinds and level counts come from the profile; columns added since (step 7 features) are counted
    from ``data``.
    """
    prof = _load_profile(state_dir)
    target = str(prof["target"]["name"])
    names = _columns_of(data) or list(prof["columns"])
    new_cols = [c for c in names if c not in prof["columns"]]
    extra: dict = {}
    if new_cols:
        sample, _ = load_sample(data, columns=None) if not isinstance(data, pd.DataFrame) else (data, len(data))
        sample = sample.rename(columns=str)
        for col in new_cols:
            kind = _column_kind(sample[col])
            if kind in ("categorical", "bool"):
                values = sample[col].dropna().astype(str)
                extra[col] = {"kind": kind, **_level_stats(values.value_counts(), len(values))}
    rows = []
    for col in names:
        info = prof["columns"].get(col) or extra.get(col)
        if col == target or not info or info.get("kind") not in ("categorical", "bool"):
            continue
        n_levels = int(info.get("n_levels") or info.get("n_unique") or 0)
        rows.append({"column": col, "n_levels": str(n_levels), "rare_levels": str(int(info.get("rare_levels") or 0)),
                     "high_cardinality": "yes" if n_levels > HIGH_CARD_LEVELS else "no"})
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


def _label_facts(data, target: str, id_cols: set) -> dict:
    """Rows, missing labels, class counts and label conflicts over every row, in one streaming pass.

    A conflict is a group of rows identical in every non-target, non-id column with more than one label.
    Memory: 16 bytes per labelled row for the two hash streams.
    """
    n = n_missing = 0
    counts = pd.Series(dtype="int64")
    feature_hashes, label_hashes = [], []
    for batch in scan(data):
        batch = batch.rename(columns=str)
        n += len(batch)
        y = batch[target]
        labelled = y.notna()
        n_missing += int((~labelled).sum())
        counts = counts.add(y[labelled].astype(str).value_counts(), fill_value=0)
        features = [c for c in batch.columns if c != target and c not in id_cols]
        if features and labelled.any():
            feature_hashes.append(_row_hashes(batch.loc[labelled, features]))
            label_hashes.append(pd.util.hash_pandas_object(y[labelled], index=False).to_numpy(dtype=np.uint64))
    rows = groups = 0
    if feature_hashes:
        fh, lh = np.concatenate(feature_hashes), np.concatenate(label_hashes)
        order = np.lexsort((lh, fh))
        fh, lh = fh[order], lh[order]
        new_pair = np.ones(len(fh), dtype=bool)
        new_pair[1:] = (fh[1:] != fh[:-1]) | (lh[1:] != lh[:-1])
        pair_keys = fh[new_pair]                     # one entry per distinct (features, label) pair
        keys, n_labels = np.unique(pair_keys, return_counts=True)
        conflicted = keys[n_labels > 1]
        groups, rows = int(len(conflicted)), int(np.isin(fh, conflicted).sum())
    return {"n": n, "missing": n_missing, "class_counts": counts.astype("int64"), "conflict_rows": rows,
            "conflict_groups": groups}


def _label_block(facts: dict, target_info: dict) -> str:
    lines = [_COMPUTED_START, f"- rows: {facts['n']:,}", f"- missing labels: {facts['missing']:,}"]
    if target_info["task"] == "regression":
        lines.append(f"- target: min {target_info['min']}, max {target_info['max']}, mean {target_info['mean']}, "
                     f"std {target_info['std']}")
    else:
        counts = facts["class_counts"].sort_values(ascending=False, kind="mergesort")
        total = int(counts.sum())
        lines.append("- classes: " + "; ".join(f"{k} = {int(c):,} ({c / total:.1%})" for k, c in counts.items()))
    lines += [f"- identical feature rows with different labels: {facts['conflict_rows']:,} rows in "
              f"{facts['conflict_groups']:,} groups", _COMPUTED_END]
    return "\n".join(lines)


def label_audit(data, *, target: str, state_dir: str | Path = "ml_pipeline") -> Path:
    """Step 1: the label's computed facts (every row) plus five sections the agent writes; later calls refresh
    the facts only, never the writing."""
    target = str(target)
    try:
        prof = _load_profile(state_dir)
    except LeakageError:
        prof = None
    sample, _ = load_sample(data)
    sample = sample.rename(columns=str)
    if prof and set(prof["columns"]) == set(sample.columns):
        id_cols = {c for c, i in prof["columns"].items() if i.get("kind") == "id"}
    else:
        dates = datetime_columns(sample.drop(columns=[target]))
        id_cols = {c for c in sample.columns if c != target and _column_kind(sample[c], dates.get(c)) == "id"}
    facts = _label_facts(data, target, id_cols)
    if prof and prof["target"].get("name") == target and prof["target"].get("task") == "regression":
        target_info = prof["target"]
    else:
        target_info = _target_info(sample[target], facts["class_counts"])
    block = _label_block(facts, target_info)
    path = Path(state_dir) / LABEL_AUDIT_FILE
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


# ====================================================================== preparation (steps 4-8)
CLEANING_FILE = "cleaning_log.csv"
CLEANING_HEADER = ["rule", "reason", "rows_before", "rows_after", "cells_changed", "columns_changed", "date"]
ASSOCIATIONS_FILE = "associations.json"


def log_cleaning(rule: str, before: pd.DataFrame, after: pd.DataFrame, *, reason: str,
                 state_dir: str | Path = "ml_pipeline") -> pd.DataFrame:
    """Step 4: record one cleaning rule with its before/after counts; returns ``after``.

    Rows are matched by index label and cells compared on the shared columns, NaN equal to NaN. For data
    cleaned out of core, record the rule with frames of the affected rows, or the line
    ``- Cleaning: none needed - reason: ...`` when nothing needs cleaning.
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


def _split_state(state_dir) -> dict:
    """The frozen split's state; raises when there is none or it predates train/validation sizes."""
    path = Path(state_dir) / STATE_FILE
    if not path.is_file():
        raise LeakageError("no split state: call guard.split() or guard.split_to_files() at step 6 first")
    state = json.loads(path.read_text())
    if "n_train" not in state:
        raise LeakageError("split state is from an older guard: re-run guard.split()")
    return state


def _rows(data) -> int:
    """How many rows ``data`` holds: ``shape[0]`` for frames, arrays, and sparse matrices, a streamed count for
    files, else ``len()``."""
    if _is_path(data):
        return sum(len(batch) for batch in scan(data))
    shape = getattr(data, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) else len(data)


def _split_frame(data, expected: int, what: str, error) -> tuple[pd.DataFrame, int]:
    """(frame to compute on, total rows) for a split given as a frame or a file; raises ``error`` on a wrong size."""
    total = _rows(data)
    if total != expected:
        raise error(f"{total:,} rows, but the frozen {what} split has {expected:,}: compute this on the {what} "
                    "split only")
    frame = data if isinstance(data, pd.DataFrame) else load_sample(data)[0]
    return frame.rename(columns=str), total


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


def train_associations(train, *, target: str, state_dir: str | Path = "ml_pipeline") -> dict:
    """Step 7: correlations and feature-target strength on the frozen training split only (a frame or the
    train file split_to_files wrote; a large file is measured on a uniform sample of its rows)."""
    state = _split_state(state_dir)
    frame, total = _split_frame(train, int(state["n_train"]), "training", NotTrainingSplit)
    target = str(target)
    task = _target_info(frame[target])["task"]
    try:
        known = {c: i["kind"] for c, i in _load_profile(state_dir)["columns"].items()}
    except LeakageError:
        known = {}
    kinds = {c: known.get(c) or _column_kind(frame[c]) for c in frame.columns if c != target}
    usable = [c for c, k in kinds.items() if k in ("numeric", "bool", "categorical")]
    strengths = []
    for col in usable:
        measure, value = _feature_target(frame[col], frame[target], kinds[col], task)
        entry = {"column": str(col), "measure": measure, "value": round(_finite(value), 4)}
        if kinds[col] == "categorical" and frame[col].nunique() > HIGH_CARD_LEVELS:
            entry["high_cardinality"] = True
        strengths.append(entry)
    strengths.sort(key=lambda e: e["value"], reverse=True)
    numeric = [c for c in usable if kinds[c] in ("numeric", "bool")]
    corr = frame[numeric].astype(float).rank().corr() if len(numeric) >= 2 else pd.DataFrame()
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            rho = float(corr.loc[a, b])
            if rho == rho and abs(rho) > COLLINEAR_RHO:
                pairs.append({"a": str(a), "b": str(b), "rho": round(rho, 4)})
    computed_on = {"split": "train", "n_rows": int(total), "date": date.today().isoformat()}
    if len(frame) < total:
        computed_on["sample_rows"] = int(len(frame))
    result = {"schema_version": "1", "computed_on": computed_on, "target": target, "task": task,
              "feature_target": strengths, "collinear_pairs": pairs}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / ASSOCIATIONS_FILE).write_text(json.dumps(result, indent=2))
    _association_figures(corr, strengths, pairs, total, state_dir)
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


# ====================================================================== training receipts and importance (steps 9-12)
TRAINING_FILE = "training_runs.jsonl"
IMPORTANCE_FILE = "importance.json"


def _append_run(state_dir, name: str, n_rows: int, rows_from: int | None = None) -> None:
    state = _split_state(state_dir)
    source = n_rows if rows_from is None else rows_from
    line = {"model": str(name), "n_rows": int(n_rows), "n_train": int(state["n_train"]),
            "full": int(source) == int(state["n_train"]), "date": date.today().isoformat()}
    if rows_from is not None:
        line["rows_from"] = int(rows_from)
    with (Path(state_dir) / TRAINING_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def fit(model, X, y=None, *, name: str | None = None, rows_from=None, state_dir: str | Path = "ml_pipeline",
        **fit_kwargs):
    """Steps 9-10: train ``model`` and record how many rows it saw, so Gate C can prove a full-data run.

    After resampling (``fit_resample(X_train, y_train)``) pass ``rows_from=X_train``: the run is full when
    the frame it was resampled from is the whole training split, and the receipt keeps both counts.
    """
    _split_state(state_dir)  # no split, no training - checked before the model is touched
    n_rows = _rows(X)
    source = None if rows_from is None else _rows(rows_from)
    if y is None:
        model.fit(X, **fit_kwargs)
    else:
        model.fit(X, y, **fit_kwargs)
    _append_run(state_dir, name or type(model).__name__, n_rows, source)
    return model


def record_training(data, *, name: str, rows_from=None, state_dir: str | Path = "ml_pipeline") -> None:
    """Custom training loops (PyTorch, Keras, out-of-core): record the rows ``data`` holds - the dataset or the
    train file, not a batch loader - so the count is measured, never typed."""
    _append_run(state_dir, name, _rows(data), None if rows_from is None else _rows(rows_from))


def importance(predict_fn, val, *, target: str, metric_fn, greater_is_better: bool = True,
               n_repeats: int = 5, seed: int = 0, state_dir: str | Path = "ml_pipeline") -> dict:
    """Step 12: permutation importance on the frozen validation split (a frame or the val file), with a figure."""
    state = _split_state(state_dir)
    frame, total = _split_frame(val, int(state["n_val"]), "validation", NotValidationSplit)
    target = str(target)
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy()
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
    computed_on = {"split": "validation", "n_rows": int(total), "date": date.today().isoformat()}
    if len(frame) < total:
        computed_on["sample_rows"] = int(len(frame))
    result = {"schema_version": "1", "computed_on": computed_on, "metric": metric,
              "baseline_score": round(baseline, 6), "n_repeats": int(n_repeats), "features": features}
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
    pipeline_md: str | Path | None = None,
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
    _refuse_resplit_after_test(state_dir, pipeline_md)

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
        "n_train": int(len(train)),
        "n_val": int(len(val)),
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


def _test_touches(state_dir: Path, pipeline_md: Path) -> int:
    """How often the frozen test set was scored: the state file's count or the ledger's final-test lines,
    whichever is larger (all runs: a new ``## Run:`` heading never resets single use)."""
    state_path = Path(state_dir) / STATE_FILE
    touches = int(json.loads(state_path.read_text()).get("test_touches", 0)) if state_path.is_file() else 0
    ledgers = {Path(pipeline_md).resolve(), (Path(state_dir) / "PIPELINE.md").resolve()}
    logged = max((len(_FINAL_LINE.findall(p.read_text())) for p in ledgers if p.is_file()), default=0)
    return max(touches, logged)


def _run_tag(pipeline_md: Path) -> str:
    """The newest ``## Run:`` heading, as the approval hook tags approvals with it ('' when there is none)."""
    if not Path(pipeline_md).is_file():
        return ""
    heads = re.findall(r"^[ \t]*##[ \t]+Run[ \t]*:[^\n]*", Path(pipeline_md).read_text(), re.I | re.M)
    return " ".join(heads[-1].split()) if heads else ""


def _final_test_overrides(state_dir: Path, pipeline_md: Path) -> tuple[int, int]:
    """(all, newest run) user-approved ``final test`` overrides: strict ledger lines, and - when the approval hook
    keeps a record - no more than the user granted in chat (in that run, for the newest-run count)."""
    ledgers = {Path(pipeline_md).resolve(), (Path(state_dir) / "PIPELINE.md").resolve()}
    every = max((_override_lines(p, "final test", current_run=False) for p in ledgers), default=0)
    newest = max((_override_lines(p, "final test", current_run=True) for p in ledgers), default=0)
    record = _approvals(state_dir)
    if record is None:
        return every, newest
    granted = [o for o in record.get("overrides", []) if str(o.get("what", "")).lower() == "final test"]
    run = _run_tag(pipeline_md)
    return min(every, len(granted)), min(newest, sum(1 for o in granted if str(o.get("run") or "") == run))


def _test_used_up(state_dir: Path, pipeline_md: Path) -> bool:
    """True when the frozen test set may not be scored (or re-split) again: every touch after the first needs one
    more user-approved override over the whole ledger, and one in the newest run."""
    touches = _test_touches(state_dir, pipeline_md)
    every, newest = _final_test_overrides(state_dir, pipeline_md)
    return touches >= 1 and (touches >= 1 + every or newest == 0)


def _refuse_resplit_after_test(state_dir, pipeline_md) -> None:
    """A re-split would hand out a fresh test set; once the frozen one was scored that needs an unused override."""
    state_dir = Path(state_dir)
    ledger = Path(pipeline_md) if pipeline_md is not None else state_dir / "PIPELINE.md"
    if _test_used_up(state_dir, ledger):
        raise TestSetAlreadyUsed(
            "the frozen test set was already evaluated; re-splitting would reset that. Agree a new test strategy "
            "with the user, who says 'override final test' in chat, and record `- Override: final test - ...` first")


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
    pipeline_md: str | Path | None = None,
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
    _refuse_resplit_after_test(state_dir, pipeline_md)
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
    train_fp, n_train = _fingerprint_source(paths["train"])
    val_fp, n_val = _fingerprint_source(paths["val"])
    kind = "time" if time_col is not None else "group" if group_col is not None else "hashed"
    _write_state(state_dir, {
        "fingerprint_version": 2,
        "test_fingerprint": test_fp,
        "train_fingerprint": train_fp,
        "val_fingerprint": val_fp,
        "n_train": int(n_train),
        "n_val": int(n_val),
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
def _approvals(state_dir: Path) -> dict | None:
    path = Path(state_dir) / APPROVALS_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


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
    touches = _test_touches(state_dir, pipeline_md)
    if _test_used_up(state_dir, pipeline_md):
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
