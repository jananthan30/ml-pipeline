#!/usr/bin/env python3
"""PreToolUse hook: no model training until the pipeline gates say so, one step at a time.

Claude Code runs this before each matched tool call and pipes the call to stdin as JSON. Every
string in ``tool_input`` is scanned - and, for shell commands, every local script or notebook the
command runs (plus the local modules it imports) - for fitting, training, tuning, and evaluation
calls. The call is denied unless ``ml_pipeline/PIPELINE.md`` records the gate that unlocks it AND
the user approved that gate in chat (recorded by the UserPromptSubmit hook in
``ml_pipeline/.approvals.json``; the agent cannot approve its own gates):

    no gate approved  ->  only exploratory fits (PCA/t-SNE/clustering/distribution fits) for EDA
    Gate A approved   ->  preprocessing fits (scalers, encoders, imputers) allowed
    Gate B approved   ->  baseline models; other models after '- Step 9 done'; tuning after '- Step 10 done'
    Gate C approved   ->  evaluating on the test split allowed (once, at step 14)
    always            ->  fitting on the test split is denied; the guard's state files are read-only

Recording a step or gate line is checked too: steps go in order, each step's artifacts must
exist, and a gate needs its steps, its artifacts, and the user's approval.

The hook fails open: an unexpected error allows the call and reports the error, so a bug here can
never lock anyone out of their own tools. ``ML_PIPELINE_ENFORCE=0`` switches it off for projects
that are not ML pipelines; ``ML_PIPELINE_REQUIRE_USER_APPROVAL=0`` falls back to ledger-only gates.
Stdlib only; no subprocesses, no network.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

PIPELINE_REL = Path("ml_pipeline") / "PIPELINE.md"
SEARCH_UP = 4  # look in cwd and this many parent directories
DOC_SUFFIXES = {".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
SKIP_KEYS = {"old_string", "file_path", "path", "notebook_path", "description", "url", "pattern", "glob", "cwd"}
SNIPPET = 80
APPROVALS_FILE = ".approvals.json"
STATE_FILE = ".guard_state.json"
RUNS_FILE = "runs.jsonl"

# ------------------------------------------------------------------ detection
# `<receiver>.fit(` where receiver is `name` or `Name(...)`; also partial_fit, fit_transform, fit_predict.
FIT_CALL = re.compile(r"(?:\b([A-Za-z_]\w*)\s*(?:\([^()]*\))?)?\.\s*(?:partial_)?fit(?:_transform|_predict)?\s*\(")
# The fit method taken as a value (`f = clf.fit`, `getattr(clf, "fit")`): a way to call it unseen.
FIT_REF = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*(?:partial_)?fit(?:_transform|_predict)?\b(?!\s*\()")
GETATTR_FIT = re.compile(r"getattr\s*\(\s*([A-Za-z_]\w*)\s*,\s*[\"'](?:partial_)?fit(?:_transform|_predict)?[\"']")
# APIs that only ever train models.
TRAIN_API = re.compile(
    r"\b(?:xgb|xgboost|lgb|lightgbm|catboost)\.(?:train|cv)\s*\(|\btrainer\.train\s*\(|\bTrainer\s*\(|"
    r"\.fit_generator\s*\(|\.fit_one_cycle\s*\(|\bloss\.backward\s*\(|\boptimizer\.step\s*\("
)
# Helpers that fit a model inside them (cross-validation, learning curves).
TRAIN_HELPER = re.compile(r"\b(?:cross_val_score|cross_validate|cross_val_predict|learning_curve|validation_curve|"
                          r"permutation_test_score)\s*\(")
# The guard library's step 14 call scores the frozen test set.
FINAL_TEST_CALL = re.compile(r"\bfinal_test\s*\(")
# Hyperparameter search: step 11, after the candidates of step 10.
TUNING = re.compile(
    r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|HalvingRandomSearchCV|BayesSearchCV|TuneSearchCV)\s*\(|"
    r"\boptuna\.create_study\s*\(|\.optimize\s*\(\s*\w+|\btune\.(?:run|Tuner)\s*\(|\bhyperopt\.fmin\s*\(|(?<![\w.])fmin\s*\("
)
# Evaluating a model: predict/score methods and the usual sklearn metric functions.
EVAL_CALL = re.compile(
    r"(?:\.(?:predict|predict_proba|predict_log_proba|decision_function|score|evaluate)|"
    r"\b(?:accuracy_score|balanced_accuracy_score|roc_auc_score|average_precision_score|f1_score|"
    r"precision_score|recall_score|log_loss|mean_squared_error|mean_absolute_error|"
    r"root_mean_squared_error|r2_score|classification_report|confusion_matrix|cross_val_score|"
    r"cross_validate|matthews_corrcoef|brier_score_loss|cohen_kappa_score|mean_absolute_percentage_error))\s*\("
)
ESTIMATOR = re.compile(
    r"\b\w*(?:Classifier|Regressor|Regression|Ridge|Lasso|ElasticNet|SVC|SVR|NB|KNeighbors|MLP|"
    r"XGB|LGBM|CatBoost|DecisionTree|RandomForest|GradientBoosting|ExtraTrees|AdaBoost|"
    r"Perceptron|SGD|GaussianProcess|Sequential|keras|Booster|TabularPredictor|Prophet|ARIMA|SARIMAX)\w*\b"
)
# Models that count as the step 9 baseline (a dummy plus one simple model).
BASELINE = re.compile(
    r"\b(?:Dummy(?:Classifier|Regressor)|LogisticRegression(?:CV)?|LinearRegression|Ridge(?:CV|Classifier(?:CV)?)?|"
    r"Lasso(?:CV)?|ElasticNet(?:CV)?|DecisionTree(?:Classifier|Regressor)|LinearSVC|LinearSVR|"
    r"(?:Gaussian|Multinomial|Bernoulli|Complement)NB|SGD(?:Classifier|Regressor)|NaiveForecaster|ExponentialSmoothing)\b"
)
TRANSFORMER = re.compile(
    r"\b\w*(?:Scaler|Imputer|Encoder|Vectorizer|PCA|Normalizer|Binarizer|Discretizer|"
    r"Transformer|SelectKBest|TruncatedSVD|PolynomialFeatures)\w*\b"
)
_VAR_SUFFIX = r"(?:_?\d+|_(?:num|cat|x|y|text|txt|feat|features?|cols?|train|all|in|out))?"
TRANSFORMER_VAR = re.compile(
    r"^(?:scaler|std_?scaler|mms|imputer|imp|encoder|enc|ohe|ordinal_?encoder|vectorizer|vec|tfidf|svd|preproc|"
    r"preprocessor|prep|ct|col_?transformer|transformer|selector|poly|poly_?features|binner|pipe_prep)" + _VAR_SUFFIX + r"$"
    r"|^\w+_(?:scaler|imputer|encoder|vectorizer|tfidf|preprocessor|transformer|selector)$",
    re.I,
)
# Unsupervised fits that EDA needs (embeddings, clustering, outlier scores, distribution fits).
EXPLORATORY = re.compile(
    r"\b(?:TSNE|UMAP|KMeans|MiniBatchKMeans|DBSCAN|HDBSCAN|OPTICS|AgglomerativeClustering|GaussianMixture|"
    r"KernelDensity|PCA|IncrementalPCA|KernelPCA|NMF|FactorAnalysis|Isomap|LocallyLinearEmbedding|MDS|"
    r"IsolationForest|LocalOutlierFactor|EllipticEnvelope|lowess)\b|\bstats\.\w+\b"
)
EXPLORATORY_VAR = re.compile(
    r"^(?:tsne|umap|reducer|embedder|kmeans|km|clusterer|clustering|gmm|kde|iforest|iso_forest|lof|pca|"
    r"norm|lognorm|expon|gamma|beta|weibull_min|weibull_max|pareto|poisson|nbinom|lowess)(?:_?\d+|_[23]d)?$",
    re.I,
)
_TEST_WORDS = (r"\w+_test|test_(?!size\b|ratio\b|frac\w*\b|split\b|fraction\b|index\b|idx\b|indices\b|fold\w*\b|ids?\b)\w+|test|"
               r"\w*hold_?out\w*|[A-Za-z]+_te|te_[A-Za-z]+|\w+_tst|test[A-Z]\w*|[A-Za-z]+Test\b")
TEST_SPLIT = re.compile(r"\b(?:" + _TEST_WORDS + r")\b")
COMMENT = re.compile(r"#[^\n]*")
STRING = re.compile(r"'[^'\n]*'|\"[^\"\n]*\"")


def _test_split_pattern() -> re.Pattern:
    extra = [n.strip() for n in os.environ.get("ML_PIPELINE_TEST_NAMES", "").split(",") if n.strip()]
    if not extra:
        return TEST_SPLIT
    return re.compile(r"\b(?:" + _TEST_WORDS + "|" + "|".join(map(re.escape, extra)) + r")\b")


# ------------------------------------------------------------------ gate state
GATE_LINE = re.compile(r"^.*\bgate\s*[\[(]?([abcd])\b.*$", re.I | re.M)
APPROVED = re.compile(r"\b(?:approved|approved-gate)\b", re.I)
NEGATED = re.compile(r"\b(?:not|pending|awaiting|waiting|todo|unapproved|blocked|in progress)\b", re.I)
OVERRIDE_ANY = re.compile(r"^\s*-?\s*override\s*:.*$", re.I | re.M)
OVERRIDE_TARGET = re.compile(r"override\s*:\s*(gate\s*[abcd]|red\s*flag\s+[a-z][a-z-]*|final\s*test)\b", re.I)
STEP_LINE = re.compile(r"(?:^|(?<=[\'\"\s]))\s*-\s*Step\s+(\d{1,2})\s+(done|skipped)\b\s*[:\-–—]\s*\S", re.I | re.M)
FINAL_TEST_LINE = re.compile(r"(?:^|(?<=[\'\"\s]))\s*-\s*Step 14 final test:", re.I | re.M)
PROTECTED_LINE = re.compile(r"^\s*-\s*(?:Gate\s*[ABCD]\s*:.*\bapproved\b|Override\s*:|Step\s+\d{1,2}\s+(?:done|skipped)\b|"
                            r"Step 14 final test:).*$", re.I | re.M)


def _norm(what: str) -> str:
    return re.sub(r"\s+", " ", what.strip().lower())


def _status_part(line: str) -> str:
    """The part of a ledger line that states its status: up to 'approved', before any free-text reason."""
    cut = len(line)
    for marker in (re.search(r"\breason\b", line, re.I), re.search(r"[(\[]", line)):
        if marker:
            cut = min(cut, marker.start())
    approved = APPROVED.search(line[:cut])
    return line[: approved.end()] if approved else line[:cut]


def override_targets(text: str) -> list[str]:
    """Normalized override targets ('gate b', 'red flag x', 'final test') on non-negated override lines."""
    out = []
    for line in OVERRIDE_ANY.finditer(text):
        m = OVERRIDE_TARGET.search(line.group(0))
        if m and not NEGATED.search(_status_part(line.group(0))):
            out.append(_norm(m.group(1)))
    return out


# ------------------------------------------------------------------ messages
NO_PIPELINE_MSG = (
    "ml-pipeline: model training detected ({0!r}) but there is no ml_pipeline/PIPELINE.md. "
    "Training starts at step 1, not step 10 - run the ml-pipeline skill: inspect the data, do EDA, "
    "define the prediction problem, and get the user's approval at Gate A. "
    "Not an ML project? Set ML_PIPELINE_ENFORCE=0."
)
GATE_A_MSG = (
    "ml-pipeline: nothing is fit before Gate A ({0!r}) except exploratory fits for EDA. Finish steps 1-3 "
    "(data inspection, EDA, prediction-problem contract), get the user's explicit approval in chat, and "
    "record `- Gate A: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md."
)
GATE_B_MSG = (
    "ml-pipeline: model training is locked until Gate B ({0!r}). Finish steps 4-8 (cleaning, "
    "engineering, split, features, preprocessing), get the user's explicit approval in chat, and record "
    "`- Gate B: approved YYYY-MM-DD` in ml_pipeline/PIPELINE.md. If the user explicitly chose to skip "
    "ahead, they say 'override gate B' in chat and you record `- Override: Gate B - ...`."
)
GATE_C_MSG = (
    "ml-pipeline: evaluating on the test split is locked until Gate C ({0!r}). The test set is touched "
    "exactly once, at step 14, after the user approves Gate C - tune and select on validation only."
)
TEST_MSG = (
    "ml-pipeline: this fits on the test split ({0!r}). The test set is touched exactly once, at "
    "step 14, and is never fit on. Fit on the training split only."
)
NO_PROOF_MSG = (
    " PIPELINE.md says Gate {0} is approved, but the user has not approved it in chat. Show them the gate "
    "report and ask; their reply (for example 'approve gate {0}') is recorded automatically."
)
BASELINE_MSG = (
    "ml-pipeline: baseline first ({0!r}). Step 9 trains a dummy predictor and one simple model "
    "(logistic/linear regression or a small tree), logs them with guard.log_run(kind='baseline'), and "
    "records '- Step 9 done: ...'. Then the other candidates. Override only if the user agreed: "
    "'- Override: red flag baseline-first - ...'."
)
TUNING_MSG = (
    "ml-pipeline: tuning comes after training candidates ({0!r}). Record '- Step 10 done: ...' with the "
    "candidates compared on validation (logged with guard.log_run(kind='candidate')) before a hyperparameter "
    "search. Override only if the user agreed: '- Override: red flag tune-before-train - ...'."
)
PROTECTED_MSG = (
    "ml-pipeline: {0} is written only by the guard library and by the user's chat approvals. "
    "Read it with guard.status() instead; never edit, move, or delete it. To bring the guard library into "
    "the project, copy the file with cp rather than writing its contents."
)
LEDGER_MSG = (
    "ml-pipeline: this would remove recorded lines from PIPELINE.md ({0}). Gate approvals, overrides, step "
    "lines, and final-test lines are an append-only record: add new lines, never delete or rewrite these."
)

# ------------------------------------------------------------------ gate and step recording
PROFILE_FILE = "data_profile.json"
FIGURES_DIR = "figures"
MIN_PNG_BYTES = 1024
GATE_REQUIREMENTS = {
    "A": {"steps": (1, 2, 3), "profile": True, "figures": {"01": 1, "02": 2}, "contract": True},
    "B": {"steps": (4, 5, 6, 7, 8), "figures": {"04": 1}, "rationale": True, "split_state": True},
    "C": {"steps": (9, 10, 11, 12), "figures": {"12": 1}, "runs": ("baseline",)},
    "D": {"steps": (13, 14), "figures": {"13": 1}, "final_test_line": True},
}
STEP_GATE = {**{s: "A" for s in range(4, 9)}, **{s: "B" for s in range(9, 13)}, **{s: "C" for s in range(13, 17)}}
MANDATORY_STEPS = {1, 2, 3, 6, 9, 12, 14}
STEP_ARTIFACTS = {
    1: {"profile": True},
    2: {"figures": {"01": 1, "02": 2}},
    3: {"contract": True},
    6: {"split_state": True},
    9: {"runs": ("baseline",)},
    10: {"runs": ("candidate",)},
    12: {"figures": {"12": 1}},
    13: {"figures": {"13": 1}},
    14: {"final_test_line": True},
}
RATIONALE_HEAD = re.compile(r"^\s*Model rationale:\s*$", re.I | re.M)
RATIONALE_BULLETS = ("traits", "baseline", "candidates", "ruled out", "metric")
CONTRACT_HEAD = re.compile(r"^\s*Prediction contract:\s*$", re.I | re.M)
CONTRACT_BULLETS = ("target", "unit", "horizon", "available", "metric")
LEDGER = re.compile(r"^\s*-\s*Figure\s+(\S+\.png):\s*\S", re.I | re.M)
GATE_RECORD_MSG = "ml-pipeline: Gate {0} can't be recorded: {1}."
STEP_RECORD_MSG = "ml-pipeline: Step {0} can't be recorded yet: {1}."


def require_proof() -> bool:
    return os.environ.get("ML_PIPELINE_REQUIRE_USER_APPROVAL", "1").strip().lower() not in {"0", "false", "off", "no"}


def load_approvals(state_dir: Path | None) -> dict:
    """What the user approved in chat, as recorded by hooks/record_approval.py."""
    if state_dir is None:
        return {"gates": {}, "overrides": []}
    path = state_dir / APPROVALS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"gates": data.get("gates") or {}, "overrides": data.get("overrides") or []}
    except (OSError, ValueError):
        return {"gates": {}, "overrides": []}


def user_approved(approvals: dict, gate: str) -> bool:
    return not require_proof() or gate.upper() in approvals.get("gates", {})


def user_overrode(approvals: dict, what: str) -> bool:
    return not require_proof() or any(_norm(str(o.get("what", ""))) == _norm(what) for o in approvals.get("overrides", []))


def approved_gates_in(text: str) -> list[str]:
    """Gate letters that ``text`` records as approved (non-negated lines only)."""
    gates = []
    for match in GATE_LINE.finditer(text):
        line = match.group(0)
        if APPROVED.search(line) and not NEGATED.search(_status_part(line)) and not OVERRIDE_ANY.match(line):
            gates.append(match.group(1).upper())
    return gates


def steps_in(text: str) -> dict[int, str]:
    """{step number: 'done' | 'skipped'} recorded in ``text``; the final-test line counts as step 14 done."""
    steps = {int(m.group(1)): m.group(2).lower() for m in STEP_LINE.finditer(text)}
    if FINAL_TEST_LINE.search(text):
        steps.setdefault(14, "done")
    return steps


def _block_missing(text: str, head: re.Pattern, bullets: tuple[str, ...]) -> list[str]:
    match = head.search(text)
    if not match:
        return list(bullets)
    window = "\n".join(text[match.end():].splitlines()[:15])
    return [b for b in bullets if not re.search(r"^\s*-\s*" + re.escape(b) + r"\b[^:\n]*:\s*\S", window, re.I | re.M)]


def rationale_missing(text: str) -> list[str]:
    """Bullets absent from the Model rationale block; all five when the block itself is absent."""
    return _block_missing(text, RATIONALE_HEAD, RATIONALE_BULLETS)


def contract_missing(text: str) -> list[str]:
    return _block_missing(text, CONTRACT_HEAD, CONTRACT_BULLETS)


def _run_kinds(state_dir: Path) -> set[str]:
    path = state_dir / RUNS_FILE
    kinds = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                kinds.add(str(json.loads(line).get("kind")))
            except ValueError:
                continue
    if kinds & {"tuned", "final"}:
        kinds.add("candidate")
    return kinds


def artifact_problems(required: dict, state_dir: Path, ledger_text: str, check_all_figures: bool = True) -> list[str]:
    """Everything still missing for ``required``, in the words the agent will read."""
    problems: list[str] = []
    if required.get("profile") and not (state_dir / PROFILE_FILE).is_file():
        problems.append(f"{PROFILE_FILE} missing (run guard.profile)")
    figures_dir = state_dir / FIGURES_DIR
    figures = sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else []
    for prefix, need in required.get("figures", {}).items():
        have = [p for p in figures if p.name.startswith(prefix + "_")]
        if len(have) < need:
            problems.append(f"{FIGURES_DIR}/{prefix}_*.png {len(have)} found (need {need})")
    if check_all_figures:
        ledgered = {m.group(1) for m in LEDGER.finditer(ledger_text)}
        for png in figures:
            size = png.stat().st_size
            if size < MIN_PNG_BYTES:
                problems.append(f"{FIGURES_DIR}/{png.name} is {size} bytes (empty?)")
            if png.name not in ledgered:
                problems.append(f"{png.name} has no '- Figure {png.name}:' line")
    if required.get("contract"):
        missing = contract_missing(ledger_text)
        if missing:
            problems.append("Prediction contract block missing lines: " + ", ".join(missing))
    if required.get("rationale"):
        missing = rationale_missing(ledger_text)
        if missing:
            problems.append("Model rationale block missing lines: " + ", ".join(missing))
    if required.get("split_state") and not (state_dir / STATE_FILE).is_file():
        problems.append("no frozen split (run guard.split or guard.split_to_files at step 6)")
    kinds = _run_kinds(state_dir) if required.get("runs") else set()
    for kind in required.get("runs", ()):
        if kind not in kinds:
            problems.append(f"no '{kind}' run in {RUNS_FILE} (log it with guard.log_run(kind='{kind}'))")
    if required.get("final_test_line") and not FINAL_TEST_LINE.search(ledger_text):
        problems.append("no '- Step 14 final test:' line (written by guard.final_test)")
    return problems


def gate_problems(gate: str, state_dir: Path, ledger_text: str, approvals: dict | None = None) -> list[str]:
    """Everything still missing before ``gate`` may be recorded."""
    required = GATE_REQUIREMENTS.get(gate, {})
    steps = steps_in(ledger_text)
    problems = [f"step {s} not recorded ('- Step {s} done: <result>')" for s in required.get("steps", ()) if s not in steps]
    problems += artifact_problems(required, state_dir, ledger_text)
    if approvals is not None and not user_approved(approvals, gate):
        problems.append(f"the user has not approved Gate {gate} in chat - show them the gate report and ask "
                        f"(their reply, e.g. 'approve gate {gate}', is recorded automatically)")
    return problems


def step_problems(step: int, status: str, state_dir: Path, ledger_text: str, gates: dict) -> list[str]:
    """Why ``step`` can't be recorded yet: earlier steps, the phase gate, and the step's own artifacts."""
    steps = steps_in(ledger_text)
    problems = [f"step {s} comes first" for s in range(1, step) if s not in steps]
    gate = STEP_GATE.get(step)
    if gate and not (gates.get(gate) or (gate == "B" and gates.get("override_b"))):
        problems.append(f"Gate {gate} is not approved by the user yet")
    if status == "skipped" and step in MANDATORY_STEPS:
        problems.append(f"step {step} can't be skipped")
    elif status == "done":
        problems += artifact_problems(STEP_ARTIFACTS.get(step, {}), state_dir, ledger_text, check_all_figures=False)
    return problems


def _state_dir_for(tool_input: dict, cwd: str) -> Path:
    """The ml_pipeline/ directory this call concerns: next to the edited PIPELINE.md, else found from cwd."""
    target = str(tool_input.get("file_path") or "")
    if target.endswith("PIPELINE.md"):
        return Path(target).resolve().parent
    found = find_pipeline(cwd)
    return found.parent if found else Path(cwd or ".").resolve() / "ml_pipeline"


# ------------------------------------------------------------------ red flags (need a profile)
RED_FLAGS = [
    ("neural-net-small-data", lambda t: bool(t.get("small_data")),
     re.compile(r"\b(?:MLPClassifier|MLPRegressor|Sequential|keras|torch\.nn|nn\.Module|tensorflow|tf\.keras)\b"),
     "a neural net on {n_rows:,} rows overfits and hides it - start with a regularized linear model or gradient boosting"),
    ("random-split-temporal", lambda t: bool(t.get("has_datetime")),
     re.compile(r"\b(?:train_test_split|KFold|StratifiedKFold|ShuffleSplit|RepeatedKFold|RepeatedStratifiedKFold)\s*\("),
     "the data has a date column, so a random split leaks the future into training - use guard.split(time_col=...), "
     "guard.cv_splits(time_col=...) or TimeSeriesSplit"),
    ("group-split", lambda t: bool(t.get("has_groups")),
     re.compile(r"\b(?:train_test_split|KFold|StratifiedKFold|ShuffleSplit|RepeatedKFold|RepeatedStratifiedKFold|TimeSeriesSplit)\s*\("),
     "rows share an entity ({groups}), so any split that is not group-aware leaks - use guard.split(group_col=...), "
     "guard.cv_splits(group_col=...) or GroupKFold"),
    ("accuracy-imbalanced", lambda t: bool(t.get("imbalanced")),
     re.compile(r"scoring\s*=\s*[\"']accuracy[\"']"),
     "the minority class is {minority_pct} of rows, so accuracy rewards ignoring it - select on average_precision, f1, or balanced_accuracy"),
]
RESAMPLE = re.compile(r"fit_resample\s*\(([^)]*)\)")
OVERRIDE_FLAG = re.compile(r"^.*\boverrid(?:e|den)\b.*\bred flag\s+([a-z-]+)\b.*$", re.I | re.M)
RED_FLAG_MSG = ("ml-pipeline: red flag {0}: {1}. Override only after the user agrees (they say 'override red flag {0}' "
                "in chat) with '- Override: red flag {0} - ...' in ml_pipeline/PIPELINE.md.")


def _load_profile(state_dir: Path) -> dict | None:
    path = state_dir / PROFILE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None


def overridden_flags(markdown: str, approvals: dict | None = None) -> set[str]:
    flags = {m.group(1).lower() for m in OVERRIDE_FLAG.finditer(markdown) if not NEGATED.search(_status_part(m.group(0)))}
    if approvals is not None:
        flags = {f for f in flags if user_overrode(approvals, f"red flag {f}")}
    return flags


def red_flag(code: str, traits: dict, overrides: set[str]) -> str | None:
    """The first red flag the code trips given the profile's traits, or None."""
    values = {"n_rows": int(traits.get("n_rows") or 0),
              "minority_pct": f"{float(traits.get('minority_frac') or 0):.1%}",
              "groups": ", ".join(traits.get("group_candidates") or []) or "group_col"}
    trains = ".fit" in code or bool(TRAIN_API.search(code))
    for name, applies, pattern, why in RED_FLAGS:
        if name in overrides or not applies(traits) or not pattern.search(code):
            continue
        if name == "neural-net-small-data" and not trains:
            continue
        return RED_FLAG_MSG.format(name, why.format(**values))
    if "resample-before-split" not in overrides:
        for match in RESAMPLE.finditer(code):
            if not re.search(r"train|\btr_|_tr\b|\btrn\b|_trn\b", match.group(1), re.I):
                return RED_FLAG_MSG.format("resample-before-split",
                                           f"resampling must fit on the training split only ({match.group(0)[:60]}) - split first, then fit_resample(X_train, y_train)")
    return None


def _strings(obj, out: list[str] | None = None) -> list[str]:
    """Every string value in a tool_input tree, minus keys that describe rather than contain code."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in SKIP_KEYS:
                _strings(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _strings(value, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def _all_strings(obj, out: list[str] | None = None) -> list[str]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for value in obj.values():
            _all_strings(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _all_strings(value, out)
    elif isinstance(obj, str):
        out.append(obj)
    return out


# ------------------------------------------------------------------ scripts a shell command runs
RUNNER = re.compile(
    r"(?:^|[\s;&|(`$])(?:python[\d.]*|py|ipython|uv|poetry|pipenv|conda|pixi|hatch|marimo|jupyter|papermill|"
    r"torchrun|accelerate|deepspeed|bash|sh|zsh|source|Rscript|julia|make|nohup|exec)\b|"
    r"(?:^|[\s;&|])\.{1,2}/\S+|%run\b|runpy\.|exec\s*\(|open\s*\("
)
DATA_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".pkl", ".pickle", ".joblib", ".pt", ".pth",
                 ".h5", ".hdf5", ".npy", ".npz", ".zip", ".gz", ".bz2", ".xz", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
                 ".onnx", ".bin", ".safetensors", ".db", ".sqlite"}
# The plugin's own libraries are recognized by their opening docstring, not their file name.
GUARD_MARKERS = (b"Runtime guard for the ml-pipeline skill", b"A dataset with a known ceiling: beat it and you leaked")
IMPORT_LINE = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:\(([^)]*)\)|([\w \t,]+))|import[ \t]+([\w.]+(?:[ \t]*,[ \t]*[\w.]+)*))", re.M)
MODULE_FLAG = re.compile(r"(?:^|\s)-m\s+([\w.]+)")
CD = re.compile(r"^\s*(?:pushd|cd)\s+([\"']?)([^\"';&|]+)\1\s*$")
SEGMENT = re.compile(r"&&|\|\||;|\n")
MAX_FILES, MAX_BYTES = 40, 5_000_000


def _is_guard_lib(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(400)
    except OSError:
        return False
    return any(marker in head for marker in GUARD_MARKERS)


def _read_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".ipynb":
        try:
            cells = json.loads(text).get("cells", [])
            return "\n".join("".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                             for c in cells if c.get("cell_type") == "code")
        except ValueError:
            return text
    return text


def _candidate_paths(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, comments=True, posix=True)
    except ValueError:
        tokens = command.split()
    tokens += re.findall(r"'([^'\n]{1,300})'", command) + re.findall(r'"([^"\n]{1,300})"', command)
    tokens += re.findall(r"(?:open|run_path|%run|exec_file|source)\s*\(?\s*['\"]?([\w./~-]+\.\w+)", command)
    out = []
    for tok in tokens:
        tok = tok.split("=", 1)[1] if tok.startswith("--") and "=" in tok else tok
        if tok and not tok.startswith("-") and len(tok) < 300:
            out.append(tok)
    return out


def _module_files(name: str, roots: list[Path]) -> list[Path]:
    parts = name.split(".")
    out = []
    for root in roots:
        out += [root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__main__.py"),
                root.joinpath(*parts, "__init__.py")]
    return out


def _take(path: Path, seen: set, sources: list[str]) -> str | None:
    """Read ``path`` once (bounded) unless it is data or the plugin's own library."""
    try:
        if not path.is_file() or path.suffix.lower() in DATA_SUFFIXES:
            return None
        path = path.resolve()
        if path in seen or len(seen) >= MAX_FILES or path.stat().st_size > MAX_BYTES or _is_guard_lib(path):
            return None
    except OSError:
        return None
    seen.add(path)
    text = _read_source(path)
    sources.append(text)
    return text


def executed_sources(command: str, cwd: str, depth: int = 2, seen: set | None = None) -> list[str]:
    """Contents of local scripts/notebooks a shell command runs, and the local modules they import.

    Follows `cd dir && ...`, `python -m pkg.mod`, `from pkg import mod`, `exec(open(...))`, and shell
    scripts that run other scripts (``depth`` levels).
    """
    seen = set() if seen is None else seen
    if depth < 0 or not RUNNER.search(command):
        return []
    base = Path(cwd or ".")
    sources: list[str] = []
    for segment in SEGMENT.split(command):
        cd = CD.match(segment)
        if cd:
            target = Path(os.path.expanduser(cd.group(2).strip()))
            base = target if target.is_absolute() else base / target
            continue
        if not RUNNER.search(segment):
            continue
        paths = [Path(t) if Path(t).is_absolute() else base / t for t in _candidate_paths(segment)]
        for name in MODULE_FLAG.findall(segment):
            paths += _module_files(name, [base, Path(cwd or ".")])
        for path in paths:
            text = _take(path, seen, sources)
            if text is None:
                continue
            here = path.resolve().parent
            sources += _imported_sources(text, here, base, seen)
            sources += executed_sources(text, str(here), depth - 1, seen)
    return sources


def _imported_sources(text: str, here: Path, base: Path, seen: set, depth: int = 2) -> list[str]:
    if depth < 0:
        return []
    out: list[str] = []
    for match in IMPORT_LINE.finditer(text):
        if match.group(1):
            listed = match.group(2) if match.group(2) is not None else match.group(3)
            names = [n.split()[0] for n in listed.replace("\n", " ").split(",") if n.strip()]
            candidates = [match.group(1)] + [f"{match.group(1)}.{n}" for n in names]
        else:
            candidates = [n.strip() for n in match.group(4).split(",")]
        for name in candidates:
            for path in _module_files(name, [here, base]):
                found = _take(path, seen, out)
                if found is not None:
                    out += _imported_sources(found, path.resolve().parent, base, seen, depth - 1)
    return out


# ------------------------------------------------------------------ analysis
def _args(text: str, start: int, limit: int = 400) -> str:
    """The argument list starting just after an opening paren, up to its matching close."""
    depth, i = 1, start
    while i < len(text) and i - start < limit:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start:i]


def _constructor(receiver: str, code_before: str) -> str | None:
    """'Name(args...)' from the last `receiver = Name(...)` before the call, or None."""
    if not receiver:
        return None
    last = None
    for m in re.finditer(r"(?<![\w.])" + re.escape(receiver) + r"\s*=\s*([A-Za-z_][\w.]*)\s*\(", code_before):
        last = m
    if last is None:
        return None
    return last.group(1) + "(" + _args(code_before, last.end())


def _classify(receiver: str, code_before: str = "") -> str:
    """'explore', 'transformer', or 'model' for a fit on ``receiver``.

    What the receiver was built from decides first (`km = RandomForestClassifier()` is a model whatever
    its name), then the receiver's own name. Anything unknown counts as a model: the conservative
    reading is the one that keeps the test set honest.
    """
    built = _constructor(receiver, code_before)
    if built is not None:
        name = built.split("(", 1)[0].split(".")[-1]
        if ESTIMATOR.search(built) or TRAIN_HELPER.search(built):
            return "model"
        if EXPLORATORY.fullmatch(name) or EXPLORATORY.search(built.split("(", 1)[0]):
            return "explore"
        if TRANSFORMER.search(built):
            return "transformer"
        return "model"
    if ESTIMATOR.search(receiver):
        return "model"
    if EXPLORATORY_VAR.match(receiver) or (receiver and EXPLORATORY.fullmatch(receiver)):
        return "explore"
    if TRANSFORMER_VAR.match(receiver) or TRANSFORMER.search(receiver):
        return "transformer"
    return "model"


def _code(text: str) -> str:
    """Code to scan: comments removed, and literal \\n / \\t (one-line `python -c "a\\nb"`) turned into real ones."""
    return COMMENT.sub("", text.replace("\\n", "\n").replace("\\t", "\t"))


def analyse(text: str) -> dict[str, str | None]:
    """First matching snippet for each kind of finding, or None."""
    code = _code(text)
    test_split = _test_split_pattern()
    found: dict[str, str | None] = {"model": None, "transformer": None, "explore": None, "fit_on_test": None,
                                    "eval_on_test": None, "tuning": None, "non_baseline": None}
    api = TRAIN_API.search(code)
    if api:
        found["model"] = found["non_baseline"] = api.group(0)
    for match in FIT_CALL.finditer(code):
        args = _args(code, match.end())
        snippet = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
        if test_split.search(STRING.sub('""', args)) and found["fit_on_test"] is None:
            found["fit_on_test"] = snippet
        kind = _classify(match.group(1) or "", code[: match.start()])
        if found[kind] is None:
            found[kind] = snippet
    for pattern in (GETATTR_FIT, FIT_REF):
        for match in pattern.finditer(code):
            kind = _classify(match.group(1) or "", code[: match.start()])
            if found[kind] is None:
                found[kind] = code[match.start(): match.end()][:SNIPPET]
    helper = TRAIN_HELPER.search(code)
    if helper:
        found["model"] = found["model"] or helper.group(0)
    tuning = TUNING.search(code)
    if tuning:
        found["tuning"] = tuning.group(0)
        found["model"] = found["model"] or tuning.group(0)
    if found["model"]:
        for match in ESTIMATOR.finditer(STRING.sub('""', code)):  # a plot title saying "Regression" is not a model
            if not BASELINE.fullmatch(match.group(0)) and not re.fullmatch(r"\w*(?:Transformer|Encoder)\w*", match.group(0)):
                found["non_baseline"] = found["non_baseline"] or match.group(0)
                break
    for match in EVAL_CALL.finditer(code):
        args = _args(code, match.end())
        if test_split.search(STRING.sub('""', args)) and found["eval_on_test"] is None:
            found["eval_on_test"] = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
    final = FINAL_TEST_CALL.search(code)
    if final and found["eval_on_test"] is None:
        found["eval_on_test"] = code[final.start(): final.end() + 60][:SNIPPET]
    return found


def gate_state(markdown: str, approvals: dict | None = None) -> dict[str, bool]:
    """Which gates count as approved: a non-negated ledger line AND (unless disabled) the user's chat approval.

    ``ledger_*`` keys say what the ledger alone claims, so a denial can explain a missing chat approval.
    """
    approvals = approvals if approvals is not None else {"gates": {}, "overrides": []}
    state = {"A": False, "B": False, "C": False, "D": False, "override_b": False, "override_c": False}
    for gate in approved_gates_in(markdown):
        state[f"ledger_{gate}"] = True
        if user_approved(approvals, gate):
            state[gate] = True
    targets = override_targets(markdown)
    state["override_b"] = "gate b" in targets and user_overrode(approvals, "gate b")
    state["override_c"] = "gate c" in targets and user_overrode(approvals, "gate c")
    # backwards-compatible alias used by older callers
    state["override"] = state["override_b"]
    return state


def find_pipeline(cwd: str) -> Path | None:
    override = os.environ.get("ML_PIPELINE_FILE")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    here = Path(cwd or ".").resolve()
    for _ in range(SEARCH_UP + 1):
        candidate = here / PIPELINE_REL
        if candidate.is_file():
            return candidate
        if here.parent == here:
            break
        here = here.parent
    return None


def _proof_hint(gates: dict, gate: str) -> str:
    return NO_PROOF_MSG.format(gate) if gates.get(f"ledger_{gate}") and not gates.get(gate) else ""


def decide(found: dict[str, str | None], gates: dict[str, bool] | None, steps: dict[int, str] | None = None,
           overrides: set[str] | None = None) -> str | None:
    """A denial reason, or None to allow. ``gates`` is None when no PIPELINE.md exists."""
    steps = steps or {}
    overrides = overrides or set()
    if found["fit_on_test"]:
        return TEST_MSG.format(found["fit_on_test"])
    if found.get("eval_on_test") and not (gates and (gates["C"] or gates["override_c"])):
        return GATE_C_MSG.format(found["eval_on_test"]) + (_proof_hint(gates, "C") if gates else "")
    if not (found["model"] or found["transformer"]):
        return None
    if gates is None:
        return NO_PIPELINE_MSG.format(found["model"]) if found["model"] else None
    if gates["B"] or gates["override_b"]:
        if found["model"]:
            if found.get("tuning") and 10 not in steps and "tune-before-train" not in overrides:
                return TUNING_MSG.format(found["tuning"])
            if found.get("non_baseline") and 9 not in steps and "baseline-first" not in overrides:
                return BASELINE_MSG.format(found["non_baseline"])
        return None
    if gates["A"]:
        return (GATE_B_MSG.format(found["model"]) + _proof_hint(gates, "B")) if found["model"] else None
    if not found["model"] and found.get("explore"):
        return None  # scaling only to feed an EDA embedding/clustering (e.g. StandardScaler -> t-SNE) is exploration
    return GATE_A_MSG.format(found["model"] or found["transformer"]) + _proof_hint(gates, "A")


# ------------------------------------------------------------------ tamper protection
# The state files by name, by the library's constants and writers, and by common string-splitting.
PROTECTED_NAME = re.compile(
    r"\.guard_state\.json|\.approvals\.json|\bAPPROVALS_FILE\b|\bSTATE_FILE\b|\b_write_state\s*\(|\brecord_approval\b|"
    r"[\"'][^\"'\n]*(?:\.appr|approvals?\.js|guard_st|_state\.js)[^\"'\n]*[\"']"
)
READ_ONLY_TOOL = re.compile(r"^\s*(?:cat|less|more|head|tail|jq|ls|stat|wc|file|grep|egrep|fgrep|rg|sort|uniq|cut|column|"
                            r"diff|echo|printf|pwd|which|du|df|tree|basename|dirname|realpath|nl)\b")
DESTRUCTIVE_CMD = re.compile(r"(?:^|[;&|(`]\s*|\bsudo\s+|\bxargs\s+)(?:rm|mv|dd|truncate|shred|unlink)\s|"
                             r"\bsed\s+(?:-\w*\s+)*-\w*i|\bperl\s+-\w*i|\bcp\b[^\n;&|]*PIPELINE\.md\s*(?:$|[;&|])|"
                             r"\btee\s+(?!-a\b)(?!--append\b)[^\n;&|]*PIPELINE\.md")
DESTRUCTIVE_WRITE = re.compile(r"(?<![>&\d])>(?!>)\s*[\"']?[^\s|;&]*PIPELINE\.md|\bopen\s*\([^)]*PIPELINE\.md[^)]*[\"'][wx]|"
                               r"PIPELINE\.md[^\n]*\.write_text\s*\(")
GIT_RESTORE = re.compile(r"(?:^|[;&|(]\s*)git\s+(?:checkout|restore|reset|stash|clean|switch)\b[^\n;&|]*")


def read_only_command(command: str) -> bool:
    """A command made only of reading tools: no redirection, substitution, or execution."""
    if not command.strip() or re.search(r"(?<![\d&])>|`|\$\(|<\(", command):
        return False
    if re.search(r"\bfind\b.*-(?:exec|delete|ok)\b", command):
        return False
    return all(READ_ONLY_TOOL.match(part) for part in re.split(r"\|\|?|&&|;|\n", command) if part.strip())


def _ledger_recorded(cwd: str) -> bool:
    ledger = find_pipeline(cwd)
    return ledger is not None and bool(PROTECTED_LINE.search(ledger.read_text(encoding="utf-8", errors="replace")))


def protected_violation(tool_name: str, tool_input: dict, cwd: str = "", executed: list[str] | None = None) -> str | None:
    """Deny writes to the guard's state files and deletion of recorded PIPELINE.md lines."""
    command = str(tool_input.get("command") or "")
    for text in _all_strings(tool_input) + list(executed or []):
        match = PROTECTED_NAME.search(text)
        if match:
            return PROTECTED_MSG.format(match.group(0).strip("'\""))
    target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if target.endswith("PIPELINE.md"):
        edits = tool_input.get("edits") if isinstance(tool_input.get("edits"), list) else []
        pairs = [(tool_input.get("old_string"), tool_input.get("new_string"))] + [(e.get("old_string"), e.get("new_string")) for e in edits if isinstance(e, dict)]
        for old, new in pairs:
            if isinstance(old, str):
                lost = [m.group(0).strip() for m in PROTECTED_LINE.finditer(old) if m.group(0).strip() not in (new or "")]
                if lost:
                    return LEDGER_MSG.format(lost[0][:80])
        if isinstance(tool_input.get("content"), str):
            path = Path(target)
            existing = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            lost = [m.group(0).strip() for m in PROTECTED_LINE.finditer(existing) if m.group(0).strip() not in tool_input["content"]]
            if lost:
                return LEDGER_MSG.format(lost[0][:80])
    if command:
        unquoted = STRING.sub("''", command)
        rewrites = "PIPELINE.md" in command and (DESTRUCTIVE_CMD.search(unquoted) or DESTRUCTIVE_WRITE.search(command))
        restore = GIT_RESTORE.search(command)
        restores_state = restore and re.search(r"ml_pipeline|--hard|\s--\s+\.|\bclean\b|\bstash\b", restore.group(0))
        if (rewrites or restores_state) and _ledger_recorded(cwd):  # a ledger with nothing recorded yet is fair game
            what = "shell command rewrites, moves, or deletes PIPELINE.md" if rewrites else \
                "git would restore ml_pipeline/ to an older state (resetting the test-set count and approvals)"
            return LEDGER_MSG.format(what)
    for source in executed or []:
        if "PIPELINE.md" in source and DESTRUCTIVE_WRITE.search(source) and _ledger_recorded(cwd):
            return LEDGER_MSG.format("a script this command runs rewrites PIPELINE.md")
    return None


def _after_text(tool_input: dict, existing: str, text: str) -> str:
    """PIPELINE.md as it would read after this call."""
    if isinstance(tool_input.get("content"), str):
        return tool_input["content"]
    pairs = [(tool_input.get("old_string"), tool_input.get("new_string"))]
    pairs += [(e.get("old_string"), e.get("new_string")) for e in tool_input.get("edits") or [] if isinstance(e, dict)]
    after, edited = existing, False
    for old, new in pairs:
        if isinstance(old, str) and isinstance(new, str):
            after, edited = (after.replace(old, new, 1) if old else after + new), True
    return after if edited else existing + "\n" + text


def recording_problem(tool_input: dict, text: str, cwd: str) -> str | None:
    """Recording a gate, override, or step line in PIPELINE.md: the prerequisites must already hold."""
    target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if target and not target.endswith("PIPELINE.md"):
        return None  # a step or gate line in another file is not a recording
    if not target and not (STEP_LINE.search(text) or GATE_LINE.search(text) or OVERRIDE_ANY.search(text)):
        return None  # shell commands are checked whatever file name they spell, since names can be built
    state_dir = _state_dir_for(tool_input, cwd)
    pipeline_md = state_dir / "PIPELINE.md"
    existing = pipeline_md.read_text(encoding="utf-8", errors="replace") if pipeline_md.is_file() else ""
    after = _after_text(tool_input, existing, text)
    approvals = load_approvals(state_dir)
    before_steps, after_steps = steps_in(existing), steps_in(after)
    gates_now = gate_state(existing, approvals)
    for step, status in sorted(after_steps.items()):
        if step in before_steps:
            continue
        problems = step_problems(step, status, state_dir, after, gates_now)
        if problems:
            return STEP_RECORD_MSG.format(step, "; ".join(problems))
    for gate in sorted(set(approved_gates_in(after)) - set(approved_gates_in(existing))):
        problems = gate_problems(gate, state_dir, after, approvals)
        if problems:
            return GATE_RECORD_MSG.format(gate, "; ".join(problems))
    wanted = override_targets(after)
    for what in sorted(set(wanted)):
        granted = sum(1 for o in approvals.get("overrides", []) if _norm(str(o.get("what", ""))) == what)
        if require_proof() and wanted.count(what) > granted and wanted.count(what) > override_targets(existing).count(what):
            return (f"ml-pipeline: '- Override: {what}' can't be recorded: the user has not asked for it in chat. "
                    f"Explain the risk in 2-3 sentences and ask; if they reply 'override {what}', it is recorded "
                    "automatically and this line is then accepted.")
    return None


def _emit(reason: str | None) -> int:
    if reason is None:
        print("{}")
    else:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
            "systemMessage": reason,
        }))
    return 0


def main() -> int:
    try:
        if os.environ.get("ML_PIPELINE_ENFORCE", "1").strip().lower() in {"0", "false", "off", "no"}:
            return _emit(None)
        payload = json.load(sys.stdin)
        tool_input = payload.get("tool_input") or {}
        tool_name = str(payload.get("tool_name") or "")
        cwd = payload.get("cwd") or os.getcwd()
        text = "\n".join(_strings(tool_input))
        command = tool_input.get("command") if isinstance(tool_input.get("command"), str) else None
        # Reading (cat, grep, jq, ...) changes nothing and runs nothing.
        if command is not None and read_only_command(command):
            return _emit(None)
        # Shell commands are judged with the local scripts, notebooks and modules they run.
        executed = executed_sources(command, cwd) if command is not None else []
        # 0. The guard's own state and the ledger's recorded lines are not editable.
        reason = protected_violation(tool_name, tool_input, cwd, executed)
        if reason:
            return _emit(reason)
        # 1. Recording a gate, override, or step: its prerequisites must already hold.
        reason = recording_problem(tool_input, "\n".join([text, *executed]), cwd)
        if reason:
            return _emit(reason)
        # 2. Prose files are not code (running one is checked when a command executes it).
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if target and Path(target).suffix.lower() in DOC_SUFFIXES:
            return _emit(None)
        text = "\n".join([text, *executed])
        # 2b. Red flags: wrong applications the data profile can prove.
        state_dir = _state_dir_for(tool_input, cwd)
        pipeline = find_pipeline(cwd)
        if pipeline is not None:
            state_dir = pipeline.parent
        approvals = load_approvals(state_dir)
        markdown = pipeline.read_text(encoding="utf-8", errors="replace") if pipeline else ""
        overrides = overridden_flags(markdown, approvals)
        profile = _load_profile(state_dir)
        if profile is not None:
            reason = red_flag(_code(text), profile.get("traits") or {}, overrides)
            if reason:
                return _emit(reason)
        # 3. Fast path: nothing to look at.
        if (not re.search(r"\.fit\b|fit\w*\s*\(|fit[\"']", text) and not TRAIN_API.search(text)
                and not EVAL_CALL.search(text) and not TUNING.search(text) and not TRAIN_HELPER.search(text)
                and not FINAL_TEST_CALL.search(text)):
            return _emit(None)
        found = analyse(text)
        if not any(found.values()):
            return _emit(None)
        gates = gate_state(markdown, approvals) if pipeline else None
        return _emit(decide(found, gates, steps_in(markdown), overrides))
    except Exception as exc:  # fail open: never lock the user out because of us
        print(json.dumps({"systemMessage": f"ml-pipeline hook error (call allowed): {exc}"}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
