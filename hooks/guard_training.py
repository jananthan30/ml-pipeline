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
exist, and a gate needs its steps, its artifacts, its per-column records (gate_records.py), and the
user's approval. Only strict ledger lines in the newest run count: ``- Gate X: approved ...`` and
``- Override: <what> - ...``, after the last ``## Run:`` heading (or anywhere when there is none); for
each gate the last such line decides.

The hook fails open: an unexpected error allows the call and reports the error, so a bug here can
never lock anyone out of their own tools - except that a crash while checking a gate's records
denies that one ledger write, so a gate is never recorded unchecked. The user (never the agent)
sets ``ML_PIPELINE_ENFORCE=0`` to switch it off for projects that are not ML pipelines, or
``ML_PIPELINE_REQUIRE_USER_APPROVAL=0`` to fall back to ledger-only gates. Stdlib only; no
subprocesses, no network.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_records  # noqa: E402

PIPELINE_REL = Path("ml_pipeline") / "PIPELINE.md"
SEARCH_UP = 4  # look in cwd and this many parent directories
# Prose and config are not code; running one is still checked. A .txt file is scanned like code: it can be run.
DOC_SUFFIXES = {".md", ".markdown", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
PROSE_SUFFIXES = {".md", ".markdown", ".rst"}
# The plugin runs in every project; a fit/step/train call in these languages is not model training.
OTHER_LANGUAGE_SUFFIXES = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".html", ".htm", ".css", ".scss",
    ".java", ".kt", ".scala", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".rb",
    ".php", ".dart", ".lua", ".sql",
}
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
    r"\b(?:xgb|xgboost|lgb|lightgbm|catboost)\s*\.\s*(?:train|cv)\s*\(|\btrainer\s*\.\s*train\s*\(|"
    r"(?<!class )\b\w*Trainer\s*\(|"
    r"\.fit_generator\s*\(|\.fit_one_cycle\s*\(|\bloss\.backward\s*\(|\boptimizer\.step\s*\("
)
# A gradient step under any names: a backward pass plus an optimizer step in the same code.
BACKWARD = re.compile(r"\.\s*backward\s*\(")
OPT_STEP = re.compile(r"\.\s*step\s*\(")
# Helpers that fit a model inside them (cross-validation, learning curves).
TRAIN_HELPER = re.compile(r"\b(?:cross_val_score|cross_validate|cross_val_predict|learning_curve|validation_curve|"
                          r"permutation_test_score|permutation_importance)\s*\(")
# A slice of the rows inside a model-training call, or nrows= in a reader in code that trains: a pilot.
# Row slices with a positive stop or a negative start (`[:5000]`, `.iloc[:500]`, `[-5000:]`), head() and sample().
SLICE = re.compile(r"\[\s*-?\d*\s*:\s*\d+\s*\]|\[\s*-\d+\s*:\s*\]|\.\s*i?loc\s*\[\s*-?\d*\s*:\s*\d+|"
                   r"\.\s*(?:head|sample)\s*\(")
FULL_SHUFFLE = re.compile(r"\bfrac\s*=\s*1(?:\.0*)?\b")
READER = re.compile(r"\bread_(?:csv|table|excel|json|fwf|parquet|feather|sas|stata|spss|sql)\s*\(")
NROWS = re.compile(r"\bnrows\s*=\s*(?!0\s*(?:[,)]|$))")
SELECTOR_NAMES = r"SelectKBest|SelectPercentile|GenericUnivariateSelect|SelectFpr|SelectFdr|SelectFwe"
SELECTOR = re.compile(r"\b(?:" + SELECTOR_NAMES + r")\b")
SELECTION_SCORE = re.compile(
    r"\b(?:mutual_info_classif|mutual_info_regression|f_classif|f_regression|chi2|r_regression)\s*\(")
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
RUN_HEAD = re.compile(r"^[ \t]*##[ \t]+Run[ \t]*:[^\n]*", re.I | re.M)
# Strict shapes, at a line start or right after a quote (as in `echo '- Gate A: approved ...' >> PIPELINE.md`).
_LINE_START = r"(?:^|(?<=[\'\"]))[ \t]*-[ \t]*"
GATE_RECORD = re.compile(_LINE_START + r"Gate[ \t]+([A-D])[ \t]*:[ \t]*(\S*)", re.I | re.M)
OVERRIDE_WHAT = r"gate[ \t]+[abcd]|red[ \t]+flag[ \t]+[a-z][a-z-]*[a-z]|final[ \t]+test|pilot|partial[ \t]+data"
OVERRIDE_LINE = re.compile(_LINE_START + r"Override[ \t]*:[ \t]*(" + OVERRIDE_WHAT + r")[ \t]*(?:-|$)", re.I | re.M)
OVERRIDE_ANY = re.compile(_LINE_START + r"Override[ \t]*:", re.I | re.M)
GATE_LINE = re.compile(_LINE_START + r"Gate[ \t]+[A-D][ \t]*:", re.I | re.M)
STEP_LINE = re.compile(r"(?:^|(?<=[\'\"\s]))\s*-\s*Step\s+(\d{1,2})\s+(done|skipped)\b\s*[:\-–—]\s*\S", re.I | re.M)
FINAL_TEST_LINE = re.compile(r"(?:^|(?<=[\'\"\s]))\s*-\s*Step 14 final test:", re.I | re.M)
PROTECTED_LINE = re.compile(r"^\s*-\s*(?:Gate\s*[ABCD]\s*:.*\bapproved\b|Override\s*:|Step\s+\d{1,2}\s+(?:done|skipped)\b|"
                            r"Step 14 final test:).*$", re.I | re.M)


def _norm(what: str) -> str:
    return re.sub(r"\s+", " ", what.strip().lower())


def current_run(markdown: str) -> str:
    """The newest run: everything from the last ``## Run:`` heading on, or the whole file when there is none."""
    heads = list(RUN_HEAD.finditer(markdown))
    return markdown[heads[-1].start():] if heads else markdown


def run_label(markdown: str) -> str:
    """The newest ``## Run:`` heading ('' when there is none): approvals belong to the run they were given in."""
    heads = list(RUN_HEAD.finditer(markdown))
    return " ".join(heads[-1].group(0).split()) if heads else ""


def override_targets(text: str) -> list[str]:
    """Normalized targets ('gate b', 'red flag x', 'final test', 'pilot', 'partial data') of strict override lines."""
    return [_norm(m.group(1)) for m in OVERRIDE_LINE.finditer(text)]


# ------------------------------------------------------------------ messages
NO_PIPELINE_MSG = (
    "ml-pipeline: model training detected ({0!r}) but there is no ml_pipeline/PIPELINE.md. "
    "Training starts at step 1, not step 10 - run the ml-pipeline skill: inspect the data, do EDA, "
    "define the prediction problem, and get the user's approval at Gate A. "
    "Not an ML project? Ask the user to switch enforcement off."
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
    "ahead, they say 'override gate B' in chat and you record `- Override: Gate B - ...`; an override needs the "
    "same records as an approval."
)
GATE_C_MSG = (
    "ml-pipeline: evaluating on the test split is locked until Gate C ({0!r}). The test set is touched "
    "exactly once, at step 14, after the user approves Gate C - tune and select on validation only. If the user "
    "explicitly chose otherwise, they say 'override gate C' in chat and you record `- Override: Gate C - ...`; an "
    "override needs the same records as an approval."
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
SELF_DISABLE_MSG = (
    "ml-pipeline: switching enforcement or the user-approval rule off, or pointing the hook at a different ledger, "
    "is the user's decision - an agent may not change that setting. If this project is not an ML pipeline, ask the "
    "user to switch enforcement off themselves."
)
PILOT_MSG = (
    "ml-pipeline: this trains on a slice of the data ({0!r}). Train on the full training split. If the user "
    "explicitly agreed to a quick pilot, they say 'override pilot' in chat and you record "
    "`- Override: pilot - <size> - user approved YYYY-MM-DD - reason: ...`; Gate C still needs a full-data training run."
)
TARGET_STATS_WHY = ("target-based feature selection must use the training split only ({0}) - select features inside "
                    "the training data or a CV fold")
GATE_CHECK_FAILED_MSG = ("ml-pipeline: could not check the Gate {0} records ({1}); fix the file it names, re-run the "
                         "guard helper that writes it, and retry.")
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
LEDGER = re.compile(r"^\s*-\s*Figure\s+(\S+\.png):[ \t]*(\S[^\n]*)$", re.I | re.M)
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


def _in_run(entry, run: str) -> bool:
    return isinstance(entry, dict) and str(entry.get("run") or "") == run


def user_approved(approvals: dict, gate: str, run: str = "") -> bool:
    """The user approved ``gate`` in chat during the run labelled ``run`` (see run_label)."""
    return not require_proof() or _in_run(approvals.get("gates", {}).get(gate.upper()), run)


def granted_overrides(approvals: dict, what: str, run: str = "") -> int:
    return sum(1 for o in approvals.get("overrides", []) if _in_run(o, run) and _norm(str(o.get("what", ""))) == _norm(what))


def user_overrode(approvals: dict, what: str, run: str = "") -> bool:
    return not require_proof() or granted_overrides(approvals, what, run) > 0


def _gate_lines(text: str) -> list[tuple[str, str, str]]:
    """(gate, 'approved' | 'override' | 'other', whole stripped line) for each strict gate line, in order."""
    found = [(m.start(), m.group(1).upper(), "approved" if m.group(2).lower().startswith("approved") and
              re.fullmatch(r"approved[.,;:]?", m.group(2), re.I) else "other") for m in GATE_RECORD.finditer(text)]
    found += [(m.start(), m.group(1)[-1].upper(), "override") for m in OVERRIDE_LINE.finditer(text)
              if m.group(1).lower().startswith("gate")]
    out = []
    for start, gate, status in sorted(found):
        line_start = text.rfind("\n", 0, start) + 1
        end = text.find("\n", start)
        out.append((gate, status, text[line_start: end if end != -1 else len(text)].strip()))
    return out


def _last_status(text: str) -> dict[str, str]:
    """For each gate, the status its last strict line in the newest run gives."""
    last: dict[str, str] = {}
    for gate, status, _ in _gate_lines(current_run(text)):
        last[gate] = status
    return last


def approved_gates_in(text: str) -> list[str]:
    """Gate letters the newest run in ``text`` records as approved: the last strict line is `- Gate X: approved`."""
    return [g for g, status in sorted(_last_status(text).items()) if status == "approved"]


def steps_in(text: str) -> dict[int, str]:
    """{step: 'done' | 'skipped'} recorded in the newest run of ``text``; the final-test line counts as step 14 done."""
    run = current_run(text)
    steps = {int(m.group(1)): m.group(2).lower() for m in STEP_LINE.finditer(run)}
    if FINAL_TEST_LINE.search(run):
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


RECORD_CHECKS = {"A": gate_records.gate_a_problems, "B": gate_records.gate_b_problems, "C": gate_records.gate_c_problems}


def gate_problems(gate: str, state_dir: Path, ledger_text: str, approvals: dict | None = None,
                  override: bool = False) -> list[str]:
    """Everything still missing before ``gate`` may be recorded (approved, or overridden: same records either way)."""
    required = GATE_REQUIREMENTS.get(gate, {})
    steps = steps_in(ledger_text)
    problems = [f"step {s} not recorded ('- Step {s} done: <result>')" for s in required.get("steps", ()) if s not in steps]
    problems += artifact_problems(required, state_dir, ledger_text)
    if gate in RECORD_CHECKS:
        problems += RECORD_CHECKS[gate](state_dir, current_run(ledger_text))
    run = run_label(ledger_text)
    if approvals is not None and override and not user_overrode(approvals, f"gate {gate.lower()}", run):
        problems.append(f"the user has not asked to override Gate {gate} in chat - explain the risk and ask "
                        f"(their reply, e.g. 'override gate {gate}', is recorded automatically)")
    elif approvals is not None and not override and not user_approved(approvals, gate, run):
        problems.append(f"the user has not approved Gate {gate} in chat - show them the gate report and ask "
                        f"(their reply, e.g. 'approve gate {gate}', is recorded automatically)")
    return problems


def step_problems(step: int, status: str, state_dir: Path, ledger_text: str, gates: dict) -> list[str]:
    """Why ``step`` can't be recorded yet: earlier steps, the phase gate, and the step's own artifacts."""
    steps = steps_in(ledger_text)
    problems = [f"step {s} comes first" for s in range(1, step) if s not in steps]
    gate = STEP_GATE.get(step)
    if gate and not (gates.get(gate) or gates.get(f"override_{gate.lower()}")):
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
    """Red flags the newest run overrides with a strict ``- Override: red flag <name> - ...`` line (and, with
    ``approvals``, that the user asked for in chat during this run)."""
    run = run_label(markdown)
    flags = {t[len("red flag "):] for t in override_targets(current_run(markdown)) if t.startswith("red flag ")}
    if approvals is not None:
        flags = {f for f in flags if user_overrode(approvals, f"red flag {f}", run)}
    return flags


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
    for match in re.finditer(r"\b(?:" + SELECTOR_NAMES + r")\s*\([^()]*\)\s*\.\s*fit\w*\s*\(", code):
        args = _args(code, match.end())
        if "train" not in args.lower():
            return code[match.start(): match.end() + len(args) + 1].strip()[:60]
    return None


def red_flag(code: str, traits: dict, overrides: set[str]) -> str | None:
    """The first red flag the code trips given the profile's traits, or None."""
    values = {"n_rows": int(traits.get("n_rows") or 0),
              "minority_pct": f"{float(traits.get('minority_frac') or 0):.1%}",
              "groups": ", ".join(traits.get("group_candidates") or []) or "group_col"}
    trains = ".fit" in code or bool(TRAIN_API.search(code)) or bool(BACKWARD.search(code) and OPT_STEP.search(code))
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
    if "target-stats-outside-train" not in overrides:
        snippet = _target_stats_outside_train(code)
        if snippet:
            return RED_FLAG_MSG.format("target-stats-outside-train", TARGET_STATS_WHY.format(snippet))
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
    r"(?:^|[\s;&|(`$/])(?:python[\d.]*|py|ipython|uv|poetry|pipenv|conda|pixi|hatch|marimo|jupyter|papermill|"
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
# Converting a notebook without executing it runs nothing.
NOT_A_RUN = re.compile(r"^\s*jupyter\s+nbconvert\b(?!.*--execute)")
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
        try:
            cd = CD.match(segment)
            if cd:
                target = Path(os.path.expanduser(cd.group(2).strip()))
                base = target if target.is_absolute() else base / target
                continue
            if not RUNNER.search(segment) or NOT_A_RUN.search(segment):
                continue
            for code in re.findall(r"(?:^|\s)-\w*c\s+(?:'([^']*)'|\"([^\"]*)\")", segment):
                inner = code[0] or code[1]  # sh -c '...', bash -lc "...": the string is a command line
                sources.append(inner)
                sources += executed_sources(inner, str(base), depth - 1, seen)
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
        except Exception:  # noqa: BLE001 - one unreadable path must not fail the whole hook open
            continue
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


def _slices_rows(args: str) -> bool:
    """True when a fit call's arguments take a slice of the rows (not columns, not a shift, not a shuffle)."""
    for match in SLICE.finditer(args):
        before = args[:match.start()]
        if before.count("[") > before.count("]") or before.rstrip().endswith(".columns"):
            continue  # inside another subscript, or a slice of column names: column selection
        if "sample" in match.group(0) and FULL_SHUFFLE.search(_args(args, match.end())):
            continue  # sample(frac=1) shuffles every row
        return True
    return False


def _code(text: str) -> str:
    """Code to scan: comments removed, and literal \\n / \\t (one-line `python -c "a\\nb"`) turned into real ones."""
    return COMMENT.sub("", text.replace("\\n", "\n").replace("\\t", "\t"))


def analyse(text: str) -> dict[str, str | None]:
    """First matching snippet for each kind of finding, or None."""
    code = _code(text)
    test_split = _test_split_pattern()
    found: dict[str, str | None] = {"model": None, "transformer": None, "explore": None, "fit_on_test": None,
                                    "eval_on_test": None, "tuning": None, "non_baseline": None, "pilot": None}
    api = TRAIN_API.search(code)
    if api:
        found["model"] = found["non_baseline"] = api.group(0)
    backward = BACKWARD.search(code)
    if found["model"] is None and backward and OPT_STEP.search(code):
        line_start = code.rfind("\n", 0, backward.start()) + 1
        found["model"] = found["non_baseline"] = code[line_start: backward.end() + 1].strip()[:SNIPPET]
    for match in FIT_CALL.finditer(code):
        args = _args(code, match.end())
        snippet = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
        if test_split.search(STRING.sub('""', args)) and found["fit_on_test"] is None:
            found["fit_on_test"] = snippet
        kind = _classify(match.group(1) or "", code[: match.start()])
        if kind == "model" and found["pilot"] is None and _slices_rows(args):
            found["pilot"] = snippet.strip()
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
    if found["model"] and found["pilot"] is None:
        for match in READER.finditer(code):
            args = _args(code, match.end())
            if NROWS.search(args):
                found["pilot"] = code[match.start(): match.end() + len(args) + 1][:SNIPPET]
                break
    return found


def gate_state(markdown: str, approvals: dict | None = None) -> dict[str, bool]:
    """Which gates count as approved: a non-negated ledger line AND (unless disabled) the user's chat approval.

    ``ledger_*`` keys say what the ledger alone claims, so a denial can explain a missing chat approval.
    """
    approvals = approvals if approvals is not None else {"gates": {}, "overrides": []}
    run = run_label(markdown)
    state = {"A": False, "B": False, "C": False, "D": False, "override_b": False, "override_c": False}
    for gate, status in _last_status(markdown).items():
        if status == "approved":
            state[f"ledger_{gate}"] = True
            state[gate] = user_approved(approvals, gate, run)
        elif status == "override" and gate in "BC":
            state[f"override_{gate.lower()}"] = user_overrode(approvals, f"gate {gate.lower()}", run)
    state["pilot"] = "pilot" in override_targets(current_run(markdown)) and user_overrode(approvals, "pilot", run)
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
            if found.get("pilot") and not gates.get("pilot"):
                return PILOT_MSG.format(found["pilot"])
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


# ------------------------------------------------------------------ the audit tool's exception
# The audit scores saved predictions and (when trusted) models on test data, which is what this guard denies before
# Gate C. So exactly one command shape is let through: an interpreter, the plugin's OWN audit tool (resolved inside
# this plugin's install folder), and known flags with plain data values - one command, nothing chained, no shell
# expansion, no environment settings. Any other command that names the tool is denied; look-alikes elsewhere are
# judged like any other script. ``--trust`` and ``--retrain`` also need the user's own chat approval.
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
AUDIT_TOOL = PLUGIN_ROOT / "skills" / "ml-audit" / "lib" / "mlpipeline_audit.py"
AUDIT_NAME = re.compile(r"mlpipeline_audit", re.I)
AUDIT_FAMILIES = {"inventory", "honest-scores", "data-problems", "leakage", "model-safety", "weak-spots", "test-misuse",
                  "retrain"}
_AUDIT_NAME_VALUE = re.compile(r"^[\w.-]{1,200}$")
AUDIT_FLAGS = {"--claim": None, "--review": None, "--trust": _AUDIT_NAME_VALUE, "--retrain": _AUDIT_NAME_VALUE,
               "--check": re.compile("^(?:" + "|".join(sorted(AUDIT_FAMILIES)) + ")$"),
               "--sample-rows": re.compile(r"^\d{1,9}$"), "--timeout": re.compile(r"^\d{1,9}$")}
PY_SAFE_FLAGS = {"-I", "-B", "-u", "-E", "-s"}
INTERPRETER = re.compile(r"^python(?:\d+(?:\.\d+)?)?$")
_UNQUOTED_META = set(";&|<>()$`\\\n\r*?[]{}~!#")
_DOUBLE_QUOTED_META = set("$`\\!\n\r")
AUDIT_APPROVALS = ".audit_approvals.json"
AUDIT_SHAPE_MSG = (
    "ml-pipeline: the audit tool runs only as one plain command: `python3 <plugin>/skills/ml-audit/lib/mlpipeline_audit.py "
    "<folder> [--claim ...] [--review ...] [--check ...]` - the plugin's own copy, nothing chained before or after it, "
    "no pipes, redirection, variables, substitutions, or environment settings, and only its documented flags."
)
AUDIT_APPROVAL_MSG = (
    "ml-pipeline: `{0} {1}` needs the user's own approval in chat. Explain what it does and ask; their reply (for "
    "example '{2} {1}') is recorded automatically, then run the same command again."
)
AUDIT_TAMPER_MSG = "ml-pipeline: the plugin's audit tool is not editable from a session; read it, don't change it."


def _plain_words(command: str) -> list[str] | None:
    """Shell words of a command made of plain words and quotes only, or None (any operator, expansion, or escape)."""
    words, cur, quote, have = [], [], None, False
    for ch in command:
        if quote == "'":
            if ch == "'":
                quote = None
            elif ch in "\n\r":
                return None
            else:
                cur.append(ch)
        elif quote == '"':
            if ch == '"':
                quote = None
            elif ch in _DOUBLE_QUOTED_META:
                return None
            else:
                cur.append(ch)
        elif ch in "'\"":
            quote, have = ch, True
        elif ch in " \t":
            if have or cur:
                words.append("".join(cur))
            cur, have = [], False
        elif ch in _UNQUOTED_META:
            return None
        else:
            cur.append(ch)
            have = True
    if quote is not None:
        return None
    if have or cur:
        words.append("".join(cur))
    return words


def _real_interpreter(word: str, cwd: str) -> bool:
    """A python interpreter by name (PATH), or a path to a binary called python (not a script wrapper)."""
    if not INTERPRETER.match(os.path.basename(word)) or "=" in word:
        return False
    if "/" not in word:
        return True
    path = Path(word) if os.path.isabs(word) else Path(cwd or ".") / word
    real = path.resolve()
    if not real.is_file() or real.suffix.lower() in {".py", ".sh", ".txt"}:
        return False
    with real.open("rb") as fh:
        return fh.read(2) != b"#!"


def audit_invocation(command: str, cwd: str) -> dict | None:
    """The parsed exact invocation of the plugin's audit tool, or None. Any doubt (or error) means None."""
    try:
        if not AUDIT_TOOL.is_file():
            return None
        words = _plain_words(command.strip())
        if not words or len(words) < 3 or not _real_interpreter(words[0], cwd):
            return None
        rest = words[1:]
        while rest and rest[0] in PY_SAFE_FLAGS:
            rest = rest[1:]
        if len(rest) < 2:
            return None
        tool = Path(rest[0]) if os.path.isabs(rest[0]) else Path(cwd or ".") / rest[0]
        if tool.resolve() != AUDIT_TOOL.resolve():
            return None
        folder, flags, args, i = None, [], rest[1:], 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                name, eq, value = arg.partition("=")
                if name not in AUDIT_FLAGS:
                    return None
                if not eq:
                    i += 1
                    if i >= len(args):
                        return None
                    value = args[i]
                check = AUDIT_FLAGS[name]
                if check is not None and not check.match(value):
                    return None
                flags.append((name, value))
            elif arg.startswith("-") or folder is not None or not arg:
                return None
            else:
                folder = arg
            i += 1
        if folder is None:
            return None
        return {"folder": folder, "flags": flags}
    except Exception:  # noqa: BLE001 - never exempt a command we could not fully read
        return None


def audit_approved(kind: str, name: str, folder: str, cwd: str) -> bool:
    """The user approved ``kind`` ('trust' or 'retrain') for project ``name`` in chat (hooks/record_approval.py),
    and ``name`` is the audited folder's own name."""
    base = Path(cwd or ".")
    target = (Path(folder) if os.path.isabs(folder) else base / folder).resolve()
    if target.name != name:
        return False
    for state in {base.resolve() / "ml_audit", target / "ml_audit"}:
        try:
            data = json.loads((state / AUDIT_APPROVALS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if any(isinstance(e, dict) and e.get("project") == name for e in data.get(kind) or []):
            return True
    return False


def audit_problem(tool_input: dict, cwd: str) -> tuple[bool, str | None]:
    """(exact audit invocation?, denial reason). Exact invocations skip the training checks: the tool takes no code."""
    target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if target:
        try:
            path = Path(target) if os.path.isabs(target) else Path(cwd or ".") / target
            if path.resolve() == AUDIT_TOOL.resolve():
                return False, AUDIT_TAMPER_MSG
        except OSError:
            pass
    command = tool_input.get("command") if isinstance(tool_input.get("command"), str) else None
    if command is None:
        return False, None
    parsed = audit_invocation(command, cwd)
    if parsed is None:
        return False, (AUDIT_SHAPE_MSG if AUDIT_NAME.search(command) else None)
    for flag, kind, phrase in (("--trust", "trust", "trust models in"), ("--retrain", "retrain", "approve audit retrain for")):
        for name, value in parsed["flags"]:
            if name == flag and require_proof() and not audit_approved(kind, value, parsed["folder"], cwd):
                return True, AUDIT_APPROVAL_MSG.format(flag, value, phrase)
    return True, None


# ------------------------------------------------------------------ self-disable
SWITCH = r"ML_PIPELINE_(?:ENFORCE|FILE|REQUIRE_USER_APPROVAL)"
# Assigning a switch (`X=0`, `"X": "0"`, `environ["X"] = ...`, `setenv X`), never reading it
# (`$X`, `${X:-1}`, `"$X" == "0"`, `os.environ.get("X")`).
SELF_DISABLE = re.compile(
    SWITCH + r"[\"']?\]?\s*(?:=(?!=)|:(?!-))|\b(?:setenv|setdefault|putenv)\b\W*[\"']?" + SWITCH
)

# ------------------------------------------------------------------ tamper protection
# The state files by name, by the library's constants and writers, and by common string-splitting.
PROTECTED_NAME = re.compile(
    r"\.guard_state\.json|\.approvals\.json|\.audit_approvals\.json|\bAPPROVALS_FILE\b|\bAUDIT_APPROVALS_FILE\b|"
    r"\bSTATE_FILE\b|\b_write_state\s*\(|\brecord_approval\b|"
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


def _new_gate_lines(existing: str, after: str) -> list[tuple[str, str]]:
    """(gate, status) for each strict gate line that ``after`` adds to the newest run.

    A line counts as new when it occurs more often in the newest run after the call than before it, so
    rewriting the whole ledger, or anchoring an edit on a recorded gate line, does not re-check that gate,
    while the same line under a new ``## Run:`` heading is a new recording.
    """
    same_run = run_label(existing) == run_label(after)
    before = [line for _, _, line in _gate_lines(current_run(existing))] if same_run else []
    found = _gate_lines(current_run(after))
    lines = [line for _, _, line in found]
    out: list[tuple[str, str]] = []
    for gate, status, line in found:
        if lines.count(line) > before.count(line) and (gate, status) not in out:
            out.append((gate, status))
    return out


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
    same_run = run_label(existing) == run_label(after)
    before_steps, after_steps = (steps_in(existing) if same_run else {}), steps_in(after)
    gates_now = gate_state(existing if same_run else "", approvals)
    for step, status in sorted(after_steps.items()):
        if step in before_steps:
            continue
        problems = step_problems(step, status, state_dir, after, gates_now)
        if problems:
            return STEP_RECORD_MSG.format(step, "; ".join(problems))
    for gate, status in _new_gate_lines(existing, after):
        if status == "other":
            continue
        try:
            problems = gate_problems(gate, state_dir, after, approvals, override=status == "override")
        except Exception as exc:  # noqa: BLE001 - a gate is never recorded unchecked; only this ledger write is denied
            return GATE_CHECK_FAILED_MSG.format(gate, exc)
        if problems:
            return GATE_RECORD_MSG.format(gate, "; ".join(problems))
    run = run_label(after)
    wanted = override_targets(current_run(after))
    had = override_targets(current_run(existing)) if same_run else []
    for what in sorted(set(wanted)):
        granted = granted_overrides(approvals, what, run)
        if require_proof() and wanted.count(what) > granted and wanted.count(what) > had.count(what):
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
        target = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        suffix = Path(target).suffix.lower() if target else ""
        # Reading (cat, grep, jq, ...) changes nothing and runs nothing.
        if command is not None and read_only_command(command):
            return _emit(None)
        # The audit tool: its exact invocation is allowed (it takes no code); anything else naming it is denied.
        exact_audit, reason = audit_problem(tool_input, cwd)
        if reason or exact_audit:
            return _emit(reason)
        # Shell commands are judged with the local scripts, notebooks and modules they run.
        executed = executed_sources(command, cwd) if command is not None else []
        # A script that runs the audit tool would hide its command line from the shape check above.
        if any(AUDIT_NAME.search(source) for source in executed):
            return _emit(AUDIT_SHAPE_MSG)
        # 0. Only the user switches enforcement or the approval rule off; settings files included.
        if suffix not in PROSE_SUFFIXES and any(SELF_DISABLE.search(t) for t in [text, *executed]):
            return _emit(SELF_DISABLE_MSG)
        # 0b. The guard's own state and the ledger's recorded lines are not editable.
        reason = protected_violation(tool_name, tool_input, cwd, executed)
        if reason:
            return _emit(reason)
        # 1. Recording a gate, override, or step: its prerequisites must already hold.
        reason = recording_problem(tool_input, "\n".join([text, *executed]), cwd)
        if reason:
            return _emit(reason)
        # 2. Prose, config, and other languages are not Python (running a file is checked when a command runs it).
        if suffix in DOC_SUFFIXES or suffix in OTHER_LANGUAGE_SUFFIXES:
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
                and not FINAL_TEST_CALL.search(text) and not (BACKWARD.search(text) and OPT_STEP.search(text))):
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
