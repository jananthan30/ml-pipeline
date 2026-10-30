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
RUNS = D + "runs.jsonl"


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
                      IMPORTANCE: json.dumps(importance()),
                      RUNS: json.dumps({"kind": "baseline"}) + "\n" + json.dumps({"kind": "candidate"}) + "\n"})
    return files


def ledger(upto: str = "A") -> str:
    """PIPELINE.md text with every figure line (and, from Gate B, the model rationale) Gates A..``upto`` need."""
    text = LEDGER_A
    if upto in ("B", "C"):
        text += LEDGER_B + RATIONALE
    if upto == "C":
        text += LEDGER_C
    return text


# The combined hook also checks the step lines, the prediction contract, and the earlier gates: process(gate)
# is that part of the ledger, everything a gate needs besides its records and figure lines.
CONTRACT = ("Prediction contract:\n- target: y = 1 if readmitted within 30 days\n- unit: one row per admission\n"
            "- horizon: 30 days after discharge\n- available: features known at discharge\n- metric: balanced accuracy\n")
STEP_LINES = {s: f"- Step {s} done: step {s} finished, see the records\n" for s in range(1, 14)}
GATE_STEPS = {"A": (1, 2, 3), "B": (4, 5, 6, 7, 8), "C": (9, 10, 11, 12), "D": (13,)}


def process(gate: str) -> str:
    """Step lines, the contract, and approval lines of the gates before ``gate``, in recording order."""
    text = ""
    for g in "ABCD":
        text += "".join(STEP_LINES[s] for s in GATE_STEPS[g]) + (CONTRACT if g == "A" else "")
        if g == gate:
            return text
        text += f"- Gate {g}: approved 2026-09-23\n"
    return text


def write(root: Path, files: dict) -> None:
    for rel, data in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
