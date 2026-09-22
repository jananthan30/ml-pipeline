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
        check = getattr(gr, f"gate_{gate.lower()}_problems")
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
                lambda r: r[0].update(figure_note="age: looks fine overall, there are 7 things to check here"),
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


if __name__ == "__main__":
    unittest.main()
