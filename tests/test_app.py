"""Exercise real Streamlit widgets and the strict CSV upload path locally."""

import csv
from io import StringIO
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from app_data import (
    MAX_TEST_ROWS, MAX_UPLOAD_BYTES, MODEL_FILES, SESSION_EVALUATION_KEY,
    evaluate_all, evaluate_session, load_models, model_revision, parse_test_csv,
)
from dataset import ROOT, TARGET
from evaluation import evaluate_model


class CsvUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = (ROOT / "test_data.csv").read_bytes()
        cls.rows = list(csv.reader(StringIO(cls.content.decode())))

    def test_bom_and_reordered_columns_preserve_values(self):
        original = pd.read_csv(ROOT / "test_data.csv", dtype=str)
        reordered = original[original.columns[::-1]].head(20)
        parsed = parse_test_csv(b"\xef\xbb\xbf" + reordered.to_csv(index=False).encode())
        pd.testing.assert_frame_equal(parsed, reordered.reset_index(drop=True))

    def test_rejects_duplicate_header_bad_width_empty_and_malformed_files(self):
        header = ",".join(self.rows[0])
        duplicate = self.rows[0].copy()
        duplicate[1] = duplicate[0]
        cases = [
            b"", (header + "\n").encode(),
            (",".join(duplicate) + "\n").encode(),
            (header + "\n" + ",".join(self.rows[1][:-1])).encode(),
            (header + "\n" + ",".join(self.rows[1] + ["extra"])).encode(),
            (header + '\n"unclosed').encode(),
            b"\xff\xfe", b"unknown,outcome\nf,won\n",
        ]
        for content in cases:
            with self.subTest(prefix=content[:30]):
                with self.assertRaises(ValueError):
                    parse_test_csv(content)

    def test_rejects_oversized_files_and_too_many_rows(self):
        with self.assertRaisesRegex(ValueError, "2 MB"):
            parse_test_csv(b"x" * (MAX_UPLOAD_BYTES + 1))
        content = (",".join(self.rows[0]) + "\n" +
                   (",".join(self.rows[1]) + "\n") * (MAX_TEST_ROWS + 1)).encode()
        with self.assertRaisesRegex(ValueError, "at most"):
            parse_test_csv(content)


class SessionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = (ROOT / "test_data.csv").read_bytes()
        cls.models = load_models()
        cls.revision = model_revision()

    def test_reuses_results_and_invalidates_on_data_or_model_change(self):
        state = {}
        with patch("app_data.evaluate_all", wraps=evaluate_all) as evaluate:
            original = evaluate_session(self.content, self.revision, self.models, state)
            self.assertIs(evaluate_session(self.content, self.revision, self.models, state), original)
            self.assertEqual(evaluate.call_count, 1)
            subset = original[0].head(10).to_csv(index=False).encode()
            changed = evaluate_session(subset, self.revision, self.models, state)
            self.assertEqual(len(changed[0]), 10)
            self.assertEqual(evaluate.call_count, 2)
            evaluate_session(subset, self.revision + (("changed", 1, 1),), self.models, state)
            self.assertEqual(evaluate.call_count, 3)

    def test_sessions_are_isolated_and_invalid_data_clears_previous_result(self):
        first, second = {}, {}
        with patch("app_data.evaluate_all", wraps=evaluate_all) as evaluate:
            a = evaluate_session(self.content, self.revision, self.models, first)
            b = evaluate_session(self.content, self.revision, self.models, second)
            self.assertIsNot(a, b)
            self.assertEqual(evaluate.call_count, 2)
        with self.assertRaises(ValueError):
            evaluate_session(b"invalid,outcome\nf,won", self.revision, self.models, first)
        self.assertNotIn(SESSION_EVALUATION_KEY, first)
        self.assertIn(SESSION_EVALUATION_KEY, second)


class StreamlitInteractionTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def assert_healthy(self, app):
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertFalse(app.error, [error.value for error in app.error])

    def comparison(self, app):
        return next(element.value for element in app.dataframe
                    if "ML Model Name" in element.value.columns)

    def predictions(self, app):
        return next(element.value for element in app.dataframe
                    if "Test row" in element.value.columns)

    def upload(self, app, content):
        app.radio(key="data_source").set_value("Upload CSV").run()
        app.file_uploader(key="test_csv").set_value(("positions.csv", content, "text/csv")).run()
        return app

    def test_all_dropdown_choices_show_offline_metrics_and_all_model_comparison(self):
        app = self.make_app()
        offline = pd.read_csv(ROOT / "results/metrics.csv")
        for name in MODEL_FILES:
            with self.subTest(model=name):
                app.selectbox(key="selected_model").set_value(name).run()
                self.assert_healthy(app)
                expected = offline[offline["ML Model Name"] == name].iloc[0]
                self.assertEqual(len(app.metric), 6)
                for metric in app.metric:
                    value = expected[metric.label]
                    formatted = f"{value:.2%}" if metric.label == "Accuracy" else f"{value:.4f}"
                    self.assertEqual(metric.value, formatted)
                self.assertEqual(len(self.comparison(app)), 5)
                self.assertEqual(len(app.get("vega_lite_chart")), 3)
        expected = offline.drop(columns=["CV F1 Mean", "CV F1 Std", "Training F1"])
        pd.testing.assert_frame_equal(self.comparison(app), expected)

    def test_upload_recomputes_results_without_fitting_and_filters_errors(self):
        app = self.make_app()
        subset = pd.read_csv(ROOT / "test_data.csv", dtype=str).head(40)
        with patch.object(Pipeline, "fit", side_effect=AssertionError("The app must not retrain")):
            self.upload(app, subset.to_csv(index=False).encode())
        self.assert_healthy(app)
        expected = evaluate_model(joblib.load(ROOT / "model/svm.joblib"), subset)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "F1"),
                         f"{expected['metrics']['F1']:.4f}")
        self.assertEqual(self.predictions(app)["Predicted"].tolist(), expected["predictions"])
        self.assertEqual(len(self.predictions(app)), 40)
        app.checkbox(key="only_errors").check().run()
        self.assert_healthy(app)
        self.assertTrue(self.predictions(app)["Result"].ne("Correct").all())
        # Changing classifiers keeps the current upload, rather than restoring the bundle.
        app.selectbox(key="selected_model").set_value("Naive Bayes").run()
        app.checkbox(key="only_errors").uncheck().run()
        self.assert_healthy(app)
        self.assertEqual(len(self.predictions(app)), 40)

    def test_dropdown_and_filter_changes_reuse_session_evaluation(self):
        with patch("app_data.evaluate_all", wraps=evaluate_all) as evaluate:
            app = self.make_app()
            app.selectbox(key="selected_model").set_value("KNN").run()
            app.checkbox(key="only_errors").check().run()
            self.assert_healthy(app)
            self.assertEqual(evaluate.call_count, 1)
            self.upload(app, pd.read_csv(ROOT / "test_data.csv", dtype=str).head(15).to_csv(index=False).encode())
            self.assert_healthy(app)
            self.assertEqual(evaluate.call_count, 2)

    def test_invalid_upload_removes_stale_results_and_can_be_corrected(self):
        app = self.make_app()
        self.upload(app, b"feature,outcome\nf,won\n")
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.metric), 0)
        self.assertFalse(any("ML Model Name" in item.value.columns for item in app.dataframe))
        app.file_uploader(key="test_csv").set_value(
            ("test_data.csv", (ROOT / "test_data.csv").read_bytes(), "text/csv")
        ).run()
        self.assert_healthy(app)
        self.assertEqual(len(app.metric), 6)
        for column, value in [("outcome", "draw"), ("bkblk", "invalid"), ("bkblk", "")]:
            frame = pd.read_csv(ROOT / "test_data.csv", dtype=str).head(3)
            frame.loc[0, column] = value
            app.file_uploader(key="test_csv").set_value(
                ("invalid.csv", frame.to_csv(index=False).encode(), "text/csv")
            ).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(len(app.metric), 0)

    def test_single_class_upload_and_cleared_uploader(self):
        app = self.make_app()
        source = pd.read_csv(ROOT / "test_data.csv", dtype=str)
        subset = source[source[TARGET] == "won"].head(12)
        self.upload(app, subset.to_csv(index=False).encode())
        self.assert_healthy(app)
        self.assertEqual(next(metric.value for metric in app.metric if metric.label == "AUC"), "N/A")
        self.assertTrue(any("requires both" in info.value for info in app.info))
        self.assertEqual(len(app.get("vega_lite_chart")), 2)
        app.file_uploader(key="test_csv").clear().run()
        self.assert_healthy(app)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("Upload a labeled" in info.value for info in app.info))

    def test_missing_model_has_an_actionable_error(self):
        # Missing artifacts must produce an owner-facing error rather than a traceback.
        with self.assertLogs(level="ERROR"), patch(
            "app_data.model_revision", side_effect=FileNotFoundError("model missing")
        ):
            app = self.make_app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("saved classifiers are unavailable" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
