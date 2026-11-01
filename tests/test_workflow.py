"""Integration checks for saved models, held-out rows, and CSV validation.

Run after training: python -m unittest discover -s tests -v
"""

import hashlib
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from dataset import ROOT, TARGET, load_dataset
from evaluation import evaluate_model


class SavedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test = pd.read_csv(ROOT / "test_data.csv", dtype=str)
        cls.train = pd.read_csv(ROOT / "data/train_data.csv", dtype=str)
        cls.manifest = json.loads((ROOT / "results/manifest.json").read_text())
        cls.results = json.loads((ROOT / "results/evaluation.json").read_text())

    def test_split_is_disjoint_complete_and_reconstructable(self):
        source = load_dataset()
        train_ids = self.manifest["train_indices"]
        test_ids = self.manifest["test_indices"]
        self.assertEqual((len(train_ids), len(test_ids)), (2556, 640))
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertEqual(set(train_ids) | set(test_ids), set(source.index))
        pd.testing.assert_frame_equal(source.loc[train_ids].reset_index(drop=True), self.train)
        pd.testing.assert_frame_equal(source.loc[test_ids].reset_index(drop=True), self.test)
        for filename, expected in self.manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), expected)
        for filename, expected in self.manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), expected)
        for filename, expected in self.manifest["model_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "model" / filename).read_bytes()).hexdigest(), expected)

    def test_five_saved_models_reproduce_recorded_metrics(self):
        self.assertEqual(len(self.results), 5)
        table = pd.read_csv(ROOT / "results/metrics.csv").set_index("ML Model Name")
        self.assertEqual(self.manifest["selected_model"], table["CV F1 Mean"].idxmax())
        predictions = pd.read_csv(ROOT / "results/test_predictions.csv")
        for key, recorded in self.results.items():
            with self.subTest(model=key):
                model = joblib.load(ROOT / f"model/{key}.joblib")
                self.assertEqual(len(model.feature_names_in_), 36)
                self.assertNotIn(TARGET, model.feature_names_in_)
                actual = evaluate_model(model, self.test)
                for metric, value in actual["metrics"].items():
                    self.assertAlmostEqual(value, recorded["metrics"][metric])
                    self.assertAlmostEqual(value, table.loc[recorded["name"], metric])
                self.assertEqual(actual["predictions"], predictions[recorded["name"]].tolist())
                self.assertEqual(actual["confusion_matrix"], recorded["confusion_matrix"])
                # Confirm label counts and F1 directly from confusion cells.
                tn, fp, fn, tp = np.asarray(actual["confusion_matrix"]).ravel()
                self.assertEqual(int(tn + fp + fn + tp), 640)
                self.assertAlmostEqual(actual["metrics"]["F1"], 2 * tp / (2 * tp + fp + fn))
                y = self.test[TARGET].eq("won").astype(int)
                self.assertAlmostEqual(actual["metrics"]["AUC"], roc_auc_score(y, actual["scores"]))
                # Encoders in saved artifacts must have been fitted on train rows.
                for name, categories in zip(model.feature_names_in_, model.named_steps["encode"].categories_):
                    self.assertEqual(set(categories), set(self.train[name]))

    def test_csv_validation_and_column_order(self):
        model = joblib.load(ROOT / "model/logistic_regression.joblib")
        expected = evaluate_model(model, self.test)
        self.assertEqual(evaluate_model(model, self.test[self.test.columns[::-1]]), expected)
        invalid_frames = [
            self.test.iloc[:0], self.test.drop(columns=TARGET),
            self.test.assign(unexpected="x"), self.test.assign(outcome="draw"),
            self.test.assign(bkblk="invalid"), self.test.assign(bkblk=None),
        ]
        for frame in invalid_frames:
            with self.subTest(columns=list(frame.columns), rows=len(frame)):
                with self.assertRaises(ValueError):
                    evaluate_model(model, frame)

    def test_knn_predictions_and_probabilities_are_stable_across_query_paths(self):
        # Categorical indicators create many equally distant neighbors. The old
        # brute-force model changed scores under these calculation settings.
        pipeline = joblib.load(ROOT / "model/knn.joblib")
        X = pipeline.named_steps["encode"].transform(self.test.drop(columns=TARGET))
        classifier = pipeline.named_steps["model"]
        expected_labels = classifier.predict(X)
        expected_probabilities = classifier.predict_proba(X)
        for cython in [True, False]:
            for threads in [1, 2]:
                with self.subTest(cython=cython, threads=threads), config_context(
                    enable_cython_pairwise_dist=cython,
                ), threadpool_limits(limits=threads):
                    np.testing.assert_array_equal(classifier.predict(X), expected_labels)
                    np.testing.assert_array_equal(
                        classifier.predict_proba(X), expected_probabilities,
                    )
        for size in [1, 17, len(X)]:
            with self.subTest(batch_size=size):
                labels = np.concatenate([
                    classifier.predict(X[start:start + size]) for start in range(0, len(X), size)
                ])
                probabilities = np.concatenate([
                    classifier.predict_proba(X[start:start + size])
                    for start in range(0, len(X), size)
                ])
                np.testing.assert_array_equal(labels, expected_labels)
                np.testing.assert_array_equal(probabilities, expected_probabilities)

    def test_single_class_test_data_has_explicitly_unavailable_auc(self):
        model = joblib.load(ROOT / "model/svm.joblib")
        subset = self.test[self.test[TARGET] == "won"]
        result = evaluate_model(model, subset)
        self.assertIsNone(result["metrics"]["AUC"])
        self.assertIsNotNone(result["auc_note"])
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (2, 2))

    def test_rare_valid_category_absent_from_training_is_supported(self):
        from sklearn.base import clone
        # Remove the single rare position to exercise inference on a category
        # that is valid in this domain but unseen by the fitted encoder.
        source = load_dataset()
        rare = source[source["spcop"] == "t"]
        ordinary = source[source["spcop"] == "f"]
        model = clone(joblib.load(ROOT / "model/naive_bayes.joblib"))
        model.fit(ordinary.drop(columns=TARGET), ordinary[TARGET].eq("won").astype(int))
        self.assertEqual(len(evaluate_model(model, rare)["predictions"]), 1)


if __name__ == "__main__":
    unittest.main()
