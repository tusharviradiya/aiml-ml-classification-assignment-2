"""CSV parsing and saved-model inference for the Streamlit interface.

Uploads remain in memory. This module only loads the repository's own models;
it never fits a classifier or accepts a model file from the visitor.
"""

import csv
import hashlib
from io import StringIO

import joblib
import pandas as pd

from dataset import ROOT, TARGET, feature_names
from evaluation import evaluate_model


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_TEST_ROWS = 5000
SESSION_EVALUATION_KEY = "chess_evaluation"


def parse_test_csv(content):
    """Read UTF-8 CSV strictly, including duplicate-header and row-width checks."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The test CSV must be 2 MB or smaller.")
    try:
        reader = csv.reader(StringIO(content.decode("utf-8-sig")), strict=True)
        header = next(reader, [])
        if not header:
            raise ValueError("The CSV is empty. Download the example test CSV to see the format.")
        if len(header) != len(set(header)):
            raise ValueError("The CSV has duplicate column names; each column must appear once.")
        expected = set(feature_names() + [TARGET])
        missing = sorted(expected - set(header))
        extra = sorted(set(header) - expected)
        if missing or extra:
            raise ValueError(f"CSV columns do not match. Missing: {missing}; extra: {extra}.")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"CSV record ending at line {reader.line_num} has {len(row)} values; "
                    f"expected {len(header)}. Check the commas and quotes."
                )
            rows.append(row)
            if len(rows) > MAX_TEST_ROWS:
                raise ValueError(f"Upload test data only: at most {MAX_TEST_ROWS:,} positions.")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV with UTF-8 encoding and try again.") from error
    except csv.Error as error:
        raise ValueError("The CSV has malformed quotes or fields. Check its formatting.") from error
    if not rows:
        raise ValueError("The CSV contains a header but no test positions.")
    return pd.DataFrame(rows, columns=header, dtype=str)


def model_revision():
    """Invalidate cached models when a training run replaces saved artifacts."""
    return tuple(
        (name, path.stat().st_mtime_ns, path.stat().st_size)
        for name, filename in MODEL_FILES.items()
        for path in [ROOT / "model" / filename]
    )


def load_models():
    return {name: joblib.load(ROOT / "model" / filename) for name, filename in MODEL_FILES.items()}


def evaluate_all(models, frame):
    details = {name: evaluate_model(model, frame) for name, model in models.items()}
    comparison = pd.DataFrame([
        {"ML Model Name": name, **result["metrics"]} for name, result in details.items()
    ])
    return details, comparison


def evaluate_session(content, revision, models, state):
    """Reuse the current evaluation only within this visitor's session.

    File content and model revision both participate in the key. A new upload
    replaces the previous result, and invalid data clears it before validation.
    This avoids repeated inference for dropdown/filter interactions without
    placing uploaded data into Streamlit's shared resource or data caches.
    """
    key = (hashlib.sha256(content).hexdigest(), revision)
    cached = state.get(SESSION_EVALUATION_KEY)
    if cached is not None and cached["key"] == key:
        return cached["value"]
    state.pop(SESSION_EVALUATION_KEY, None)
    frame = parse_test_csv(content)
    details, comparison = evaluate_all(models, frame)
    value = (frame, details, comparison)
    state[SESSION_EVALUATION_KEY] = {"key": key, "value": value}
    return value


def prediction_table(frame, result):
    table = pd.DataFrame({
        "Test row": range(1, len(frame) + 1),
        "Actual": frame[TARGET].tolist(),
        "Predicted": result["predictions"],
    })
    table["Result"] = [
        "Correct" if actual == predicted else ("False win" if predicted == "won" else "Missed win")
        for actual, predicted in zip(table["Actual"], table["Predicted"])
    ]
    return table
