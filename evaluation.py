"""Shared inference and binary evaluation for saved pipelines and labeled CSVs.

This module never fits a model. The app can call evaluate_model with an uploaded
DataFrame containing the 36 raw categorical features plus the outcome column.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import LABELS, TARGET, category_domains


def evaluate_model(pipeline, frame):
    names = list(pipeline.feature_names_in_)
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    missing = sorted(set(names + [TARGET]) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(names + [TARGET]))
    if missing or extra:
        raise ValueError(f"CSV columns do not match. Missing: {missing}; extra: {extra}")
    if frame.isna().any().any():
        raise ValueError("Missing values are not supported; provide complete test rows.")
    if not frame[TARGET].isin(LABELS).all():
        raise ValueError("outcome must contain only 'won' or 'nowin'.")
    X = frame.loc[:, names]
    for name, categories in category_domains().items():
        if not X[name].isin(categories).all():
            raise ValueError(f"Unknown category in {name}; expected {list(categories)}.")
    y = frame[TARGET].map(LABELS).to_numpy()
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected model classes [0, 1], where won is positive.")
    predicted = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X)[:, 1]
    else:
        # SVC decision scores are sufficient for binary ROC AUC.
        scores = pipeline.decision_function(X)
    metrics = {
        "Accuracy": float(accuracy_score(y, predicted)),
        "AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "F1": float(f1_score(y, predicted, zero_division=0)),
        # MCC is conventionally 0 when both vectors contain one constant class.
        "MCC": (0.0 if len(np.unique(y)) == len(np.unique(predicted)) == 1
                else float(matthews_corrcoef(y, predicted))),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, predicted, labels=[0, 1], target_names=["nowin", "won"],
            output_dict=True, zero_division=0,
        ),
        "predictions": ["won" if value == 1 else "nowin" for value in predicted],
        "scores": scores.tolist(),
        "auc_note": "AUC requires both target classes." if len(np.unique(y)) < 2 else None,
    }
