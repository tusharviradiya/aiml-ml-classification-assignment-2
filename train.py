"""Reproduce the split, tune five classifiers, and save pipelines and results.

Run: python train.py
Settings are chosen by five-fold training CV F1; the test set never enters CV.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataset import LABELS, ROOT, TARGET, load_dataset
from evaluation import evaluate_model


SEED = 42
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def model_specs():
    return [
        ("logistic_regression", "Logistic Regression",
         LogisticRegression(max_iter=2000, random_state=SEED), {"model__C": [0.1, 1, 10]}),
        ("decision_tree", "Decision Tree",
         DecisionTreeClassifier(random_state=SEED),
         {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [1, 5, 10]}),
        # On one-hot inputs Manhattan distances are sums of exact 0/1
        # differences. An explicit tree search avoids the varying brute-force
        # calculation paths observed in the earlier run. Preserve row order
        # and library versions: equally distant neighbors can still tie.
        ("knn", "KNN", KNeighborsClassifier(
            algorithm="kd_tree", metric="manhattan", n_jobs=1,
        ),
         {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}),
        ("naive_bayes", "Naive Bayes", MultinomialNB(), {"model__alpha": [0.1, 1, 10]}),
        ("svm", "Support Vector Machine", SVC(kernel="rbf", random_state=SEED),
         {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]}),
    ]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render_observations(details, winner):
    knn = details["knn"]["model_parameters"]
    knn_training_note = (
        "With distance weighting, a training row is its own zero-distance neighbor; "
        "a perfect training score therefore does not establish generalization."
        if knn["weights"] == "distance" else
        "Uniform weighting gives each selected neighbor the same vote; "
        "the training score remains optimistic because each row can be its own neighbor."
    )
    explanations = {
        "logistic_regression": "The small training/test F1 gap suggests limited overfitting "
        "on this split. Its additive linear boundary makes more test errors than the tree "
        "and RBF SVM, which can represent feature interactions.",
        "decision_tree": "The unrestricted-depth tree fits the training labels perfectly, "
        "but its high CV and test F1 also support strong generalization on this split. "
        "Training perfection alone is not evidence of severe overfitting. The learned "
        "conjunctions of descriptor conditions suit this rule-based endgame task.",
        "knn": f"CV selected k={knn['n_neighbors']} and {knn['weights']} weighting. "
        f"{knn_training_note} The training/CV gap shows why held-out evaluation matters. "
        "Manhattan distance and an explicit KD-tree search are used; regression checks "
        "compare predictions and probabilities across calculation settings and batch sizes.",
        "naive_bayes": "Both training and CV F1 are lower than for the other classifiers, "
        "suggesting limited model fit rather than only a large train/test overfitting gap. "
        "Its conditional-independence assumption may be restrictive for related board "
        "descriptors; this is a possible explanation, not an established cause.",
        "svm": "The RBF kernel allows nonlinear feature interactions. High training, CV "
        "and test F1 support the fit on these positions, while the high test MCC shows "
        "strong agreement across both labels. Its ranking is based on training CV, "
        "not on choosing the best held-out score.",
    }
    lines = ["## Observations", "", "| ML Model Name | Observation about model performance |",
             "| --- | --- |"]
    for key, result in details.items():
        tn, fp = result["confusion_matrix"][0]
        fn, tp = result["confusion_matrix"][1]
        lines.append(
            f"| {result['name']} | Test F1 {result['metrics']['F1']:.4f}; "
            f"{fp} false wins and {fn} missed wins. "
            f"Training F1 {result['training_metrics']['F1']:.4f}; "
            f"CV F1 {result['cv_f1_mean']:.4f} +/- {result['cv_f1_std']:.4f}. "
            f"{explanations[key]} |"
        )
    ranked = sorted(details.values(), key=lambda result: result["cv_f1_mean"], reverse=True)
    best, runner_up = ranked[:2]
    best_errors = sum(best["confusion_matrix"][i][1 - i] for i in range(2))
    runner_errors = sum(runner_up["confusion_matrix"][i][1 - i] for i in range(2))
    lines.append(f"| Overall winner for this dataset | **{winner}**, selected by the "
                 f"highest mean training CV F1. Its advantage over {runner_up['name']} "
                 f"is {best['cv_f1_mean'] - runner_up['cv_f1_mean']:.4f}; "
                 f"the test error counts are {best_errors} versus {runner_errors}. "
                 "The runner-up is competitive, and this small difference does not "
                 "establish statistical superiority. |")
    lines += ["", "A false win means predicting won for a nowin position; a missed win means "
              "predicting nowin for a won position. Small score differences on one split "
              "do not establish statistical superiority. The test set was not used to "
              "select hyperparameters or the winner. These results concern one split "
              "of KRKPA7, not arbitrary chess positions."]
    return lines


def update_readme(report):
    readme = ROOT / "README.md"
    start = "<!-- MODEL_RESULTS_START -->"
    end = "<!-- MODEL_RESULTS_END -->"
    if readme.exists():
        text = readme.read_text()
        if start not in text or end not in text:
            raise ValueError("README result markers are missing; refusing to overwrite it.")
        prefix, remainder = text.split(start, 1)
        _, suffix = remainder.split(end, 1)
        readme.write_text(prefix + start + "\n\n" + report + "\n" + end + suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", choices=["local", "bits-lab", "unspecified"], default="unspecified",
        help="Record where this run was performed; screenshots provide separate lab evidence.",
    )
    args = parser.parse_args()
    frame = load_dataset()
    train, test = train_test_split(
        frame, test_size=0.2, random_state=SEED, stratify=frame[TARGET],
    )
    train.to_csv(ROOT / "data/train_data.csv", index=False)
    test.to_csv(ROOT / "test_data.csv", index=False)
    model_dir = ROOT / "model"
    results_dir = ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    summary, details = [], {}
    predictions = test[[TARGET]].copy()
    for key, name, classifier, grid in model_specs():
        print(f"Training {name}...", flush=True)
        # Full one-hot encoding treats categories symmetrically. Its values are
        # 0/1, so extra numeric scaling is unnecessary. All models use the same
        # raw features and split. MultinomialNB sees one category-count per field.
        pipeline = Pipeline([
            # spcop=t appears only once in the source. Ignore valid categories
            # absent from a fold, encoding them as all zeros for that field.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("model", classifier),
        ])
        search = GridSearchCV(
            pipeline, grid, scoring="f1", cv=cv, n_jobs=1,
            return_train_score=True, error_score="raise",
        )
        search.fit(train.drop(columns=TARGET), train[TARGET].map(LABELS))
        fitted = search.best_estimator_
        joblib.dump(fitted, model_dir / f"{key}.joblib")
        # Verify that saved artifacts, not only in-memory estimators, work.
        restored = joblib.load(model_dir / f"{key}.joblib")
        result = evaluate_model(restored, test)
        train_result = evaluate_model(restored, train)
        summary.append({
            "ML Model Name": name, **result["metrics"],
            "CV F1 Mean": float(search.best_score_),
            "CV F1 Std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "Training F1": train_result["metrics"]["F1"],
        })
        predictions[name] = result.pop("predictions")
        predictions[f"{name} score"] = result.pop("scores")
        details[key] = {
            "name": name, "best_parameters": search.best_params_,
            "model_parameters": restored.named_steps["model"].get_params(deep=False),
            "cv_f1_mean": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "training_metrics": train_result["metrics"], **result,
        }
        pd.DataFrame(search.cv_results_).to_csv(results_dir / f"{key}_cv.csv", index=False)
        print(f"  CV F1={search.best_score_:.4f}; test F1={result['metrics']['F1']:.4f}", flush=True)

    table = pd.DataFrame(summary)
    table.to_csv(results_dir / "metrics.csv", index=False)
    predictions.rename_axis("source_row_index").to_csv(results_dir / "test_predictions.csv")
    write_json(results_dir / "evaluation.json", details)
    # Select the overall model using training CV rather than test-set ranking.
    winner = max(summary, key=lambda row: row["CV F1 Mean"])["ML Model Name"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "execution_environment": args.environment,
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
        "seed": SEED, "test_fraction": 0.2, "cv_folds": 5,
        "selection_metric": "Mean stratified training CV F1 (won is positive)",
        "selected_model": winner,
        "label_mapping": LABELS,
        "features": list(frame.drop(columns=TARGET).columns),
        "feature_categories": {name: sorted(frame[name].unique().tolist()) for name in frame.columns if name != TARGET},
        "train_indices": train.index.tolist(), "test_indices": test.index.tolist(),
        "train_class_counts": train[TARGET].value_counts().to_dict(),
        "test_class_counts": test[TARGET].value_counts().to_dict(),
        "sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                   for path in ["data/raw/kr-vs-kp.data", "data/raw/kr-vs-kp.names", "test_data.csv"]},
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in ["train.py", "dataset.py", "evaluation.py"]},
        "model_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(model_dir.glob("*.joblib"))},
    }
    write_json(results_dir / "manifest.json", manifest)
    lines = ["# Model evaluation", "", "Generated by `python train.py`.", "",
             f"Recorded execution environment: {args.environment}. Python: {platform.python_version()}.", "",
             f"Training rows: {len(train)}. Test rows: {len(test)}. Seed: {SEED}.",
             "Positive class: won. Settings selected by five-fold training CV F1.", "",
             "| ML Model Name | " + " | ".join(METRICS) + " |",
             "| --- | " + " | ".join(["---:"] * len(METRICS)) + " |"]
    for row in summary:
        lines.append("| " + row["ML Model Name"] + " | " + " | ".join(f"{row[metric]:.4f}" for metric in METRICS) + " |")
    lines += ["", "| Model | CV F1 mean | CV F1 std | Training F1 | Test F1 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['ML Model Name']} | {row['CV F1 Mean']:.4f} | {row['CV F1 Std']:.4f} | {row['Training F1']:.4f} | {row['F1']:.4f} |")
    lines += ["", f"Selected model: **{winner}**, based on training CV F1.", "",
              "Precision, recall and F1 use won as the positive class. AUC uses positive-class "
              "probabilities, or decision scores for SVM. MCC uses predicted labels. "
              "Precision/recall/F1 are set to zero when their denominator is zero.", "",
              "SVM scores are decision values, not probabilities. Confusion matrices in "
              "evaluation.json use row=true, column=predicted, with order [nowin, won].", "",
              "These are results for one held-out split of this restricted endgame dataset. "
              "They do not establish performance on arbitrary chess positions."]
    lines += [""] + render_observations(details, winner)
    report = "\n".join(lines) + "\n"
    (results_dir / "MODEL_EVALUATION.md").write_text(report)
    update_readme(report.replace("# Model evaluation\n", "", 1).lstrip())
    print(f"\nSelected by training CV: {winner}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
