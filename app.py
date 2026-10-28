"""Interactive KRKPA7 test-set evaluation. Launch with streamlit run app.py."""

import json
import logging

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from app_data import (
    METRICS, MODEL_FILES, evaluate_all, load_models, model_revision,
    parse_test_csv, prediction_table,
)
from dataset import ROOT, TARGET, category_domains


st.set_page_config(page_title="Chess Endgame Lab", page_icon="♜", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_models(revision):
    # revision participates in Streamlit's cache key; uploads are never cached.
    return load_models()


def confusion_chart(matrix):
    labels = ["Cannot win · nowin", "Can win · won"]
    cells = pd.DataFrame([
        {"Actual": labels[row], "Predicted": labels[column], "Positions": count}
        for row, values in enumerate(matrix) for column, count in enumerate(values)
    ])
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0, title="Predicted outcome")),
        y=alt.Y("Actual:N", sort=labels, title="Actual outcome"),
        tooltip=["Actual:N", "Predicted:N", "Positions:Q"],
    )
    tiles = base.mark_rect(stroke="#faf8f3", strokeWidth=4).encode(
        color=alt.Color("Positions:Q", scale=alt.Scale(range=["#e5ede5", "#174b39"]), legend=None),
    )
    numbers = base.mark_text(fontSize=26, fontWeight="bold").encode(
        text="Positions:Q",
        color=alt.condition(alt.datum.Positions > max(max(row) for row in matrix) / 2,
                            alt.value("#ffffff"), alt.value("#183c2d")),
    )
    return (tiles + numbers).properties(height=280).configure_view(stroke=None)


def main():
    st.markdown(
        "<style>h1 {font-family: Georgia, serif; letter-spacing: -0.035em;} "
        "[data-testid='stMetricValue'] {font-variant-numeric: tabular-nums;} "
        ".block-container {padding-top: 2rem; padding-bottom: 2rem;}</style>",
        unsafe_allow_html=True,
    )
    st.caption("♔  ♖   /   ♚  ♟     ·     KRKPA7 CLASSIFICATION")
    st.title("Chess Endgame Lab")
    st.write("Explore whether White can win. Compare five classifiers on the same endgame positions.")
    st.caption("King and rook vs. king and pawn on a7 · White to move · 36 categorical descriptors")

    with st.sidebar:
        st.header("Set up your evaluation")
        source = st.radio("1. Choose test data", ["Bundled test set", "Upload CSV"], key="data_source")
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload a labeled test CSV", type=["csv"], key="test_csv", max_upload_size=2,
                help="Up to 2 MB and 5,000 test rows. Include all 36 features and outcome.",
            )
        st.download_button(
            "Download example test CSV", data=(ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv", mime="text/csv", key="example_download",
        )
        selected = st.selectbox(
            "2. Choose a classifier", list(MODEL_FILES), index=4, key="selected_model",
        )
        st.caption("Saved models evaluate your test rows. No training runs in this app.")
        with st.expander("CSV format and labels"):
            st.write("Include all 36 feature columns and the target column `outcome`.")
            st.write("Use `won` for White can win and `nowin` for White cannot win.")
            st.caption("Column order may vary. Values are case-sensitive. Use held-out test rows.")
            st.dataframe(pd.DataFrame([
                {"Feature": name, "Allowed values": ", ".join(values)}
                for name, values in category_domains().items()
            ]), hide_index=True)
        st.markdown("[Dataset and citation](https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn)")

    if source == "Upload CSV" and uploaded is None:
        st.info("Upload a labeled test CSV to begin, or choose the bundled test set.")
        st.stop()
    try:
        content = uploaded.getvalue() if uploaded is not None else (ROOT / "test_data.csv").read_bytes()
        frame = parse_test_csv(content)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    try:
        models = cached_models(model_revision())
    except Exception:
        logging.exception("Failed to load the saved chess classifiers")
        st.error("The saved classifiers are unavailable. Please contact the project owner.")
        st.stop()
    try:
        details, comparison = evaluate_all(models, frame)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    result = details[selected]
    wins = int(frame[TARGET].eq("won").sum())
    st.divider()
    st.caption(
        f"{'Bundled held-out test set' if source == 'Bundled test set' else 'Uploaded test set'}"
        f"  ·  {len(frame):,} positions  ·  {wins:,} won / {len(frame) - wins:,} nowin"
    )
    st.subheader(selected)
    if result["auc_note"]:
        st.info(result["auc_note"] + " The other five metrics are still shown.")
    for column, metric in zip(st.columns(6), METRICS):
        value = result["metrics"][metric]
        display = "N/A" if value is None else (f"{value:.2%}" if metric == "Accuracy" else f"{value:.4f}")
        column.metric(metric, display)
    st.caption("Positive class: won. Precision, recall and F1 describe White-can-win predictions.")

    evaluation_tab, comparison_tab, positions_tab = st.tabs([
        "Model evaluation", "Compare all five models", "Inspect test positions",
    ])
    with evaluation_tab:
        chart_col, report_col = st.columns([1, 1])
        with chart_col:
            st.subheader("Confusion matrix")
            st.altair_chart(confusion_chart(result["confusion_matrix"]), width="stretch")
            st.caption("Rows are actual outcomes; columns are predictions. Hover over a cell for its count.")
        with report_col:
            st.subheader("Classification report")
            report = result["classification_report"]
            st.dataframe(pd.DataFrame([
                {"Class": label, **report[label]} for label in ["nowin", "won", "macro avg", "weighted avg"]
            ]), hide_index=True, column_config={
                key: st.column_config.NumberColumn(format="%.4f") for key in ["precision", "recall", "f1-score"]
            })
            tn, fp = result["confusion_matrix"][0]
            fn, tp = result["confusion_matrix"][1]
            st.write(f"**{tp + tn:,} correct predictions** out of {len(frame):,} positions.")
            st.write(f"**False wins: {fp:,}** — predicted won when the label is nowin.")
            st.write(f"**Missed wins: {fn:,}** — predicted nowin when the label is won.")
        with st.expander("ROC curve and metric definitions"):
            if result["metrics"]["AUC"] is not None:
                fpr, tpr, _ = roc_curve(frame[TARGET].eq("won").astype(int), result["scores"])
                curve = alt.Chart(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})).mark_line(
                    color="#1e5c47",
                ).encode(
                    x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["False positive rate:Q", "True positive rate:Q"],
                ).properties(height=240)
                st.altair_chart(curve, width="stretch")
            st.write(
                "Accuracy measures the fraction correct. Precision measures how often a predicted win is right; "
                "recall measures how many labeled wins are found. F1 balances precision and recall. "
                "AUC measures ranking across thresholds. MCC summarizes agreement on a scale from −1 to 1."
            )
            st.caption("AUC uses probabilities for four models and decision values for SVM. "
                       "Undefined precision, recall, F1 and MCC follow the evaluator's zero convention.")

    with comparison_tab:
        st.subheader("One test set. Five classifiers.")
        st.caption("Every value below is calculated from the currently selected test data.")
        st.dataframe(comparison, hide_index=True, column_config={
            metric: st.column_config.NumberColumn(format="%.4f") for metric in METRICS
        })
        score_chart = alt.Chart(comparison).mark_bar(color="#1e5c47", cornerRadiusEnd=3).encode(
            x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("ML Model Name:N", sort="-x", title=None),
            tooltip=["ML Model Name:N", alt.Tooltip("F1:Q", format=".4f")],
        ).properties(height=220)
        st.altair_chart(score_chart, width="stretch")
        st.download_button(
            "Download comparison CSV", comparison.to_csv(index=False), "model_comparison.csv", "text/csv",
            key="comparison_download",
        )
        manifest = json.loads((ROOT / "results/manifest.json").read_text())
        st.caption(f"Original training CV selection: {manifest['selected_model']}. "
                   "A ranking on an uploaded subset does not replace training validation.")

    with positions_tab:
        st.subheader("Inspect predictions")
        predictions = prediction_table(frame, result)
        only_errors = st.checkbox("Show errors only", key="only_errors")
        visible = predictions[predictions["Result"] != "Correct"] if only_errors else predictions
        if visible.empty:
            st.success("No errors for this model on these test positions.")
        st.dataframe(visible, hide_index=True)
        st.caption("Test row numbers start at 1 and refer to data rows in this CSV.")
        st.download_button(
            "Download selected model predictions", predictions.to_csv(index=False),
            "model_predictions.csv", "text/csv", key="predictions_download",
        )
        with st.expander("Preview input features"):
            st.dataframe(frame.head(50), hide_index=True)
            st.caption("Showing up to 50 rows. These are descriptors supplied by the dataset, not board coordinates.")

    st.divider()
    st.caption("Educational evaluation of one restricted chess endgame. Results do not establish performance "
               "on arbitrary chess positions. Use held-out data; training rows give optimistic results.")


if __name__ == "__main__":
    main()
