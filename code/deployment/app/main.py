"""Streamlit front-end for the Titanic survival predictor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Append the repository root (`.../PMLDLAssignment1.2`) so imports like
# `code.deployment.*` resolve when running from the Streamlit app directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from code.deployment.api.main import (  # noqa: E402
    PredictionRequest,
    PredictionResponse,
    get_model_package,
    predict as run_prediction,
)


def _format_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted[key] = f"{value:.3f}"
        else:
            formatted[key] = str(value)
    return formatted


def _collect_optional_float(label: str, help_text: str | None = None) -> float | None:
    """Allow users to leave a numeric field blank to keep it unknown."""

    raw_value = st.text_input(label, value="", help=help_text)
    stripped = raw_value.strip()
    if not stripped:
        return None

    try:
        return float(stripped)
    except ValueError:
        st.warning(f"Could not parse '{raw_value}' as a number; leaving it blank.")
        return None


def _collect_optional_int(label: str, help_text: str | None = None) -> int | None:
    raw_value = st.text_input(label, value="", help=help_text)
    stripped = raw_value.strip()
    if not stripped:
        return None

    try:
        return int(float(stripped))
    except ValueError:
        st.warning(f"Could not parse '{raw_value}' as an integer; leaving it blank.")
        return None


def _predict(payload: Dict[str, Any]) -> PredictionResponse:
    request_payload = PredictionRequest(**payload)
    return run_prediction(request_payload)


def main() -> None:
    st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

    st.title("Titanic Survival Prediction")
    st.write(
        "Enter passenger details below to estimate the probability of survival using the "
        "trained logistic regression model."
    )

    package = get_model_package()
    metadata: Dict[str, Any] = package.get("metadata", {})
    metrics: Dict[str, Any] = metadata.get("metrics", {})

    with st.sidebar:
        st.header("Model Snapshot")
        st.write(f"Created at: {metadata.get('created_at', 'unknown')}")
        st.write(f"Model type: {metadata.get('model_type', 'unknown')}")

        if metrics:
            st.subheader("Evaluation metrics")
            formatted_metrics = _format_metrics(metrics)
            for metric_name, metric_value in formatted_metrics.items():
                st.metric(metric_name.replace("_", " ").title(), metric_value)
        else:
            st.info("Model metrics are unavailable.")

    with st.form("prediction_form"):
        st.subheader("Passenger details")
        pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=0, help="Ticket class (1 = First)")
        sex = st.selectbox("Sex", options=["female", "male"], index=1)

        age = _collect_optional_float("Age", help_text="Leave blank if unknown.")
        sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0, step=1)
        fare = _collect_optional_float("Fare", help_text="Ticket fare in the original currency.")
        embarked = st.selectbox(
            "Port of embarkation",
            options=["", "C", "Q", "S"],
            format_func=lambda v: "Unknown" if v == "" else v,
        )

        submitted = st.form_submit_button("Predict survival")

    if submitted:
        payload: Dict[str, Any] = {
            "pclass": int(pclass),
            "sex": sex,
            "age": age,
            "sibsp": int(sibsp),
            "parch": int(parch),
            "fare": fare,
            "embarked": embarked or None,
        }

        try:
            response = _predict(payload)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f"Prediction failed: {exc}")
            return

        st.success("Prediction complete")
        col_prob, col_label = st.columns(2)
        col_prob.metric("Survival probability", f"{response.probability:.2%}")
        col_label.metric("Predicted outcome", "Survived" if response.survived == 1 else "Did not survive")

        if response.model_timestamp:
            st.caption(f"Model package timestamp: {response.model_timestamp}")


if __name__ == "__main__":
    main()
