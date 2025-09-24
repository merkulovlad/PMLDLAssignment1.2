"""Train the Titanic classifier from preprocessed datasets.

The preprocessing DAG is expected to generate the train/test CSVs and the
pipeline artifact under ``data/processed``. This script loads those files,
fits a logistic regression model, persists the outputs, and logs everything
to MLflow so the run history is traceable.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

import joblib
import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
PIPELINE_PATH = PROCESSED_DIR / "preprocess_pipeline.joblib"
MODELS_DIR = REPO_ROOT / "models"
MLRUNS_DIR = REPO_ROOT / "mlruns"
TARGET_COLUMN = "survived"


def load_processed_data(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed train/test CSV files and split features/target."""

    if not train_path.exists():
        raise FileNotFoundError(f"Missing processed training data at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing processed test data at {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in train data")
    if target_column not in test_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data")

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column].astype(int)

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].astype(int)

    return x_train, x_test, y_train, y_test


def load_preprocessing_artifacts(path: Path = PIPELINE_PATH) -> Dict[str, object]:
    """Load preprocessing artifacts saved by the preprocessing DAG."""

    if path.exists():
        return joblib.load(path)
    return {}


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a logistic regression classifier."""

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(x_train, y_train.values)
    return model


def evaluate_model(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Sequence[int]]:
    """Compute evaluation metrics and return metrics, report, and predictions."""

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
    }

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return metrics, report, y_pred


def build_model_package(
    model: LogisticRegression,
    preprocessing_artifacts: Dict[str, object],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    """Aggregate trained components and metadata for persistence."""

    return {
        "model": model,
        "preprocessing": preprocessing_artifacts,
        "metadata": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model_type": "LogisticRegression",
            "metrics": metrics,
        },
    }


def save_model_artifacts(
    model: LogisticRegression,
    package: Dict[str, object],
) -> Tuple[Path, Path]:
    """Persist the trained estimator and the packaged bundle."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "titanic_model.joblib"
    package_path = MODELS_DIR / "titanic_model_package.joblib"

    joblib.dump(model, model_path)
    joblib.dump(package, package_path)

    return model_path, package_path


def log_to_mlflow(
    model: LogisticRegression,
    x_train: pd.DataFrame,
    metrics: Dict[str, float],
    params: Dict[str, object],
    evaluation_report: Dict[str, Dict[str, float]],
    package_path: Path,
    preprocessing_path: Path,
    y_test: pd.Series,
    y_pred: Sequence[int],
) -> None:
    """Send metrics, model, and artifacts to MLflow."""

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("titanic_survival")

    signature = infer_signature(x_train, model.predict_proba(x_train))
    input_example = x_train.head(5)

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            signature=signature,
            input_example=input_example,
        )
        mlflow.log_artifact(str(package_path), artifact_path="artifacts")
        if preprocessing_path.exists():
            mlflow.log_artifact(str(preprocessing_path), artifact_path="artifacts")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            report_path = tmp_path / "classification_report.json"
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(evaluation_report, report_file, indent=2)
            mlflow.log_artifact(str(report_path), artifact_path="artifacts")

            confusion_path = tmp_path / "confusion_matrix.json"
            cm = confusion_matrix(y_test, y_pred).tolist()
            with open(confusion_path, "w", encoding="utf-8") as cm_file:
                json.dump({"confusion_matrix": cm}, cm_file, indent=2)
            mlflow.log_artifact(str(confusion_path), artifact_path="artifacts")


def run_training(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    pipeline_path: Path = PIPELINE_PATH,
) -> Dict[str, object]:
    """End-to-end training routine returning metadata about persisted assets."""

    x_train, x_test, y_train, y_test = load_processed_data(train_path, test_path)
    preprocessing_artifacts = load_preprocessing_artifacts(pipeline_path)

    model = train_model(x_train, y_train)
    metrics, report, y_pred = evaluate_model(model, x_test, y_test)

    package = build_model_package(model, preprocessing_artifacts, metrics)
    model_path, package_path = save_model_artifacts(model, package)

    params = {
        "model_type": "LogisticRegression",
        "solver": model.solver,
        "max_iter": model.max_iter,
        "random_state": model.random_state,
        "n_features": x_train.shape[1],
    }

    log_to_mlflow(
        model=model,
        x_train=x_train,
        metrics=metrics,
        params=params,
        evaluation_report=report,
        package_path=package_path,
        preprocessing_path=pipeline_path,
        y_test=y_test,
        y_pred=y_pred,
    )

    return {
        "model_path": str(model_path),
        "package_path": str(package_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic model from preprocessed data")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="Path to processed training CSV")
    parser.add_argument("--test", type=Path, default=TEST_PATH, help="Path to processed test CSV")
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=PIPELINE_PATH,
        help="Path to preprocessing pipeline joblib",
    )
    args = parser.parse_args()

    result = run_training(train_path=args.train, test_path=args.test, pipeline_path=args.pipeline)

    print("Model saved to:", result["model_path"])  # noqa: T201
    print("Model package saved to:", result["package_path"])  # noqa: T201
    print("Evaluation metrics:")  # noqa: T201
    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
