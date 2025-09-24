"""Airflow DAG for training the Titanic survival model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import importlib.util

from airflow import DAG
from airflow.operators.python import PythonOperator


_HELPER_PATH = Path(__file__).resolve().parent / "_utils.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("airflow_dag_utils", _HELPER_PATH)
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
ensure_repo_root_on_path = _HELPER_MODULE.ensure_repo_root_on_path


REPO_ROOT = ensure_repo_root_on_path(Path(__file__))

from code.models.train_model import PIPELINE_PATH, TEST_PATH, TRAIN_PATH, run_training  # noqa: E402


def train_model_task(**context):
    """Invoke the training pipeline and return artifact metadata."""

    return run_training(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        pipeline_path=PIPELINE_PATH,
    )


with DAG(
    dag_id="train_model",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    run_training_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
    )
