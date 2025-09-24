"""Airflow DAG chaining preprocessing, training, and deployment packaging."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import importlib.util
import subprocess
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator  # OK for Airflow 3.x

# --- utils bootstrap ---
_HELPER_PATH = Path(__file__).resolve().parent / "_utils.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("airflow_dag_utils", _HELPER_PATH)
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
ensure_repo_root_on_path = _HELPER_MODULE.ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path(Path(__file__))

from services.airflow.dags.preproccess import run_preprocessing  # noqa: E402
from code.models.train_model import run_training  # noqa: E402

# --- compose file content ---
DOCKER_COMPOSE_PATH = REPO_ROOT / "code" / "deployment" / "docker-compose.yml"
DOCKER_COMPOSE_TEMPLATE = """version: "3.9"

services:
  api:
    build:
      context: ../..
      dockerfile: code/deployment/api/Dockerfile
    container_name: titanic_api
    ports:
      - "8000:8000"
    volumes:
      - ../../models:/app/models:ro
      - ../../data:/app/data:ro
    restart: unless-stopped

  app:
    build:
      context: ../..
      dockerfile: code/deployment/app/Dockerfile
    container_name: titanic_app
    depends_on:
      - api
    ports:
      - "8501:8501"
    volumes:
      - ../../models:/app/models:ro
    restart: unless-stopped
"""

def run_training_task(**context):
    return run_training()

def generate_docker_compose(**_context):
    DOCKER_COMPOSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCKER_COMPOSE_PATH.write_text(DOCKER_COMPOSE_TEMPLATE, encoding="utf-8")

    return str(DOCKER_COMPOSE_PATH.resolve())

# Bring services up in detached mode and log a quick health check
def compose_up(compose_payload: Any = None, **_context):
    if isinstance(compose_payload, dict):
        compose_path = compose_payload.get("docker_compose_path")
    elif compose_payload:
        compose_path = str(compose_payload)
    else:
        compose_path = None

    if not compose_path:
        compose_path = str(DOCKER_COMPOSE_PATH.resolve())

    compose_file = Path(compose_path)
    if not compose_file.exists():
        raise FileNotFoundError(f"compose_up: docker compose file not found at {compose_file}")
    workdir = compose_file.parent  # run docker compose from this directory

    # Build & start in detached mode
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d", "--build"],
        check=True,
        cwd=str(workdir),
    )

    # Optional: quick health probe for Streamlit and FastAPI
    # Note: curl may not be present; ignore failures.
    try:
        subprocess.run(["curl", "-I", "http://localhost:8501/_stcore/health"], cwd=str(workdir), check=False)
    except Exception:
        pass
    try:
        subprocess.run(["curl", "-I", "http://localhost:8000/docs"], cwd=str(workdir), check=False)
    except Exception:
        pass

    return {"compose_workdir": str(workdir)}

with DAG(
    dag_id="full_ml_pipeline_v1",
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/5 * * * *",  # CHANGED: keep if you want 5-min schedule
    catchup=False,
    tags=["titanic", "ml"],
) as dag:
    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocessing,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=run_training_task,
    )

    compose = PythonOperator(
        task_id="make_docker_compose",
        python_callable=generate_docker_compose,
    )

    compose_up_task = PythonOperator(
        task_id="compose_up",
        python_callable=compose_up,
        op_kwargs={"compose_payload": compose.output},
    )

    preprocess >> train >> compose >> compose_up_task
