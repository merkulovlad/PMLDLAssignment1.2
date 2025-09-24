# Airflow + Docker ML Pipeline

This repository packages an end-to-end Titanic survival workflow. Apache Airflow orchestrates data preprocessing and model training, and Docker keeps the serving stack (FastAPI + Streamlit) reproducible.

## Repository layout
- `services/airflow/dags/` – Airflow DAG definitions (`full_pipeline_v1.py`, `preproccess.py`, `train_model.py`) and shared helpers.
- `services/airflow/airflow_home/` – Local Airflow metadata, logs, and credentials created when you run Airflow. The folder is ignored by git so each developer can keep their own state.
- `code/deployment/` – Docker artefacts. The `docker-compose.yml` file starts the FastAPI prediction service and the Streamlit UI.
- `models/` & `data/` – Model artefacts and datasets that Airflow tasks read from and write to. They are mounted into the Docker containers.

## Prerequisites
- Python 3.11 (for authoring DAG logic and local utilities)
- Docker Desktop or Docker Engine with the Compose plugin (`docker compose` CLI)
- Optional but recommended: A dedicated virtual environment (the repo uses `airflow-venv/` locally)

## Airflow quickstart
1. Create a virtual environment and install Airflow (pin the version you want to work with):
   ```bash
   python -m venv airflow-venv
   source airflow-venv/bin/activate
   pip install --upgrade pip "apache-airflow==2.9.1"
   ```
2. Set `AIRFLOW_HOME` so Airflow writes its metadata inside the repository:
   ```bash
   export AIRFLOW_HOME=$(pwd)/services/airflow/airflow_home
   ```
3. Initialize and launch the local instance. `airflow standalone` is the easiest way to get the webserver, scheduler, and a default admin user running:
   ```bash
   airflow standalone
   ```
   The command prints the login and password for the web UI (http://localhost:8080). Keep the process running while you experiment.
4. Upload DAGs if needed: with `AIRFLOW_HOME` pointing at `services/airflow/airflow_home`, Airflow automatically finds the DAG files inside `services/airflow/dags/`.

### Working with the DAGs
- `process_data` preprocesses the raw Titanic dataset into train/test splits and persists the preprocessing pipeline (`data/processed/`).
- `train_model` fits the logistic-regression model defined in `code/models/train_model.py` and stores the packaged artefacts under `models/`.
- `full_ml_pipeline_v1` chains preprocessing → training → a Docker Compose refresh. The final task rewrites `code/deployment/docker-compose.yml` (so it always reflects the expected services) and can optionally bring the stack up by shelling out to `docker compose`.

Trigger runs from the UI or the CLI (e.g., `airflow dags trigger full_ml_pipeline_v1`). Logs live under `services/airflow/airflow_home/logs/` and stay out of version control.

## Docker workflow
The repository contains everything you need to serve the trained model locally with containers.

1. Ensure the latest artefacts exist (`models/` and `data/processed/`). Running the Airflow pipeline is the easiest path.
2. Build and start the services:
   ```bash
   cd code/deployment
   docker compose up --build
   ```
3. Visit the apps:
   - FastAPI docs: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

The Compose file mounts the `models/` directory read-only so you can retrain with Airflow and instantly refresh the running containers. Shut everything down with `docker compose down`.

## Environment variables & secrets
- Place sensitive settings (database URIs, API keys) in a `.env` file next to `docker-compose.yml`. The file is ignored by git; add a `.env.example` if you need to document defaults.
- Airflow uses the SQLite metadata database by default. If you switch to Postgres, add the connection string via the Airflow UI or environment variables rather than committing secrets.

## Troubleshooting tips
- If Docker complains about stale containers after an Airflow-triggered deploy, run `docker compose down` manually before re-running the pipeline.
- When developing new DAGs, keep an eye on `services/airflow/airflow_home/logs/` for detailed task logs.
- Regenerate the virtual environment (`airflow-venv/`) whenever dependencies change; the directory is ignored, so each contributor maintains their own copy.
