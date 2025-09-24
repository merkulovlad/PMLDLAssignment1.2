"""FastAPI application for serving the Titanic survival model."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LOGGER = logging.getLogger(__name__)
_FILE_PATH = Path(__file__).resolve()
# Jump to repository root (`.../PMLDLAssignment1.2`) rather than `.../code` so we
# align with the training script that persists artifacts under `<repo>/models`.
REPO_ROOT = _FILE_PATH.parents[3]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PACKAGE_PATH = MODELS_DIR / "titanic_model_package.joblib"


class PredictionRequest(BaseModel):
    """Schema representing the raw passenger features."""

    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1-3)")
    sex: str = Field(..., description="Passenger sex (e.g. male/female)")
    age: float | None = Field(None, ge=0, description="Passenger age in years")
    sibsp: int | None = Field(None, ge=0, description="# of siblings or spouses aboard")
    parch: int | None = Field(None, ge=0, description="# of parents or children aboard")
    fare: float | None = Field(None, ge=0, description="Ticket fare")
    embarked: str | None = Field(None, description="Port of embarkation")


class PredictionResponse(BaseModel):
    """Prediction payload returning class and probability."""

    survived: int = Field(..., description="Predicted survival label (0/1)")
    probability: float = Field(..., ge=0, le=1, description="Probability of survival")
    model_timestamp: str | None = Field(None, description="Model package creation timestamp")


def _load_package() -> Dict[str, Any]:
    if not MODEL_PACKAGE_PATH.exists():
        raise FileNotFoundError(
            f"Model package missing at {MODEL_PACKAGE_PATH}. Train the model before serving."
        )

    package = joblib.load(MODEL_PACKAGE_PATH)
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("Unexpected model package format.")
    return package


@lru_cache(maxsize=1)
def get_model_package() -> Dict[str, Any]:
    """Cached access to the packaged model and preprocessing artifacts."""

    LOGGER.info("Loading model package from %s", MODEL_PACKAGE_PATH)
    return _load_package()


def _prepare_input_frame(payload: PredictionRequest) -> pd.DataFrame:
    data = payload.dict()
    df = pd.DataFrame([data])
    return df


def _apply_preprocessing(
    df: pd.DataFrame,
    preprocessing: Dict[str, Any],
) -> pd.DataFrame:
    num_imputer = preprocessing.get("num_imputer")
    cat_imputer = preprocessing.get("cat_imputer")
    encoder = preprocessing.get("encoder")
    scaler = preprocessing.get("scaler")

    if scaler is None or num_imputer is None:
        raise ValueError("Preprocessing artifacts are incomplete; retrain preprocessing pipeline.")

    num_features = getattr(num_imputer, "feature_names_in_", None)
    num_cols: Sequence[str] = list(num_features) if num_features is not None else []

    cat_features = getattr(cat_imputer, "feature_names_in_", None)
    cat_cols: Sequence[str] = list(cat_features) if cat_features is not None else []

    missing_cols: List[str] = [col for col in num_cols if col not in df.columns]
    missing_cols += [col for col in cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input payload missing required columns: {missing_cols}")

    df_num = df[num_cols].apply(pd.to_numeric, errors="coerce")
    num_array = num_imputer.transform(df_num)
    x_num = pd.DataFrame(num_array, columns=num_cols, index=df.index)

    if cat_cols and cat_imputer is not None and encoder is not None:
        df_cat = df[cat_cols].astype(str)
        cat_array = cat_imputer.transform(df_cat)
        ohe_array = encoder.transform(cat_array)
        ohe_cols = encoder.get_feature_names_out(cat_cols)
        x_ohe = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
    else:
        x_ohe = pd.DataFrame(index=df.index)

    x_combined = pd.concat([x_num, x_ohe], axis=1)

    scaler_features = getattr(scaler, "feature_names_in_", None)
    scaler_feature_names: Sequence[str]
    if scaler_features is None:
        scaler_feature_names = list(x_combined.columns)
    else:
        scaler_feature_names = list(scaler_features)

    x_combined = x_combined.reindex(columns=scaler_feature_names, fill_value=0)

    scaled = scaler.transform(x_combined)
    return pd.DataFrame(scaled, columns=scaler_feature_names, index=df.index)


def predict(payload: PredictionRequest) -> PredictionResponse:
    package = get_model_package()
    model = package["model"]
    preprocessing = package.get("preprocessing", {})
    metadata = package.get("metadata", {})

    try:
        input_df = _prepare_input_frame(payload)
        transformed_df = _apply_preprocessing(input_df, preprocessing)
        proba = float(model.predict_proba(transformed_df)[0][1])
        pred = int(model.predict(transformed_df)[0])
    except Exception as exc:  # pragma: no cover - FastAPI will surface message
        LOGGER.exception("Failed to run prediction")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        survived=pred,
        probability=proba,
        model_timestamp=metadata.get("created_at"),
    )


app = FastAPI(title="Titanic Survival API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        get_model_package()
    except Exception as exc:  # pragma: no cover - fails fast on boot
        LOGGER.error("Unable to load model package: %s", exc)
        raise


@app.get("/health", tags=["internal"])
def health_check() -> Dict[str, Any]:
    package = get_model_package()
    metadata = package.get("metadata", {})
    return {
        "status": "ok",
        "model_timestamp": metadata.get("created_at"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    return predict(request)


__all__ = ["app", "predict", "PredictionRequest", "PredictionResponse"]
