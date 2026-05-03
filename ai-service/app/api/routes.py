"""
FastAPI API Routes for the Credit Scoring AI Service.

Endpoints:
  POST /api/v1/score            — Single model scoring (best model by AUC)
  POST /api/v1/score/multi      — Champion LR + Challenger XGB + ensemble
  GET  /api/v1/models           — List loaded models + metrics
  GET  /api/v1/model-info       — Model metadata
  GET  /api/v1/algorithms       — Algorithm documentation
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.models.schemas import (
    ErrorCode,
    ErrorResponse,
    ScoringRequest,
    ScoringResponse,
)
from app.services.scoring_service import ScoringService

logger = logging.getLogger("credit-ai-service.api")

_scoring_service: ScoringService | None = None


def get_scoring_service() -> ScoringService:
    """FastAPI dependency: return the shared service instance."""
    if _scoring_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service not initialised",
        )
    return _scoring_service


@asynccontextmanager
async def lifespan_scoring_service():
    """Initialise the scoring service when the app starts."""
    global _scoring_service
    models_dir = str(Path(settings.MODEL_PATH))
    logger.info("Initialising ScoringService (models dir: %s)", models_dir)
    _scoring_service = ScoringService(models_dir=models_dir)
    loaded = _scoring_service.load_models()
    logger.info("Models loaded: %s", loaded)
    yield
    logger.info("Disposing ScoringService")
    _scoring_service = None


router = APIRouter()


# ── POST /api/v1/score ───────────────────────────────────────
@router.post(
    "/score",
    response_model=ScoringResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        422: {"model": ErrorResponse, "description": "Pydantic validation failed"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Score a credit application",
    description=(
        "Receives a validated credit application payload, runs the XGBoost model, "
        "computes SHAP feature attributions, and returns the credit score, "
        "probability of default, risk level, and per-feature explanations."
    ),
    tags=["Scoring"],
)
async def score_credit_application(
    request: ScoringRequest,
) -> ScoringResponse:
    """
    Primary endpoint: score one credit application.

    Input is pre-validated by Pydantic before reaching this handler.
    """
    service = get_scoring_service()

    try:
        response = service.predict(request)
        return response

    except FileNotFoundError as exc:
        logger.exception("Model file not found during prediction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded",
        )
    except Exception as exc:
        logger.exception("Unexpected error during scoring")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during prediction",
        )


# ── GET /api/v1/model-info ───────────────────────────────────
@router.get(
    "/model-info",
    summary="Get model metadata",
    description="Returns the loaded model version and feature list.",
    tags=["Scoring"],
)
async def model_info() -> dict:
    """Return static model metadata (no scoring needed)."""
    return {
        "model_version":  settings.MODEL_VERSION,
        "feature_count":  len(settings.FEATURE_COLUMNS),
        "features":       settings.FEATURE_COLUMNS,
        "score_range":    [settings.SCORE_MIN, settings.SCORE_MAX],
    }


# ── GET /api/v1/models ────────────────────────────────────────
@router.get(
    "/models",
    summary="List all trained models",
    description="Returns metadata and metrics for each loaded model.",
    tags=["Scoring"],
)
async def list_models() -> dict:
    """Return list of available models with their performance metrics."""
    service = get_scoring_service()
    return {
        "available_models": service.get_available_models(),
        "model_version":    settings.MODEL_VERSION,
    }


# ── POST /api/v1/score/multi ─────────────────────────────────
@router.post(
    "/score/multi",
    summary="Score with all models",
    description=(
        "Runs Champion LR + Challenger XGB on the same input. "
        "Returns per-model scores, ensemble average, SHAP explanations, and voting results."
    ),
    tags=["Scoring"],
)
async def score_multi_model(request: ScoringRequest) -> dict:
    """
    Multi-model scoring endpoint.

    Runs both models:
      1. Champion Logistic Regression — linear, highly interpretable, regulatory-friendly
      2. Challenger XGBoost — gradient boosting, best AUC for tabular data

    Returns:
      - Per-model credit_score, probability, risk_level, SHAP explanations
      - Ensemble: AUC-weighted average of all model outputs
      - Voting: how many models voted approve vs reject
    """
    service = get_scoring_service()

    try:
        result = service.predict_multi(request)
        return result

    except FileNotFoundError as exc:
        logger.exception("Model file not found")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded",
        )
    except Exception as exc:
        logger.exception("Unexpected error during multi-model scoring")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during multi-model prediction",
        )


# ── GET /api/v1/algorithms ───────────────────────────────────
@router.get(
    "/algorithms",
    summary="Get algorithm descriptions",
    description="Returns documentation about each algorithm in the pipeline.",
    tags=["Scoring"],
)
async def algorithms() -> dict:
    """
    Returns human-readable descriptions of all algorithms used.
    """
    return {
        "algorithms": [
            {
                "id":       "champion_lr",
                "name":     "Logistic Regression (Champion)",
                "category": "Classification (Linear)",
                "role":     "Champion — highly interpretable, regulatory-friendly",
                "strengths": ["Fast", "Interpretable coefficients", "Probability-calibrated"],
                "limitations": ["Assumes linear decision boundary"],
            },
            {
                "id":       "challenger_xgb",
                "name":     "XGBoost (Challenger)",
                "category": "Ensemble Learning (Gradient Boosting)",
                "role":     "Challenger — best-in-class for tabular data, Optuna-tuned",
                "strengths": [
                    "Sequential error correction (each tree fixes previous mistakes)",
                    "Highest AUC-ROC typically",
                    "Fast with large datasets",
                ],
                "limitations": ["Black-box", "Requires SHAP for interpretability"],
            },
        ],
        "preprocessing": [
            {
                "id":    "standardscaler",
                "name":  "StandardScaler",
                "role":  "Normalises features (zero mean, unit variance) — required for LR",
            },
        ],
        "explainability": [
            {
                "id":    "shap",
                "name":  "SHAP (SHapley Additive exPlanations)",
                "role":  "Game-theory-based feature attribution — explains each prediction",
            },
        ],
    }

