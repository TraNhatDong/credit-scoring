"""
Credit Scoring AI Service - FastAPI Entry Point.

Expects model artifacts produced by run_pipeline.py:
  - champion_lr_model.{version}.pkl   (Logistic Regression)
  - challenger_xgb_model.{version}.pkl  (XGBoost)
  - pipeline_metadata.json

Each artifact is a self-contained bundle:
  {model, calibrator, preprocessor_state, feature_order, threshold, metrics, ...}
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exceptions import register_handlers
from app.api.routes import get_scoring_service, lifespan_scoring_service, router as api_router
from app.core.config import settings
from app.core.health import HealthCheckResponse

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("credit-ai-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → running → shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("Credit Scoring AI Service starting up")
    logger.info("  MODEL_PATH:    %s", settings.MODEL_PATH)
    logger.info("  MODEL_VERSION: %s", settings.MODEL_VERSION)
    logger.info("  FEATURES:      %s", settings.FEATURE_COLUMNS)
    logger.info("=" * 60)

    async with lifespan_scoring_service():
        logger.info("ScoringService ready")
        yield

    logger.info("Credit Scoring AI Service shutting down")


app = FastAPI(
    title="Credit Scoring AI Service",
    description=(
        "Champion LR + Challenger XGBoost credit risk scoring with SHAP explainable AI. "
        "Provides FICO-like credit scores (300–850), probability of default, risk level "
        "classification, and per-feature SHAP explanations."
    ),
    version="2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_handlers(app)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health():
    """Liveness probe — returns UP if the process is running."""
    return HealthCheckResponse(status="UP", service="credit-ai-service")


@app.get("/ready", response_model=HealthCheckResponse, tags=["Health"])
async def ready():
    """
    Readiness probe — returns UP only when models have loaded successfully.

    Returns DOWN if models failed to load, allowing orchestration (k8s, docker-compose)
    to detect unhealthy instances and stop routing traffic to them.
    """
    try:
        svc = get_scoring_service()
        if svc._loaded and svc._models:
            return HealthCheckResponse(status="UP", service="credit-ai-service")
    except Exception:
        pass
    return HealthCheckResponse(status="DOWN", service="credit-ai-service")
