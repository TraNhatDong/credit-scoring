# Credit Scoring AI Service

Champion LR + Challenger XGBoost inference API with SHAP explainable AI.

## Architecture

```
POST /api/v1/score          → best model (highest AUC) + SHAP explanations
POST /api/v1/score/multi    → both models + ensemble + voting
GET  /api/v1/models         → loaded model metadata + metrics
GET  /api/v1/model-info     → static feature/version info
GET  /api/v1/algorithms     → algorithm documentation
GET  /health                → liveness probe
GET  /ready                 → readiness probe (models loaded?)
```

## Models

| Role | Model | Features | Notes |
|---|---|---|---|
| Champion | Logistic Regression | 12 (IV-filtered) | Linear, isotonic-calibrated, regulatory-friendly |
| Challenger | XGBoost | 16 (all engineered) | Gradient boosting, isotonic-calibrated, best AUC |

## Preprocessing Pipeline (shared with training)

Identical to `_wrapper._predict_core`:

1. **Impute** — `NumberOfDependents` → 0; all others → median from training
2. **Engineer** — `monthly_debt`, `late_payments_total`, `debt_per_person`, `income_per_credit_line`, `utilization_per_line`, `is_income_missing`
3. **Clip** — age `(18, 100)`, RevolvingUtil `(0, 5)`, DebtRatio `(0, 5)` or `(0, p99)` for income-missing rows

## Inference Pipeline

```
Request (10 raw features)
  → Preprocessor.transform()
  → feature_order reindex
  → dtype float32 (XGB) / float64 (LR)
  → [scaler.transform] for LR
  → model.predict_proba()
  → calibrator.predict()
  → NaN + range guard
  → threshold → binary prediction
  → FICO score mapping
  → SHAP explanations
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (requires raw CSV at data/cs-training.csv)
python -m app.scripts.run_pipeline

# Run service
MODEL_PATH=./models uvicorn app.main:app --host 0.0.0.0 --port 8001

# Run with Docker
docker build -t credit-ai-service .
docker run -v ./models:/app/models credit-ai-service
```

## Docker Compose

The root `docker-compose.yml` includes the AI service alongside `postgres`, `core-backend`, and `frontend`.

```bash
docker compose up ai-service
```

Health checks: `/ready` returns DOWN until models load successfully.

## Environment Variables

See `.env.example` for all supported variables. Key ones:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/app/models` | Directory containing `.pkl` artifact files |
| `MODEL_VERSION` | `v4.1_20260503` | Version tag (matches artifact filenames) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `PORT` | `8001` | Server port |

## Artifact Files

Model artifacts are single-file bundles produced by `run_pipeline.py`:

```
models/
  champion_lr_model.v4.1_YYYYMMDD.pkl      # LR bundle
  challenger_xgb_model.v4.1_YYYYMMDD.pkl  # XGB bundle
  pipeline_metadata.json                    # shared metadata
```

Each bundle contains: `{model, calibrator, preprocessor_state, feature_order, threshold, metrics, ...}`

## API Example

```bash
curl -X POST http://localhost:8001/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.5,
    "MonthlyIncome": 9120,
    "NumberOfOpenCreditLinesAndLoans": 13,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 6,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

Response:

```json
{
  "credit_score": 724,
  "risk_probability": 0.0823,
  "risk_level": "LOW",
  "shap_explanations": [
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "value": 0.3, "contribution": 28.4, "direction": "POSITIVE"},
    ...
  ],
  "model_version": "Logistic Regression (Champion)",
  "inference_ms": 12
}
```
