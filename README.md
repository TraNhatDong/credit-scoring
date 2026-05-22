# Credit Scoring System — Enterprise Microservices

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (React)                       │
│            Dashboard · Glassmorphism UI                  │
│                        :3000                              │
└──────────────┬──────────────────────────┬─────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    Core Backend          │  │      AI Service          │
│    (Java 25 + SB3)       │◄─┤    (Python 3.10 + FastAPI)│
│    :8080                 │  │    :8001                 │
│    REST · JPA · WebClient│  │    XGBoost + SHAP        │
└──────────────┬───────────┘  └──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│              PostgreSQL 16 (JSONB + Partitioning)        │
│                      :5432                               │
│  customers · credit_applications · ai_audit_log         │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Git
- 8 GB RAM recommended

### Run all services

```bash
# 1. Clone / navigate to project
cd credit-scoring

# 2. Generate training data + train model (optional, uses mock if skipped)
cd ai-service
pip install -r requirements.txt
python -m app.scripts.generate_data --rows 50000 --output data/credit_train.csv
python -m app.scripts.train_model --data data/credit_train.csv --model models/

# 3. Build & start all containers
cd ..
docker-compose up --build

# Services will be available at:
#   Frontend     http://localhost:3000
#   Backend API  http://localhost:8080/api/v1
#   AI Service   http://localhost:8001/docs
#   PostgreSQL   localhost:5432
```

## Services

### AI Service (`ai-service/`)
- **FastAPI** REST API with Pydantic validation
- **XGBoost** binary classifier for default prediction
- **SHAP TreeExplainer** for per-feature attribution
- Mock mode (no model file needed for development)
- Multi-stage Dockerfile, non-root user

```
POST /api/v1/score          → credit_score, risk_probability, shap_explanations
GET  /api/v1/model-info     → model metadata
GET  /health                → liveness probe
```

### Core Backend (`core-backend/`)
- **Spring Boot 3.3** with Java 25
- **Spring Data JPA** + PostgreSQL
- **WebClient** for async AI service calls with retry
- `@RestControllerAdvice` global exception handling
- JSONB columns for AI audit trail

```
POST /api/v1/customers           → create customer
GET  /api/v1/customers           → list customers
POST /api/v1/applications        → create application
POST /api/v1/applications/submit → submit to AI scoring
POST /api/v1/applications/decide → approve/reject
```

### Frontend (`frontend/`)
- **React 18** + **Vite** + **TypeScript**
- **Glassmorphism** design system (CSS-only, no UI library)
- Scoring page with animated score gauge + SHAP waterfall bars
- Full CRUD for customers and applications

### Database (`db/migrations/`)
- PostgreSQL 16 with `uuid-ossp` extension
- 3NF schema: `customers`, `credit_applications`, `ai_audit_log`
- `ai_audit_log` table is **partitioned by month** (rolling 12-month window)
- JSONB columns for request/response audit trail
- Auto-update `updated_at` triggers

## Key Business Rules

| Condition | Action |
|---|---|
| Score ≥ 700 **AND** Risk = LOW | Auto-approve (SYSTEM_AUTO) |
| Score < 580 **OR** Risk = CRITICAL | Manual review required |
| Any status except COMPLETED | Cannot decide |

## Score Bands

| Probability | Score Range | Risk Level |
|---|---|---|
| < 5% | 740–850 | LOW |
| 5–15% | 580–740 | MEDIUM |
| 15–35% | 500–580 | HIGH |
| > 35% | 300–500 | CRITICAL |

## Environment Variables

### Core Backend (`application.yml`)
```yaml
SPRING_DATASOURCE_URL: jdbc:postgresql://postgres:5432/credit_scoring
SPRING_DATASOURCE_USERNAME: credit_user
SPRING_DATASOURCE_PASSWORD: credit_pass_secure_2026
AI_SERVICE_URL: http://ai-service:8001
```

### AI Service (`config.py`)
```env
MODEL_PATH=/app/models/credit_xgb_model.pkl
LOG_LEVEL=INFO
```

## Testing

### AI Service
```bash
cd ai-service
pip install pytest
pytest tests/ -v
```

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, TypeScript, Lucide Icons |
| Backend | Java 25, Spring Boot 3.3 |
| AI | Python 3.10, FastAPI, XGBoost, SHAP |
| Database | PostgreSQL 16 |
| Container | Docker, Docker Compose |
