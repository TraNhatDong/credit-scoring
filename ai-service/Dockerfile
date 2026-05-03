# ============================================================
# AI Service — FastAPI + XGBoost credit scoring
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# Install system + Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --create-home appuser

# Copy requirements + install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Pre-warm models at startup
ENV PYTHONUNBUFFERED=1

# Non-root user
USER appuser

# Expose port
EXPOSE 8001

# Readiness check — only mark healthy once models are loaded
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8001/ready || exit 1

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
