"""
Health check response model.
"""
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status:   str
    service:  str = "credit-ai-service"
