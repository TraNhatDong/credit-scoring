"""
Pydantic request/response schemas for the Credit Scoring AI Service.
All inputs are strictly validated using Pydantic.
"""
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ErrorCode(str, Enum):
    """Standardized error codes returned by the AI service."""
    INVALID_AGE      = "INVALID_AGE"
    INVALID_INCOME   = "INVALID_INCOME"
    INVALID_DTI      = "INVALID_DTI"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    INTERNAL_ERROR   = "INTERNAL_ERROR"


# ──────────────────────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────────────────────
class ScoringRequest(BaseModel):
    """
    Input schema for a single credit scoring prediction.
    All fields use GMSC (Give Me Some Credit) dataset column names.

    Validated ranges are based on real-world banking criteria and dataset statistics.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0.0,
        description="Total balance on credit cards and personal lines of credit / sum of credit limits",
    )
    age: int = Field(..., ge=0, le=120, description="Borrower age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ...,
        description="Number of times borrower has been 30-59 days past due but no worse in last 2 years",
    )
    DebtRatio: float = Field(
        ..., ge=0.0,
        description="Monthly debt payments / monthly gross income",
    )
    MonthlyIncome: float = Field(..., gt=0, description="Monthly gross income")
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0,
        description="Number of open credit lines and loans",
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0,
        description="Number of times borrower has been 90+ days past due",
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0,
        description="Number of mortgage and real estate loans including home equity lines of credit",
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ...,
        description="Number of times borrower has been 60-89 days past due but no worse in last 2 years",
    )
    NumberOfDependents: float = Field(
        ..., ge=0,
        description="Number of dependents in family excluding themselves",
    )

    @field_validator("MonthlyIncome")
    @classmethod
    def income_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("MonthlyIncome must be strictly positive")
        return round(v, 2)

    @model_validator(mode="after")
    def cross_field_validation(self) -> "ScoringRequest":
        if self.NumberOfTime60_89DaysPastDueNotWorse > self.NumberOfTime30_59DaysPastDueNotWorse:
            raise ValueError(
                "NumberOfTime60_89DaysPastDueNotWorse cannot exceed NumberOfTime30_59DaysPastDueNotWorse"
            )
        if self.NumberOfTimes90DaysLate > self.NumberOfTime30_59DaysPastDueNotWorse:
            raise ValueError(
                "NumberOfTimes90DaysLate cannot exceed NumberOfTime30_59DaysPastDueNotWorse"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.3,
                "age": 45,
                "NumberOfTime30_59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.5,
                "MonthlyIncome": 9120,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        }
    }


# ──────────────────────────────────────────────────────────────
# SHAP Explanation
# ──────────────────────────────────────────────────────────────
class ShapExplanation(BaseModel):
    """
    Individual SHAP explanation for one feature.

    - contribution > 0  → feature pushes score UP  (POSITIVE)
    - contribution < 0  → feature pushes score DOWN (NEGATIVE)
    - contribution = 0  → feature has neutral effect
    """
    feature:      str = Field(..., description="Feature name")
    value:        float = Field(..., description="Actual feature value in this prediction")
    contribution: float = Field(..., description="SHAP contribution to prediction (can be negative)")
    direction:    str  = Field(...,
                                pattern="^(POSITIVE|NEGATIVE|NEUTRAL)$",
                                description="Whether this feature adds or subtracts from score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "feature": "DebtRatio",
                "value": 0.5,
                "contribution": 42.1,
                "direction": "POSITIVE",
            }
        }
    }


# ──────────────────────────────────────────────────────────────
# Response
# ──────────────────────────────────────────────────────────────
class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class ScoringResponse(BaseModel):
    """
    Full scoring response including the score, probability, risk level,
    and per-feature SHAP explanations.
    """
    credit_score:       int               = Field(..., ge=300, le=850,
                                                   description="Credit score (FICO-like range 300–850)")
    risk_probability:   float             = Field(...,
                                                   ge=0.0, le=1.0,
                                                   description="Probability of default (0.0–1.0)")
    risk_level:         RiskLevel          = Field(...,
                                                   description="Categorical risk bucket")
    shap_explanations:  list[ShapExplanation] = Field(...,
                                                   description="Per-feature SHAP breakdown, sorted by absolute contribution desc")
    model_version:      str               = Field(...,
                                                   description="Model version tag used for this prediction")
    inference_ms:       int               = Field(...,
                                                   ge=0,
                                                   description="Total inference time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "credit_score": 724,
                "risk_probability": 0.0823,
                "risk_level": "LOW",
                "shap_explanations": [
                    {
                        "feature": "DebtRatio",
                        "value": 0.5,
                        "contribution": 42.1,
                        "direction": "POSITIVE",
                    },
                    {
                        "feature": "NumberOfTime60-89DaysPastDueNotWorse",
                        "value": 0,
                        "contribution": 28.4,
                        "direction": "POSITIVE",
                    },
                ],
                "model_version": "1.0.0",
                "inference_ms": 87,
            }
        }
    }


# ──────────────────────────────────────────────────────────────
# Error response
# ──────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standardized error returned on validation or server failures."""
    error_code:    ErrorCode = Field(..., description="Machine-readable error code")
    message:       str       = Field(..., description="Human-readable description")
    detail:        str | None = Field(default=None,
                                       description="Optional extended detail for debugging")

    model_config = {"json_schema_extra": {
        "example": {
            "error_code": "INVALID_DTI",
            "message": "DTI ratio is out of accepted range",
            "detail": "Expected 0.0–5.0, got 6.1",
        }
    }}
