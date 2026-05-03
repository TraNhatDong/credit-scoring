"""
Global exception handlers for the Credit Scoring AI Service.

Ensures every non-2xx response has a consistent JSON structure:
  { "error_code": "...", "message": "...", "detail": "..." }
"""
import logging
import sys
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models.schemas import ErrorCode, ErrorResponse

logger = logging.getLogger("credit-ai-service.exceptions")


# ──────────────────────────────────────────────────────────────
# Register exception handlers with FastAPI
# ──────────────────────────────────────────────────────────────
def register_handlers(app: FastAPI) -> None:

    # ── Pydantic / FastAPI validation errors ──────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = exc.errors()
        logger.warning("Validation error on %s %s: %s", request.method, request.url, errors)

        # Map first error to a meaningful error_code
        first = errors[0] if errors else {}
        loc   = ".".join(str(l) for l in (first.get("loc") or []))
        msg   = first.get("msg", "Validation failed")

        code_map: dict[str, ErrorCode] = {
            "age":         ErrorCode.INVALID_AGE,
            "MonthlyIncome": ErrorCode.INVALID_INCOME,
            "DebtRatio":   ErrorCode.INVALID_DTI,
        }
        error_code = next(
            (code for key, code in code_map.items() if key in loc),
            ErrorCode.INTERNAL_ERROR,
        )

        body = ErrorResponse(
            error_code=error_code,
            message=msg,
            detail=f"Field '{loc}': {msg} — {first.get('input', 'N/A')}",
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body.model_dump(),
        )

    # ── Generic Python exceptions ──────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        body = ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An internal server error occurred",
            detail=str(exc) if not isinstance(exc, AssertionError) else None,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(exclude_none=True),
        )
