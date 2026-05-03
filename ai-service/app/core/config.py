"""
Core configuration loaded from environment variables.
Uses Pydantic Settings for type-safe validation.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # ML Model
    MODEL_PATH: str = "/app/models/"
    MODEL_VERSION: str = "1.0.0"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Feature columns — must match GMSC dataset column names exactly
    FEATURE_COLUMNS: list[str] = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]

    # Score thresholds
    SCORE_MIN: int = 300
    SCORE_MAX: int = 850
    RISK_PROB_THRESHOLD: float = 0.5

    @property
    def base_url(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
