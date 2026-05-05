"""
Preprocessor — stateful, deterministic preprocessing for credit scoring.

Single source of truth for both training scripts and the inference service.
DO NOT modify independently — any change must apply to both.

Pipeline: engineer → impute → clip (all deterministic, no random)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class Preprocessor:
    """
    Stateful, deterministic preprocessor for credit scoring features.

    fit()       — captures statistics from training data only (no leakage)
    transform() — applies impute → engineer → clip using fitted state
    """

    def __init__(self):
        self.median_values: dict[str, float] = {}
        self.p99_debt_ratio: float = 0.0
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        """
        Compute and store fitted state from training data only.

        Captures:
          - median for every numeric column (used by SimpleImputer)
          - p99 of DebtRatio where income > 0 (used for domain-aware clipping)
          - feature column names
        """
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        self.median_values = X_train[numeric_cols].median().to_dict()

        debt_income_valid = X_train.loc[
            ~X_train["MonthlyIncome"].isna() & (X_train["MonthlyIncome"] > 0),
            "DebtRatio",
        ]
        p99 = debt_income_valid.quantile(0.99)
        self.p99_debt_ratio = float(p99) if not np.isnan(p99) else 5.0

        self.feature_names = X_train.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to input DataFrame.

        Pipeline:
          1. Enforce fitted column order
          2. Coerce every column to numeric (silent strings → NaN)
          3. Guard: must be fitted before use
          4. Capture income-missing flag BEFORE impute
          5. Impute original features
          6. Engineer row-wise features
          7. Clip outliers
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = X[self.feature_names].copy()

        X = X.apply(pd.to_numeric, errors="coerce")

        if not self.median_values:
            raise RuntimeError(
                "Preprocessor must be fitted before transform(). "
                "Call preprocessor.fit(X_train) first."
            )

        income_missing = X["MonthlyIncome"].isna()

        self._impute_all(X)
        self._engineer(X, income_missing)
        self._clip(X, income_missing)

        return X

    def _engineer(self, X: pd.DataFrame, income_missing: pd.Series) -> None:
        """
        Row-wise feature engineering.

        is_income_missing must use the flag captured BEFORE imputation,
        otherwise it is always zero because MonthlyIncome was already filled.

        Denominators are clipped to >= 0 before adding 1 to prevent
        division by zero or negative denominators from dirty data.
        """
        X["monthly_debt"] = X["MonthlyIncome"] * X["DebtRatio"]
        X["late_payments_total"] = (
            X["NumberOfTime30-59DaysPastDueNotWorse"]
            + X["NumberOfTime60-89DaysPastDueNotWorse"]
            + X["NumberOfTimes90DaysLate"]
        )
        dependents_denom  = X["NumberOfDependents"].clip(lower=0) + 1
        credit_lines_denom = X["NumberOfOpenCreditLinesAndLoans"].clip(lower=0) + 1
        X["debt_per_person"]        = X["monthly_debt"] / dependents_denom
        X["income_per_credit_line"] = X["MonthlyIncome"] / credit_lines_denom
        X["utilization_per_line"]   = X["RevolvingUtilizationOfUnsecuredLines"] / credit_lines_denom
        X["is_income_missing"]      = income_missing.astype(int)

    def _impute_all(self, X: pd.DataFrame) -> None:
        """
        Impute original features BEFORE engineering.

        NumberOfDependents → 0; all other numeric columns → median.

        Special values 96/98 in late-payment columns are sentinel codes
        (observed in the GiveMeSomeCredit dataset — likely data-entry errors
        or system placeholders). They appear simultaneously in all three
        columns (264 rows) and have a bad_rate of 54% — the risk signal is
        real but the value 96/98 is erroneous. Clipping to 15 preserves
        the high-risk nature without distorting the scale.
        """
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(0)

        # Clip sentinel codes 96/98 to a reasonable maximum (15).
        # The 264 sentinel rows have bad_rate=54%, indicating genuine high risk.
        # Replacing with 0 (imputation) would erase the risk signal entirely.
        # Clipping to 15 keeps the extreme-but-valid signal.
        _SENTINEL_MAX = 15
        for col in (
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ):
            if col in X.columns:
                X.loc[X[col].isin({96, 98}), col] = float(_SENTINEL_MAX)

        for col, median_val in self.median_values.items():
            if col != "NumberOfDependents":
                X[col] = X[col].fillna(median_val)

    def _clip(self, X: pd.DataFrame, income_missing: pd.Series) -> None:
        """
        Clip extreme values using fitted state.

        Domain logic for DebtRatio:
          - income_valid (income present AND > 0) → clip DebtRatio to (0, 5)
          - income_missing → clip DebtRatio to (0, p99) [absolute debt]

        The income_missing flag must be captured BEFORE imputation.
        """
        X["age"] = X["age"].clip(lower=18, upper=100)
        X["RevolvingUtilizationOfUnsecuredLines"] = (
            X["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=5)
        )
        income_valid = ~income_missing & (X["MonthlyIncome"] > 0)
        X.loc[income_valid, "DebtRatio"] = (
            X.loc[income_valid, "DebtRatio"].clip(lower=0, upper=5)
        )
        X.loc[~income_valid, "DebtRatio"] = (
            X.loc[~income_valid, "DebtRatio"].clip(lower=0, upper=self.p99_debt_ratio)
        )

    def get_state(self) -> dict[str, Any]:
        """Return serializable state dict for artifact bundling."""
        return {
            "median_values":   self.median_values,
            "p99_debt_ratio": self.p99_debt_ratio,
            "feature_names":   self.feature_names,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Preprocessor":
        """Reconstruct Preprocessor from serialized state."""
        inst = cls()
        inst.median_values   = state["median_values"]
        inst.p99_debt_ratio = state["p99_debt_ratio"]
        inst.feature_names   = state["feature_names"]
        return inst
