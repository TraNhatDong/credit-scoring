"""
_transform.py — Preprocessor class (STATEFUL, deterministic).

Implements: engineer → impute → clip
Fitted state is captured during fit() and reused during transform().

Used by:
  - _train.py: fit on train, transform all 3 sets
  - _wrapper.py: loaded via Preprocessor.from_state() for production inference
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

    All operations are deterministic: no random sampling, no runtime config.
    """

    def __init__(self):
        self.median_values: dict[str, float] = {}
        self.p99_debt_ratio: float = 0.0
        self.feature_names: list[str] = []

    # ── fit ────────────────────────────────────────────────────────────────────
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
        # Guard: if all income is missing/zero, debt_income_valid is empty → quantile returns NaN.
        # Fall back to the generic DebtRatio clip ceiling (5.0) used everywhere else.
        self.p99_debt_ratio = float(p99) if not np.isnan(p99) else 5.0

        self.feature_names = X_train.columns.tolist()
        return self

    # ── transform ───────────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to input DataFrame.

        All steps use fitted state from fit() — no new statistics computed.

        Pipeline:
          1. Enforce fitted column order (guards against caller-side reordering)
          2. Coerce every column to numeric (silent strings / bad dtypes → NaN)
          3. Guard: must be fitted before use
          4. Guard: reject missing required columns
          5. Capture income-missing flag BEFORE impute
          6. Impute original features
          7. Engineer row-wise features (safe — all values present)
          8. Clip outliers (age, RevolvingUtil, DebtRatio)
        """
        # ── Step 1: enforce column order ──────────────────────────────────────
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = X[self.feature_names].copy()

        # ── Step 2: coerce to numeric ───────────────────────────────────────────
        X = X.apply(pd.to_numeric, errors="coerce")

        # ── Step 3: fitted-state guard ────────────────────────────────────────
        if not self.median_values:
            raise RuntimeError(
                "Preprocessor must be fitted before transform(). "
                "Call preprocessor.fit(X_train) first."
            )

        # ── Step 4 (captured): income-missing flag before imputation ───────────
        # This flag drives both is_income_missing and the DebtRatio clip branch.
        income_missing = X["MonthlyIncome"].isna()

        # ── Step 5: Impute original features BEFORE engineering ───────────────
        self._impute_all(X)

        # ── Step 6: Engineer row-wise features ────────────────────────────────
        self._engineer(X, income_missing)

        # ── Step 7: Clip outliers ─────────────────────────────────────────────
        self._clip(X, income_missing)

        return X

    # ── private helpers ─────────────────────────────────────────────────────────
    def _engineer(self, X: pd.DataFrame, income_missing: pd.Series) -> None:
        """
        Row-wise feature engineering — applied identically to every set.

        is_income_missing must use the flag captured before imputation,
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
        # clip denominators to >= 0 before adding 1
        dependents_denom = X["NumberOfDependents"].clip(lower=0) + 1
        credit_lines_denom = X["NumberOfOpenCreditLinesAndLoans"].clip(lower=0) + 1
        X["debt_per_person"]        = X["monthly_debt"] / dependents_denom
        X["income_per_credit_line"] = X["MonthlyIncome"] / credit_lines_denom
        X["utilization_per_line"]   = X["RevolvingUtilizationOfUnsecuredLines"] / credit_lines_denom
        X["is_income_missing"]      = income_missing.astype(int)

    def _impute_all(self, X: pd.DataFrame) -> None:
        """
        Impute original features BEFORE engineering.

        At this point engineered features do not yet exist; only raw columns
        are present. NumberOfDependents → 0; all other numeric columns → median.
        """
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(0)
        for col, median_val in self.median_values.items():
            if col != "NumberOfDependents":
                X[col] = X[col].fillna(median_val)

    def _clip(self, X: pd.DataFrame, income_missing: pd.Series) -> None:
        """
        Clip extreme values using fitted state.

        Domain logic for DebtRatio:
          - income_valid (income present AND > 0) → clip DebtRatio to (0, 5)
          - income_missing → clip DebtRatio to (0, p99) [absolute debt]

        The income_missing flag must be captured BEFORE imputation; after
        imputation all NaN values are filled, making the flag always False.
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

    # ── serialization helpers ──────────────────────────────────────────────────
    def get_state(self) -> dict[str, Any]:
        """
        Return serializable state dict for artifact bundling.

        Prefer this over pickling the whole object — it is stable across
        code-version changes and explicit about what is persisted.
        """
        return {
            "median_values":    self.median_values,
            "p99_debt_ratio":   self.p99_debt_ratio,
            "feature_names":    self.feature_names,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Preprocessor":
        """Reconstruct Preprocessor from serialized state."""
        inst = cls()
        inst.median_values   = state["median_values"]
        inst.p99_debt_ratio  = state["p99_debt_ratio"]
        inst.feature_names   = state["feature_names"]
        return inst
