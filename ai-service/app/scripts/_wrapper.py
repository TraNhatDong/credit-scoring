"""
_wrapper.py — CreditScoringModel production inference wrapper.

Dependency: _config.py + _transform.py only.
Does NOT import _train.py or _monitor.py.

Wrapper is strictly read-only: transform → predict → calibrate.
No training logic, no fallback hacks, no branching complexity.

Public API (3 methods only):
  - CreditScoringModel.load(path) → instance
  - model.predict(features) → dict  (entry point)
  - model._predict_core(X) → dict   (private, ~20 lines)
"""
from __future__ import annotations

import joblib
from typing import Any

import numpy as np
import pandas as pd

from ._config import (
    log, pd_to_credit_score, RiskBands,
    RAW_FEATURE_COLUMNS,
)
from ._transform import Preprocessor


class CreditScoringModel:
    """
    Production inference wrapper for credit scoring models.

    Usage:
        model = CreditScoringModel.load("models/champion_lr_model.v4.1_YYYYMMDD.pkl")
        result = model.predict(input_df)

    Artifact structure (single-file bundle):
        {
            "model":       XGBClassifier / LogisticRegression,
            "calibrator":  IsotonicRegression / _PlattScaler,
            "preprocessor": Preprocessor (fitted, bundled),
            "feature_names": [...],
            "champion":    bool,
            "model_version": str,
            ...
        }
    """

    REQUIRED_FEATURES = RAW_FEATURE_COLUMNS  # single source of truth from _config.py

    def __init__(self, artifact: dict):
        """
        Construct from a single artifact bundle.

        All state is extracted from the bundle — no separate files needed.
        """
        self.model        = artifact["model"]
        self.calibrator   = artifact["calibrator"]
        self.preprocessor = Preprocessor.from_state(artifact["preprocessor_state"])
        self.feature_names = artifact["feature_names"]
        self.feature_order = artifact.get("feature_order", self.feature_names)
        self.champion     = artifact.get("champion", False)
        self.model_type   = artifact.get("model_type", "tree")
        self.model_version = artifact.get("model_version", "unknown")
        self.scaler       = artifact.get("scaler", None)
        self.threshold    = artifact.get("threshold", 0.5)
        # Training scores for quantile banding — fitted RiskBands are deterministic
        # from the training distribution; no leakage risk since scores are predictions.
        self._scores_train = artifact.get("scores_train", None)
        self._risk_bands: "RiskBands | None" = None

    def get_risk_bands(self) -> "RiskBands":
        """
        Lazily fit quantile-based RiskBands from training score distribution.

        Band boundaries are fixed at fit time and never change during inference.
        Single authoritative banding system: no PD thresholds, no legacy PD-fixed bands.
        """
        if self._risk_bands is None:
            if self._scores_train is not None and len(self._scores_train) > 0:
                self._risk_bands = RiskBands.fit(
                    np.asarray(self._scores_train, dtype=float), n_bands=7
                )
            else:
                log.warning(
                    "scores_train not in artifact — cannot fit quantile RiskBands. "
                    "Falling back to score in [300, 850]."
                )
                self._risk_bands = None
        return self._risk_bands

    @classmethod
    def load(cls, model_path: str) -> "CreditScoringModel":
        """
        Load model from a single artifact bundle file.

        Args:
            model_path: Path to champion_lr or challenger_xgb model pickle.
        """
        log.info("Loading CreditScoringModel from: %s", model_path)
        artifact = joblib.load(model_path)
        instance = cls(artifact)
        log.info(
            "  Loaded: %s | champion=%s | version=%s",
            artifact.get("model_type", "?"),
            artifact.get("champion", "?"),
            instance.model_version,
        )
        return instance

    # ── Layer 1: input schema validation ───────────────────────────────────────
    def _validate_schema(self, X: pd.DataFrame) -> list[str]:
        """
        Layer 1 validation — returns list of error messages (empty = valid).
        Raises on missing columns or empty DataFrame.
        """
        errors: list[str] = []
        missing = [f for f in self.REQUIRED_FEATURES if f not in X.columns]
        if missing:
            errors.append(f"Missing features: {missing}")
        if len(X) == 0:
            errors.append("Empty DataFrame")
        if "age" in X.columns:
            age = pd.to_numeric(X["age"], errors="coerce")
            if ((age < 0) | age.isna()).any():
                errors.append("Invalid/negative age values")
        return errors

    # ── Layer 2: output sanity check ────────────────────────────────────────────
    def _check_output_range(self, pd_calibrated) -> None:
        """
        Layer 2 validation — raise if calibrated PD is out of [0, 1] range.

        Accepts np.ndarray or list (list is returned after .tolist() in predict()).
        """
        arr = np.asarray(pd_calibrated)
        if np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(
                f"Calibrated PD out of range: min={arr.min():.6f}, max={arr.max():.6f}"
            )

    # ── Core inference ─────────────────────────────────────────────────────────
    def _predict_core(self, X: pd.DataFrame) -> dict[str, Any]:
        """
        Core inference — no validation (assumes valid input).

        Pipeline:
          1. Preprocessor.transform() → engineer + impute + clip
          2. Guard missing engineered features + explicit column ordering (single source: feature_order)
          3. Dtype optimisation (float32 for tree, float64 for linear)
          4. model.predict_proba() with capability check
          5. Safe calibration (guard for None)
          6. NaN check
          7. Threshold application
          8. Credit score conversion
        """
        n_samples = len(X)

        # Step 1: Full preprocessing via bundled Preprocessor
        X_proc = self.preprocessor.transform(X)

        # Step 2: Guard missing engineered features, then enforce column order
        # feature_order is the single source of truth — no double-reorder risk
        if self.feature_order is not None:
            missing = [f for f in self.feature_order if f not in X_proc.columns]
            if missing:
                raise ValueError(f"Missing engineered features after transform: {missing}")
            X_proc = X_proc.reindex(columns=self.feature_order, fill_value=0)

        # Step 3: Dtype — float32 halves RAM for XGB; float64 for LR precision
        dtype = np.float32 if self.model_type == "tree" else np.float64

        # Step 4: Feature selection (scaler for champion LR)
        if self.champion and self.scaler is not None:
            X_lr = X_proc.values.astype(dtype)
            X_lr = self.scaler.transform(X_lr)
            features = X_lr
        else:
            features = X_proc.values.astype(dtype)

        # Step 5: Raw model probabilities (capability check)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                f"Model {type(self.model).__name__} does not support predict_proba()"
            )
        pd_raw = self.model.predict_proba(features)[:, 1]

        # Step 6: Calibration with None guard + warning
        if self.calibrator is not None:
            pd_calibrated = self.calibrator.predict(pd_raw)
        else:
            log.warning("Calibrator is None — using raw probabilities (model: %s)", self.model_type)
            pd_calibrated = pd_raw

        # Step 7: NaN guard
        if np.isnan(pd_calibrated).any():
            raise ValueError(
                f"NaN in calibrated prediction output: "
                f"pd_raw min={pd_raw.min():.6f} max={pd_raw.max():.6f}"
            )

        # Step 8: Threshold and credit score
        prediction = (pd_calibrated >= self.threshold).astype(int)
        scores = pd_to_credit_score(pd_calibrated, clamp_min=True)

        bands = []
        risk_bands = self.get_risk_bands()
        for s in scores:
            if risk_bands is not None:
                bands.append(risk_bands.get_band(int(s)))
            else:
                # Final safety fallback: score clipped to [300, 850] by pd_to_credit_score
                bands.append(str(int(s)))

        log.debug(
            "Inference: model=%s | n_samples=%d | mean_pd_raw=%.4f | mean_pd_cal=%.4f | "
            "mean_score=%.1f | pos_rate=%.2f",
            self.model_type,
            n_samples,
            float(pd_raw.mean()),
            float(pd_calibrated.mean()),
            float(scores.mean()),
            float(prediction.mean()),
        )

        return {
            "pd_raw":          pd_raw.tolist(),          # API-friendly list
            "pd_calibrated":   pd_calibrated.tolist(),   # API-friendly list
            "prediction":       prediction.tolist(),      # 0/1 per row
            "credit_score":    scores.tolist(),           # API-friendly list
            "risk_band":       bands,
            "model_version":   self.model_version,
            "threshold":       self.threshold,
            "errors":          [],
        }

    # ── Public entry point ─────────────────────────────────────────────────────
    def predict(self, features: pd.DataFrame | dict) -> dict[str, Any]:
        """
        Predict on raw input.

        Two-layer validation:
          Layer 1: input schema (missing columns, empty DataFrame, invalid age)
          Layer 2: output sanity check (calibrated PD in [0, 1])

        Args:
            features: DataFrame with REQUIRED_FEATURES columns, or dict of {col: value}

        Returns:
            {
                "pd_raw":          raw model probabilities (list),
                "pd_calibrated":   calibrated probabilities (list),
                "prediction":       binary decisions 0/1 using final_threshold (list),
                "credit_score":    consumer credit scores 300-850 (list),
                "risk_band":       risk band per row (list),
                "model_version":   model version tag,
                "threshold":       decision threshold used,
                "errors":          validation errors (empty list = valid),
            }
        """
        # Accept dict input
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Layer 1: input schema validation
        schema_errors = self._validate_schema(features)
        if schema_errors:
            return {
                "pd_raw": None, "pd_calibrated": None,
                "prediction": None,
                "credit_score": None, "risk_band": None,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "errors": schema_errors,
            }
        X = features[list(self.REQUIRED_FEATURES)]

        # Core inference
        result = self._predict_core(X)

        # Layer 2: output sanity check — raise on out-of-range calibrated PD
        self._check_output_range(result["pd_calibrated"])

        return result
