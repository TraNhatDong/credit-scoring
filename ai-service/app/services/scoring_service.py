"""
Credit Scoring Service — Champion LR + Challenger XGB Scoring.

Inference pipeline (mirrors _wrapper._predict_core exactly):
  1. Load artifact bundle per model (model + calibrator + preprocessor_state)
  2. Build SHAP Explainer (LinearExplainer for LR, callable-wrapped for XGB)
  3. For each scoring call:
     a. Preprocessor.transform()  — impute → engineer → clip
     b. Reindex to feature_order  — column order matches training
     c. Select IV-filtered columns for LR (or all 16 for XGB)
     d. dtype float32 (XGB) / float64 (LR)
     e. model.predict_proba() → raw prob
     f. calibrator.predict()   → calibrated prob
     g. NaN guard + range validation
     h. threshold → binary prediction
     i. FICO score mapping
  4. Ensemble: AUC-weighted average + majority voting
"""
import json as _json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import shap.maskers
from numpy.typing import NDArray

from app.core.config import settings
from app.core.preprocessor import Preprocessor
from app.models.schemas import (
    RiskLevel,
    ScoringRequest,
    ScoringResponse,
    ShapExplanation,
)

logger = logging.getLogger("credit-ai-service.scoring")

# Pipeline model keys — matched to actual artifact filenames saved by run_pipeline.py:
#   champion_xgb_model.{version}.pkl  → champion (best AUC)
#   benchmark_lr_model.{version}.pkl   → challenger (benchmark)
_MODEL_KEYS = [
    "champion_xgb",
    "benchmark_lr",
]


def _xgb_callable(model: Any, X: np.ndarray) -> np.ndarray:
    """Plain callable bypassing XGBoost >= 2.0 '[5E-1]' base_score parsing bug."""
    import xgboost
    if isinstance(X, xgboost.DMatrix):
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


class ScoringService:
    """
    Inference service for Champion LR + Challenger XGB.

    Each artifact bundle provides everything needed for a prediction:
      model, calibrator, preprocessor_state, feature_order, threshold.

    Mock predictions are used when model files are not found so the API
    remains testable in development.
    """

    def __init__(self, models_dir: str = "/app/models"):
        self._models_dir:    Path = Path(models_dir)
        self._models:       dict[str, dict[str, Any]] = {}
        self._preprocessors: dict[str, Any] = {}        # key → Preprocessor instance
        self._shap_exp:     dict[str, shap.Explainer] = {}
        self._metadata:     dict[str, Any] = {}
        self._loaded = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def load_models(self) -> dict[str, bool]:
        """
        Load all pipeline artifacts.

        Returns a dict of {model_name: loaded_successfully}.
        """
        if self._loaded:
            return {name: True for name in self._models}

        if not self._models_dir.is_dir():
            logger.warning(
                "Models directory not found: %s — using mock predictions",
                self._models_dir,
            )
            self._loaded = True
            return {}

        # Load pipeline metadata
        meta_path = self._models_dir / "pipeline_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self._metadata = _json.load(f)
                logger.info("Loaded pipeline metadata: version=%s",
                            self._metadata.get("pipeline_version"))
            except Exception as exc:
                logger.warning("Could not load pipeline_metadata.json: %s", exc)

        import glob as _glob

        load_status: dict[str, bool] = {}

        for key in _MODEL_KEYS:
            pattern = str(self._models_dir / f"{key}_model.*.pkl")
            matches = sorted(_glob.glob(pattern), reverse=True)
            if not matches:
                logger.warning("Model file not found: %s", pattern)
                load_status[key] = False
                continue

            path = Path(matches[0])
            try:
                data = joblib.load(path)
                self._models[key] = data
                load_status[key] = True
                logger.info("Loaded: %s from %s (type=%s, champion=%s)",
                            key, path.name, data.get("model_type", "?"),
                            data.get("champion", "?"))

                # Reconstruct per-model Preprocessor from bundled state
                preprocessor_state = data.get("preprocessor_state")
                if preprocessor_state is not None:
                    self._preprocessors[key] = Preprocessor.from_state(preprocessor_state)
                    logger.info("  Preprocessor ready for %s", key)
                else:
                    logger.warning("  No preprocessor_state in artifact for %s", key)

            except Exception as exc:
                logger.error("Failed to load %s: %s", path, exc)
                load_status[key] = False

        # Build SHAP explainers
        for key, data in self._models.items():
            model_type = data.get("model_type", "tree")
            try:
                self._shap_exp[key] = self._build_shap_explainer(
                    data["model"], model_type, data=data,
                )
                logger.info("SHAP explainer ready for: %s", key)
            except Exception as exc:
                logger.warning("Could not build SHAP explainer for %s: %s", key, exc)

        self._loaded = True
        return load_status

    # ── Single-model predict ───────────────────────────────────────────────────

    def predict(self, req: ScoringRequest) -> ScoringResponse:
        """
        Run the champion model (best by Test AUC) with SHAP explanations.

        Uses best_model_key from pipeline metadata, not hardcoded XGBoost.
        """
        t0 = time.perf_counter()

        # Map pipeline key (xgb/lr) → service key (champion_xgb/benchmark_lr)
        _KEY_MAP = {"xgb": "champion_xgb", "lr": "benchmark_lr"}

        raw_best_key = self._metadata.get("best_model_key", None)
        best_key = _KEY_MAP.get(raw_best_key) if raw_best_key else None

        if best_key is None or best_key not in self._shap_exp:
            best_key = max(
                (k for k in self._models if k in self._shap_exp),
                key=lambda k: self._models[k].get("metrics", {}).get("auc", 0),
                default=None,
            )

        if best_key and best_key in self._shap_exp:
            data = self._models[best_key]
            model_type = data.get("model_type", "tree")
            feature_order = data.get("feature_order", None)

            raw_prob, calibrated_prob = self._predict_core(req, data, model_type, feature_order, best_key)
            explainer = self._shap_exp[best_key]
            shap_exp, _ = self._call_shap_explainer(
                explainer, req, data, model_type, calibrated_prob=calibrated_prob,
            )
            risk_prob = calibrated_prob
            model_name = data.get("model_name", best_key)
        else:
            shap_exp, risk_prob = self._mock_explain(req)
            model_name = "mock"

        credit_score = self._prob_to_score(risk_prob, clamp_min=True)
        risk_level   = self._score_to_risk_level(credit_score)
        elapsed_ms   = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "Scored (%s): score=%d prob=%.4f risk=%s elapsed=%dms",
            model_name, credit_score, risk_prob, risk_level.value, elapsed_ms,
        )

        return ScoringResponse(
            application_id=req.application_id,
            credit_score=credit_score,
            risk_probability=round(risk_prob, 4),
            risk_level=risk_level,
            shap_explanations=[ShapExplanation(**e) for e in shap_exp],
            model_version=model_name,
            inference_ms=elapsed_ms,
        )

    # ── Multi-model predict ───────────────────────────────────────────────────

    def predict_multi(self, req: ScoringRequest) -> dict[str, Any]:
        """
        Run all loaded classification models on the same input.

        Returns:
            {
              "models": {
                <model_key>: {
                  "credit_score", "risk_probability", "risk_level",
                  "shap_explanations", "prediction", "probability", "metrics"
                }
              },
              "ensemble": {
                "credit_score", "risk_probability", "risk_level",
                "shap_explanations", "voting", "weighted_probability"
              },
              "pipeline_metadata": {...},
              "inference_ms": <int>
            }
        """
        t0 = time.perf_counter()

        if not self._models:
            logger.warning("No real models loaded — returning mock predictions")
            single = self.predict(req)
            return self._mock_multi_response(single, req)

        model_results: dict[str, Any] = {}
        probabilities: list[tuple[str, float, float]] = []   # (key, prob, auc)

        for key in _MODEL_KEYS:
            data = self._models.get(key)
            if not data:
                continue

            model_type    = data.get("model_type", "tree")
            feature_order = data.get("feature_order", None)
            metrics       = data.get("metrics", {})
            auc           = metrics.get("auc", 0.5)

            _, prob = self._predict_core(req, data, model_type, feature_order, key)

            threshold = data.get("threshold",
                                data.get("metrics", {}).get("final_threshold", 0.5))
            pred  = int(prob >= threshold)
            score = self._prob_to_score(prob, clamp_min=True)
            risk  = self._score_to_risk_level(score)

            probabilities.append((key, prob, auc))

            shap_exp, _ = self._get_shap_explanations(
                key, req, data, model_type, calibrated_prob=prob,
            )

            model_results[key] = {
                "credit_score":     score,
                "risk_probability": round(prob, 4),
                "risk_level":      risk.value,
                "shap_explanations": shap_exp,
                "prediction":      "GOOD" if pred == 0 else "DEFAULT",
                "probability":     round(prob, 4),
                "metrics":         metrics,
                "model_name":      data.get("model_name", key),
                "model_type":      model_type,
            }
            logger.info("  %s: score=%d prob=%.4f pred=%s AUC=%.4f",
                        key, score, prob, "GOOD" if pred == 0 else "DEFAULT", auc)

        ensemble = self._build_ensemble(model_results, probabilities, req)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "application_id": req.application_id,
            # Top-level fields — alias for the ensemble result (convenience for callers)
            "creditScore":     ensemble.get("credit_score"),
            "riskProbability": ensemble.get("risk_probability"),
            "riskLevel":       ensemble.get("risk_level"),
            "shapExplanations": ensemble.get("shap_explanations"),
            "models":             model_results,
            "ensemble":          ensemble,
            "pipeline_metadata": self._metadata,
            "inference_ms":      elapsed_ms,
        }

    def get_available_models(self) -> list[dict[str, Any]]:
        """Return metadata about loaded models."""
        return [
            {
                "name":       key,
                "model_type": data.get("model_type", "unknown"),
                "model_name": data.get("model_name", key),
                "metrics":    data.get("metrics", {}),
                "shap_ready": key in self._shap_exp,
            }
            for key, data in self._models.items()
        ]

    # ── Core inference (mirrors _wrapper._predict_core) ──────────────────────

    def _predict_core(
        self,
        req: ScoringRequest,
        data: dict[str, Any],
        model_type: str,
        feature_order: list[str] | None,
        model_key: str,
    ) -> tuple[float, float]:
        """
        Core inference — identical pipeline to _wrapper._predict_core.

        Steps:
          1. Preprocessor.transform()   — impute → engineer → clip
          2. Reindex to feature_order  — column order matches training
          3. dtype float32 (tree) / float64 (linear)
          4. Feature selection for LR (champion only)
          5. model.predict_proba()
          6. calibrator.predict()
          7. NaN guard + range validation
        """
        # ── Step 1: Full preprocessing via bundled Preprocessor ─────────────────
        preprocessor = self._preprocessors.get(model_key)

        if preprocessor is not None:
            req_data = req.model_dump(exclude={"application_id"})
            df = pd.DataFrame([req_data])
            # Snake_case (Pydantic) → PascalCase/hyphens (Preprocessor expects)
            df = df.rename(columns={
                "revolving_utilization_of_unsecured_lines": "RevolvingUtilizationOfUnsecuredLines",
                "number_of_time30_59days_past_due_not_worse": "NumberOfTime30-59DaysPastDueNotWorse",
                "debt_ratio":                               "DebtRatio",
                "monthly_income":                            "MonthlyIncome",
                "number_of_open_credit_lines_and_loans":     "NumberOfOpenCreditLinesAndLoans",
                "number_of_times90days_late":                "NumberOfTimes90DaysLate",
                "number_real_estate_loans_or_lines":        "NumberRealEstateLoansOrLines",
                "number_of_time60_89days_past_due_not_worse": "NumberOfTime60-89DaysPastDueNotWorse",
                "number_of_dependents":                      "NumberOfDependents",
            })
            df = preprocessor.transform(df)
        else:
            # Fallback: manual preprocessing (dev only, no preprocessor_state in artifact)
            df = self._manual_transform(req)

        # ── Step 2: Enforce column order (single source of truth: feature_order) ──
        if feature_order is not None:
            missing = [f for f in feature_order if f not in df.columns]
            if missing:
                raise ValueError(f"Missing engineered features after transform: {missing}")
            df = df.reindex(columns=feature_order, fill_value=0)

        # ── Step 3: dtype — float32 halves RAM for XGB; float64 for LR precision ─
        dtype = np.float32 if model_type == "tree" else np.float64

        # ── Step 4: Feature selection — LR (champion) uses scaled IV-filtered features ─
        champion = data.get("champion", False)
        scaler   = data.get("scaler", None)

        if champion and scaler is not None:
            X = df.values.astype(dtype)
            X = scaler.transform(X)
        else:
            X = df.values.astype(dtype)

        # ── Step 5: Raw probability ───────────────────────────────────────────────
        if not hasattr(data["model"], "predict_proba"):
            raise TypeError(
                f"Model {type(data['model']).__name__} does not support predict_proba()"
            )
        pd_raw = data["model"].predict_proba(X)[:, 1]

        # ── Step 6: Calibration ─────────────────────────────────────────────────
        calibrator = data.get("calibrator")
        if calibrator is not None:
            pd_calibrated = calibrator.predict(pd_raw)
        else:
            logger.warning("Calibrator is None — using raw probabilities (model: %s)", model_type)
            pd_calibrated = pd_raw

        pd_calibrated = float(np.asarray(pd_calibrated).item())

        # ── Step 6b: Calibration sanity check ──────────────────────────────────
        # Isotonic regression can overfit at tails (especially for class-imbalanced data
        # with ~6.7% default rate).  Detect pathological cases and blend with raw prob.
        #
        # Thresholds derived from the GMSC training data distribution:
        #   - Very low P (e.g. 0.001) is often trustworthy for "clean" borrowers
        #   - Very high P (> 0.5) is trustworthy for clearly risky borrowers
        #   - Mid-range P (0.01–0.5) where isotonic can swing wildly → check ratio
        raw_prob = float(pd_raw)
        CALIB_SANITY_MIN = 0.001   # P below this → skip blending (trust raw)
        CALIB_SANITY_MAX = 0.500   # P above this → skip blending (trust raw)
        CALIB_SANITY_RATIO = 10.0  # if calibrated/raw > 10x → suspicious isotonic overshoot

        if (CALIB_SANITY_MIN <= raw_prob <= CALIB_SANITY_MAX
                and pd_calibrated > 0
                and pd_calibrated / raw_prob > CALIB_SANITY_RATIO):
            logger.warning(
                "Suspicious isotonic calibration: raw=%.4f calibrated=%.4f (ratio=%.1fx) "
                "-- blending 70%% raw + 30%% calibrated",
                raw_prob, pd_calibrated, pd_calibrated / raw_prob,
            )
            pd_calibrated = 0.70 * raw_prob + 0.30 * pd_calibrated

        # ── Step 7: NaN guard + range validation ─────────────────────────────────
        if np.isnan(pd_calibrated):
            raise ValueError(
                f"NaN in calibrated prediction: pd_raw={float(pd_raw):.6f}"
            )
        if pd_calibrated < 0.0 or pd_calibrated > 1.0:
            raise ValueError(
                f"Calibrated PD out of range: {pd_calibrated:.6f} (raw={float(pd_raw):.6f})"
            )

        return float(pd_raw), pd_calibrated

    # ── Manual preprocessing (fallback only — no preprocessor_state in artifact) ─

    def _manual_transform(self, req: ScoringRequest) -> pd.DataFrame:
        """
        Fallback when artifact has no preprocessor_state.

        Applies impute → engineer → clip manually.
        This is NOT used in production — it only exists as a dev safety net.
        """
        # Start with snake_case columns (Pydantic alias names)
        d = req.model_dump(exclude={"application_id"})
        df = pd.DataFrame([d])

        # Rename snake_case (Pydantic) → PascalCase/hyphens (GMSC dataset names)
        # so that subsequent engineer/clip operations use the same names as
        # the real preprocessor trained on the original GMSC dataset.
        df = df.rename(columns={
            "revolving_utilization_of_unsecured_lines":  "RevolvingUtilizationOfUnsecuredLines",
            "number_of_time30_59days_past_due_not_worse": "NumberOfTime30-59DaysPastDueNotWorse",
            "debt_ratio":                               "DebtRatio",
            "monthly_income":                            "MonthlyIncome",
            "number_of_open_credit_lines_and_loans":     "NumberOfOpenCreditLinesAndLoans",
            "number_of_times90days_late":                "NumberOfTimes90DaysLate",
            "number_real_estate_loans_or_lines":         "NumberRealEstateLoansOrLines",
            "number_of_time60_89days_past_due_not_worse": "NumberOfTime60-89DaysPastDueNotWorse",
            "number_of_dependents":                      "NumberOfDependents",
        })

        # Impute
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

        # Engineer (before clip, income_missing = False since req requires gt=0)
        income_missing = pd.Series([False])
        df["monthly_debt"] = df["MonthlyIncome"] * df["DebtRatio"]
        df["late_payments_total"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )
        df["debt_per_person"] = df["monthly_debt"] / (df["NumberOfDependents"].clip(lower=0) + 1)
        df["income_per_credit_line"] = df["MonthlyIncome"] / (df["NumberOfOpenCreditLinesAndLoans"].clip(lower=0) + 1)
        df["utilization_per_line"] = df["RevolvingUtilizationOfUnsecuredLines"] / (df["NumberOfOpenCreditLinesAndLoans"].clip(lower=0) + 1)
        df["is_income_missing"] = 0

        # Clip
        df["age"] = df["age"].clip(lower=18, upper=100)
        df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=5)
        df["DebtRatio"] = df["DebtRatio"].clip(lower=0, upper=5)

        return df

    # ── SHAP helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _rescale_shap_to_prob(
        shap_values: np.ndarray,
        base_value: float,
        target_prob: float,
    ) -> tuple[float, np.ndarray]:
        """
        Rescale SHAP contributions so they sum to target_prob.

        SHAP values from an individual model explainer are relative to that
        model's base_value.  When we want SHAP explanations for an ensemble
        probability (which differs from any single model's output), we scale
        both base_value and the per-feature contributions proportionally so
        that base_value + sum(contributions) == target_prob.

        This keeps the relative importance of each feature unchanged while
        anchoring the absolute contribution to the correct probability.
        """
        contributions = shap_values
        total_contrib = float(contributions.sum())
        current_prob = base_value + total_contrib
        if abs(current_prob) < 1e-9:
            return base_value, contributions
        scale = (target_prob - base_value) / current_prob
        # Guard: only rescale if scale is reasonable (avoid pathological cases)
        if 0.2 <= scale <= 5.0:
            return base_value, contributions * scale
        return base_value, contributions

    def _build_shap_explainer(
        self,
        model: Any,
        model_type: str,
        data: dict[str, Any] | None = None,
    ) -> shap.Explainer:
        """Build a SHAP Explainer matching the model's feature dimensionality."""
        n_feat = self._n_artifact_features(data, model_type) if data else 16

        background = np.random.randn(50, n_feat).astype(np.float64)
        masker = shap.maskers.Independent(background)

        if model_type == "linear":
            return shap.LinearExplainer(model, masker)
        else:
            return shap.Explainer(
                lambda x: _xgb_callable(model, x),
                masker,
            )

    def _n_artifact_features(self, data: dict[str, Any] | None, model_type: str) -> int:
        """
        Number of features the artifact was trained on.

        LR: iv_features length from artifact (or pipeline metadata)
        XGB: all 16 engineered features
        """
        if model_type == "linear" and data is not None:
            iv = data.get("iv_features", None)
            if iv is not None:
                return len(iv)
            meta_iv = self._metadata.get("lr_features", None)
            if meta_iv is not None:
                return len(meta_iv)
            return self._metadata.get("n_lr_features", 12)
        return self._metadata.get("n_xgb_features", 16)

    def _get_X_for_model(
        self,
        req: ScoringRequest,
        data: dict[str, Any],
        model_type: str,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build the correctly-shaped feature array for a model's SHAP explainer.

        Returns (X, feature_names) matching exactly what the model was trained on.
        """
        # Build full 16-feature array via preprocessor
        preprocessor_key = next((k for k, d in self._models.items() if d is data), None)
        preprocessor = self._preprocessors.get(preprocessor_key)

        if preprocessor is not None:
            req_data = req.model_dump(exclude={"application_id"})
            df = pd.DataFrame([req_data])
            df = df.rename(columns={
                "revolving_utilization_of_unsecured_lines": "RevolvingUtilizationOfUnsecuredLines",
                "number_of_time30_59days_past_due_not_worse": "NumberOfTime30-59DaysPastDueNotWorse",
                "debt_ratio":                               "DebtRatio",
                "monthly_income":                            "MonthlyIncome",
                "number_of_open_credit_lines_and_loans":     "NumberOfOpenCreditLinesAndLoans",
                "number_of_times90days_late":                "NumberOfTimes90DaysLate",
                "number_real_estate_loans_or_lines":        "NumberRealEstateLoansOrLines",
                "number_of_time60_89days_past_due_not_worse": "NumberOfTime60-89DaysPastDueNotWorse",
                "number_of_dependents":                      "NumberOfDependents",
            })
            df = preprocessor.transform(df)
        else:
            df = self._manual_transform(req)

        # Reindex to feature_order
        feature_order = data.get("feature_order", None)
        if feature_order is not None:
            df = df.reindex(columns=feature_order, fill_value=0)

        dtype = np.float32 if model_type == "tree" else np.float64
        champion = data.get("champion", False)
        scaler   = data.get("scaler", None)

        if champion and scaler is not None:
            X_scaled = scaler.transform(df.values.astype(dtype))
            # SHAP for LR: need scaled values to match what LinearExplainer expects
            feature_names = list(df.columns)
            return X_scaled.astype(np.float64), feature_names
        else:
            return df.values.astype(dtype), list(df.columns)

    def _call_shap_explainer(
        self,
        explainer: shap.Explainer,
        req: ScoringRequest,
        data: dict[str, Any],
        model_type: str,
        calibrated_prob: float | None = None,
        target_prob: float | None = None,
    ) -> tuple[list[dict], float]:
        """
        Call SHAP Explainer and return structured explanations + probability.

        The feature array passed to the explainer must match the exact shape
        and column order the model was trained on (handled by _get_X_for_model).

        If target_prob is provided (e.g. ensemble probability) and differs from
        calibrated_prob (single-model probability), SHAP values are rescaled
        proportionally so base_value + sum(contrib) == target_prob.  This keeps
        feature importance relative while anchoring contributions to the correct
        absolute probability.

        Probability returned is always the calibrated (single-model or ensemble)
        probability, NOT reconstructed from sigmoid(base + sum(shap)).
        """
        X, feature_names = self._get_X_for_model(req, data, model_type)
        X = np.atleast_2d(X).astype(np.float64)

        n = self._n_artifact_features(data, model_type) if data else 16

        result = explainer(X)

        shap_values = result.values
        base_value = float(result.base_values[0]) if hasattr(result, "base_values") else 0.0

        # Handle SHAP output shape variations
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            shap_values = shap_values[0]  # Take first row for single prediction

        # Ensure 1D array
        shap_values = np.asarray(shap_values).flatten()

        prob = calibrated_prob if calibrated_prob is not None else (
            float(np.clip(1 / (1 + np.exp(-(base_value + shap_values.sum()))), 0.0, 1.0))
        )

        # Rescale SHAP contributions if explaining a different probability (e.g. ensemble)
        if target_prob is not None and target_prob != prob:
            base_value, shap_values = self._rescale_shap_to_prob(shap_values, base_value, target_prob)

        # Slice to the model's feature dimensionality
        n_features = min(n, len(shap_values), len(feature_names))
        explanations = self._build_explanations(
            shap_values[:n_features],
            X[0][:n_features],
            feature_names[:n_features],
        )
        return explanations, prob

    def _get_shap_explanations(
        self,
        model_key: str,
        req: ScoringRequest,
        data: dict[str, Any],
        model_type: str,
        calibrated_prob: float | None = None,
        target_prob: float | None = None,
    ) -> tuple[list[dict], float]:
        """Get SHAP explanations for a specific model, with mock fallback."""
        if model_key in self._shap_exp:
            return self._call_shap_explainer(
                self._shap_exp[model_key], req, data, model_type,
                calibrated_prob=calibrated_prob,
                target_prob=target_prob,
            )
        return self._mock_explain(req)

    def _build_explanations(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: list[str] | None = None,
        neutral_threshold: float = 0.05,
    ) -> list[dict]:
        """
        Convert raw SHAP values into sorted explanation dicts.

        Direction convention (matches SHAP standard — class=1 = default):
          - POSITIVE contribution → pushes probability of default UP → red badge (higher risk)
          - NEGATIVE contribution → pushes probability of default DOWN → green badge (lower risk)

        neutral_threshold=0.05: features with |contribution| < 0.05 are marked NEUTRAL
        (XGBoost SHAP values are typically in [-1.0, +1.0]; 0.05 avoids over-colouring minor features).
        """
        contributions = []
        for i, feat_name in enumerate(feature_names or []):
            raw_val = shap_values[i] if i < len(shap_values) else 0.0
            # Handle multi-dimensional arrays from some SHAP explainers
            while isinstance(raw_val, np.ndarray) and raw_val.size > 1:
                raw_val = raw_val.flat[0] if raw_val.ndim > 1 else raw_val[0]
            contribution = float(raw_val)
            abs_c = abs(contribution)
            if abs_c < neutral_threshold:
                direction = "NEUTRAL"
            elif contribution > 0:
                direction = "POSITIVE"
            else:
                direction = "NEGATIVE"

            contributions.append({
                "feature":      feat_name,
                "value":        float(feature_values[i]),
                "contribution": round(contribution, 4),
                "direction":    direction,
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contributions

    # ── Ensemble ───────────────────────────────────────────────────────────

    def _build_ensemble(
        self,
        model_results: dict[str, Any],
        probabilities: list[tuple[str, float, float]],
        req: ScoringRequest,
    ) -> dict[str, Any]:
        """Build ensemble using AUC-weighted average of calibrated probabilities."""
        if not probabilities:
            return self._mock_ensemble(req)

        total_auc = sum(auc for _, _, auc in probabilities)
        if total_auc > 0:
            weighted_prob = sum(prob * auc / total_auc for _, prob, auc in probabilities)
        else:
            weighted_prob = sum(p for _, p, _ in probabilities) / len(probabilities)

        votes = {"approved": 0, "rejected": 0}
        for result in model_results.values():
            if result["prediction"] == "GOOD":
                votes["approved"] += 1
            else:
                votes["rejected"] += 1

        ensemble_score = self._prob_to_score(weighted_prob, clamp_min=True)
        ensemble_risk  = self._score_to_risk_level(ensemble_score)

        # SHAP from best model (highest AUC) — rescale to ensemble probability
        best_key = max(probabilities, key=lambda x: x[2])[0]
        best_prob = next(p for k, p, _ in probabilities if k == best_key)
        data = self._models[best_key]
        model_type = data.get("model_type", "tree")
        best_shap, _ = self._get_shap_explanations(
            best_key, req, data, model_type,
            calibrated_prob=best_prob,
            target_prob=weighted_prob,
        )

        return {
            "credit_score":         ensemble_score,
            "risk_probability":     round(weighted_prob, 4),
            "risk_level":          ensemble_risk.value,
            "shap_explanations":   best_shap,
            "voting":              votes,
            "weighted_probability": round(weighted_prob, 4),
            "best_model_key":      best_key,
        }

    # ── Mock helpers ───────────────────────────────────────────────────────

    def _mock_explain(
        self,
        req: ScoringRequest,
    ) -> tuple[list[dict], float]:
        """
        Deterministic mock SHAP explanations when no model is loaded.

        Uses business-logic-based contributions so the mock is visually meaningful:
          POSITIVE contribution → increases default risk (red)
          NEGATIVE contribution → decreases default risk (green)
        """
        d = req.model_dump(exclude={"application_id"})
        explanations = []

        # Approximate log-odds contribution per unit of each feature
        # (sign conventions match the real trained models: class=1 = default)
        contrib_map = {
            "revolving_utilization_of_unsecured_lines":  8.0,   # higher utilization → risk
            "number_of_time30_59days_past_due_not_worse": 5.0,  # any late payment → risk
            "number_of_time60_89days_past_due_not_worse": 6.0,  # worse late payment → risk
            "number_of_times90days_late":              10.0,  # 90+ days → most severe
            "debt_ratio":                               3.0,   # higher DTI → risk
            "age":                                      -0.2,  # older → slightly lower risk
            "monthly_income":                           -1e-4, # higher income → lower risk (per VND)
            "number_of_open_credit_lines_and_loans":     0.3,  # more lines → slight risk
            "number_real_estate_loans_or_lines":       -0.2,  # real estate → lower risk
            "number_of_dependents":                     0.5,   # more dependents → slight risk
        }

        total_contrib = 0.0
        for name, val in d.items():
            v = float(val)
            contrib = contrib_map.get(name, 0.0) * v
            total_contrib += contrib
            direction = "POSITIVE" if contrib > 0.05 else "NEGATIVE" if contrib < -0.05 else "NEUTRAL"
            explanations.append({
                "feature":      name,
                "value":        round(v, 4),
                "contribution": round(contrib, 4),
                "direction":    direction,
            })

        explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # Approximate calibrated probability from total log-odds contribution
        log_odds = total_contrib
        prob = 1 / (1 + np.exp(-log_odds))
        prob = float(np.clip(prob, 0.001, 0.999))
        return explanations, round(prob, 4)

    def _mock_multi_response(
        self,
        single: ScoringResponse,
        req: ScoringRequest,
    ) -> dict[str, Any]:
        """Build a full mock multi-model response with correct key names."""
        shap_exp = [e.model_dump() for e in single.shap_explanations]

        mock_models = {
            "champion_xgb": {
                "credit_score":       single.credit_score,
                "risk_probability":   single.risk_probability,
                "risk_level":         single.risk_level.value,
                "shap_explanations":  shap_exp,
                "prediction":         "DEFAULT" if single.risk_probability > 0.5 else "GOOD",
                "probability":        single.risk_probability,
                "metrics":            {},
                "model_name":         "XGBoost (Champion)",
                "model_type":         "tree",
            },
            "benchmark_lr": {
                "credit_score":       single.credit_score,
                "risk_probability":   single.risk_probability,
                "risk_level":         single.risk_level.value,
                "shap_explanations":  shap_exp,
                "prediction":         "DEFAULT" if single.risk_probability > 0.5 else "GOOD",
                "probability":        single.risk_probability,
                "metrics":            {},
                "model_name":         "Logistic Regression",
                "model_type":         "linear",
            },
        }

        return {
            "application_id": req.application_id,
            "models":             mock_models,
            "ensemble": {
                "credit_score":         single.credit_score,
                "risk_probability":     single.risk_probability,
                "risk_level":           single.risk_level.value,
                "shap_explanations":   shap_exp,
                "voting":              {"approved": 1, "rejected": 0}
                    if single.risk_probability > 0.5
                    else {"approved": 0, "rejected": 1},
                "weighted_probability": single.risk_probability,
                "best_model_key":      "champion_xgb",
            },
            "pipeline_metadata": {},
            "inference_ms":      single.inference_ms,
        }

    def _mock_ensemble(self, req: ScoringRequest) -> dict[str, Any]:
        """Fallback ensemble when no models are loaded."""
        shap_exp, prob = self._mock_explain(req)
        score = self._prob_to_score(prob, clamp_min=True)
        risk  = self._prob_to_risk_level(prob)
        mock_pred = "DEFAULT" if prob > 0.5 else "GOOD"
        return {
            "credit_score":         score,
            "risk_probability":     round(prob, 4),
            "risk_level":          risk.value,
            "shap_explanations":   shap_exp,
            "voting":              {"approved": 1, "rejected": 0}
                if mock_pred == "GOOD" else {"approved": 0, "rejected": 1},
            "weighted_probability": round(prob, 4),
            "best_model_key":      None,
        }

    # ── Scoring helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _prob_to_score(prob: float, clamp_min: bool = False) -> int:
        """
        Map P(default) → FICO-like credit score (300–850).

        Formula identical to pipeline's pd_to_credit_score():
          PD =  0.8% → Score = 850 (Excellent — v4.1 tightened anchor)
          PD = 50.0% → Score = 300 (Very Poor)

        Args:
            prob:      probability of default in [0, 1]
            clamp_min: if True, score is always >= 300 (even for extreme probabilities);
                       if False, score can go below 300 (e.g. P=0.999 → score ≈ 285).
                       Use clamp_min=True for production-facing scores.
        """
        TARGET_MIN_PD = 0.008  # 0.8% — matches pipeline's pd_to_credit_score() anchor
        TARGET_MAX_PD = 0.50   # 50%   — matches pipeline's pd_to_credit_score() anchor
        MIN_SCORE = 300
        MAX_SCORE = 850

        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))

        lo_max = np.log((1 - TARGET_MIN_PD) / TARGET_MIN_PD)
        lo_min = np.log((1 - TARGET_MAX_PD) / TARGET_MAX_PD)
        log_odds = np.log((1 - prob) / prob)

        grade_range = MAX_SCORE - MIN_SCORE
        log_range   = lo_max - lo_min
        score = MAX_SCORE - grade_range * (log_odds - lo_max) / log_range

        raw = round(score)
        if clamp_min:
            raw = max(raw, MIN_SCORE)
        return int(np.clip(raw, MIN_SCORE, MAX_SCORE))

    @staticmethod
    def _prob_to_risk_level(prob: float) -> RiskLevel:
        """Map P(default) → categorical risk bucket."""
        if   prob < 0.05: return RiskLevel.LOW
        elif prob < 0.15: return RiskLevel.MEDIUM
        elif prob < 0.35: return RiskLevel.HIGH
        else:             return RiskLevel.CRITICAL

    @staticmethod
    def _score_to_risk_level(score: int) -> RiskLevel:
        """
        Map credit score → categorical risk bucket.

        Uses the FICO-like score scale (300–850) which is the Basel II standard.
        Thresholds are calibrated to the PD→score anchor used by _prob_to_score:
          score >= 700 → LOW      (PD <= 5.4%)
          score >= 600 → MEDIUM    (PD <= 15.6%)
          score >= 500 → HIGH      (PD <= 29.3%)
          score <  500 → CRITICAL (PD > 29.3%)

        NOTE: riskLevel is derived from score (not probability) to guarantee
        consistency — score and riskLevel always agree by construction.
        """
        if   score >= 700: return RiskLevel.LOW
        elif score >= 600: return RiskLevel.MEDIUM
        elif score >= 500: return RiskLevel.HIGH
        else:              return RiskLevel.CRITICAL
