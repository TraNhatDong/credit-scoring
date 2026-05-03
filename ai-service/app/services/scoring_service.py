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

# Pipeline model keys (must match run_pipeline.py step 6 naming)
_MODEL_KEYS = [
    "champion_lr",
    "challenger_xgb",
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
        self._preprocessors: dict[str, Preprocessor] = {}
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

        # Map pipeline key (lr/xgb) → service key (champion_lr/challenger_xgb)
        _KEY_MAP = {"lr": "champion_lr", "xgb": "challenger_xgb"}

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

            raw_prob, calibrated_prob = self._predict_core(req, data, model_type, feature_order)
            explainer = self._shap_exp[best_key]
            shap_exp, _ = self._call_shap_explainer(
                explainer, req, data, model_type, calibrated_prob=calibrated_prob,
            )
            risk_prob = calibrated_prob
            model_name = data.get("model_name", best_key)
        else:
            shap_exp, risk_prob = self._mock_explain(req)
            model_name = "mock"

        credit_score = self._prob_to_score(risk_prob)
        risk_level   = self._prob_to_risk_level(risk_prob)
        elapsed_ms   = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "Scored (%s): score=%d prob=%.4f risk=%s elapsed=%dms",
            model_name, credit_score, risk_prob, risk_level.value, elapsed_ms,
        )

        return ScoringResponse(
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

            _, prob = self._predict_core(req, data, model_type, feature_order)

            threshold = data.get("threshold",
                                data.get("metrics", {}).get("final_threshold", 0.5))
            pred  = int(prob >= threshold)
            score = self._prob_to_score(prob)
            risk  = self._prob_to_risk_level(prob)

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
    ) -> tuple[float, float]:
        """
        Core inference — identical pipeline to _wrapper._predict_core.

        Steps:
          1. Preprocessor.transform()   — impute → engineer → clip
          2. Reindex to feature_order    — column order matches training
          3. dtype float32 (tree) / float64 (linear)
          4. Feature selection for LR
          5. model.predict_proba()
          6. calibrator.predict()
          7. NaN guard + range validation
        """
        # ── Step 1: Full preprocessing via bundled Preprocessor ─────────────────
        preprocessor = self._preprocessors.get(
            next((k for k, d in self._models.items() if d is data), None),
            None,
        )

        if preprocessor is not None:
            df = pd.DataFrame([req.model_dump()])
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
        d = req.model_dump()
        df = pd.DataFrame([d])

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
            df = pd.DataFrame([req.model_dump()])
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
    ) -> tuple[list[dict], float]:
        """
        Call SHAP Explainer and return structured explanations + probability.

        The feature array passed to the explainer must match the exact shape
        and column order the model was trained on (handled by _get_X_for_model).

        Probability is the calibrated value from _predict_core, not reconstructed
        from sigmoid(base + sum(shap)).
        """
        X, feature_names = self._get_X_for_model(req, data, model_type)
        X = np.atleast_2d(X).astype(np.float64)

        n = self._n_artifact_features(data, model_type) if data else 16

        result = explainer(X)

        shap_values = result.values
        base_value = float(result.base_values[0]) if hasattr(result, "base_values") else 0.0

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # Use calibrated probability — SHAP explains model behaviour, not calibration
        prob = calibrated_prob if calibrated_prob is not None else (
            float(np.clip(1 / (1 + np.exp(-(base_value + shap_values[0].sum()))), 0.0, 1.0))
        )

        # Slice to the model's feature dimensionality
        explanations = self._build_explanations(
            shap_values[0][:n],
            X[0][:n],
            feature_names[:n],
        )
        return explanations, prob

    def _get_shap_explanations(
        self,
        model_key: str,
        req: ScoringRequest,
        data: dict[str, Any],
        model_type: str,
        calibrated_prob: float | None = None,
    ) -> tuple[list[dict], float]:
        """Get SHAP explanations for a specific model, with mock fallback."""
        if model_key in self._shap_exp:
            return self._call_shap_explainer(
                self._shap_exp[model_key], req, data, model_type,
                calibrated_prob=calibrated_prob,
            )
        return self._mock_explain(req)

    def _build_explanations(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: list[str] | None = None,
        neutral_threshold: float = 1.0,
    ) -> list[dict]:
        """Convert raw SHAP values into sorted explanation dicts."""
        contributions = []
        for i, feat_name in enumerate(feature_names or []):
            contribution = float(shap_values[i])
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

        ensemble_score = self._prob_to_score(weighted_prob)
        ensemble_risk  = self._prob_to_risk_level(weighted_prob)

        # SHAP from best model (highest AUC) — pass calibrated probability
        best_key = max(probabilities, key=lambda x: x[2])[0]
        best_prob = next(p for k, p, _ in probabilities if k == best_key)
        data = self._models[best_key]
        model_type = data.get("model_type", "tree")
        best_shap, _ = self._get_shap_explanations(
            best_key, req, data, model_type, calibrated_prob=best_prob,
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
        """Deterministic mock SHAP explanations when no model is loaded."""
        import random
        d = req.model_dump()
        arr = np.array(list(d.values()), dtype=np.float64)
        seed = int(sum(arr) * 1000) % (2**31)
        rng = random.Random(seed)

        explanations = []
        for name, val in d.items():
            c = round(rng.uniform(-30, 30), 2)
            explanations.append({
                "feature":      name,
                "value":        round(float(val), 4),
                "contribution": c,
                "direction":    "POSITIVE" if c > 0 else "NEGATIVE",
            })

        explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        revolver   = float(req.RevolvingUtilizationOfUnsecuredLines)
        delay_60   = float(req.NumberOfTime60_89DaysPastDueNotWorse)
        prob = min(0.99, max(0.01, revolver * 0.1 + delay_60 * 0.05))
        return explanations, round(prob, 4)

    def _mock_multi_response(
        self,
        single: ScoringResponse,
        req: ScoringRequest,
    ) -> dict[str, Any]:
        """Build a full mock multi-model response."""
        shap_exp = [e.model_dump() for e in single.shap_explanations]

        return {
            "models": {
                "mock_model": {
                    "credit_score":       single.credit_score,
                    "risk_probability":   single.risk_probability,
                    "risk_level":         single.risk_level.value,
                    "shap_explanations":  shap_exp,
                    "prediction":         "DEFAULT" if single.risk_probability > 0.5 else "GOOD",
                    "probability":        single.risk_probability,
                    "metrics":            {},
                    "model_name":         "Mock Model",
                    "model_type":         "mock",
                }
            },
            "ensemble": {
                "credit_score":         single.credit_score,
                "risk_probability":     single.risk_probability,
                "risk_level":           single.risk_level.value,
                "shap_explanations":   shap_exp,
                "voting":              {"approved": 1, "rejected": 0}
                    if single.risk_probability > 0.5
                    else {"approved": 0, "rejected": 1},
                "weighted_probability": single.risk_probability,
                "best_model_key":      "mock_model",
            },
            "pipeline_metadata": {},
        }

    def _mock_ensemble(self, req: ScoringRequest) -> dict[str, Any]:
        """Fallback ensemble when no models are loaded."""
        shap_exp, prob = self._mock_explain(req)
        score = self._prob_to_score(prob)
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
    def _prob_to_score(prob: float) -> int:
        """
        Map P(default) → FICO-like credit score (300–850).

        Formula identical to pipeline's pd_to_credit_score():
          PD =  2% → Score = 850 (Excellent)
          PD = 40% → Score = 300 (Very Poor)
        """
        TARGET_MIN_PD = 0.02
        TARGET_MAX_PD = 0.40
        MIN_SCORE = 300
        MAX_SCORE = 850

        prob = float(np.clip(prob, 1e-6, 1 - 1e-6))

        lo_max = np.log((1 - TARGET_MIN_PD) / TARGET_MIN_PD)
        lo_min = np.log((1 - TARGET_MAX_PD) / TARGET_MAX_PD)
        log_odds = np.log((1 - prob) / prob)

        grade_range = MAX_SCORE - MIN_SCORE
        log_range   = lo_max - lo_min
        score = MAX_SCORE - grade_range * (log_odds - lo_max) / log_range

        return int(np.clip(round(score), MIN_SCORE, MAX_SCORE))

    @staticmethod
    def _prob_to_risk_level(prob: float) -> RiskLevel:
        """Map P(default) → categorical risk bucket."""
        if   prob < 0.05: return RiskLevel.LOW
        elif prob < 0.15: return RiskLevel.MEDIUM
        elif prob < 0.35: return RiskLevel.HIGH
        else:             return RiskLevel.CRITICAL
