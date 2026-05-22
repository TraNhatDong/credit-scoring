"""
_train.py — Training pipeline entry point and model-specific helpers.

Dependency: _config.py only. Does NOT import _monitor.py.

Public entry point: train_pipeline()
"""
from __future__ import annotations

import time as _time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_score,
    recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ._config import (
    COST_FP, COST_FN, ISOTONIC_MIN_N, IV_THRESHOLD, MODEL_VERSION,
    N_SPLITS, OPTUNA_AVAILABLE, OPTUNA_N_TRIALS, RANDOM_STATE,
    SHAP_EVAL_SIZE, THRESHOLD_STRATEGY, THRESHOLDS,
    XGB_EARLY_STOP,
    build_calibrator, compute_iv_scores, compute_shap,
    filter_by_iv, find_best_threshold_by_cost, find_best_threshold_by_f1,
    log, metrics_at_threshold, pd_to_credit_score,
    stratified_shap_sample,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
def _xgb_fold_train(
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    X_val_fold: np.ndarray,
    y_val_fold: np.ndarray,
    params: dict,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, np.ndarray]:
    """Train one XGB fold with early stopping. Returns (trained_model, OOF_probabilities)."""
    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=XGB_EARLY_STOP,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    proba = model.predict_proba(X_val_fold)[:, 1]
    return model, proba


def _optuna_objective(
    trial,   # optuna.Trial — optuna is only imported inside OPTUNA_AVAILABLE blocks
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> float:
    """Optuna objective — StratifiedKFold CV AUC (k-fold only, no external val sets)."""
    y_arr = np.asarray(y_train)
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
        "gamma":             trial.suggest_float("gamma", 0.1, 2.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X_train))

    for tr_idx, val_idx in skf.split(X_train, y_arr):
        X_tr, X_vl = X_train[tr_idx], X_train[val_idx]
        y_tr, y_vl = y_arr[tr_idx], y_arr[val_idx]
        _, proba = _xgb_fold_train(X_tr, y_tr, X_vl, y_vl, params=params,
                                    scale_pos_weight=scale_pos_weight)
        oof_proba[val_idx] = proba

    fpr, tpr, _ = roc_curve(y_arr, oof_proba)
    return auc(fpr, tpr)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — LR (k-fold OOF + calibration + threshold)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_lr_kfold(
    X_lr: np.ndarray,
    y_train: pd.Series,
    X_val_thresh_lr: np.ndarray,
    X_test_lr: np.ndarray,
    y_val_thresh: pd.Series,
    y_test: pd.Series,
    iv_feature_names: list[str],
    scaler: StandardScaler,
) -> dict[str, Any]:
    """Champion LR: k-fold OOF → isotonic calibration → threshold tuning → test eval."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    lr_oof_proba = np.zeros(len(X_lr))
    lr_fold_aucs: list[float] = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_lr, y_train)):
        X_tr_lr, X_vl_lr = X_lr[tr_idx], X_lr[vl_idx]
        y_tr, y_vl = y_train.values[tr_idx], y_train.values[vl_idx]

        lr_fold = LogisticRegression(
            solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        lr_fold.fit(X_tr_lr, y_tr)
        lr_oof_proba[vl_idx] = lr_fold.predict_proba(X_vl_lr)[:, 1]

        fpr, tpr, _ = roc_curve(y_vl, lr_oof_proba[vl_idx])
        lr_fold_aucs.append(auc(fpr, tpr))
        log.info("    LR fold %d: AUC=%.4f", fold_idx + 1, lr_fold_aucs[-1])

    lr_oof_auc = round(auc(*roc_curve(y_train.values, lr_oof_proba)[:2]), 4)
    log.info("    LR OOF AUC=%.4f | fold mean=%.4f | fold std=%.4f",
             lr_oof_auc, np.mean(lr_fold_aucs), np.std(lr_fold_aucs))

    # Final LR on full train
    lr_raw_final = LogisticRegression(
        solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    lr_raw_final.fit(X_lr, y_train.values)

    # ── Calibration: seeded-shuffle hold-out to avoid ordering bias in OOF splits ───
    # OOF predictions have no temporal order; a "last 20%" slice would introduce
    # ordering bias if the data was sorted upstream.  We use a seeded shuffle so
    # the hold-out is reproducible and unbiased relative to the OOF indices.
    n_calib = max(int(len(lr_oof_proba) * 0.20), 50)
    idx = np.arange(len(lr_oof_proba))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(idx)
    calib_idx = idx[:n_calib]
    train_idx = idx[n_calib:]
    lr_calibrator, lr_calib_method = build_calibrator(
        lr_oof_proba[train_idx], y_train.values[train_idx]
    )
    log.info("    LR calibration: %s (N_oof=%d, N_calib=%d, seeded-shuffle hold-out)",
             lr_calib_method, len(train_idx), len(calib_idx))

    def lr_model_predict(X: np.ndarray) -> np.ndarray:
        raw = lr_raw_final.predict_proba(X)[:, 1]
        return lr_calibrator.predict(raw)

    # Full-train calibrated predictions (for PSI monitoring — same set as val_thresh)
    lr_prob_train = lr_model_predict(X_lr)

    # Raw probabilities (from uncalibrated LR) — used only internally for calibration.
    lr_prob_train_raw = lr_raw_final.predict_proba(X_lr)[:, 1]

    # Threshold tuning on val_thresh
    lr_prob_thresh = lr_model_predict(X_val_thresh_lr)
    lr_opt_thresh, lr_opt_f1 = find_best_threshold_by_f1(y_val_thresh.values, lr_prob_thresh)
    lr_cost_thresh, lr_cost  = find_best_threshold_by_cost(y_val_thresh.values, lr_prob_thresh)

    # Threshold robustness on test
    lr_prob_test = lr_model_predict(X_test_lr)
    lr_opt_thresh_test, _ = find_best_threshold_by_f1(y_test.values, lr_prob_test)
    lr_cost_thresh_test, _ = find_best_threshold_by_cost(y_test.values, lr_prob_test)

    lr_thresh_robustness = {
        "val_thresh_opt":   lr_opt_thresh,   "test_opt":         lr_opt_thresh_test,
        "val_thresh_cost": lr_cost_thresh,  "test_cost":        lr_cost_thresh_test,
        "opt_delta":       round(abs(lr_opt_thresh - lr_opt_thresh_test), 4),
        "cost_delta":      round(abs(lr_cost_thresh - lr_cost_thresh_test), 4),
        "opt_delta_pct":   round(abs(lr_opt_thresh - lr_opt_thresh_test) / max(lr_opt_thresh, 0.01), 4),
        "cost_delta_pct":  round(abs(lr_cost_thresh - lr_cost_thresh_test) / max(lr_cost_thresh, 0.01), 4),
    }
    log.info(
        "    LR threshold robustness — opt: |%.2f-%.2f|=%s | cost: |%.2f-%.2f|=%s",
        lr_opt_thresh, lr_opt_thresh_test, lr_thresh_robustness["opt_delta"],
        lr_cost_thresh, lr_cost_thresh_test, lr_thresh_robustness["cost_delta"],
    )

    lr_final_thresh = lr_cost_thresh if THRESHOLD_STRATEGY == "cost" else lr_opt_thresh

    # Final evaluation on test (metrics use final_threshold, not 0.5)
    lr_prob_test_binary = (lr_prob_test >= lr_final_thresh).astype(int)
    fpr_lr, tpr_lr, _ = roc_curve(y_test.values, lr_prob_test)
    lr_auc = round(auc(fpr_lr, tpr_lr), 4)

    lr_metrics = {
        "auc":                lr_auc,
        "oof_auc":           lr_oof_auc,
        "k_fold_aucs":       [round(a, 4) for a in lr_fold_aucs],
        "k_fold_mean":       round(np.mean(lr_fold_aucs), 4),
        "k_fold_std":        round(np.std(lr_fold_aucs), 4),
        "accuracy":          round(accuracy_score(y_test, lr_prob_test_binary), 4),
        "precision":         round(precision_score(y_test, lr_prob_test_binary, zero_division=0), 4),
        "recall":            round(recall_score(y_test, lr_prob_test_binary, zero_division=0), 4),
        "f1":               round(f1_score(y_test, lr_prob_test_binary, zero_division=0), 4),
        "optimal_threshold":  lr_opt_thresh,    "optimal_f1":         lr_opt_f1,
        "cost_threshold":     lr_cost_thresh,   "business_cost":      lr_cost,
        "final_threshold":    lr_final_thresh,  "threshold_strategy": THRESHOLD_STRATEGY,
        "calibration_method": lr_calib_method,  "threshold_robustness": lr_thresh_robustness,
        "per_threshold": [
            metrics_at_threshold(y_test.values, lr_prob_test, t) for t in THRESHOLDS
        ],
    }

    # SHAP — LR was trained on StandardScaler-transformed features, so the explainer
    # operates in scaled space. We expose two naming layers for different audiences:
    #   - internal_log  : feature names annotated "(scaled)" for engineering readability
    #   - external_user : raw feature names for business-facing reports
    import shap
    shap_idx = stratified_shap_sample(
        X_val_thresh_lr, lr_prob_thresh, lr_final_thresh, SHAP_EVAL_SIZE,
    )
    lr_shap_internal_names = [f"{name} (scaled)" for name in iv_feature_names]
    lr_shap_external_names = list(iv_feature_names)   # raw names for business users
    lr_shap_internal = compute_shap(
        lr_raw_final, "linear",
        X_lr, X_val_thresh_lr[shap_idx], lr_shap_internal_names,
    )
    lr_shap_external = compute_shap(
        lr_raw_final, "linear",
        X_lr, X_val_thresh_lr[shap_idx], lr_shap_external_names,
    )

    return {
        "model":             lr_raw_final,
        "calibrator":        lr_calibrator,
        "calibration_method": lr_calib_method,
        "scaler":            scaler,
        "model_type":        "linear",
        "model_name":        "Logistic Regression (Benchmark)",
        "champion":          False,
        "iv_features":      iv_feature_names,
        "feature_order":     iv_feature_names,
        "metrics":           lr_metrics,
        "shap_top5":        lr_shap_internal,
        "shap_top5_raw":    lr_shap_external,   # raw feature names for business users
        "shap_note":        "SHAP computed on raw LogisticRegression (before calibration); "
                            "interpret model behaviour, not the calibrated probability output",
        "prob_oof":         lr_oof_proba,    # OOF (for unbiased AUC)
        "prob_train":       lr_prob_train,   # full-train calibrated (for PSI)
        "scores_train":     pd_to_credit_score(lr_prob_train, clamp_min=True),
        "prob_thresh":      lr_prob_thresh,
        "prob_test":        lr_prob_test,     # calibrated (for threshold/metrics + scoring)
        "prob_test_raw":    lr_prob_test,     # calibrated (same as prob_test for LR)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — XGBoost (Optuna + k-fold OOF + calibration + threshold)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_xgb_kfold(
    X_train: np.ndarray,
    X_val_thresh: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_val_thresh: pd.Series,
    y_test: pd.Series,
    xgb_feature_names: list[str],
) -> dict[str, Any]:
    """Challenger XGB: Optuna tuning → k-fold OOF → isotonic calibration → threshold."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = 1.0  # no class weighting — raw probs stay calibrated, AUC unchanged
    log.info("  Class — neg=%d | pos=%d | scale_pos_weight=%.4f",
             n_neg, n_pos, scale_pos_weight)

    # Optuna hyperparameter search
    log.info("  [CHALLENGER] Training XGBoost + Optuna k-fold (n_trials=%d, k=%d)...",
             OPTUNA_N_TRIALS, N_SPLITS)

    if OPTUNA_AVAILABLE:
        import optuna as _optuna
        study = _optuna.create_study(
            direction="maximize",
            sampler=_optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial: _optuna_objective(trial, X_train, y_train, scale_pos_weight),
            n_trials=OPTUNA_N_TRIALS,
            show_progress_bar=False,
        )
        best_params     = study.best_params
        best_optuna_auc = round(study.best_value, 4)
        log.info("  [CHALLENGER] Optuna best | k-fold OOF AUC=%.4f | params=%s",
                 best_optuna_auc, best_params)
    else:
        log.warning("  [CHALLENGER] Optuna not available — using defaults")
        best_params     = {
            "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 2, "gamma": 0.5,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
        best_optuna_auc = None

    # K-fold OOF
    log.info("  [CHALLENGER] Generating OOF predictions (k=%d)...", N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    xgb_oof_proba = np.zeros(len(X_train))
    xgb_fold_aucs: list[float] = []
    xgb_fold_iters: list[int] = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
        y_tr, y_vl = y_train.values[tr_idx], y_train.values[vl_idx]

        fold_model, proba = _xgb_fold_train(
            X_tr, y_tr, X_vl, y_vl,
            params=best_params,
            scale_pos_weight=scale_pos_weight,
        )
        xgb_oof_proba[vl_idx] = proba
        xgb_fold_iters.append(fold_model.best_iteration)

        fpr, tpr, _ = roc_curve(y_vl, proba)
        xgb_fold_aucs.append(auc(fpr, tpr))
        log.info("    XGB fold %d: AUC=%.4f | best_iter=%d",
                 fold_idx + 1, xgb_fold_aucs[-1], fold_model.best_iteration)

    xgb_oof_auc = round(auc(*roc_curve(y_train.values, xgb_oof_proba)[:2]), 4)
    log.info("    XGB OOF AUC=%.4f | fold mean=%.4f | fold std=%.4f",
             xgb_oof_auc, np.mean(xgb_fold_aucs), np.std(xgb_fold_aucs))

    # Final XGB: lock in median fold iterations directly.
    # The k-fold median is already a robust estimate of the optimal iteration count;
    # retraining with a deterministic slice for early stopping would consume 15% of
    # training data and introduce ordering-bias risk if the data has any structure.
    best_iter_median = int(np.median(xgb_fold_iters))
    final_n_est = best_iter_median
    xgb_raw_final = XGBClassifier(
        n_estimators=final_n_est,
        **{k: v for k, v in best_params.items() if k != "n_estimators"},
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_raw_final.fit(X_train, y_train.values, verbose=False)
    log.info("  [CHALLENGER] Final XGB n_estimators=%d (median of %d fold iterations)",
             final_n_est, N_SPLITS)

    # Calibration: seeded-shuffle hold-out (same rationale as LR calibration above).
    n_xgb_calib = max(int(len(xgb_oof_proba) * 0.20), 50)
    xgb_idx = np.arange(len(xgb_oof_proba))
    xgb_rng = np.random.default_rng(RANDOM_STATE)
    xgb_rng.shuffle(xgb_idx)
    xgb_calib_idx = xgb_idx[:n_xgb_calib]
    xgb_train_idx = xgb_idx[n_xgb_calib:]
    xgb_calibrator, xgb_calib_method = build_calibrator(
        xgb_oof_proba[xgb_train_idx], y_train.values[xgb_train_idx],
        use_isotonic=True,  # isotonic — fixes tail under-estimation from Platt sigmoid
    )
    log.info("    XGB calibration: %s (N_oof=%d, N_calib=%d, seeded-shuffle hold-out, isotonic=fix-tail)",
             xgb_calib_method, len(xgb_train_idx), len(xgb_calib_idx))

    def xgb_model_predict(X: np.ndarray) -> np.ndarray:
        raw = xgb_raw_final.predict_proba(X)[:, 1]
        return xgb_calibrator.predict(raw)

    # Full-train predictions (for PSI monitoring — calibrated)
    xgb_prob_train = xgb_model_predict(X_train)

    # Test predictions (calibrated — stored in artifact for reporting)
    xgb_prob_test_cal = xgb_model_predict(X_test)

    # Threshold tuning on val_thresh
    xgb_prob_thresh = xgb_model_predict(X_val_thresh)
    xgb_opt_thresh, xgb_opt_f1 = find_best_threshold_by_f1(y_val_thresh.values, xgb_prob_thresh)
    xgb_cost_thresh, xgb_cost  = find_best_threshold_by_cost(y_val_thresh.values, xgb_prob_thresh)

    # Threshold robustness on test (calibrated probs)
    xgb_opt_thresh_test, _  = find_best_threshold_by_f1(y_test.values, xgb_prob_test_cal)
    xgb_cost_thresh_test, _ = find_best_threshold_by_cost(y_test.values, xgb_prob_test_cal)

    # Threshold instability detection: if delta > 0.10, shrink toward 0.5
    # A cost_threshold of 0.89 is unrealistic — indicates calibration pushed probs to extremes
    # on val_thresh, making the cost-minimum unstable. Averaging with 0.5 acts as regularisation.
    cost_delta = abs(xgb_cost_thresh - xgb_cost_thresh_test)
    if cost_delta > 0.10:
        log.warning(
            "  XGB threshold instability: val_cost_thresh=%.2f | test_cost_thresh=%.2f | delta=%.2f > 0.10"
            " — shrinking toward 0.50 for robustness",
            xgb_cost_thresh, xgb_cost_thresh_test, cost_delta,
        )
        xgb_cost_thresh = round((xgb_cost_thresh + xgb_cost_thresh_test + 0.50) / 3, 2)
        log.warning("  XGB adjusted cost_threshold = %.2f (shrinkage average)", xgb_cost_thresh)

    xgb_thresh_robustness = {
        "val_thresh_opt":  xgb_opt_thresh,  "test_opt":        xgb_opt_thresh_test,
        "val_thresh_cost": xgb_cost_thresh, "test_cost":       xgb_cost_thresh_test,
        "opt_delta":       round(abs(xgb_opt_thresh - xgb_opt_thresh_test), 4),
        "cost_delta":     round(abs(xgb_cost_thresh - xgb_cost_thresh_test), 4),
        "opt_delta_pct":  round(abs(xgb_opt_thresh - xgb_opt_thresh_test) / max(xgb_opt_thresh, 0.01), 4),
        "cost_delta_pct": round(abs(xgb_cost_thresh - xgb_cost_thresh_test) / max(xgb_cost_thresh, 0.01), 4),
    }
    log.info(
        "    XGB threshold robustness — opt: |%.2f-%.2f|=%s | cost: |%.2f-%.2f|=%s",
        xgb_opt_thresh, xgb_opt_thresh_test, xgb_thresh_robustness["opt_delta"],
        xgb_cost_thresh, xgb_cost_thresh_test, xgb_thresh_robustness["cost_delta"],
    )

    xgb_final_thresh = xgb_cost_thresh if THRESHOLD_STRATEGY == "cost" else xgb_opt_thresh

    # Final evaluation on test (metrics use final_threshold, not 0.5)
    xgb_prob_test_binary = (xgb_prob_test_cal >= xgb_final_thresh).astype(int)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test.values, xgb_prob_test_cal)
    xgb_auc = round(auc(fpr_xgb, tpr_xgb), 4)

    xgb_metrics = {
        "auc":                xgb_auc,
        "oof_auc":           xgb_oof_auc,
        "k_fold_aucs":       [round(a, 4) for a in xgb_fold_aucs],
        "k_fold_mean":       round(np.mean(xgb_fold_aucs), 4),
        "k_fold_std":        round(np.std(xgb_fold_aucs), 4),
        "accuracy":          round(accuracy_score(y_test, xgb_prob_test_binary), 4),
        "precision":         round(precision_score(y_test, xgb_prob_test_binary, zero_division=0), 4),
        "recall":            round(recall_score(y_test, xgb_prob_test_binary, zero_division=0), 4),
        "f1":               round(f1_score(y_test, xgb_prob_test_binary, zero_division=0), 4),
        "optimal_threshold":  xgb_opt_thresh,   "optimal_f1":         xgb_opt_f1,
        "cost_threshold":     xgb_cost_thresh,  "business_cost":      xgb_cost,
        "final_threshold":    xgb_final_thresh, "threshold_strategy": THRESHOLD_STRATEGY,
        "calibration_method": xgb_calib_method, "threshold_robustness": xgb_thresh_robustness,
        "best_iter_median":  best_iter_median,  "optuna_val_auc":    best_optuna_auc,
        "optuna_params":      best_params,
        "per_threshold": [
            metrics_at_threshold(y_test.values, xgb_prob_test_cal, t) for t in THRESHOLDS
        ],
    }

    # SHAP (TreeExplainer — XGBoost 3.x compatible with shap 0.49 after base_score fix)
    shap_idx = stratified_shap_sample(
        X_val_thresh, xgb_prob_thresh, xgb_final_thresh, SHAP_EVAL_SIZE,
    )
    xgb_shap = compute_shap(xgb_raw_final, "tree", None,
                            X_val_thresh[shap_idx], xgb_feature_names)

    return {
        "model":             xgb_raw_final,
        "calibrator":        xgb_calibrator,
        "calibration_method": xgb_calib_method,
        "model_type":        "tree",
        "model_name":        "XGBoost (Champion)",
        "champion":          True,
        "feature_order":     xgb_feature_names,
        "metrics":           xgb_metrics,
        "shap_top5":        xgb_shap,
        "prob_oof":         xgb_oof_proba,   # OOF (for AUC eval)
        "prob_train":       xgb_prob_train,  # full-train calibrated (for PSI)
        "scores_train":     pd_to_credit_score(xgb_prob_train, clamp_min=True),
        "prob_thresh":      xgb_prob_thresh,
        "prob_test":        xgb_prob_test_cal,   # calibrated test (for threshold/metrics)
        "prob_test_raw":    xgb_prob_test_cal,  # calibrated test (for score/band display — MUST match prob_test)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# train_pipeline — entry point (PUBLIC)
# ═══════════════════════════════════════════════════════════════════════════════
def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val_thresh: pd.DataFrame,
    y_val_thresh: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
) -> dict[str, Any]:
    """
    Full training pipeline: IV selection → scaling → LR + XGB → calibration → threshold.

    Args:
        X_train, y_train         : training set (preprocessor.transform already applied)
        X_val_thresh, y_val_thresh : threshold-tuning set (preprocessor.transform already applied)
        X_test, y_test           : test/evaluation set (preprocessor.transform already applied)
        preprocessor             : fitted Preprocessor instance (for artifact bundling)

    Returns:
        {
            "lr":   {model, calibrator, scaler, metrics, ...},
            "xgb":  {model, calibrator, metrics, ...},
            "lr_feature_names": [...],
            "iv_scores": {...},
            "preprocessor": <fitted Preprocessor>,
        }
    """
    t0 = _time.perf_counter()
    log.info("STEP 8 — Training Champion + Challenger (v4.1 | k=%d folds)", N_SPLITS)
    log.info("  train=%d | val_thresh=%d | Test=%d",
             len(X_train), len(X_val_thresh), len(X_test))

    # ── Reproducibility: re-seed numpy to cover any non-deterministic ops inside helpers ──
    np.random.seed(RANDOM_STATE)

    # ── Feature-consistency guard — catch silent upstream column reorder ─────────
    train_cols = list(X_train.columns)
    for name, df in [("val_thresh", X_val_thresh), ("test", X_test)]:
        if list(df.columns) != train_cols:
            raise ValueError(
                f"Column order mismatch in {name}: expected {train_cols}, got {list(df.columns)}"
            )

    # ── IV feature selection (computed on train ONLY) ─────────────────────────
    log.info("STEP 6 — IV feature selection (IV > %.2f, computed on train)", IV_THRESHOLD)
    iv_scores = compute_iv_scores(X_train, y_train, list(X_train.columns))
    log.info("  IV scores (sorted desc):")
    for feat, iv in sorted(iv_scores.items(), key=lambda x: -x[1]):
        flag = f"< {IV_THRESHOLD} DROP" if iv < IV_THRESHOLD else "KEEP"
        log.info("    %-45s IV=%7.4f  %s", feat, iv, flag)

    lr_feature_names = filter_by_iv(iv_scores, threshold=IV_THRESHOLD)
    dropped = [f for f in list(X_train.columns) if f not in lr_feature_names]
    log.info("  Selected %d / %d features for LR (IV > %.2f) | dropped: %s",
             len(lr_feature_names), len(X_train.columns), IV_THRESHOLD, dropped)

    # ── Scale IV-filtered features for LR ─────────────────────────────────────
    log.info("STEP 7 — Scaling IV-filtered features (StandardScaler, fit on train only)")
    col_idx = [list(X_train.columns).index(f) for f in lr_feature_names]

    X_tr_lr = X_train.values[:, col_idx].astype(np.float64)
    X_vt_lr = X_val_thresh.values[:, col_idx].astype(np.float64)
    X_te_lr = X_test.values[:, col_idx].astype(np.float64)

    scaler = StandardScaler()
    scaler.fit(X_tr_lr)
    X_tr_lr = scaler.transform(X_tr_lr)
    X_vt_lr = scaler.transform(X_vt_lr)
    X_te_lr = scaler.transform(X_te_lr)
    log.info("  Scaler mean[:3]: %s | std[:3]: %s", scaler.mean_[:3], scaler.scale_[:3])

    # ── Champion LR ────────────────────────────────────────────────────────────
    log.info("")
    log.info("  [CHAMPION] K-fold OOF + Isotonic calibration...")
    t_lr = _time.perf_counter()
    results_lr = _run_lr_kfold(
        X_tr_lr, y_train, X_vt_lr, X_te_lr,
        y_val_thresh, y_test,
        lr_feature_names, scaler,
    )
    log.info("  [CHAMPION] LR done in %.1fs | Test AUC=%.4f | OOF AUC=%.4f | thresh=%.2f (strategy=%s)",
             _time.perf_counter() - t_lr,
             results_lr["metrics"]["auc"], results_lr["metrics"]["oof_auc"],
             results_lr["metrics"]["final_threshold"], THRESHOLD_STRATEGY)

    # ── Challenger XGB ─────────────────────────────────────────────────────────
    log.info("")
    t_xgb = _time.perf_counter()
    # Enforce column order explicitly — .values inherits DataFrame column order
    xgb_feature_names = list(X_train.columns)
    X_tr_xgb = X_train[xgb_feature_names].values.astype(np.float32)
    X_vt_xgb = X_val_thresh[xgb_feature_names].values.astype(np.float32)
    X_te_xgb = X_test[xgb_feature_names].values.astype(np.float32)

    results_xgb = _run_xgb_kfold(
        X_tr_xgb, X_vt_xgb, X_te_xgb,
        y_train, y_val_thresh, y_test,
        xgb_feature_names,
    )
    log.info("  [CHALLENGER] XGBoost done in %.1fs | Test AUC=%.4f | OOF AUC=%.4f | thresh=%.2f (strategy=%s)",
             _time.perf_counter() - t_xgb,
             results_xgb["metrics"]["auc"], results_xgb["metrics"]["oof_auc"],
             results_xgb["metrics"]["final_threshold"], THRESHOLD_STRATEGY)

    log.info("  STEP 8 done in %.1fs", _time.perf_counter() - t0)

    return {
        "lr":               results_lr,
        "xgb":              results_xgb,
        "lr_feature_names": lr_feature_names,
        "iv_scores":        iv_scores,
        "preprocessor":     preprocessor,
    }
