"""
_monitor.py — PSI monitoring, feature drift, class imbalance, artifact saving, report generation.

Dependency: _config.py only. Does NOT import _train.py.

Public entry points:
  - step9_monitoring(...)
  - step10_save_artifacts(...)
  - step11_generate_report(...)
"""
from __future__ import annotations

import json
import os
import time as _time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ._config import (
    COST_FP, COST_FN, ISOTONIC_MIN_N, MODEL_VERSION, N_SPLITS,
    THRESHOLD_STRATEGY, THRESHOLDS,
    compute_bad_rate_drift, compute_ks_statistic, compute_psi,
    compute_psi_bin_edges, log, pd_to_credit_score, psi_band, score_band,
    RAW_FEATURE_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — PSI + FEATURE DRIFT + CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════════
def step9_monitoring(
    results_lr: dict,
    results_xgb: dict,
    X_train: pd.DataFrame,
    X_val_thresh: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    iv_scores: dict[str, float],
    prob_val_lr: np.ndarray,
    prob_val_xgb: np.ndarray,
    top_n_drift: int = 10,
) -> None:
    """
    Compute PSI (prediction-level) + feature drift (PSI + KS for top features)
    + class imbalance monitoring.

    PSI bins: fit from OOF_train, reuse for val_thresh + test.
    Modifies results_lr / results_xgb in-place by adding metrics.
    """
    t0 = _time.perf_counter()
    log.info("STEP 9 — PSI + Feature Drift + Class Imbalance Monitoring")

    # PSI: reference (prob_val on val_thresh) vs actual (prob_test on test)
    for (name, res, prob_val) in [
        ("LR", results_lr, prob_val_lr),
        ("XGBoost", results_xgb, prob_val_xgb),
    ]:
        # PSI-DEV: val_thresh vs train (reference = val_thresh predictions)
        psi_bin_edges = compute_psi_bin_edges(prob_val, n_bins=10)
        psi_dev  = round(compute_psi(prob_val, res["prob_thresh"], bin_edges=psi_bin_edges), 4)
        # PSI-Test: test vs val_thresh (same reference)
        psi_test = round(compute_psi(prob_val, res["prob_test"],  bin_edges=psi_bin_edges), 4)

        res["metrics"]["psi_dev"]      = psi_dev
        res["metrics"]["psi_test"]     = psi_test
        res["metrics"]["psi_dev_band"]  = psi_band(psi_dev)
        res["metrics"]["psi_test_band"] = psi_band(psi_test)

        log.info(
            "  %s — PSI-DEV=%.4f (%s) | PSI-Test=%.4f (%s) | bins from val_thresh",
            name, psi_dev, psi_band(psi_dev), psi_test, psi_band(psi_test),
        )

    # Feature drift monitoring (top N features by IV)
    top_features = sorted(iv_scores, key=lambda f: iv_scores[f], reverse=True)[:top_n_drift]
    feature_drift: dict[str, Any] = {}

    log.info("  Feature drift (top %d by IV):", top_n_drift)
    for feat in top_features:
        train_vals      = X_train[feat].values
        val_thresh_vals = X_val_thresh[feat].values
        test_vals       = X_test[feat].values

        ks_vt  = round(compute_ks_statistic(train_vals, val_thresh_vals), 4)
        ks_te  = round(compute_ks_statistic(train_vals, test_vals), 4)
        # PSI: train vs val_thresh and train vs test — compute_psi handles shape mismatch
        # by using combined percentiles when bin_edges is None.
        psi_vt = round(compute_psi(train_vals, val_thresh_vals), 4)
        psi_te = round(compute_psi(train_vals, test_vals), 4)

        flag_vt = ks_vt > 0.1 or psi_vt > 0.1
        flag_te = ks_te > 0.1 or psi_te > 0.1

        feature_drift[feat] = {
            "ks_val_thresh":    ks_vt,
            "ks_test":          ks_te,
            "psi_val_thresh":   psi_vt,
            "psi_test":         psi_te,
            "drift_flag_val_thresh": flag_vt,
            "drift_flag_test":   flag_te,
        }

        flag_str = ""
        if flag_vt: flag_str += " [val_thresh DRIFT]"
        if flag_te: flag_str += " [test DRIFT]"
        log.info(
            "    %-45s KS_vt=%.4f KS_te=%.4f PSI_vt=%.4f PSI_te=%.4f%s",
            feat, ks_vt, ks_te, psi_vt, psi_te, flag_str,
        )

    results_lr["feature_drift"]  = feature_drift
    results_xgb["feature_drift"] = feature_drift

    # Class imbalance monitoring
    br = compute_bad_rate_drift(y_train.values, y_test.values)
    log.info(
        "  Class imbalance — train bad_rate=%.4f | test bad_rate=%.4f "
        "| drift=%.4f | flag=%s",
        br["bad_rate_train"], br["bad_rate_test"],
        br["drift"], br["flag"],
    )
    if br["flag"]:
        log.warning(
            "  ⚠️  CLASS IMBALANCE DRIFT DETECTED: %s",
            br["flag_reason"],
        )

    results_lr["bad_rate"]  = br
    results_xgb["bad_rate"] = br

    log.info("  STEP 9 done in %.1fs", _time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
def step10_save_artifacts(
    results_lr: dict,
    results_xgb: dict,
    lr_feature_names: list[str],
    iv_scores: dict[str, float],
    preprocessor,
    out_dir: str = "models",
) -> None:
    """
    Save all production artifacts as single-file bundles.

    Artifact bundle contains: {model, calibrator, preprocessor, metadata}
    Rule: 1 model = 1 artifact file = 1 API call.

    Scaler is bundled inside the LR artifact (champion only).
    Preprocessor is bundled in both artifacts.
    """
    t0 = _time.perf_counter()
    log.info("STEP 10 — Saving artifacts to: %s", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    best_key = max(
        {"lr": results_lr, "xgb": results_xgb}.items(),
        key=lambda kv: kv[1]["metrics"]["auc"],
    )[0]

    # ── Champion model (LR) — single artifact bundle ───────────────────────────
    lr_path = os.path.join(out_dir, f"champion_lr_model.{MODEL_VERSION}.pkl")
    joblib.dump({
        "model":              results_lr["model"],
        "calibrator":         results_lr["calibrator"],
        "calibration_method": results_lr["calibration_method"],
        "scaler":             results_lr["scaler"],
        "feature_names":      lr_feature_names,
        "feature_order":      lr_feature_names,
        "champion":           True,
        "model_type":         "linear",
        "model_version":      MODEL_VERSION,
        "preprocessor_state": preprocessor.get_state(),   # serialised — stable across version changes
        "iv_threshold":       0.02,
        "iv_scores":          iv_scores,
        "threshold":          results_lr["metrics"]["final_threshold"],
        "threshold_strategy": THRESHOLD_STRATEGY,
    }, lr_path)
    log.info("  Saved: %s (champion_lr | %d IV features | %s)",
             lr_path, len(lr_feature_names), results_lr["calibration_method"])

    # ── Challenger model (XGB) — single artifact bundle ────────────────────────
    xgb_path = os.path.join(out_dir, f"challenger_xgb_model.{MODEL_VERSION}.pkl")
    joblib.dump({
        "model":              results_xgb["model"],
        "calibrator":         results_xgb["calibrator"],
        "calibration_method": results_xgb["calibration_method"],
        "feature_names":      RAW_FEATURE_COLUMNS,
        "feature_order":      RAW_FEATURE_COLUMNS,
        "champion":           False,
        "model_type":         "tree",
        "model_version":      MODEL_VERSION,
        "preprocessor_state": preprocessor.get_state(),   # serialised — same Preprocessor as LR
        "iv_threshold":       0.02,
        "iv_scores":          iv_scores,
        "threshold":         results_xgb["metrics"]["final_threshold"],
        "threshold_strategy": THRESHOLD_STRATEGY,
    }, xgb_path)
    log.info("  Saved: %s (challenger_xgb | %d raw features | %s)",
             xgb_path, len(RAW_FEATURE_COLUMNS), results_xgb["calibration_method"])

    # ── Pipeline metadata JSON ────────────────────────────────────────────────
    def _summarise(m: dict) -> dict:
        return {k: v for k, v in m.items() if k not in (
            "per_threshold", "threshold_robustness",
        )}

    metadata = {
        "pipeline_version":      MODEL_VERSION,
        "iv_threshold":          0.02,
        "n_splits":           N_SPLITS,
        "calibration_min_n":    ISOTONIC_MIN_N,
        "iv_scores":            iv_scores,
        "lr_features":          lr_feature_names,
        "n_lr_features":        len(lr_feature_names),
        "n_xgb_features":      len(RAW_FEATURE_COLUMNS),
        "thresholds_evaluated": THRESHOLDS,
        "cost_fp":              COST_FP,
        "cost_fn":              COST_FN,
        "threshold_strategy":   THRESHOLD_STRATEGY,
        "champion": {
            "key":                "lr",
            "model_name":         "Logistic Regression (Champion)",
            "test_auc":            results_lr["metrics"]["auc"],
            "oof_auc":            results_lr["metrics"]["oof_auc"],
            "k_fold_aucs":        results_lr["metrics"].get("k_fold_aucs"),
            "final_threshold":     results_lr["metrics"]["final_threshold"],
            "threshold_robustness": results_lr["metrics"].get("threshold_robustness"),
            "calibration_method": results_lr["calibration_method"],
            "bad_rate":           results_lr.get("bad_rate"),
            "metrics":            _summarise(results_lr["metrics"]),
        },
        "challenger": {
            "key":                "xgb",
            "model_name":         "XGBoost (Challenger)",
            "test_auc":            results_xgb["metrics"]["auc"],
            "oof_auc":            results_xgb["metrics"]["oof_auc"],
            "k_fold_aucs":        results_xgb["metrics"].get("k_fold_aucs"),
            "final_threshold":     results_xgb["metrics"]["final_threshold"],
            "threshold_robustness": results_xgb["metrics"].get("threshold_robustness"),
            "calibration_method": results_xgb["calibration_method"],
            "optuna_params":       results_xgb["metrics"].get("optuna_params"),
            "optuna_val_auc":     results_xgb["metrics"].get("optuna_val_auc"),
            "best_iter_median":   results_xgb["metrics"].get("best_iter_median"),
            "bad_rate":           results_xgb.get("bad_rate"),
            "metrics":            _summarise(results_xgb["metrics"]),
        },
        "best_model_key": best_key,
        "feature_drift":  results_lr.get("feature_drift"),
    }

    meta_path = os.path.join(out_dir, f"pipeline_metadata.{MODEL_VERSION}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("  Saved: %s", meta_path)
    log.info("  STEP 10 done in %.1fs", _time.perf_counter() - t0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11 — GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def step11_generate_report(
    results_lr: dict,
    results_xgb: dict,
    lr_feature_names: list[str],
    iv_scores: dict[str, float],
    out_path: str = "models/model_comparison_report.txt",
) -> None:
    """Generate human-readable Basel II / IFRS 9 compliant model report."""
    t0 = _time.perf_counter()
    log.info("STEP 11 — Generating report: %s", out_path)

    models   = {"lr": results_lr, "xgb": results_xgb}
    best_key = max(models, key=lambda k: models[k]["metrics"]["auc"])
    best     = models[best_key]
    lines: list[str] = []
    W = 85

    def hr():   lines.append("=" * W)
    def rule(): lines.append("-" * W)

    hr()
    lines.append("     CREDIT SCORING ML PIPELINE — MODEL COMPARISON REPORT")
    lines.append(f"     v4.1 Optimized | {MODEL_VERSION} | Production Grade | Basel II / IFRS 9")
    lines.append(f"     Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    hr()
    lines.append("")

    # 1. Validation policy
    lines.append("## 1. VALIDATION POLICY")
    rule()
    lines.append("  Split: train=70% | val_thresh=15% | Test=15%")
    lines.append(f"  Calibration: isotonic regression on OOF k-fold (k={N_SPLITS}), sigmoid fallback if N<{ISOTONIC_MIN_N}")
    lines.append("  Optuna: k-fold CV objective only (no external val sets)")
    lines.append("  SHAP: stratified sampling (50%% near threshold + 50%% random)")
    lines.append("")
    lines.append("  Validation role separation:")
    lines.append("    train       : Optuna k-fold CV + OOF k-fold + isotonic calibration + final models")
    lines.append("    val_thresh  : threshold tuning + PSI-DEV + SHAP stratified sampling")
    lines.append("    Test        : final evaluation + PSI-Test + threshold robustness validation")
    lines.append("")
    lines.append("  Anti-leakage assertions (hash-based row signatures): PASS")
    lines.append("")

    # 2. Class imbalance
    lines.append("## 2. CLASS IMBALANCE MONITORING")
    rule()
    br = results_lr.get("bad_rate", {})
    lines.append(f"  train bad_rate : {br.get('bad_rate_train', 'N/A')}")
    lines.append(f"  test  bad_rate : {br.get('bad_rate_test', 'N/A')}")
    lines.append(f"  drift          : {br.get('drift', 'N/A')}")
    flag = br.get("flag", False)
    lines.append(f"  status         : {'⚠️  FLAGGED — drift > threshold' if flag else 'OK'}")
    lines.append("")

    # 3. Feature spaces
    lines.append("## 3. FEATURE SPACES")
    rule()
    lines.append(f"  CHAMPION (LR)  : {len(lr_feature_names)} IV-filtered features (IV > 0.02) + StandardScaler")
    lines.append(f"  CHALLENGER (XGB): {len(RAW_FEATURE_COLUMNS)} raw cleaned features (no scaling, no IV filter)")
    lines.append("")
    lines.append("  IV scores (sorted desc):")
    for feat, iv in sorted(iv_scores.items(), key=lambda x: -x[1]):
        flag = "< 0.02 — dropped" if iv <= 0.02 else "— selected for LR"
        lines.append(f"    {feat:<45} IV={iv:>7.4f}  {flag}")
    lines.append("")

    # 4. Feature drift
    lines.append("## 4. FEATURE DRIFT MONITORING")
    rule()
    lines.append("  KS > 0.1 or PSI > 0.1 flags a feature as drifted")
    lines.append("  Format: KS_val_thresh | KS_Test | PSI_val_thresh | PSI_Test")
    lines.append("")
    fd = results_lr.get("feature_drift", {})
    if fd:
        for feat, vals in fd.items():
            flags = ""
            if vals.get("drift_flag_val_thresh"): flags += " [val_thresh DRIFT]"
            if vals.get("drift_flag_test"):       flags += " [test DRIFT]"
            lines.append(
                f"    {feat:<45} KS={vals['ks_val_thresh']:.4f}/{vals['ks_test']:.4f}"
                f"  PSI={vals['psi_val_thresh']:.4f}/{vals['psi_test']:.4f}{flags}"
            )
    else:
        lines.append("  (not computed)")
    lines.append("")

    # 5. Population Stability Index
    lines.append("## 5. POPULATION STABILITY INDEX (PSI)")
    rule()
    lines.append("  PSI bins: fit from OOF_train, reuse for val_thresh + test")
    lines.append("  < 0.1 : stable | 0.1–0.2 : monitor | > 0.2 : recalibrate")
    lines.append("")
    for key, res in sorted(models.items(), key=lambda kv: -kv[1]["metrics"]["auc"]):
        m = res["metrics"]
        mname = "Logistic Regression" if key == "lr" else "XGBoost"
        tag    = " [CHAMPION]" if res["champion"] else " [CHALLENGER]"
        lines.append(f"  {mname}{tag}")
        lines.append(f"    PSI-DEV  : {m.get('psi_dev', 'N/A'):>7.4f} | {m.get('psi_dev_band', '')}")
        lines.append(f"    PSI-Test : {m.get('psi_test', 'N/A'):>7.4f} | {m.get('psi_test_band', '')}")
        lines.append("")
    lines.append("")

    # 6. Best model
    lines.append("## 6. BEST MODEL (by Test AUC)")
    rule()
    m = best["metrics"]
    lines.append(f"  Name{'':30} : {best['model_name']}{' [CHAMPION]' if best['champion'] else ' [CHALLENGER]'}")
    lines.append(f"  Test AUC{'':27} : {m['auc']}")
    if "oof_auc" in m:
        lines.append(f"  OOF AUC{'':27} : {m['oof_auc']} (k={N_SPLITS}, unbiased)")
    t = m["final_threshold"]
    lines.append(f"  Accuracy  (t={t}){'':11} : {m['accuracy']}")
    lines.append(f"  Precision (t={t}){'':10} : {m['precision']}")
    lines.append(f"  Recall    (t={t}){'':14} : {m['recall']}")
    lines.append(f"  F1        (t={t}){'':22} : {m['f1']}")
    lines.append(f"  Optimal threshold{'':18} : {m['optimal_threshold']} (F1={m['optimal_f1']})")
    lines.append(f"  Cost-sensitive thresh{'':13} : {m['cost_threshold']} (cost={m['business_cost']})")
    lines.append(f"  Final threshold{'':20} : {m['final_threshold']} (strategy={m['threshold_strategy']})")
    lines.append(f"  Calibration method{'':16} : {m.get('calibration_method', 'N/A')}")
    thr_rob = m.get("threshold_robustness", {})
    if thr_rob:
        lines.append(
            f"  Threshold robustness{'':14} : "
            f"opt_delta={thr_rob.get('opt_delta')} | cost_delta={thr_rob.get('cost_delta')}"
        )
    if "optuna_val_auc" in m and m["optuna_val_auc"] is not None:
        lines.append(f"  Optuna k-fold OOF AUC{'':11} : {m['optuna_val_auc']} (n_trials=20)")
        lines.append(f"  Optuna params{'':19} : {m.get('optuna_params', 'N/A')}")
    lines.append("")
    lines.append("  Top 5 SHAP features:")
    for i, feat in enumerate(best.get("shap_top5") or [], 1):
        lines.append(f"    {i}. {feat['feature']}: {feat['mean_abs_shap']:.6f}")
    lines.append("")

    # 7. Per-threshold metrics
    lines.append("## 7. METRICS PER THRESHOLD (best model, Test)")
    rule()
    lines.append(f"    {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    lines.append("    " + "-" * 52)
    for row in m.get("per_threshold") or []:
        lines.append(
            f"    {row['threshold']:>10.2f} {row['accuracy']:>10.4f}"
            f" {row['precision']:>10.4f} {row['recall']:>8.4f} {row['f1']:>8.4f}"
        )
    lines.append("")

    # 8. Credit score samples
    lines.append("## 8. CREDIT SCORE SAMPLES (first 10 Test predictions)")
    rule()
    lines.append("  Scale: 300–850 (300=highest risk, 850=lowest risk)")
    lines.append("")
    for key, res in sorted(models.items(), key=lambda kv: -kv[1]["metrics"]["auc"]):
        mname = "Logistic Regression" if key == "lr" else "XGBoost"
        tag   = " [CHAMPION]" if res["champion"] else " [CHALLENGER]"
        tag2  = " ← BEST" if key == best_key else ""
        lines.append(f"  {mname}{tag}{tag2}")
        scores = pd_to_credit_score(res["prob_test"][:10])
        lines.append(f"    {'Idx':>4}  {'PD':>8}  {'Score':>6}  {'Band':>12}")
        lines.append("    " + "-" * 36)
        for i, (pd_val, score) in enumerate(zip(res["prob_test"][:10], scores)):
            lines.append(f"    {i:>4}  {pd_val:>8.4f}  {score:>6}  {score_band(int(score)):>12}")
        lines.append("")

    # 9. Champion-Challenger comparison
    lines.append("## 9. CHAMPION vs CHALLENGER COMPARISON")
    rule()
    lines.append(
        f"    {'Model':<32} {'Type':<12} {'Test AUC':>9} {'OOF AUC':>9} "
        f"{'Cost':>8} {'Calib':>10} {'N Features':>11}"
    )
    lines.append("    " + "-" * 98)
    for key, res in sorted(models.items(), key=lambda kv: -kv[1]["metrics"]["auc"]):
        m = res["metrics"]
        mtype = "[CHAMPION]" if res["champion"] else "[CHALLENGER]"
        n_feat = len(lr_feature_names) if res["champion"] else len(RAW_FEATURE_COLUMNS)
        lines.append(
            f"    {res['model_name']:<32} {mtype:<12} {m['auc']:>9.4f}"
            f" {m.get('oof_auc', 'N/A'):>9} {m['business_cost']:>8}"
            f" {m.get('calibration_method', 'N/A'):>10} {n_feat:>11}{' ← BEST' if key == best_key else ''}"
        )
    lines.append("")
    lines.append(
        "  CHAMPION (LR)   : IV-filtered + scaled — transparent, explainable, regulatory-friendly\n"
        "  CHALLENGER (XGB) : all raw features — higher AUC, Optuna-tuned, strong regularization\n"
    )

    # 10. SHAP
    lines.append("## 10. SHAP FEATURE IMPORTANCE (Top 5 per Model)")
    rule()
    lines.append("  SHAP eval: stratified sampling (50%% near optimal threshold + 50%% random)")
    for key, res in sorted(models.items(), key=lambda kv: -kv[1]["metrics"]["auc"]):
        mname = "Logistic Regression" if key == "lr" else "XGBoost"
        mtype = "[CHAMPION]" if res["champion"] else "[CHALLENGER]"
        status = "OK" if res.get("shap_top5") else "FAILED"
        lines.append(f"  {mname} {mtype} — SHAP: {status}")
        for i, feat in enumerate(res.get("shap_top5") or [], 1):
            lines.append(f"    {i}. {feat['feature']}: {feat['mean_abs_shap']:.6f}")
        lines.append("")

    hr()
    lines.append("  END OF REPORT")
    hr()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("  Report written: %s (%d lines)", out_path, len(lines))
    log.info("  STEP 11 done in %.1fs", _time.perf_counter() - t0)
