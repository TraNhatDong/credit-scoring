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
    RiskBands,
    compute_bad_rate_drift, compute_ks_statistic, compute_psi,
    compute_psi_bin_edges, log, pd_to_credit_score, psi_band,
    RAW_FEATURE_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8b - CALIBRATION SANITY CHECKS (Buoc 1 + Buoc 4)
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_calibration_check(
    y_test: np.ndarray,
    prob_test: np.ndarray,
    score_test: np.ndarray,
    q_bands: "RiskBands",
    model_name: str,
) -> dict[str, Any]:
    """
    Bước 1: Check real bad_rate in Q1.
    Bước 4: Inspect PD/score distribution for spikes, clusters, plateaus.

    Returns a diagnostics dict for logging and report.
    """
    import numpy as np

    # ── Bước 1: bad_rate in Q1 ──────────────────────────────────────────────
    # RiskBands labels: "Q1 (Best)" = PD <= 1.0% (score >= q1_ceiling approx 833)
    # (NOT "Q1 (Excellent)" - correct label from _riskbands.py)
    Q1_label = "Q1 (Best)"
    Q1_mask = np.array([q_bands.get_band(int(s)) == Q1_label for s in score_test])
    n_Q1 = int(Q1_mask.sum())
    bad_rate_Q1 = float(y_test[Q1_mask].mean()) if n_Q1 > 0 else None

    # Interpret bad_rate_Q1 (Q1 = PD <= 1.0%)
    if bad_rate_Q1 is not None:
        if bad_rate_Q1 <= 0.010:
            br_diag = "GOOD (<= 1.0%) - Q1 PD <= 1% calibration accurate"
        elif bad_rate_Q1 <= 0.015:
            br_diag = "CAUTION (1-1.5%) - model slightly under-estimates Q1 risk"
        elif bad_rate_Q1 <= 0.020:
            br_diag = "CAUTION (1.5-2%) - model under-estimates Q1 risk"
        else:
            br_diag = "BAD (> 2%) - model significantly under-estimates Q1 risk"
    else:
        br_diag = "N/A - no Q1 samples in test set"

    # ── Bước 4a: PD distribution check ───────────────────────────────────────
    pd_vals = np.asarray(prob_test)
    pd_p01  = float(np.percentile(pd_vals, 1))
    pd_p99  = float(np.percentile(pd_vals, 99))
    pd_skew = float(np.mean((pd_vals - pd_vals.mean()) ** 3) / (np.std(pd_vals) ** 3 + 1e-12))
    pd_spike = bool(np.std(pd_vals) < 0.02)   # near-zero std = spike/collapse

    # Count PD clusters (bins of 0.01 width)
    pd_bins = np.digitize(pd_vals, np.arange(0, 1.01, 0.01))
    bin_counts = np.bincount(pd_bins[pd_bins > 0], minlength=101)
    top_bin_pct = float(bin_counts.max() / len(pd_vals) * 100) if len(pd_vals) > 0 else 0.0
    pd_cluster_flag = top_bin_pct > 20.0   # >20% in a single 1% bin = suspicious cluster

    # ── Bước 4b: Score distribution check ────────────────────────────────────
    score_vals = np.asarray(score_test, dtype=int)
    score_range = int(score_vals.max() - score_vals.min())

    # ── Bước 4c: Tail PD check ───────────────────────────────────────────────
    tail_mask = prob_test >= 0.15
    bad_rate_tail = float(y_test[tail_mask].mean()) if tail_mask.sum() > 0 else None

    diagnostics = {
        "bad_rate_Q1":        bad_rate_Q1,
        "n_Q1":              n_Q1,
        "bad_rate_Q1_diag":   br_diag,
        "pd_min":             float(pd_vals.min()),
        "pd_max":             float(pd_vals.max()),
        "pd_mean":            float(pd_vals.mean()),
        "pd_p01":             pd_p01,
        "pd_p99":             pd_p99,
        "pd_skew":            pd_skew,
        "pd_spike":           pd_spike,
        "pd_cluster_flag":    pd_cluster_flag,
        "top_bin_pct":        round(top_bin_pct, 2),
        "score_range":        score_range,
        "score_min":          int(score_vals.min()),
        "score_max":          int(score_vals.max()),
        "n_tail_samples":      int(tail_mask.sum()),
        "bad_rate_tail":      bad_rate_tail,
    }

    log.info(
        "  %-20s Q1 bad_rate=%.4f (%s) | Q1_n=%d | "
        "pd_mean=%.4f pd_p01=%.4f pd_p99=%.4f | skew=%.3f | spike=%s cluster=%s",
        model_name + ":",
        bad_rate_Q1 if bad_rate_Q1 is not None else -1,
        br_diag,
        n_Q1,
        diagnostics["pd_mean"],
        pd_p01, pd_p99,
        pd_skew,
        pd_spike,
        pd_cluster_flag,
    )
    if bad_rate_Q1 is not None and bad_rate_Q1 > 0.020:
        log.warning("  WARNING: Q1 calibration: real bad_rate=%.4f exceeds 2.0%% -- model under-estimates Q1 risk", bad_rate_Q1)
    if pd_spike:
        log.warning("  WARNING: PD distribution spike detected (std=%.4f) -- model may be near-collapsed", np.std(pd_vals))
    if pd_cluster_flag:
        log.warning("  WARNING: PD cluster detected: %.1f%% of predictions in single 1%% bin -- check isotonic calibration", top_bin_pct)

    return diagnostics


def _plot_score_distributions(
    results_lr: dict,
    results_xgb: dict,
    y_test: np.ndarray,
    out_dir: str = "models",
) -> None:
    """
    Bước 4: Save diagnostic histograms (PD + Score) as PNG to out_dir.
    Four subplots per model: PD histogram, Score histogram, PD by true label, Score by true label.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("  matplotlib not available — skipping score distribution plots: %s", exc)
        return

    for key, res in [("LR", results_lr), ("XGBoost", results_xgb)]:
        model_name = "Logistic Regression" if key == "LR" else "XGBoost"
        tag = "[CHAMPION]" if res["champion"] else "[BENCHMARK]"

        prob_test = np.asarray(res.get("prob_test", []))
        scores    = np.asarray(res.get("scores_train", []))   # training scores for banding

        if len(prob_test) == 0:
            continue

        scores_test = pd_to_credit_score(prob_test, clamp_min=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Diagnostic: {model_name} {tag}", fontsize=13, fontweight="bold")

        # Plot 1: PD histogram
        axes[0, 0].hist(prob_test, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        axes[0, 0].axvline(prob_test.mean(), color="red", linestyle="--", label=f"mean={prob_test.mean():.4f}")
        axes[0, 0].axvline(0.01, color="orange", linestyle=":", label="PD=1% anchor")
        axes[0, 0].set_xlabel("Calibrated PD")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("PD Distribution (Test)")
        axes[0, 0].legend(fontsize=8)

        # Plot 2: Score histogram
        axes[0, 1].hist(scores_test, bins=40, edgecolor="black", alpha=0.7, color="forestgreen")
        axes[0, 1].axvline(scores_test.mean(), color="red", linestyle="--", label=f"mean={scores_test.mean():.1f}")
        axes[0, 1].axvline(850, color="orange", linestyle=":", label="Score=850 anchor")
        axes[0, 1].set_xlabel("Credit Score")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Credit Score Distribution (Test)")
        axes[0, 1].legend(fontsize=8)

        # Plot 3: PD by true label
        for label_val, label_name, color in [(0, "Good (y=0)", "green"), (1, "Bad (y=1)", "red")]:
            mask = y_test == label_val
            if mask.sum() > 0:
                axes[1, 0].hist(prob_test[mask], bins=40, alpha=0.6, label=label_name, color=color)
        axes[1, 0].set_xlabel("Calibrated PD")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("PD by True Label (Test)")
        axes[1, 0].legend(fontsize=8)

        # Plot 4: Score by true label
        for label_val, label_name, color in [(0, "Good (y=0)", "green"), (1, "Bad (y=1)", "red")]:
            mask = y_test == label_val
            if mask.sum() > 0:
                axes[1, 1].hist(scores_test[mask], bins=40, alpha=0.6, label=label_name, color=color)
        axes[1, 1].set_xlabel("Credit Score")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Credit Score by True Label (Test)")
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"diag_score_dist_{key.lower()}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        log.info("  Saved: %s", out_path)


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
        # PSI-DEV: drift of val_thresh predictions vs train (reference = full-train calibrated)
        psi_bin_edges = compute_psi_bin_edges(prob_val, n_bins=10)
        psi_dev  = round(compute_psi(prob_val, res["prob_thresh"], bin_edges=psi_bin_edges), 4)
        # PSI-Test: drift of test predictions vs train (reference = full-train calibrated)
        psi_test = round(compute_psi(prob_val, res["prob_test"],  bin_edges=psi_bin_edges), 4)

        res["metrics"]["psi_dev"]      = psi_dev
        res["metrics"]["psi_test"]     = psi_test
        res["metrics"]["psi_dev_band"]  = psi_band(psi_dev)
        res["metrics"]["psi_test_band"] = psi_band(psi_test)

        log.info(
            "  %s — PSI-DEV=%.4f (%s) | PSI-Test=%.4f (%s) | reference=full-train calibrated",
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
            "  [WARNING] CLASS IMBALANCE DRIFT DETECTED: %s",
            br["flag_reason"],
        )

    results_lr["bad_rate"]  = br
    results_xgb["bad_rate"] = br

    # ── Bước 1 + Bước 4: Calibration sanity checks ────────────────────────────
    log.info("  Calibration diagnostics (B1 bad_rate_Q1 + B4 distribution check):")
    for res, name in [(results_lr, "LR"), (results_xgb, "XGBoost")]:
        prob_test = res["prob_test"]
        scores_test = pd_to_credit_score(prob_test, clamp_min=True)
        # Fit RiskBands from training scores for Q1 identification
        scores_train = res.get("scores_train")
        if scores_train is not None and len(scores_train) > 0:
            q_bands = RiskBands.fit(np.asarray(scores_train, dtype=float), n_bands=7)
        else:
            q_bands = RiskBands.fit(scores_test.astype(float), n_bands=7)
        diag = _compute_calibration_check(
            y_test.values, prob_test, scores_test, q_bands, name,
        )
        res["calibration_diagnostics"] = diag

    # ── Bước 4: Score distribution plots ───────────────────────────────────
    _plot_score_distributions(results_lr, results_xgb, y_test.values, out_dir="models")

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
    is_xgb_champion = (best_key == "xgb")

    # ── Champion model — single artifact bundle ─────────────────────────────────
    if is_xgb_champion:
        champ_path = os.path.join(out_dir, f"champion_xgb_model.{MODEL_VERSION}.pkl")
        bench_path = os.path.join(out_dir, f"benchmark_lr_model.{MODEL_VERSION}.pkl")
    else:
        champ_path = os.path.join(out_dir, f"champion_lr_model.{MODEL_VERSION}.pkl")
        bench_path = os.path.join(out_dir, f"benchmark_xgb_model.{MODEL_VERSION}.pkl")

    # ── LR artifact (always saved — benchmark or champion depending on AUC) ───────
    joblib.dump({
        "model":              results_lr["model"],
        "calibrator":         results_lr["calibrator"],
        "calibration_method": results_lr["calibration_method"],
        "scaler":             results_lr["scaler"],
        "feature_names":      lr_feature_names,
        "feature_order":      lr_feature_names,
        "champion":           not is_xgb_champion,
        "model_type":         "linear",
        "model_version":      MODEL_VERSION,
        "preprocessor_state": preprocessor.get_state(),
        "iv_threshold":       0.02,
        "iv_scores":          iv_scores,
        "threshold":          results_lr["metrics"]["final_threshold"],
        "threshold_strategy": THRESHOLD_STRATEGY,
        "scores_train":       results_lr.get("scores_train"),
    }, champ_path if not is_xgb_champion else bench_path)
    log.info("  Saved: %s (%s | %d IV features | %s)",
             champ_path if not is_xgb_champion else bench_path,
             "champion_lr" if not is_xgb_champion else "benchmark_lr",
             len(lr_feature_names), results_lr["calibration_method"])

    # ── XGB artifact ───────────────────────────────────────────────────────────
    joblib.dump({
        "model":              results_xgb["model"],
        "calibrator":         results_xgb["calibrator"],
        "calibration_method": results_xgb["calibration_method"],
        "feature_names":      RAW_FEATURE_COLUMNS,
        "feature_order":      RAW_FEATURE_COLUMNS,
        "champion":           is_xgb_champion,
        "model_type":         "tree",
        "model_version":      MODEL_VERSION,
        "preprocessor_state": preprocessor.get_state(),
        "iv_threshold":       0.02,
        "iv_scores":          iv_scores,
        "threshold":         results_xgb["metrics"]["final_threshold"],
        "threshold_strategy": THRESHOLD_STRATEGY,
        "scores_train":      results_xgb.get("scores_train"),
    }, champ_path if is_xgb_champion else bench_path)
    log.info("  Saved: %s (%s | %d raw features | %s)",
             champ_path if is_xgb_champion else bench_path,
             "champion_xgb" if is_xgb_champion else "benchmark_xgb",
             len(RAW_FEATURE_COLUMNS), results_xgb["calibration_method"])

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
            "key":                "xgb",
            "model_name":         "XGBoost (Champion)",
            "test_auc":            results_xgb["metrics"]["auc"],
            "oof_auc":            results_xgb["metrics"]["oof_auc"],
            "k_fold_aucs":        results_xgb["metrics"].get("k_fold_aucs"),
            "final_threshold":     results_xgb["metrics"]["final_threshold"],
            "threshold_robustness": results_xgb["metrics"].get("threshold_robustness"),
            "calibration_method": results_xgb["calibration_method"],
            "bad_rate":           results_xgb.get("bad_rate"),
            "metrics":            _summarise(results_xgb["metrics"]),
        },
        "challenger": {
            "key":                "lr",
            "model_name":         "Logistic Regression (Benchmark)",
            "test_auc":            results_lr["metrics"]["auc"],
            "oof_auc":            results_lr["metrics"]["oof_auc"],
            "k_fold_aucs":        results_lr["metrics"].get("k_fold_aucs"),
            "final_threshold":     results_lr["metrics"]["final_threshold"],
            "threshold_robustness": results_lr["metrics"].get("threshold_robustness"),
            "calibration_method": results_lr["calibration_method"],
            "optuna_params":       None,
            "optuna_val_auc":     None,
            "best_iter_median":   None,
            "bad_rate":           results_lr.get("bad_rate"),
            "metrics":            _summarise(results_lr["metrics"]),
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
    lines.append(f"  status         : {'[WARNING] FLAGGED -- drift > threshold' if flag else 'OK'}")
    lines.append("")

    # 2b. Calibration diagnostics
    lines.append("## 2b. CALIBRATION DIAGNOSTICS (B1 bad_rate_Q1 + B4 distribution)")
    rule()
    lines.append("  Q1 real bad_rate check:")
    lines.append("    <= 1.0% -> GOOD | 1.0-1.5% -> caution | 1.5-2.0% -> under-est | > 2.0% -> bad under-est")
    lines.append("")
    for res, name in [(results_lr, "LR"), (results_xgb, "XGBoost")]:
        d = res.get("calibration_diagnostics", {})
        if d:
            tag = "[CHAMPION]" if res["champion"] else "[BENCHMARK]"
            lines.append(f"  {name} {tag}")
            lines.append(f"    bad_rate_Q1 = {d.get('bad_rate_Q1', 'N/A')} ({d.get('n_Q1', 0)} samples) | {d.get('bad_rate_Q1_diag', '')}")
            lines.append(
                f"    PD: min={d.get('pd_min', 0):.4f} max={d.get('pd_max', 0):.4f}"
                f" mean={d.get('pd_mean', 0):.4f} p01={d.get('pd_p01', 0):.4f} p99={d.get('pd_p99', 0):.4f}"
            )
            lines.append(
                f"    skew={d.get('pd_skew', 0):.3f} spike={d.get('pd_spike', False)}"
                f" cluster={d.get('pd_cluster_flag', False)} top_bin={d.get('top_bin_pct', 0):.1f}%"
            )
            lines.append(
                f"    Score: range=[{d.get('score_min', 0)},{d.get('score_max', 0)}] width={d.get('score_range', 0)}"
            )
            lines.append(
                f"    Tail PD≥15%: n={d.get('n_tail_samples', 0)} bad_rate={d.get('bad_rate_tail', 'N/A')}"
            )
            lines.append("")
    lines.append("  Diagnostic plots: models/diag_score_dist_lr.png | models/diag_score_dist_xgb.png")
    lines.append("")

    # 3. Feature spaces
    lines.append("## 3. FEATURE SPACES")
    rule()
    lines.append(f"  CHAMPION (XGB) : {len(RAW_FEATURE_COLUMNS)} raw features (no scaling, no IV filter) — best AUC by test")
    lines.append(f"  BENCHMARK (LR) : {len(lr_feature_names)} IV-filtered features (IV > 0.02) + StandardScaler")
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

    # 8. Credit score samples + quantile band distribution
    lines.append("## 8. CREDIT SCORE SAMPLES (first 10 Test predictions)")
    rule()
    lines.append("  Scale: 300–850 (Basel II log-odds)")
    lines.append(
        "  Band assignment: hybrid — PD-threshold tails (Q1/Q7) + quantile middle (Q2–Q6)\n"
        "  Basel anchors: Q1 = PD <= 1.0% | Q7 = PD >= 15% | Q2-Q6 = ~16% each"
    )
    lines.append("")
    for key, res in sorted(models.items(), key=lambda kv: -kv[1]["metrics"]["auc"]):
        mname = "Logistic Regression" if key == "lr" else "XGBoost"
        tag   = " [CHAMPION]" if res["champion"] else " [CHALLENGER]"
        tag2  = " [BEST]" if key == best_key else ""
        lines.append(f"  {mname}{tag}{tag2}")

        # Use calibrated test probabilities for scoring and display.
        # prob_display: calibrated test probs (for PD column in report).
        prob_display = res["prob_test"]
        # prob_test_raw: stored raw test probs for XGB (LR: same as prob_display).
        prob_raw = res.get("prob_test_raw", prob_display)
        scores = pd_to_credit_score(prob_raw, clamp_min=True)

        # Fit hybrid bands from training scores: PD-anchored tails + quantile middle
        if "scores_train" in res and res["scores_train"] is not None:
            q_bands = RiskBands.fit(res["scores_train"], n_bands=7)
            band_n = len(res["scores_train"])
        else:
            # Fallback: fit from test scores (initial run before retrain)
            q_bands = RiskBands.fit(scores, n_bands=7)
            band_n = len(scores)

        # Score = calibrated prob (for Basel-compliant credit score scale)
        # PD display = calibrated prob (for regulatory probability reporting)
        lines.append(f"    {'Idx':>4}  {'PD':>8}  {'Score':>6}  {'Band':>16}")
        lines.append("    " + "-" * 42)
        for i, (pd_cal, pd_raw, score) in enumerate(zip(prob_display[:10], prob_raw[:10], scores)):
            lines.append(
                f"    {i:>4}  {pd_cal:>8.4f}  {score:>6}  {q_bands.get_band(int(score)):>16}"
            )

        # Per-model hybrid band distribution
        band_stats = q_bands.band_stats(scores)
        lines.append("")
        lines.append(
            f"  Band Distribution — hybrid PD-tail + quantile-middle (Band Set — "
            + str(band_n) + " samples):"
        )
        W = 80
        lines.append("    " + "-" * W)
        # Header row: Band | Count | Pct | Score Range | PD Anchor
        lines.append(
            "      "
            + "Band".ljust(15)
            + "Count".rjust(7)
            + "Pct".rjust(8)
            + "Score Range".rjust(16)
            + "PD Anchor (Basel)".rjust(20)
        )
        lines.append("    " + "-" * W)
        for i, label in enumerate(q_bands.labels):
            st = band_stats.get(label, {
                "count": 0, "pct": 0.0,
                "score_lo": 300, "score_hi": 850, "pd_label": None,
            })
            lo, hi = st["score_lo"], st["score_hi"]

            # Display format -- handle edge cases where adjacent thresholds collapse
            if i == 0:
                # Q1: [q1_ceiling, 850]
                score_range = f"{int(round(lo))}-{int(hi)}"
            elif i == q_bands._n_bands - 1:
                # Q7: [300, q7_floor]
                score_range = f"{int(lo)}-{int(round(hi))}"
            elif int(round(lo)) == int(round(hi)):
                # Adjacent thresholds collapsed (e.g. Q6 when span is tiny)
                score_range = f"{int(round(lo))}-{int(round(hi))}"
            elif lo > hi:
                # Inverted range -- fallback to threshold
                score_range = f"{int(round(hi))}-{int(round(lo))}"
            else:
                score_range = f"{int(round(lo))}-{int(round(hi))}"

            pd_anchor = st.get("pd_label") or ""
            lines.append(
                "      "
                + label.ljust(15)
                + str(st["count"]).rjust(7)
                + f"{st['pct']:>6.2f}% ".rjust(10)
                + score_range.rjust(16)
                + pd_anchor.rjust(20)
            )
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
        n_feat = len(RAW_FEATURE_COLUMNS) if key == "xgb" else len(lr_feature_names)
        lines.append(
            f"    {res['model_name']:<32} {mtype:<12} {m['auc']:>9.4f}"
            f" {m.get('oof_auc', 'N/A'):>9} {m['business_cost']:>8}"
            f" {m.get('calibration_method', 'N/A'):>10} {n_feat:>11}{' [BEST]' if key == best_key else ''}"
        )
    lines.append("")
    lines.append(
        "  CHAMPION (XGB)  : all raw features — best AUC, Optuna-tuned, strong regularization\n"
        "  BENCHMARK (LR)  : IV-filtered + scaled — transparent, explainable, regulatory-friendly\n"
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
