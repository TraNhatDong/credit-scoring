"""
run_pipeline.py — Orchestrator (pure, ~170 lines).

Calls one function from each module in the correct dependency order.
No business logic lives here.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── _config — imports, logging, constants, seeds, utilities ──────────────────────
from ._config import MODEL_VERSION, RANDOM_STATE, log

from ._transform import Preprocessor

from ._train import train_pipeline

from ._monitor import step10_save_artifacts, step11_generate_report, step9_monitoring


def main() -> None:
    pipeline_t0 = __import__("time").perf_counter()

    log.info("=" * 72)
    log.info("  CREDIT SCORING PIPELINE %s — v4.1 Optimized | Production Grade", MODEL_VERSION)
    log.info("  random_state=%d | strict anti-leakage | Basel II / IFRS 9", RANDOM_STATE)
    log.info("  Split: train=70%% | val_thresh=15%% | Test=15%%")
    log.info("=" * 72)

    # Resolve working directory to project root
    script_dir  = Path(__file__).parent.resolve()
    project_dir = script_dir.parent.parent
    os.chdir(project_dir)
    log.info("  Working directory: %s", project_dir)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    from ._config import step1_load_data, step2_split
    df = step1_load_data("data/cs-training.csv")
    X_train, X_val_thresh, X_test, y_train, y_val_thresh, y_test = step2_split(df)

    # ── Step 3-5: Fit Preprocessor → transform all sets ───────────────────────
    preprocessor = Preprocessor().fit(X_train)
    log.info("STEP 3-5 — Preprocessor fit: median_values=%d keys | p99_debt_ratio=%.4f",
             len(preprocessor.median_values), preprocessor.p99_debt_ratio)

    X_train      = preprocessor.transform(X_train)
    X_val_thresh = preprocessor.transform(X_val_thresh)
    X_test        = preprocessor.transform(X_test)
    log.info("  Preprocessing complete — train.shape=%s | val_thresh.shape=%s | Test.shape=%s",
             X_train.shape, X_val_thresh.shape, X_test.shape)

    # ── Step 6-8: Training pipeline ───────────────────────────────────────────
    results = train_pipeline(
        X_train, y_train,
        X_val_thresh, y_val_thresh,
        X_test, y_test,
        preprocessor,
    )

    results_lr  = results["lr"]
    results_xgb = results["xgb"]
    lr_feature_names = results["lr_feature_names"]
    iv_scores   = results["iv_scores"]

    # ── Step 9: PSI + Feature Drift + Class Imbalance ────────────────────────
    step9_monitoring(
        results_lr, results_xgb,
        X_train, X_val_thresh, X_test,
        y_train, y_test,
        iv_scores,
        prob_val_lr=results_lr["prob_thresh"],
        prob_val_xgb=results_xgb["prob_thresh"],
    )

    # ── Step 10: Save artifacts ──────────────────────────────────────────────
    step10_save_artifacts(
        results_lr, results_xgb,
        lr_feature_names, iv_scores,
        preprocessor,
    )

    # ── Step 11: Generate report ─────────────────────────────────────────────
    step11_generate_report(results_lr, results_xgb, lr_feature_names, iv_scores)

    # ── Final summary ────────────────────────────────────────────────────────
    total  = __import__("time").perf_counter() - pipeline_t0
    models_dict = {"lr": results_lr, "xgb": results_xgb}
    best_key = max(models_dict, key=lambda k: models_dict[k]["metrics"]["auc"])
    best_model_name = models_dict[best_key]["model_name"]

    log.info("")
    log.info("=" * 72)
    log.info("  PIPELINE COMPLETE — %s (total: %.1fs)", MODEL_VERSION, total)
    champion_key = best_key
    challenger_key = "xgb" if best_key == "lr" else "lr"
    champ_name = models_dict[champion_key]["model_name"]
    chal_name = models_dict[challenger_key]["model_name"]
    log.info("  Champion (%s)   Test AUC : %.4f | OOF AUC : %.4f | calib: %s",
             champ_name,
             models_dict[champion_key]["metrics"]["auc"],
             models_dict[champion_key]["metrics"]["oof_auc"],
             models_dict[champion_key]["calibration_method"])
    log.info("  Challenger (%s) Test AUC: %.4f | OOF AUC: %.4f | calib: %s",
             chal_name,
             models_dict[challenger_key]["metrics"]["auc"],
             models_dict[challenger_key]["metrics"]["oof_auc"],
             models_dict[challenger_key]["calibration_method"])
    log.info("  Best model: %s", best_model_name)
    log.info("  PSI-DEV (LR) : %.4f | PSI-Test (LR) : %.4f",
             results_lr["metrics"].get("psi_dev", "N/A"),
             results_lr["metrics"].get("psi_test", "N/A"))
    log.info("  PSI-DEV (XGB): %.4f | PSI-Test (XGB): %.4f",
             results_xgb["metrics"].get("psi_dev", "N/A"),
             results_xgb["metrics"].get("psi_test", "N/A"))
    br = results_lr.get("bad_rate", {})
    log.info("  Class drift   : bad_rate_train=%.4f | bad_rate_test=%.4f | drift=%.4f | flag=%s",
             br.get("bad_rate_train", 0), br.get("bad_rate_test", 0),
             br.get("drift", 0), br.get("flag", False))
    log.info("  Artifacts      : models/")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
