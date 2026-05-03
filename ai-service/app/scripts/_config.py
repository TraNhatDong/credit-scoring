"""
_config.py — Constants, imports, logging, seeds, and all shared utility functions.
Shared across training pipeline and inference wrapper.
"""
from __future__ import annotations

import logging
import sys
import warnings
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# ── Optional: optuna ────────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "optuna not installed — XGBoost will use default hyperparameters. "
        "Install with: pip install optuna",
        category=ImportWarning,
        stacklevel=2,
    )

warnings.filterwarnings("ignore", category=FutureWarning, module="shap")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

# ── Global configuration ────────────────────────────────────────────────────────
RANDOM_STATE = 42


def _set_all_seeds(seed: int = RANDOM_STATE) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost as xgb_lib
        xgb_lib.set_random_state(seed)
    except Exception:
        pass


_set_all_seeds(RANDOM_STATE)

# Business cost model
COST_FP = 5.0   # Approve a defaulting customer (False Positive)
COST_FN = 1.0   # Reject a good customer (False Negative)

# Hyperparameter search
OPTUNA_N_TRIALS = 20
N_SPLITS = 5            # k-fold CV for OOF + Optuna objective
SHAP_EVAL_SIZE = 2000
XGB_EARLY_STOP = 50     # early stopping rounds per fold
ISOTONIC_MIN_N = 500    # minimum N for isotonic regression; fallback sigmoid below

# Feature set — 16 total (10 original + 6 row-wise engineered)
RAW_FEATURE_COLUMNS: list[str] = [
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
    # Engineered features (row-wise only)
    "monthly_debt",
    "late_payments_total",
    "debt_per_person",
    "income_per_credit_line",
    "utilization_per_line",
    "is_income_missing",
]

IV_THRESHOLD     = 0.02    # minimum IV to keep a feature in LR (business decision)
THRESHOLD_STRATEGY = "cost"   # "cost" (business-driven) or "f1" (F1-maximising)
THRESHOLDS = [0.30, 0.40, 0.50, 0.60]

# Model version tag
MODEL_VERSION = f"v4.1_{date.today().strftime('%Y%m%d')}"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — ROW UNIQUE ID (for leakage assertions)
# ═══════════════════════════════════════════════════════════════════════════════
def _row_hash(df: pd.DataFrame) -> set[tuple]:
    """
    Return a hashable set of unique row signatures from a DataFrame.

    Each signature is (original_index, row_bytes) so rows with identical values but
    different indices are always distinct — preventing false-negative leakage assertions.
    """
    idx_arr = df.index.values
    row_arr = df.values
    return {(int(idx_arr[i]), row_arr[i].tobytes()) for i in range(len(df))}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — INFORMATION VALUE (IV)
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_iv_single_feature(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Weight-of-Evidence based Information Value for a single feature.

    IV = Σ (Distribution_Good_i - Distribution_Bad_i)
        × ln(Distribution_Good_i / Distribution_Bad_i)

    Must be computed on CLEANED data (post-imputation, post-clipping).
    Only uses train set statistics.
    """
    df = pd.DataFrame({"feature": feature.values, "target": target.values})
    df = df.dropna()

    try:
        bins = pd.qcut(df["feature"], q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.qcut(df["feature"].rank(method="first"), q=n_bins, duplicates="drop")

    total_good = (df["target"] == 0).sum()
    total_bad  = (df["target"] == 1).sum()

    if total_good == 0 or total_bad == 0:
        return 0.0

    grouped = df.groupby(bins, observed=True)["target"].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    grouped["pct_good"] = (grouped["good"] / total_good).replace(0, 1e-6)
    grouped["pct_bad"]  = (grouped["bad"]  / total_bad).replace(0, 1e-6)

    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"]  = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    return float(grouped["iv"].sum())


def compute_iv_scores(
    X: pd.DataFrame,
    y: pd.Series,
    feature_list: list[str],
) -> dict[str, float]:
    """Compute IV for each feature. Returns {feature_name: iv_value}."""
    results: dict[str, float] = {}
    for feat in feature_list:
        if feat not in X.columns:
            log.warning("  IV: feature '%s' not found — skipping", feat)
            continue
        results[feat] = round(_compute_iv_single_feature(X[feat], y), 6)
    return results


def filter_by_iv(
    iv_scores: dict[str, float],
    threshold: float = 0.02,
) -> list[str]:
    """Return features with IV strictly greater than threshold."""
    return [f for f, iv in iv_scores.items() if iv > threshold]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — POPULATION STABILITY INDEX (PSI)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bin_edges: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (Actual_pct_i - Expected_pct_i) × ln(Actual_pct_i / Expected_pct_i)

    If bin_edges is provided, reuse those edges (fit from reference = expected).
    Otherwise fit bins from the combined distribution of expected + actual.
    """
    expected = np.asarray(expected, copy=True).flatten()
    actual   = np.asarray(actual,   copy=True).flatten()

    if bin_edges is None:
        combined = np.concatenate([expected, actual])
        try:
            edges = np.unique(np.percentile(combined, np.linspace(0, 100, n_bins + 1)))
        except (ValueError, IndexError):
            return float("nan")
    else:
        edges = bin_edges

    exp_cnt = np.histogram(expected, bins=edges)[0]
    act_cnt = np.histogram(actual,   bins=edges)[0]

    exp_pct = np.where(exp_cnt / len(expected) == 0, 1e-6, exp_cnt / len(expected))
    act_pct = np.where(act_cnt / len(actual)   == 0, 1e-6, act_cnt / len(actual))

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_psi_bin_edges(expected: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute bin edges from expected (reference) distribution for PSI reuse."""
    expected = np.asarray(expected, copy=True).flatten()
    try:
        return np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
    except (ValueError, IndexError):
        raise ValueError("Cannot compute PSI bin edges from expected distribution")


def psi_band(psi_val: float) -> str:
    """Return human-readable PSI interpretation."""
    if psi_val < 0.1:  return "No shift (< 0.1)"
    if psi_val < 0.2:  return "Minor shift (0.1–0.2) — monitor"
    return "SIGNIFICANT SHIFT (> 0.2) — recalibrate"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — KOLMOGOROV-SMIRNOV TEST
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ks_statistic(
    train_feature: np.ndarray,
    val_feature: np.ndarray,
) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    stat, _ = ks_2samp(train_feature, val_feature)
    return float(stat)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — CLASS IMBALANCE MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
def compute_bad_rate_drift(
    y_train: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compute bad rate (positive rate) drift between train and test.
    Returns drift value and whether it exceeds the threshold.
    """
    br_train = float(np.mean(y_train))
    br_test  = float(np.mean(y_test))
    drift    = abs(br_test - br_train)
    return {
        "bad_rate_train":  round(br_train, 6),
        "bad_rate_test":   round(br_test, 6),
        "drift":           round(drift, 6),
        "threshold":       threshold,
        "flag":            drift > threshold,
        "flag_reason":     f"drift={drift:.4f} > threshold={threshold}" if drift > threshold else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
class _PlattScaler:
    """
    Platt (sigmoid) calibration: p_calibrated = sigmoid(A * log_odds_raw + B).

    Fit via maximum-likelihood on OOF predictions (scipy.optimize.minimize).
    Both A and B are optimised to maximise likelihood of true labels.
    Works correctly even for small datasets (no internal CV required).
    """

    def __init__(self):
        self.A: float | None = None
        self.B: float | None = None
        self._is_fitted = False

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "_PlattScaler":
        from scipy.optimize import minimize

        raw_proba = np.asarray(raw_proba, dtype=float).clip(1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=float)

        def neg_logloss(coeffs: np.ndarray) -> float:
            A, B = float(coeffs[0]), float(coeffs[1])
            log_odds = np.log(raw_proba / (1 - raw_proba))
            lp = A * log_odds + B
            p_cal = 1 / (1 + np.exp(-lp))
            p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))

        result = minimize(neg_logloss, x0=[1.0, 0.0], method="L-BFGS-B")
        self.A, self.B = float(result.x[0]), float(result.x[1])
        self._is_fitted = True
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("_PlattScaler must be fitted before predict")
        raw = np.asarray(raw_proba, dtype=float).clip(1e-6, 1 - 1e-6)
        log_odds = np.log(raw / (1 - raw))
        return (1 / (1 + np.exp(-(self.A * log_odds + self.B)))).clip(0.0, 1.0)


def build_calibrator(
    oof_proba: np.ndarray,
    y_train: np.ndarray,
    use_isotonic: bool | None = None,
) -> tuple[Any, str]:
    """
    Build a calibrator for a model using OOF predictions.

    Uses IsotonicRegression if len(oof_proba) >= ISOTONIC_MIN_N,
    otherwise falls back to _PlattScaler (maximum-likelihood sigmoid calibration).

    Returns (calibrator, method_name).
    """
    from sklearn.isotonic import IsotonicRegression

    if use_isotonic is None:
        use_isotonic = len(oof_proba) >= ISOTONIC_MIN_N

    if use_isotonic:
        calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        calibrator.fit(oof_proba, y_train)
        return calibrator, "isotonic"
    else:
        log.warning(
            "  OOF N=%d < ISOTONIC_MIN_N=%d — falling back to Platt sigmoid calibration",
            len(oof_proba), ISOTONIC_MIN_N,
        )
        calibrator = _PlattScaler()
        calibrator.fit(oof_proba, y_train)
        return calibrator, "sigmoid"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — CREDIT SCORE SCALING
# ═══════════════════════════════════════════════════════════════════════════════
def pd_to_credit_score(
    pd_values: np.ndarray,
    min_score: int = 300,
    max_score: int = 850,
    target_min_pd: float = 0.02,
    target_max_pd: float = 0.40,
) -> np.ndarray:
    """
    Convert Probability of Default (PD) to a consumer credit score.

    Calibrated so:
      PD =  2%  → Score = 850 (Excellent)
      PD = 40%  → Score = 300 (Very Poor)
    """
    pd_values = np.asarray(pd_values, dtype=float).clip(1e-6, 1 - 1e-6)

    lo_max = np.log((1 - target_min_pd) / target_min_pd)
    lo_min = np.log((1 - target_max_pd) / target_max_pd)

    grade_range = max_score - min_score
    log_range   = lo_max - lo_min

    log_odds = np.log((1 - pd_values) / pd_values)
    scores   = max_score - grade_range * (log_odds - lo_max) / log_range
    return np.round(scores).clip(min_score, max_score).astype(int)


def score_band(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 550: return "Poor"
    return "Very Poor"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1 at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(threshold, 2),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def find_best_threshold_by_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> tuple[float, float]:
    """
    Find threshold minimising business cost.

    total_cost = FP × cost_fp + FN × cost_fn

    Note: step=0.01 is intentionally fine to capture the cost-minimum precisely.
    For very large datasets where overfitting on val_thresh is a concern, increase
    the step to 0.05 (coarser grid) to reduce threshold overfit risk.
    """
    best_thresh, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost   = cost
            best_thresh = round(t, 2)
    return best_thresh, round(best_cost, 2)


def find_best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Find threshold maximising F1."""
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(t, 2)
    return best_thresh, round(best_f1, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — SHAP with stratified sampling
# ═══════════════════════════════════════════════════════════════════════════════
import shap


def compute_shap(
    model_obj: Any,
    model_type: str,
    X_background: np.ndarray,
    X_eval: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """Compute mean |SHAP| per feature. Returns top-5 list."""
    try:
        if model_type == "linear":
            # shap 0.49+: shap.Explainer with algorithm='linear' is the supported path
            bg_sample = shap.sample(X_background, min(100, len(X_background)), random_state=42) \
                if X_background is not None else None
            explainer = shap.Explainer(model_obj, bg_sample, algorithm="linear")
            result = explainer(X_eval)
        else:
            explainer = shap.TreeExplainer(model_obj)
            result = explainer(X_eval)

        # Handle both new Explanation object (shap 0.49+) and legacy numpy
        shap_vals = result.values if hasattr(result, "values") else result
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        top5_idx = np.argsort(mean_abs)[-5:][::-1]

        return [
            {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in top5_idx
        ]
    except Exception as exc:  # pragma: no cover
        log.error("  SHAP failed for %s: %s", model_type, exc)
        return []


def stratified_shap_sample(
    X_val_thresh: np.ndarray,
    proba_val_thresh: np.ndarray,
    optimal_threshold: float,
    total_size: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Stratified SHAP sampling:
      - 50% random samples from val_thresh
      - 50% samples closest to optimal_threshold from val_thresh
    """
    rng = np.random.RandomState(random_state)
    n_each = total_size // 2

    rand_idx = rng.choice(len(X_val_thresh), size=n_each, replace=False)

    dist = np.abs(proba_val_thresh - optimal_threshold)
    prox_idx = np.argsort(dist)[:n_each]

    combined = np.concatenate([rand_idx, prox_idx])
    return np.unique(combined)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY — INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def validate_raw_input(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    """
    Validate raw input DataFrame before any processing.
    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if len(df) == 0:
        errors.append("DataFrame is empty")

    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        invalid_age = ((age < 0) | age.isna()).sum()
        if invalid_age > 0:
            errors.append(f"age has {invalid_age} invalid/negative values")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
def step1_load_data(csv_path: str = "data/cs-training.csv") -> pd.DataFrame:
    """Load CSV, rename target, and validate raw input."""
    import time as _time
    t0 = _time.perf_counter()
    log.info("STEP 1 — Loading data from: %s", csv_path)

    df = pd.read_csv(csv_path)

    drop = [c for c in df.columns if "Unnamed" in c or c == ""]
    if drop:
        log.info("  Dropped unnamed columns: %s", drop)
        df = df.drop(columns=drop)

    if "SeriousDlqin2yrs" not in df.columns:
        raise ValueError("Target column 'SeriousDlqin2yrs' not found.")

    df = df.rename(columns={"SeriousDlqin2yrs": "target"})

    REQUIRED_COLS = [
        "RevolvingUtilizationOfUnsecuredLines", "age",
        "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
        "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
        "target",
    ]
    errors = validate_raw_input(df, REQUIRED_COLS)
    if errors:
        for err in errors:
            log.error("  INPUT VALIDATION FAILED: %s", err)
        raise ValueError(f"Input validation failed: {errors}")

    log.info(
        "  Loaded %d rows × %d cols | target dist:\n%s",
        df.shape[0], df.shape[1],
        df["target"].value_counts().to_string(),
    )
    log.info("  STEP 1 done in %.1fs", _time.perf_counter() - t0)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — THREE-WAY SPLIT  [BEFORE any preprocessing]
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.model_selection import StratifiedShuffleSplit


def step2_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_thresh_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Three-way stratified split — executed BEFORE cleaning/imputation.

    Roles:
      train (70%)      : Optuna k-fold CV, OOF predictions, isotonic calibration,
                         final LR + final XGB
      val_thresh (15%) : threshold tuning, PSI-DEV, SHAP stratified sampling
      Test (15%)        : final evaluation, PSI-Test, threshold robustness validation

    Leakage assertions use hash-based row signatures (NOT DataFrame index)
    because subsets have reset_index after split.
    """
    import time as _time
    t0 = _time.perf_counter()

    total_ratio = round(train_ratio + val_thresh_ratio + test_ratio, 2)
    assert total_ratio == 1.0, f"Split ratios must sum to 1.0, got {total_ratio}"

    log.info(
        "STEP 2 — Three-way stratified split "
        "(train=%.0f%% | val_thresh=%.0f%% | Test=%.0f%%)",
        train_ratio * 100, val_thresh_ratio * 100, test_ratio * 100,
    )

    # Stage 1: split full dataset → train_pool + Test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state,
    )
    train_pool_idx, test_idx = next(
        sss1.split(df.drop(columns=["target"]), df["target"])
    )

    X_test = df.iloc[test_idx].drop(columns=["target"]).reset_index(drop=True)
    y_test = df.iloc[test_idx]["target"].reset_index(drop=True)
    X_train_pool = df.iloc[train_pool_idx].drop(columns=["target"]).reset_index(drop=True)
    y_train_pool = df.iloc[train_pool_idx]["target"].reset_index(drop=True)

    # Stage 2: split train_pool → train + val_thresh
    val_thresh_share = val_thresh_ratio / (train_ratio + val_thresh_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_thresh_share,
        random_state=random_state + 1,
    )
    train_idx, val_thresh_idx = next(
        sss2.split(X_train_pool, y_train_pool)
    )

    X_train      = X_train_pool.iloc[train_idx].reset_index(drop=True)
    X_val_thresh = X_train_pool.iloc[val_thresh_idx].reset_index(drop=True)
    y_train      = y_train_pool.iloc[train_idx].reset_index(drop=True)
    y_val_thresh = y_train_pool.iloc[val_thresh_idx].reset_index(drop=True)

    log.info(
        "  train=%d (%.1f%%) | val_thresh=%d (%.1f%%) | Test=%d (%.1f%%)",
        len(X_train),  len(X_train)  / len(df) * 100,
        len(X_val_thresh), len(X_val_thresh) / len(df) * 100,
        len(X_test), len(X_test) / len(df) * 100,
    )
    log.info(
        "  Positive rate — train=%.4f | val_thresh=%.4f | Test=%.4f",
        y_train.mean(), y_val_thresh.mean(), y_test.mean(),
    )

    # Leakage assertions using hash signatures
    sig_train      = _row_hash(X_train)
    sig_val_thresh = _row_hash(X_val_thresh)
    sig_test       = _row_hash(X_test)

    assert len(sig_train & sig_val_thresh) == 0, \
        f"LEAKAGE: train overlaps val_thresh by {len(sig_train & sig_val_thresh)} rows"
    assert len(sig_train & sig_test) == 0, \
        f"LEAKAGE: train overlaps Test by {len(sig_train & sig_test)} rows"
    assert len(sig_val_thresh & sig_test) == 0, \
        f"LEAKAGE: val_thresh overlaps Test by {len(sig_val_thresh & sig_test)} rows"

    log.info("  STEP 2 done in %.1fs | leakage assertions: PASS", _time.perf_counter() - t0)
    return X_train, X_val_thresh, X_test, y_train, y_val_thresh, y_test
