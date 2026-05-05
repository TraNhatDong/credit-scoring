"""
_riskbands.py — Hybrid risk-banding system: PD-threshold tails + quantile middle.

  Q1 (Best)     : PD-anchored, PD ≤ 1.5%  (top ~10% by PD, Basel "Excellent")
  Q2 – Q6       : Quantile,   80% middle, ~16% each
  Q7 (Worst)    : PD-anchored, PD ≥ 15% (bottom ~10% by PD, Basel "Very Poor")

Hybrid benefits:
  - Tail bands are Basel-consistent (anchored to PD probability anchors)
  - Middle bands reflect actual score distribution (data-driven, no hard cutoffs)
  - Risk committee gets interpretable tail labels (Excellent / Very Poor)
  - Q1/Q7 counts vary slightly each run (reflecting actual tail population)
    instead of hardcoded 10% / 10%

Import path: from ._config import RiskBands, score_band
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ── PD-to-Score anchors (Basel-compliant, from pd_to_credit_score formula) ──
#
#   Score = 300 - B * ln(PD / (1-PD))
#   where  B = 550 / ln(99) ≈ 119.74
#
#   Anchor points used:
#     PD = 0.015 (1.5%)  → Score ≈ 782  → Q1 threshold  (Excellent)
#     PD = 0.40  (40%)   → Score = 300  → Q7 floor      (Very Poor, Basel floor)
#
_MIN_SCORE = 300
_MAX_SCORE = 850


def _pd_to_score(pd: float) -> int:
    """
    PD → score using Basel log-odds formula (same as pd_to_credit_score in _config.py).

    Formula: Score = 300 - B * ln(PD / (1-PD))
    where B = 550 / ln(99) ≈ 119.74

    Anchor points:
      PD = 0.01 (1%)  → Score = 850  (Excellent)
      PD = 0.50 (50%) → Score = 300  (Very Poor)
    """
    pd = float(np.clip(pd, 1e-6, 1 - 1e-6))
    ln_99 = float(np.log(99.0))
    B = 550.0 / ln_99
    score = float(_MIN_SCORE) - B * np.log(pd / (1.0 - pd))
    return int(np.clip(round(score), _MIN_SCORE, _MAX_SCORE))


def _score_to_pd(score: float) -> float:
    """
    Inverse: Score → PD using Basel log-odds formula.

    Score = 300 + B * ln((1-PD)/PD),  where B = 550/ln(99)
    => ln((1-PD)/PD) = (Score-300) / B
    => (1-PD)/PD = exp((Score-300)/B)
    => PD = 1 / (1 + exp((Score-300)/B))
    """
    ln_99 = float(np.log(99.0))
    B = 550.0 / ln_99
    ln_odds = (float(score) - float(_MIN_SCORE)) / B
    return 1.0 / (1.0 + np.exp(ln_odds))


# PD thresholds for anchor bands
_PD_Q1_ANCHOR = 0.010   # 1.0% — Q1 (Best / Excellent) — Basel-consistent for this dataset
_PD_Q7_ANCHOR = 0.150   # 15%  — Q7 (Worst / Very Poor)

# Score equivalents
_SCORE_Q1_ANCHOR = _pd_to_score(_PD_Q1_ANCHOR)   # ≈ 833
_SCORE_Q7_ANCHOR = _pd_to_score(_PD_Q7_ANCHOR)   # ≈ 490


class RiskBands:
    """
    Hybrid risk-banding: PD-threshold tails + quantile middle.

    Band structure (n_bands=7):
      Band 0: Q1 (Best)    — PD ≤ 1.5%  (score ≥ q1_boundary)
      Band 1: Q2           — Quantile, top ~16% of remaining
      Band 2: Q3           — Quantile, middle ~16%
      Band 3: Q4           — Quantile, middle ~16%
      Band 4: Q5           — Quantile, middle ~16%
      Band 5: Q6           — Quantile, bottom ~16% of remaining
      Band 6: Q7 (Worst)  — PD ≥ 15% (score ≤ q7_boundary)

    Usage:
      bands = RiskBands.fit(train_scores, n_bands=7)
      label = bands.get_band(score)          # e.g. "Q3" or "Q7 (Worst)"
      stats = bands.band_stats(test_scores) # per-band count, range, PD anchor
    """

    # Standard label sets
    _LABEL_SETS = {
        7: ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7 (Worst)"],
        6: ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5", "Q6 (Worst)"],
        5: ["Prime", "Standard", "Sub-Prime", "High-Risk", "Critical"],
        8: [f"Q{i}" for i in range(1, 9)],
        10: [f"Q{i}" for i in range(1, 11)],
    }

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        scores: np.ndarray,
        n_bands: int = 7,
        labels: Optional[list[str]] = None,
    ) -> "RiskBands":
        """
        Fit hybrid bands from TRAINING scores.

        Architecture:
          Tail bands (Q1, Q7) : PD-threshold anchored — stable across runs,
                                Basel-consistent, interpretable for risk committee
          Middle bands        : Quantile of scores between the two anchors —
                                data-driven, no hard cutoffs

        For n_bands=7:
          - Q1 (Best): PD ≤ 1.0%  (score ≥ max(90th_pct, _SCORE_Q1_ANCHOR))
          - Q2–Q6:    Quantile of remaining 80% (each ~16%)
          - Q7 (Worst): PD ≥ 15% (score ≤ min(10th_pct, _SCORE_Q7_ANCHOR))

        Args:
            scores: 1D array of credit scores from training data.
            n_bands: total number of bands (default 7 = hybrid).
            labels: optional custom band labels (len must == n_bands).
        """
        scores = np.asarray(scores, dtype=float).flatten()
        if len(scores) == 0:
            raise ValueError("scores cannot be empty")
        if n_bands < 2:
            raise ValueError("n_bands must be >= 2")

        scores_clamped = np.clip(scores, _MIN_SCORE, _MAX_SCORE)

        if n_bands == 7 and labels is None:
            # ── Hybrid: PD-anchored tails + quantile middle ──────────────────
            #
            # Architecture (n_bands=7):
            #   Q1 (Best)  : scores ≥ q1_ceiling
            #                q1_ceiling = min(pct_90_of_unclamped, _SCORE_Q1_ANCHOR=833)
            #                When pile-up inflates pct_90 to 850 → cap at 782 so Q1 stays
            #                Basel-consistent and Q2–Q6 are not swallowed.
            #   Q2 – Q6    : equal-width intervals across [q7_floor, q1_ceiling]
            #   Q7 (Worst) : scores < q7_floor
            #                q7_floor = max(pct_10, _SCORE_Q7_ANCHOR=490)

            # ── Unclamped reference sets (ceiling-collapse protection) ───────────
            clamped_at_max = np.sum(scores_clamped >= _MAX_SCORE)
            clamped_at_min = np.sum(scores_clamped <= _MIN_SCORE)

            q1_ref = scores_clamped[scores_clamped < _MAX_SCORE] \
                if clamped_at_max >= len(scores) * 0.15 else scores_clamped

            q7_ref = scores_clamped[scores_clamped > _MIN_SCORE] \
                if clamped_at_min >= len(scores) * 0.15 else scores_clamped

            # ── Q1 ceiling: use PD-anchored score as the fixed floor of Q1 ──────
            # The 90th percentile of unclamped scores sets the CEILING of Q1
            # (top of the Q1 band — what score is the cutoff between Q1 and Q2?).
            # But when many scores pile at 850, this percentile lands at 850,
            # which we cap at the Basel anchor to keep Q1 reasonable.
            q1_ceiling_raw = float(np.percentile(q1_ref, 90))
            # _SCORE_Q1_ANCHOR is the floor (minimum score that qualifies as Q1).
            # We use min(percentile, anchor) so Q1 never expands beyond the anchor
            # even if the model's predictions are very concentrated.
            # ── Q1 ceiling: fixed at _SCORE_Q1_ANCHOR (782) as upper bound ─────────
            q1_ceiling = min(q1_ceiling_raw, float(_SCORE_Q1_ANCHOR))

            # ── Q7 floor: fixed at _SCORE_Q7_ANCHOR (490) as lower bound ──────
            q7_floor_raw = float(np.percentile(q7_ref, 10))
            q7_floor   = max(q7_floor_raw, float(_SCORE_Q7_ANCHOR))
            q7_ceiling = float(_SCORE_Q7_ANCHOR)

            # ── Middle bands Q2–Q6: equal-width intervals across [q7_floor, q1_ceiling]
            # Q1 occupies [q1_ceiling, 850]; Q2–Q6 divide [q7_floor, q1_ceiling];
            # Q7 occupies [300, q7_ceiling].
            middle_span = q1_ceiling - q7_floor
            n_middle   = n_bands - 2   # = 5 internal boundaries for n_bands=7
            step       = middle_span / n_middle
            # n_bands=7: internal = [q1-step*1, q1-step*2, q1-step*3, q1-step*4, q1-step*5]
            # → range(1, n_bands-1) = range(1, 6) = 5 items
            internal_thresholds = [
                q1_ceiling - i * step for i in range(1, n_bands - 1)
            ]
            # Fix: if last internal boundary collides with q7_floor (due to anchor override),
            # bump it up by 1 score point so Q6 has a non-zero width and displays correctly.
            if internal_thresholds and abs(internal_thresholds[-1] - q7_floor) < 1.0:
                internal_thresholds[-1] = q7_floor + 1.0
            # 7 thresholds for 7 bands: [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
            thresholds = [q1_ceiling] + internal_thresholds + [q7_floor]
            labels = cls._LABEL_SETS.get(n_bands, [f"Band-{i+1}" for i in range(n_bands)])

        else:
            pct_pts = list(np.linspace(0, 100, n_bands + 1))[1:]
            thresholds = sorted(
                [float(np.percentile(scores_clamped, 100 - p)) for p in pct_pts],
                reverse=True,
            )
            labels = labels or cls._LABEL_SETS.get(
                n_bands, [f"Band-{i+1}" for i in range(n_bands)]
            )

        if len(labels) != n_bands:
            raise ValueError(f"labels must have {n_bands} items, got {len(labels)}")

        inst = cls.__new__(cls)
        inst._mode        = "hybrid"
        inst._n_bands    = n_bands
        inst._thresholds = thresholds
        inst._labels     = list(labels)
        inst._ref_scores = scores_clamped
        inst._observed_min = int(np.min(scores_clamped))
        inst._observed_max = int(np.max(scores_clamped))
        return inst

    @classmethod
    def from_pd_thresholds(
        cls,
        pd_thresholds: list[tuple[str, float]],
    ) -> "RiskBands":
        """Legacy PD-fixed banding — kept for backward compatibility only."""
        sorted_bands = sorted(pd_thresholds, key=lambda x: x[1])
        inst = cls.__new__(cls)
        inst._mode = "pd_fixed"
        inst._pd_thresholds = sorted_bands
        inst._n_bands = len(sorted_bands)
        inst._observed_min = _MIN_SCORE
        inst._observed_max = _MAX_SCORE
        return inst

    # ── Public API ───────────────────────────────────────────────────────────

    def get_band(self, score: int) -> str:
        """
        Return band label for a credit score.

        All modes: scans thresholds descending; first threshold <= score wins.
        """
        for i, threshold in enumerate(self._thresholds):
            if score >= threshold:
                return self._labels[i]
        return self._labels[-1]

    def band_stats(self, scores: np.ndarray) -> dict:
        """
        Per-band statistics: count, percentage, score range, PD anchor.

        For hybrid mode (n_bands=7):
          Q1 (Best):    display shows observed ceiling (≥observed_max)
          Q7 (Worst):   display shows observed min (≤observed_min)
          Q2–Q6:        lo–hi from adjacent thresholds
        """
        from collections import Counter

        scores = np.asarray(scores, dtype=float).flatten()
        total  = len(scores)
        counts: Counter[str] = Counter(self.get_band(int(s)) for s in scores)

        result = {}
        n = self._n_bands
        thresholds = self._thresholds

        for i, label in enumerate(self._labels):
            hi = thresholds[i]
            if i + 1 < len(thresholds):
                lo = thresholds[i + 1]
            else:
                lo = float(self._observed_min)

            if self._mode == "hybrid" and n == 7:
                if i == 0:
                    display_lo = hi
                    display_hi = float(self._observed_max)
                elif i == n - 1:
                    display_lo = float(self._observed_min)
                    display_hi = thresholds[-1]
                else:
                    display_lo, display_hi = lo, hi

                if i == 0:
                    pd_label = f"PD<={int(round(_PD_Q1_ANCHOR * 100))}%"
                elif i == n - 1:
                    pd_label = f"PD>={int(_PD_Q7_ANCHOR*100)}%"
                else:
                    pd_label = None
            else:
                display_lo, display_hi = lo, hi
                pd_label = None

            result[label] = {
                "count":     counts.get(label, 0),
                "pct":       round(100 * counts.get(label, 0) / total, 2),
                "score_lo":  display_lo,
                "score_hi":  display_hi,
                "threshold": hi,
                "pd_label":  pd_label,
            }
        return result

    def __repr__(self) -> str:
        parts = [f"{l}: >={int(t)}" for l, t in zip(self._labels, self._thresholds)]
        if len(self._labels) > len(self._thresholds):
            parts.append(f"{self._labels[-1]}: <{int(self._thresholds[-1])}")
        return f"RiskBands(hybrid, " + " | ".join(parts) + ")"

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def thresholds(self) -> list[float]:
        """Score thresholds (descending) for all bands."""
        return list(self._thresholds)


# ── Legacy default: PD-based bands (backward compat for score_band()) ───────
_DEFAULT_PD_BANDS = RiskBands.from_pd_thresholds([
    ("Prime",      0.005),
    ("Excellent",  0.010),
    ("Very Good", 0.020),
    ("Good",      0.050),
    ("Fair",      0.100),
    ("Poor",      0.200),
    ("Very Poor", 0.400),
    ("Critical",  0.600),
    ("VP-A",      0.800),
    ("VP-B",      1.000),
])


def score_band(score: int) -> str:
    """
    Legacy wrapper — uses default PD-fixed bands.
    For hybrid banding use: RiskBands.fit(train_scores, n_bands=7).get_band(score).
    """
    return _DEFAULT_PD_BANDS.get_band(score)
