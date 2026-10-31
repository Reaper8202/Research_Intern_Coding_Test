"""Legacy step-2 metrics, retained only for historical reproduction.

This version skips both metrics when a sample has one class. That AP convention is not the
reference-library convention. New experiments use ``evaluation.py`` and its version-2 rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC as the Mann-Whitney statistic with average ranks for ties.

    Equal to ``sklearn.metrics.roc_auc_score`` (checked in the tests) but without its per-call
    input validation, which dominates the run time of the bootstrap.
    """
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    ranks = stats.rankdata(s)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision with tied scores treated as one threshold.

    Equal to ``sklearn.metrics.average_precision_score`` (checked in the tests).
    """
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s) - 1]
    true_pos = np.cumsum(y_sorted)[last_of_tie]
    precision = true_pos / (last_of_tie + 1)
    recall_gain = np.diff(np.r_[0, true_pos]) / true_pos[-1]
    return float(np.sum(recall_gain * precision))


def _label_scores(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan"), float("nan")
    return auroc(y, s), average_precision(y, s)


def per_label_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    codes: list[str],
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """AUROC and AP for each target.

    ``mask`` (same shape as ``y_true``) selects the rows that count for each target. The
    at-risk analysis uses it to drop patients who already had the condition before the anchor.
    """
    _check_shapes(y_true, y_score, codes)
    rows = []
    for j, code in enumerate(codes):
        keep = slice(None) if mask is None else mask[:, j].astype(bool)
        y, s = y_true[keep, j], y_score[keep, j]
        auroc, ap = _label_scores(y, s)
        rows.append(
            {
                "code": code,
                "n": len(y),
                "n_pos": int(y.sum()),
                "prevalence": float(y.mean()) if len(y) else float("nan"),
                "auroc": auroc,
                "average_precision": ap,
            }
        )
    return pd.DataFrame(rows).set_index("code")


def summarize(y_true: np.ndarray, y_score: np.ndarray, codes: list[str]) -> dict[str, float]:
    table = per_label_metrics(y_true, y_score, codes)
    return {
        "macro_auroc": float(table["auroc"].mean()),
        "mAP": float(table["average_precision"].mean()),
        "chance_mAP": float(table["prevalence"].mean()),
        "n_targets_scored": int(table["auroc"].notna().sum()),
    }


def _macro_scores(y_true: np.ndarray, y_score: np.ndarray, idx: np.ndarray) -> np.ndarray:
    pairs = [_label_scores(y_true[idx, j], y_score[idx, j]) for j in range(y_true.shape[1])]
    return np.asarray(np.nanmean(np.asarray(pairs), axis=0))


def bootstrap_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """BCa bootstrap intervals for macro AUROC and mAP, resampling patients.

    BCa (bias-corrected and accelerated) is used instead of plain percentiles because average
    precision on resampled data is biased upwards for rare targets: duplicated positive patients
    raise precision at the top of the ranking.
    """
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    result = stats.bootstrap(
        (np.arange(len(y_true)),),
        lambda idx: _macro_scores(y_true, y_score, idx),
        vectorized=False,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    out = {}
    for i, name in enumerate(("macro_auroc", "mAP")):
        out[name] = {
            "ci_low": float(result.confidence_interval.low[i]),
            "ci_high": float(result.confidence_interval.high[i]),
            "bootstrap_std": float(result.standard_error[i]),
        }
    return out


def paired_bootstrap_difference(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """BCa interval for (model A - model B) in macro AUROC and mAP on the same patients.

    Both models are scored on the same resampled patients, so the interval reflects the
    difference itself. An interval above zero means that A is better than B.
    """
    if not (y_true.shape == score_a.shape == score_b.shape):
        raise ValueError("y_true, score_a and score_b must have the same shape")
    full = np.arange(len(y_true))
    point = _macro_scores(y_true, score_a, full) - _macro_scores(y_true, score_b, full)
    result = stats.bootstrap(
        (full,),
        lambda idx: _macro_scores(y_true, score_a, idx) - _macro_scores(y_true, score_b, idx),
        vectorized=False,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    out = {}
    for i, name in enumerate(("macro_auroc", "mAP")):
        out[name] = {
            "difference": float(point[i]),
            "ci_low": float(result.confidence_interval.low[i]),
            "ci_high": float(result.confidence_interval.high[i]),
        }
    return out


def _check_shapes(y_true: np.ndarray, y_score: np.ndarray, codes: list[str]) -> None:
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.shape[1] != len(codes):
        raise ValueError("number of codes does not match the number of label columns")
