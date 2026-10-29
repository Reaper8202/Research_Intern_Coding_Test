"""Multi-label evaluation: macro AUROC, mean average precision, and bootstrap intervals.

A target is skipped (``NaN``) in a sample where it has only one class, because AUROC and AP are
undefined there. Macro averages then use the remaining targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def _label_scores(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


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


def _check_shapes(y_true: np.ndarray, y_score: np.ndarray, codes: list[str]) -> None:
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.shape[1] != len(codes):
        raise ValueError("number of codes does not match the number of label columns")
