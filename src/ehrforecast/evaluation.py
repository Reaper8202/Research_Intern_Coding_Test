"""Version-2 probability metrics with explicit support and identifier alignment.

The step-2 runner keeps its original ``metrics`` module for historical reproduction.
New experiments use this module. Nonempty all-negative samples have AP 0; all-positive
samples have AP 1. AUROC requires both classes. Empty samples have no metric.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ehrforecast.contracts import align_targets, validate_binary, validate_probabilities
from ehrforecast.masking import apply_prior_mask
from ehrforecast.metrics import auroc, average_precision

METRIC_VERSION = "first-diagnosis-probabilities-v2"
METRIC_NAMES = ("macro_auroc", "mAP", "macro_brier")


@dataclass(frozen=True)
class Evaluation:
    summary: dict[str, Any]
    per_target: pd.DataFrame


def _prepare(
    labels: pd.DataFrame,
    probabilities: pd.DataFrame,
    codes: list[str],
    prior: pd.DataFrame | None,
    at_risk: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = align_targets(labels, labels.index, codes, "labels")
    probabilities = align_targets(probabilities, labels.index, codes, "probabilities")
    validate_binary(labels, "labels")
    validate_probabilities(probabilities)
    eligible = np.ones(labels.shape, dtype=bool)
    if prior is not None:
        prior = align_targets(prior, labels.index, codes, "prior mask")
        validate_binary(prior, "prior mask")
        if np.logical_and(labels.to_numpy(), prior.to_numpy()).any():
            raise ValueError("positive first-diagnosis labels overlap prior conditions")
        probabilities = apply_prior_mask(probabilities, prior, codes)
        if at_risk:
            eligible = prior.to_numpy() == 0
    elif at_risk:
        raise ValueError("at-risk evaluation requires the prior mask")
    return labels.to_numpy(dtype="int8"), probabilities.to_numpy(dtype="float64"), eligible


def _target_scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    if len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    n_pos = int(y.sum())
    auc = auroc(y, p) if 0 < n_pos < len(y) else float("nan")
    ap = average_precision(y, p) if n_pos else 0.0
    return auc, ap, float(np.mean((p - y) ** 2))


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def evaluate_predictions(
    labels: pd.DataFrame,
    probabilities: pd.DataFrame,
    codes: list[str],
    *,
    prior: pd.DataFrame | None = None,
    at_risk: bool = False,
) -> Evaluation:
    """Apply the common prior mask when supplied, then evaluate matched patients."""
    y, p, eligible = _prepare(labels, probabilities, codes, prior, at_risk)
    rows = []
    for j, code in enumerate(codes):
        keep = eligible[:, j]
        target_y, target_p = y[keep, j], p[keep, j]
        auc, ap, brier = _target_scores(target_y, target_p)
        rows.append(
            {
                "code": code,
                "n": len(target_y),
                "n_pos": int(target_y.sum()),
                "n_neg": int(len(target_y) - target_y.sum()),
                "prevalence": float(target_y.mean()) if len(target_y) else np.nan,
                "mean_probability": float(target_p.mean()) if len(target_y) else np.nan,
                "auroc": auc,
                "average_precision": ap,
                "brier": brier,
            }
        )
    table = pd.DataFrame(rows).set_index("code")
    summary: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "scope": "at_risk" if at_risk else "all_patients",
        "prior_mask_applied": prior is not None,
        "n_patients": len(labels),
        "n_targets": len(codes),
    }
    for name, column in zip(METRIC_NAMES, ("auroc", "average_precision", "brier"), strict=True):
        value = _mean(table[column].to_numpy())
        summary[name] = value if np.isfinite(value) else None
        summary[f"n_targets_{column}"] = int(table[column].notna().sum())
    return Evaluation(summary, table)


def bootstrap_comparison(
    labels: pd.DataFrame,
    probabilities: pd.DataFrame,
    codes: list[str],
    *,
    other: pd.DataFrame | None = None,
    prior: pd.DataFrame | None = None,
    at_risk: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Patient BCa intervals for one model or paired A-minus-B differences.

    Auxiliary statistics retain each resample's valid-target counts. Their intervals
    are unused. Degenerate intervals and warning messages remain visible in the report.
    """
    if len(labels) < 2 or n_boot < 2:
        raise ValueError("bootstrap needs at least two patients and two resamples")
    y, p, eligible = _prepare(labels, probabilities, codes, prior, at_risk)
    q = None if other is None else _prepare(labels, other, codes, prior, at_risk)[1]

    def statistic(indices: np.ndarray) -> np.ndarray:
        first, second = [], []
        for j in range(len(codes)):
            rows = indices[eligible[indices, j]]
            first.append(_target_scores(y[rows, j], p[rows, j]))
            if q is not None:
                second.append(_target_scores(y[rows, j], q[rows, j]))
        scores = np.asarray(first)
        point = np.array([_mean(scores[:, j]) for j in range(3)])
        if q is not None:
            other_scores = np.asarray(second)
            point -= np.array([_mean(other_scores[:, j]) for j in range(3)])
        counts = np.isfinite(scores).sum(axis=0)
        return np.r_[point, counts]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = stats.bootstrap(
            (np.arange(len(y)),),
            statistic,
            vectorized=False,
            n_resamples=n_boot,
            confidence_level=0.95,
            method="BCa",
            rng=np.random.default_rng(seed),
        )
    point = statistic(np.arange(len(y)))

    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    output: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "method": "patient BCa, 95%",
        "n_boot": n_boot,
        "seed": seed,
        "scope": "at_risk" if at_risk else "all_patients",
        "comparison": "A_minus_B" if q is not None else "single_model",
        "warnings": sorted({str(item.message) for item in caught}),
        "eligibility_note": "AUROC can omit different targets in different resamples.",
        "warning_note": "Auxiliary eligibility counts can have degenerate unused intervals.",
        "selection_note": "Intervals exclude training and model-selection uncertainty.",
    }
    for j, name in enumerate(METRIC_NAMES):
        counts = result.bootstrap_distribution[j + 3].astype(int)
        values, frequencies = np.unique(counts, return_counts=True)
        output[name] = {
            "point": finite(point[j]),
            "ci_low": finite(result.confidence_interval.low[j]),
            "ci_high": finite(result.confidence_interval.high[j]),
            "nonfinite_resamples": int((~np.isfinite(result.bootstrap_distribution[j])).sum()),
            "eligible_targets_histogram": {
                str(value): int(count) for value, count in zip(values, frequencies, strict=True)
            },
        }
    return output
