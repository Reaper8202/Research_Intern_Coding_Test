from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ehrforecast.metrics import (
    auroc,
    average_precision,
    bootstrap_summary,
    paired_bootstrap_difference,
    per_label_metrics,
    summarize,
)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    s = np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3], [0.8, 0.4, 0.2], [0.3, 0.6, 0.9]])
    return y, s


def test_perfect_ranking_scores_one_and_single_class_target_is_skipped() -> None:
    y, s = _toy()
    table = per_label_metrics(y, s, ["a", "b", "c"])
    assert table.loc["a", "auroc"] == 1.0
    assert table.loc["b", "average_precision"] == 1.0
    assert pd.isna(table.loc["c", "auroc"])  # no positives
    summary = summarize(y, s, ["a", "b", "c"])
    assert summary["macro_auroc"] == 1.0
    assert summary["n_targets_scored"] == 2
    assert summary["chance_mAP"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_mask_drops_rows_per_target() -> None:
    y, s = _toy()
    mask = np.ones_like(y)
    mask[0, 0] = 0
    table = per_label_metrics(y, s, ["a", "b", "c"], mask=mask)
    assert table.loc["a", "n"] == 3
    assert table.loc["b", "n"] == 4


def test_bootstrap_is_reproducible_and_brackets_a_strong_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(300, 3))
    s = y + rng.normal(0, 0.7, size=y.shape)
    first = bootstrap_summary(y, s, n_boot=200, seed=1)
    second = bootstrap_summary(y, s, n_boot=200, seed=1)
    assert first == second
    point = summarize(y, s, ["a", "b", "c"])["macro_auroc"]
    ci = first["macro_auroc"]
    assert ci["ci_low"] < point < ci["ci_high"]
    assert ci["ci_low"] > 0.7


def test_paired_difference_detects_the_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(300, 3))
    strong = y + rng.normal(0, 0.5, size=y.shape)
    weak = y + rng.normal(0, 3.0, size=y.shape)
    diff = paired_bootstrap_difference(y, strong, weak, n_boot=200, seed=1)
    auroc = diff["macro_auroc"]
    assert auroc["ci_low"] < auroc["difference"] < auroc["ci_high"]
    assert auroc["ci_low"] > 0
    reverse = paired_bootstrap_difference(y, weak, strong, n_boot=200, seed=1)
    assert reverse["macro_auroc"]["ci_high"] < 0


@pytest.mark.parametrize("seed", range(5))
def test_fast_metrics_match_sklearn_including_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = (rng.random(400) < 0.1).astype(int)
    s = np.round(y * 0.3 + rng.random(400), 1)  # rounding creates many ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_shape_mismatch_raises() -> None:
    y, s = _toy()
    with pytest.raises(ValueError, match="shape"):
        per_label_metrics(y, s[:, :2], ["a", "b", "c"])
