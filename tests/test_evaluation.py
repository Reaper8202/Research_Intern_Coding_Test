from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ehrforecast.evaluation import METRIC_VERSION, bootstrap_comparison, evaluate_predictions


def test_single_class_and_empty_target_support_are_explicit() -> None:
    y = pd.DataFrame({"none": [0, 0], "all": [1, 1], "mixed": [0, 1]}, index=["a", "b"])
    p = pd.DataFrame(0.5, index=y.index, columns=y.columns)
    result = evaluate_predictions(y, p, y.columns.tolist())
    table = result.per_target
    assert table.loc["none", "average_precision"] == 0
    assert table.loc["all", "average_precision"] == 1
    assert table.loc["mixed", "auroc"] == 0.5
    assert table["brier"].tolist() == [0.25] * 3
    assert result.summary["n_targets_auroc"] == 1
    assert result.summary["n_targets_average_precision"] == 3
    assert result.summary["mAP"] == 0.5
    assert result.summary["metric_version"] == METRIC_VERSION
    empty = evaluate_predictions(y.iloc[:0], p.iloc[:0], y.columns.tolist())
    assert empty.summary["macro_auroc"] is None
    assert empty.summary["mAP"] is None
    assert empty.summary["n_targets_brier"] == 0


@pytest.mark.parametrize("seed", range(4))
def test_metrics_match_reference_with_ties_and_independent_row_order(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = pd.DataFrame(rng.integers(0, 2, (80, 3)), columns=["X", "Y", "Z"])
    y.index = pd.Index([f"p{i}" for i in range(len(y))])
    p = pd.DataFrame(np.round(rng.random(y.shape), 1), index=y.index, columns=y.columns)
    result = evaluate_predictions(y, p.iloc[::-1][["Z", "X", "Y"]], y.columns.tolist())
    for code in y.columns:
        assert result.per_target.loc[code, "auroc"] == pytest.approx(
            roc_auc_score(y[code], p[code])
        )
        assert result.per_target.loc[code, "average_precision"] == pytest.approx(
            average_precision_score(y[code], p[code])
        )
        assert result.per_target.loc[code, "brier"] == pytest.approx(
            brier_score_loss(y[code], p[code])
        )


def test_prior_mask_and_at_risk_metrics_have_distinct_denominators() -> None:
    y = pd.DataFrame({"X": [0, 0, 1], "Y": [0, 0, 0]}, index=["a", "b", "c"])
    p = pd.DataFrame({"X": [0.99, 0.1, 0.5], "Y": [0.2, 0.3, 0.4]}, index=y.index)
    prior = pd.DataFrame({"X": [1, 0, 0], "Y": [1, 1, 1]}, index=y.index)
    masked = evaluate_predictions(y, p, ["X", "Y"], prior=prior)
    at_risk = evaluate_predictions(y, p, ["X", "Y"], prior=prior.iloc[::-1], at_risk=True)
    assert masked.per_target.loc["X", "auroc"] == 1
    assert masked.per_target.loc["Y", "brier"] == 0
    assert at_risk.per_target.loc["X", "n"] == 2
    assert at_risk.per_target.loc["Y", "n"] == 0
    assert at_risk.summary["n_targets_average_precision"] == 1
    prior.loc["c", "X"] = 1
    with pytest.raises(ValueError, match="overlap"):
        evaluate_predictions(y, p, ["X", "Y"], prior=prior)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_probabilities_fail(invalid: float) -> None:
    y = pd.DataFrame({"X": [0, 1]}, index=["a", "b"])
    p = pd.DataFrame({"X": [invalid, 0.5]}, index=y.index)
    with pytest.raises(ValueError, match="finite"):
        evaluate_predictions(y, p, ["X"])


def test_paired_bootstrap_is_reproducible_and_reverses_direction() -> None:
    rng = np.random.default_rng(41)
    y = pd.DataFrame(rng.integers(0, 2, (80, 2)), columns=["X", "Y"])
    y.index = pd.Index([f"p{i}" for i in range(len(y))])
    p = pd.DataFrame(
        0.3 * y.to_numpy() + 0.7 * rng.random(y.shape), index=y.index, columns=y.columns
    )
    q = pd.DataFrame(rng.random(y.shape), index=y.index, columns=y.columns)
    first = bootstrap_comparison(y, p, ["X", "Y"], other=q, n_boot=40, seed=9)
    repeat = bootstrap_comparison(y, p, ["X", "Y"], other=q, n_boot=40, seed=9)
    reverse = bootstrap_comparison(y, q, ["X", "Y"], other=p, n_boot=40, seed=9)
    assert first == repeat
    assert first["macro_auroc"]["point"] > 0
    for metric in ("macro_auroc", "mAP", "macro_brier"):
        assert first[metric]["point"] == pytest.approx(-reverse[metric]["point"])
        assert first[metric]["ci_low"] == pytest.approx(-reverse[metric]["ci_high"])
        assert first[metric]["eligible_targets_histogram"] == {"2": 40}


def test_degenerate_bootstrap_is_reported_without_inventing_an_interval() -> None:
    y = pd.DataFrame({"X": [1] + [0] * 11}, index=[f"p{i}" for i in range(12)])
    p = pd.DataFrame({"X": np.linspace(0.9, 0.1, 12)}, index=y.index)
    result = bootstrap_comparison(y, p, ["X"], n_boot=40, seed=3)
    assert result["macro_auroc"]["nonfinite_resamples"] > 0
    assert result["macro_auroc"]["eligible_targets_histogram"]["0"] > 0
    assert result["macro_auroc"]["ci_low"] is None
    assert result["mAP"]["eligible_targets_histogram"] == {"1": 40}
