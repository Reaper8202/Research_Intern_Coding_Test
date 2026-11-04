"""Per-target Platt scaling and reliability reports, on training CV only (round 3, rule C).

For each target, on at-risk rows (prior-condition cells stay 0), a logistic regression maps the
model's logit to a calibrated probability: ``sigmoid(a * logit(p) + b)``. The scaling is monotone
within a target, so per-target AUROC and AP are unchanged; only Brier score, expected calibration
error, and the reliability curve can move. Nested folds give honest out-of-fold estimates: the
scaling applied to a fold is fitted on the other four folds' out-of-fold predictions. The
weights fitted on all five folds (``platt_weights.json``) are for the frozen val bundle and the
test predictions. Official val is not read.

Usage::

    python -m ehrforecast.calibration --dataset data/processed --models models \
        --transformer configs/experiments/v2_c1.toml configs/experiments/v2_c1_seed2.toml \
        configs/experiments/v2_c1_seed3.toml --reports reports/calibration/transformer
    python -m ehrforecast.calibration --dataset data/processed --models models \
        --oof reports/stacking/clean/oof_stack_logit_mean.parquet \
        --reports reports/calibration/stack
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ehrforecast.contracts import load_training_data
from ehrforecast.evaluation import evaluate_predictions
from ehrforecast.folds import N_FOLDS, fold_patients, load_folds
from ehrforecast.lightgbm_cv import ensemble_oof
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.provenance import write_json
from ehrforecast.stacking import logit, sigmoid

log = logging.getLogger("ehrforecast.calibration")

N_BINS = 10
C_PLATT = 1.0  # one slope and one intercept per target; the fit is not sensitive to C


def fit_platt(
    probabilities: pd.DataFrame, labels: pd.DataFrame, prior: pd.DataFrame, codes: list[str]
) -> dict[str, dict[str, float]]:
    """Slope and intercept per target from at-risk rows.

    Identity (slope 1, intercept 0) when a class is missing or the fitted slope is not positive,
    so the scaling is always monotone and per-target AUROC and AP are unchanged by construction.
    """
    weights: dict[str, dict[str, float]] = {}
    p = probabilities.loc[labels.index, codes].to_numpy("float64")
    y = labels.loc[labels.index, codes].to_numpy("int8")
    at_risk = prior.loc[labels.index, codes].to_numpy() == 0
    for j, code in enumerate(codes):
        rows = at_risk[:, j]
        if len(np.unique(y[rows, j])) < 2:
            weights[code] = {"a": 1.0, "b": 0.0, "fitted": 0.0}
            continue
        model = LogisticRegression(C=C_PLATT, max_iter=1000).fit(
            logit(p[rows, j]).reshape(-1, 1), y[rows, j]
        )
        slope = float(model.coef_[0][0])
        if slope <= 0:  # the fit would reverse the ranking: keep the model's own order
            weights[code] = {"a": 1.0, "b": 0.0, "fitted": 0.0}
            continue
        weights[code] = {"a": slope, "b": float(model.intercept_[0]), "fitted": 1.0}
    return weights


def apply_platt(
    weights: dict[str, dict[str, float]],
    probabilities: pd.DataFrame,
    prior: pd.DataFrame,
    codes: list[str],
) -> pd.DataFrame:
    out = probabilities.loc[:, codes].copy()
    for code in codes:
        w = weights[code]
        out[code] = sigmoid(w["a"] * logit(out[code].to_numpy("float64")) + w["b"])
    return apply_prior_mask(out, prior.loc[out.index], codes)


def nested_platt(
    probabilities: pd.DataFrame,
    labels: pd.DataFrame,
    prior: pd.DataFrame,
    codes: list[str],
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Out-of-fold calibrated probabilities: each fold scaled by weights fitted on the others."""
    parts = []
    for fold in range(N_FOLDS):
        fit, held = fold_patients(assignments, fold)
        fit = labels.index.intersection(fit)
        held = labels.index.intersection(held)
        weights = fit_platt(probabilities.loc[fit], labels.loc[fit], prior.loc[fit], codes)
        parts.append(apply_platt(weights, probabilities.loc[held], prior.loc[held], codes))
    return pd.concat(parts).loc[labels.index]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Equal-mass bins over the given cells; |mean label - mean probability| weighted by count."""
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, len(order))):
        if len(chunk):
            total += len(chunk) * abs(float(y[chunk].mean()) - float(p[chunk].mean()))
    return total / len(y)


def reliability_table(
    y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS, label: str = ""
) -> pd.DataFrame:
    """Pooled reliability curve over all at-risk cells, equal-mass bins of predicted probability."""
    order = np.argsort(p, kind="stable")
    rows = []
    for k, chunk in enumerate(np.array_split(order, n_bins)):
        rows.append(
            {
                "model": label,
                "bin": k,
                "n": len(chunk),
                "mean_probability": float(p[chunk].mean()),
                "observed_rate": float(y[chunk].mean()),
                "p_low": float(p[chunk].min()),
                "p_high": float(p[chunk].max()),
            }
        )
    return pd.DataFrame(rows)


def per_target_calibration(
    labels: pd.DataFrame, probabilities: pd.DataFrame, prior: pd.DataFrame, codes: list[str]
) -> pd.DataFrame:
    y = labels.loc[:, codes].to_numpy("int8")
    p = probabilities.loc[labels.index, codes].to_numpy("float64")
    at_risk = prior.loc[labels.index, codes].to_numpy() == 0
    rows = []
    for j, code in enumerate(codes):
        rows.append(
            {
                "code": code,
                "n_at_risk": int(at_risk[:, j].sum()),
                "prevalence": float(y[at_risk[:, j], j].mean()),
                "mean_probability": float(p[at_risk[:, j], j].mean()),
                "ece": expected_calibration_error(y[at_risk[:, j], j], p[at_risk[:, j], j]),
            }
        )
    return pd.DataFrame(rows).set_index("code")


def run(
    dataset_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    transformer_configs: list[Path] | None = None,
    oof_path: Path | None = None,
) -> dict[str, Any]:
    if (transformer_configs is None) == (oof_path is None):
        raise ValueError("give exactly one of --transformer configs or --oof path")
    data = load_training_data(dataset_dir)
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    assignments = load_folds(dataset_dir)
    if transformer_configs is not None:
        raw = ensemble_oof(models_dir, transformer_configs, assignments).loc[labels.index]
        source = [str(p) for p in transformer_configs]
    else:
        assert oof_path is not None
        raw = pd.read_parquet(oof_path).loc[labels.index]
        source = [str(oof_path)]
    raw = apply_prior_mask(raw, prior, codes)
    calibrated = nested_platt(raw, labels, prior, codes, assignments)
    reports_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in range(N_FOLDS):
        _, held = fold_patients(assignments, fold)
        held = labels.index.intersection(held)
        before = evaluate_predictions(labels.loc[held], raw.loc[held], codes, prior=prior.loc[held])
        after = evaluate_predictions(
            labels.loc[held], calibrated.loc[held], codes, prior=prior.loc[held]
        )
        folds.append(
            {
                "fold": fold,
                "brier_raw": before.summary["macro_brier"],
                "brier_calibrated": after.summary["macro_brier"],
                "auroc_raw": before.summary["macro_auroc"],
                "auroc_calibrated": after.summary["macro_auroc"],
            }
        )
    fold_table = pd.DataFrame(folds)
    improved = int((fold_table["brier_calibrated"] < fold_table["brier_raw"]).sum())
    y_all = labels.loc[:, codes].to_numpy("int8")
    at_risk = prior.loc[labels.index, codes].to_numpy() == 0
    tables = []
    per_target = []
    summary: dict[str, Any] = {
        "scope": "training-split nested cross-validation; official val not read",
        "source": source,
        "n_bins": N_BINS,
        "fold_table": folds,
        "folds_with_lower_brier": improved,
        "adopt_rule": "calibrated Brier lower on at least 4 of 5 folds",
        "adopted": improved >= 4,
        "pooled": {},
    }
    for name, frame in (("raw", raw), ("calibrated", calibrated)):
        p_all = frame.loc[labels.index, codes].to_numpy("float64")
        evaluation = evaluate_predictions(labels, frame, codes, prior=prior)
        target_table = per_target_calibration(labels, frame, prior, codes)
        summary["pooled"][name] = {
            **evaluation.summary,
            "macro_ece": float(target_table["ece"].mean()),
            "pooled_ece_all_cells": expected_calibration_error(y_all[at_risk], p_all[at_risk]),
        }
        tables.append(reliability_table(y_all[at_risk], p_all[at_risk], label=name))
        per_target.append(target_table.add_suffix(f"_{name}"))
    pd.concat(tables).to_csv(reports_dir / "reliability.csv", index=False)
    pd.concat(per_target, axis=1).to_csv(reports_dir / "per_target.csv")
    calibrated.to_parquet(reports_dir / "oof_calibrated.parquet")
    write_json(reports_dir / "platt_weights.json", fit_platt(raw, labels, prior, codes))
    write_json(reports_dir / "calibration.json", summary)
    log.info(
        "Platt scaling: Brier %.4f -> %.4f, macro ECE %.4f -> %.4f, lower Brier on %d of %d folds",
        summary["pooled"]["raw"]["macro_brier"],
        summary["pooled"]["calibrated"]["macro_brier"],
        summary["pooled"]["raw"]["macro_ece"],
        summary["pooled"]["calibrated"]["macro_ece"],
        improved,
        N_FOLDS,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, nargs="*", default=None)
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(args.dataset, args.models, args.reports, args.transformer or None, args.oof)


if __name__ == "__main__":
    main()
