"""Stack the transformer ensemble with the LightGBM baselines, on training CV only.

Inputs are out-of-fold probabilities that already exist: the transformer seed ensemble
(``lightgbm_cv.ensemble_oof``) and the saved LightGBM feature sets. Every base prediction for a
patient comes from models that did not fit on that patient. The stacker itself is also
cross-validated: for each fold, its per-target weights are fitted on the other four folds' rows
and applied to that fold, so the reported scores are nested out-of-fold. Official val is not read.

Stackers, per target, on at-risk rows only (masked cells are exact zeros and stay zero):

* ``logit_mean``: mean of the base logits (no fitted parameter).
* ``prob_mean``: mean of the base probabilities (no fitted parameter).
* ``logistic``: L2 logistic regression on the base logits with an intercept (C fixed), fitted on
  the other folds. A target with one class among those rows falls back to ``logit_mean``.

Usage::

    python -m ehrforecast.stacking --dataset data/processed --models models \
        --transformer configs/experiments/v2_c1.toml configs/experiments/v2_c1_seed2.toml \
        configs/experiments/v2_c1_seed3.toml --lightgbm lightgbm_clean \
        --reports reports/stacking/clean
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
from ehrforecast.evaluation import bootstrap_comparison, evaluate_predictions
from ehrforecast.folds import N_FOLDS, fold_patients, load_folds
from ehrforecast.lightgbm_cv import ensemble_oof
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.provenance import write_json

log = logging.getLogger("ehrforecast.stacking")

EPS = 1e-6
STACKERS = ("logit_mean", "prob_mean", "logistic")
C_FIXED = 1.0


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def stack_target(
    base: list[np.ndarray],
    y_fit: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    stacker: str,
) -> np.ndarray:
    """Probabilities for ``apply_rows`` from base probabilities (one array per base model)."""
    z = np.column_stack([logit(b) for b in base])
    if stacker == "prob_mean":
        return np.column_stack(base)[apply_rows].mean(axis=1)
    if stacker == "logit_mean" or len(np.unique(y_fit)) < 2:
        return sigmoid(z[apply_rows].mean(axis=1))
    model = LogisticRegression(C=C_FIXED, max_iter=1000)
    model.fit(z[fit_rows], y_fit)
    return model.predict_proba(z[apply_rows])[:, 1]


def nested_stack(
    bases: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    prior: pd.DataFrame,
    codes: list[str],
    assignments: pd.DataFrame,
    stacker: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nested out-of-fold stacked probabilities and the fitted weights per fold and target."""
    if stacker not in STACKERS:
        raise ValueError(f"stacker must be one of {STACKERS}")
    index = labels.index
    arrays = [b.loc[index, codes].to_numpy("float64") for b in bases.values()]
    y = labels.loc[index, codes].to_numpy("int8")
    at_risk = prior.loc[index, codes].to_numpy() == 0
    out = np.zeros_like(arrays[0])
    weights: dict[str, Any] = {}
    position = pd.Series(np.arange(len(index)), index=index)
    for fold in range(N_FOLDS):
        fit, held = fold_patients(assignments, fold)
        fit_pos = position.loc[position.index.intersection(fit)].to_numpy()
        held_pos = position.loc[position.index.intersection(held)].to_numpy()
        for j, code in enumerate(codes):
            fit_rows = fit_pos[at_risk[fit_pos, j]]
            apply_rows = held_pos[at_risk[held_pos, j]]
            base = [a[:, j] for a in arrays]
            out[apply_rows, j] = stack_target(base, y[fit_rows, j], fit_rows, apply_rows, stacker)
            if stacker == "logistic" and len(np.unique(y[fit_rows, j])) == 2:
                z = np.column_stack([logit(b) for b in base])
                m = LogisticRegression(C=C_FIXED, max_iter=1000).fit(z[fit_rows], y[fit_rows, j])
                weights.setdefault(code, {})[f"fold{fold}"] = {
                    **{name: float(c) for name, c in zip(bases, m.coef_[0], strict=True)},
                    "intercept": float(m.intercept_[0]),
                }
    frame = pd.DataFrame(out, index=index, columns=codes)
    return apply_prior_mask(frame, prior.loc[index], codes), weights


def run(
    dataset_dir: Path,
    models_dir: Path,
    transformer_configs: list[Path],
    lightgbm_name: str,
    feature_sets: tuple[str, ...],
    reports_dir: Path,
    n_boot: int,
) -> dict[str, Any]:
    data = load_training_data(dataset_dir)
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    assignments = load_folds(dataset_dir)
    bases: dict[str, pd.DataFrame] = {
        "transformer": ensemble_oof(models_dir, transformer_configs, assignments).loc[labels.index]
    }
    for name in feature_sets:
        bases[f"lightgbm_{name}"] = pd.read_parquet(
            models_dir / lightgbm_name / f"oof_{name}.parquet"
        ).loc[labels.index]
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scope": "training-split nested cross-validation; official val not read",
        "bases": list(bases),
        "transformer_configs": [str(p) for p in transformer_configs],
        "lightgbm": lightgbm_name,
        "base_pooled": {},
        "stacked_pooled": {},
        "fold_scores": {},
        "paired_bca": {},
    }
    per_target = []
    for name, probabilities in bases.items():
        evaluation = evaluate_predictions(labels, probabilities, codes, prior=prior)
        summary["base_pooled"][name] = evaluation.summary
        per_target.append(evaluation.per_target.assign(model=name))
    # Oracle ceiling: the best base model per target, chosen on the same data (not a model).
    base_auroc = (
        pd.concat(per_target)
        .reset_index()
        .pivot_table(index="code", columns="model", values="auroc", aggfunc="first")
    )
    summary["oracle_best_base_per_target_macro_auroc"] = float(base_auroc.max(axis=1).mean())
    stacked: dict[str, pd.DataFrame] = {}
    for stacker in STACKERS:
        probabilities, weights = nested_stack(bases, labels, prior, codes, assignments, stacker)
        stacked[stacker] = probabilities
        evaluation = evaluate_predictions(labels, probabilities, codes, prior=prior)
        summary["stacked_pooled"][stacker] = evaluation.summary
        per_target.append(evaluation.per_target.assign(model=f"stack_{stacker}"))
        summary["fold_scores"][stacker] = [
            evaluate_predictions(
                labels.loc[held], probabilities.loc[held], codes, prior=prior.loc[held]
            ).summary
            for fold in range(N_FOLDS)
            for _, held in [fold_patients(assignments, fold)]
        ]
        if weights:
            write_json(reports_dir / f"weights_{stacker}.json", weights)
        log.info(
            "%s: pooled macro AUROC %.4f, mAP %.4f",
            stacker,
            evaluation.summary["macro_auroc"],
            evaluation.summary["mAP"],
        )
    pd.concat(per_target).to_csv(reports_dir / "per_target.csv")
    for stacker, probabilities in stacked.items():
        for name, reference in bases.items():
            summary["paired_bca"][f"stack_{stacker}_minus_{name}"] = bootstrap_comparison(
                labels, probabilities, codes, other=reference, prior=prior, n_boot=n_boot
            )
        probabilities.to_parquet(reports_dir / f"oof_stack_{stacker}.parquet")
    write_json(reports_dir / "stacking.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, nargs="+", required=True)
    parser.add_argument("--lightgbm", default="lightgbm_clean")
    parser.add_argument("--feature-sets", nargs="+", default=["counts", "counts_values"])
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(
        args.dataset,
        args.models,
        args.transformer,
        args.lightgbm,
        tuple(args.feature_sets),
        args.reports,
        args.n_boot,
    )


if __name__ == "__main__":
    main()
