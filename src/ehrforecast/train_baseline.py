"""Step 2: logistic-regression baselines on the leak-free dataset from step 1.

Protocol:

1. Pick the regularisation strength ``C`` by k-fold cross-validation on the training split only.
   The feature vocabulary is refitted inside every fold.
2. Fit the final model on the whole training split with the chosen ``C``.
3. Score the validation split once. Validation data never changes a model choice, so its
   metrics are an unbiased held-out estimate. The test labels are hidden, so validation is the
   only held-out evaluation available to us.

Two feature sets are compared: ``demographics`` (age and sex only) and ``codes`` (bag-of-codes
plus age and sex). The gap between them shows how much the clinical history adds.

Outputs: the fitted ``codes`` model goes to ``--models``; metrics, per-target tables,
cross-validation results, top features, and val predictions go to ``--reports``.

Usage::

    python -m ehrforecast.train_baseline --dataset data/processed --models models/baseline \
        --reports reports/baseline --log-file logs/train_baseline.log
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ehrforecast.baseline import MultiLabelLogistic
from ehrforecast.config import RANDOM_SEED
from ehrforecast.features import BagOfCodes, FeatureConfig
from ehrforecast.logging_utils import setup_logging
from ehrforecast.metrics import bootstrap_summary, per_label_metrics, summarize

log = logging.getLogger("ehrforecast.train_baseline")

FEATURE_SETS = {
    "demographics": FeatureConfig(include_codes=False),
    "codes": FeatureConfig(),
}
DEFAULT_C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)
EVENT_COLUMNS = ["patient_id", "time", "source", "kind", "code"]


def cross_validate_c(
    events: pd.DataFrame,
    train: pd.DataFrame,
    y_train: np.ndarray,
    codes: list[str],
    feature_config: FeatureConfig,
    c_grid: tuple[float, ...],
    n_folds: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Return one row per (fold, C) with macro AUROC and mAP on the held-out fold."""
    rows = []
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (fit_idx, held_idx) in enumerate(folds.split(train)):
        fit_part, held_part = train.iloc[fit_idx], train.iloc[held_idx]
        featurizer = BagOfCodes(feature_config).fit(events, fit_part)
        x_fit = featurizer.transform(events, fit_part).to_numpy()
        x_held = featurizer.transform(events, held_part).to_numpy()
        for c in c_grid:
            model = MultiLabelLogistic(C=c, n_jobs=n_jobs).fit(x_fit, y_train[fit_idx])
            scores = summarize(y_train[held_idx], model.predict_proba(x_held), codes)
            rows.append(
                {
                    "fold": fold,
                    "C": c,
                    "n_features": x_fit.shape[1],
                    "not_converged": model.n_not_converged_,
                    **scores,
                }
            )
        log.info("  fold %d/%d done (%d features)", fold + 1, n_folds, x_fit.shape[1])
    return pd.DataFrame(rows)


def select_c(cv: pd.DataFrame, c_grid: tuple[float, ...]) -> float:
    mean_auroc = cv.groupby("C")["macro_auroc"].mean()
    best = float(mean_auroc.index.to_numpy(dtype="float64")[int(np.argmax(mean_auroc.to_numpy()))])
    if best in (min(c_grid), max(c_grid)):
        log.warning("Best C=%g is on the edge of the grid; consider widening it", best)
    return best


def top_features(
    model: MultiLabelLogistic,
    feature_names: list[str],
    codes: list[str],
    descriptions: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Largest positive coefficients per target, for interpretation and leakage review."""
    lookup = {
        f"{row.source}:{row.code}": row.description for row in descriptions.itertuples(index=False)
    }
    coef = model.coefficients()
    rows = []
    for j, code in enumerate(codes):
        for rank, i in enumerate(np.argsort(-coef[j])[:k], start=1):
            name = feature_names[i]
            token = name.split(":", 1)[1] if ":" in name else name
            rows.append(
                {
                    "target": code,
                    "rank": rank,
                    "feature": name,
                    "description": lookup.get(token, ""),
                    "coefficient": round(float(coef[j, i]), 4),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    y_val: np.ndarray,
    p_val: np.ndarray,
    prior_val: np.ndarray,
    codes: list[str],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary: dict[str, Any] = summarize(y_val, p_val, codes)
    summary["bootstrap_95ci"] = bootstrap_summary(y_val, p_val, n_boot=n_boot, seed=seed)
    table = per_label_metrics(y_val, p_val, codes)
    at_risk = per_label_metrics(y_val, p_val, codes, mask=(prior_val == 0))
    table["auroc_at_risk"] = at_risk["auroc"]
    table["n_at_risk"] = at_risk["n"]
    summary["macro_auroc_at_risk"] = float(at_risk["auroc"].mean())
    summary["mAP_at_risk"] = float(at_risk["average_precision"].mean())
    return summary, table


def run(
    dataset_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    n_folds: int = 5,
    n_boot: int = 1000,
    n_jobs: int = -1,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    cohort = pd.read_parquet(dataset_dir / "cohort.parquet")
    events = pd.read_parquet(dataset_dir / "events.parquet", columns=EVENT_COLUMNS)
    labels = pd.read_parquet(dataset_dir / "labels.parquet")
    prior = pd.read_parquet(dataset_dir / "prior_conditions.parquet")
    descriptions = pd.read_parquet(dataset_dir / "code_descriptions.parquet")
    targets = pd.read_csv(dataset_dir / "targets.csv", dtype=str)
    codes = targets["CODE"].tolist()
    if list(labels.columns) != codes or list(prior.columns) != codes:
        raise ValueError("label columns do not match targets.csv")

    train = cohort[cohort["split"] == "train"]
    val = cohort[cohort["split"] == "val"]
    if set(train.index) & set(val.index):
        raise AssertionError("train and val overlap")
    y_train = labels.loc[train.index].to_numpy()
    y_val = labels.loc[val.index].to_numpy()
    prior_val = prior.loc[val.index].to_numpy()
    log.info("Train %d patients, val %d patients, %d targets", len(train), len(val), len(codes))

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "protocol": {
            "c_grid": list(c_grid),
            "n_folds": n_folds,
            "n_boot": n_boot,
            "seed": seed,
            "selection_metric": "mean cross-validated macro AUROC on train",
        },
        "feature_sets": {},
    }
    cv_tables, per_label_tables, val_predictions = [], [], {}
    for name, feature_config in FEATURE_SETS.items():
        started = time.perf_counter()
        log.info("Feature set '%s': cross-validating C on train", name)
        cv = cross_validate_c(
            events, train, y_train, codes, feature_config, c_grid, n_folds, seed, n_jobs
        )
        best_c = select_c(cv, c_grid)

        featurizer = BagOfCodes(feature_config).fit(events, train)
        x_train = featurizer.transform(events, train)
        x_val = featurizer.transform(events, val)
        model = MultiLabelLogistic(C=best_c, n_jobs=n_jobs).fit(x_train.to_numpy(), y_train)
        p_val = model.predict_proba(x_val.to_numpy())

        summary, table = evaluate(y_val, p_val, prior_val, codes, n_boot, seed)
        cv_mean = cv[cv["C"] == best_c][["macro_auroc", "mAP"]].mean()
        summary.update(
            {
                "best_C": best_c,
                "n_features": x_train.shape[1],
                "final_model_not_converged": model.n_not_converged_,
                "cv_macro_auroc_at_best_C": float(cv_mean["macro_auroc"]),
                "cv_mAP_at_best_C": float(cv_mean["mAP"]),
                "feature_config": asdict(feature_config),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        metrics["feature_sets"][name] = summary
        cv_tables.append(cv.assign(feature_set=name))
        per_label_tables.append(table.add_prefix(f"{name}_"))
        val_predictions[name] = pd.DataFrame(p_val, index=val.index, columns=codes)
        log.info(
            "  %s: C=%g  val macro AUROC %.3f [%.3f, %.3f]  val mAP %.3f [%.3f, %.3f]",
            name,
            best_c,
            summary["macro_auroc"],
            summary["bootstrap_95ci"]["macro_auroc"]["ci_low"],
            summary["bootstrap_95ci"]["macro_auroc"]["ci_high"],
            summary["mAP"],
            summary["bootstrap_95ci"]["mAP"]["ci_low"],
            summary["bootstrap_95ci"]["mAP"]["ci_high"],
        )

        if name == "codes":
            joblib.dump(
                {
                    "featurizer": featurizer,
                    "model": model,
                    "feature_names": list(x_train.columns),
                    "codes": codes,
                },
                models_dir / "baseline_codes_model.joblib",
            )
            top_features(model, list(x_train.columns), codes, descriptions).to_csv(
                reports_dir / "top_features_codes.csv", index=False
            )

    per_label = pd.concat(per_label_tables, axis=1)
    per_label.insert(0, "description", targets.set_index("CODE")["DESCRIPTION"])
    per_label.insert(1, "train_positives", y_train.sum(axis=0))
    per_label.to_csv(reports_dir / "per_condition_val.csv", index_label="code")
    pd.concat(cv_tables, ignore_index=True).to_csv(reports_dir / "cv_results.csv", index=False)
    for name, frame in val_predictions.items():
        frame.to_csv(reports_dir / f"val_predictions_{name}.csv", index_label="patient_id")
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    suspicious = per_label.index[per_label["codes_auroc"] > 0.98].tolist()
    if suspicious:
        log.warning("Very high val AUROC for %s; review top features for leakage", suspicious)
    log.info("Wrote model to %s and reports to %s", models_dir, reports_dir)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="processed-data folder")
    parser.add_argument("--models", type=Path, required=True, help="model output folder")
    parser.add_argument("--reports", type=Path, required=True, help="report output folder")
    parser.add_argument("--log-file", type=Path, default=None, help="optional log file")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(
        args.dataset,
        args.models,
        args.reports,
        n_folds=args.folds,
        n_boot=args.bootstrap,
        n_jobs=args.jobs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
