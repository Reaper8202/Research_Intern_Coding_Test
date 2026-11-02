"""Step 2: reference baselines on the leak-free dataset from step 1.

Models:

* ``incidence_age_sex``: observed five-year incidence in the patient's sex and age band.
* ``incidence_age_sex_at_risk``: the same, counting only patients without the condition, and
  probability 0 for patients who already have it.
* ``logreg_demographics``: logistic regression on age, age bands, and sex.
* ``logreg_codes``: logistic regression on bag-of-codes counts plus age and sex.

Protocol, identical for every model:

1. Choose hyperparameters by k-fold cross-validation on the training split only. All models use
   the same folds. Feature vocabularies and incidence tables are refitted inside every fold.
2. Fit the final model on the whole training split with the chosen hyperparameters.
3. Score the validation split once. Validation data never changes a model choice, so its
   metrics are an unbiased held-out estimate. The test labels are hidden, so validation is the
   only held-out evaluation available to us.
4. Compare ``logreg_codes`` with every other model on the same val patients (paired BCa
   bootstrap of the difference).

Outputs: the fitted ``logreg_codes`` model goes to ``--models``; metrics, per-target tables,
cross-validation results, top features, and val predictions go to ``--reports``.

Usage::

    python -m ehrforecast.train_baseline --dataset data/processed --models models/baseline \
        --reports reports/baseline --log-file logs/train_baseline.log
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ehrforecast.baseline import MultiLabelLogistic
from ehrforecast.config import RANDOM_SEED
from ehrforecast.features import BagOfCodes, FeatureConfig
from ehrforecast.incidence import StratifiedIncidence
from ehrforecast.logging_utils import setup_logging
from ehrforecast.metrics import (
    bootstrap_summary,
    paired_bootstrap_difference,
    per_label_metrics,
    summarize,
)

log = logging.getLogger("ehrforecast.train_baseline")

INCIDENCE_MODELS = {
    "incidence_age_sex": False,
    "incidence_age_sex_at_risk": True,
}
LOGREG_MODELS = {
    "logreg_demographics": FeatureConfig(include_codes=False),
    "logreg_codes": FeatureConfig(),
}
REFERENCE_MODEL = "logreg_codes"

DEFAULT_C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)
DEFAULT_BAND_GRID = (2.5, 5.0, 10.0, 20.0)
DEFAULT_PRIOR_STRENGTH_GRID = (1.0, 10.0, 100.0)
EVENT_COLUMNS = ["patient_id", "time", "source", "kind", "code"]


@dataclass
class Data:
    """Everything the baselines need, already split into train and val."""

    events: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    prior_train: np.ndarray
    prior_val: np.ndarray
    codes: list[str]


@dataclass
class ModelResult:
    cv: pd.DataFrame
    params: dict[str, float]
    p_val: np.ndarray
    extras: dict[str, Any]


def load_data(dataset_dir: Path) -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    """Load step 1 outputs; also return the target table and the code descriptions."""
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
    data = Data(
        events=events,
        train=train,
        val=val,
        y_train=labels.loc[train.index].to_numpy(),
        y_val=labels.loc[val.index].to_numpy(),
        prior_train=prior.loc[train.index].to_numpy(),
        prior_val=prior.loc[val.index].to_numpy(),
        codes=codes,
    )
    return data, targets, descriptions


def select_best(cv: pd.DataFrame, grids: dict[str, tuple[float, ...]]) -> dict[str, float]:
    """Return the hyperparameters with the highest mean cross-validated macro AUROC."""
    names = list(grids)
    mean_auroc = cv.groupby(names)["macro_auroc"].mean()
    best_key = mean_auroc.index[int(np.argmax(mean_auroc.to_numpy()))]
    values = best_key if isinstance(best_key, tuple) else (best_key,)
    best = {name: float(v) for name, v in zip(names, values, strict=True)}
    for name, value in best.items():
        grid = grids[name]
        if len(grid) > 1 and value in (min(grid), max(grid)):
            log.warning("Best %s=%g is on the edge of its grid", name, value)
    return best


def run_incidence(
    data: Data,
    at_risk_only: bool,
    folds: list[tuple[np.ndarray, np.ndarray]],
    band_grid: tuple[float, ...],
    strength_grid: tuple[float, ...],
) -> ModelResult:
    rows = []
    for fold, (fit_idx, held_idx) in enumerate(folds):
        fit_part, held_part = data.train.iloc[fit_idx], data.train.iloc[held_idx]
        for band, strength in itertools.product(band_grid, strength_grid):
            model = StratifiedIncidence(band, strength, at_risk_only).fit(
                fit_part, data.y_train[fit_idx], data.prior_train[fit_idx]
            )
            scores = model.predict_proba(held_part, data.prior_train[held_idx])
            rows.append(
                {
                    "fold": fold,
                    "band_years": band,
                    "prior_strength": strength,
                    **summarize(data.y_train[held_idx], scores, data.codes),
                }
            )
    cv = pd.DataFrame(rows)
    params = select_best(cv, {"band_years": band_grid, "prior_strength": strength_grid})
    model = StratifiedIncidence(params["band_years"], params["prior_strength"], at_risk_only).fit(
        data.train, data.y_train, data.prior_train
    )
    p_val = model.predict_proba(data.val, data.prior_val)
    return ModelResult(cv=cv, params=params, p_val=p_val, extras={"n_strata": len(model.strata_)})


def run_logreg(
    data: Data,
    feature_config: FeatureConfig,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_grid: tuple[float, ...],
    n_jobs: int,
) -> ModelResult:
    rows = []
    for fold, (fit_idx, held_idx) in enumerate(folds):
        fit_part, held_part = data.train.iloc[fit_idx], data.train.iloc[held_idx]
        featurizer = BagOfCodes(feature_config).fit(data.events, fit_part)
        x_fit = featurizer.transform(data.events, fit_part).to_numpy()
        x_held = featurizer.transform(data.events, held_part).to_numpy()
        for c in c_grid:
            model = MultiLabelLogistic(C=c, n_jobs=n_jobs).fit(x_fit, data.y_train[fit_idx])
            rows.append(
                {
                    "fold": fold,
                    "C": c,
                    "n_features": x_fit.shape[1],
                    "not_converged": model.n_not_converged_,
                    **summarize(data.y_train[held_idx], model.predict_proba(x_held), data.codes),
                }
            )
        log.info("  fold %d/%d done (%d features)", fold + 1, len(folds), x_fit.shape[1])
    cv = pd.DataFrame(rows)
    params = select_best(cv, {"C": c_grid})

    featurizer = BagOfCodes(feature_config).fit(data.events, data.train)
    x_train = featurizer.transform(data.events, data.train)
    x_val = featurizer.transform(data.events, data.val)
    model = MultiLabelLogistic(C=params["C"], n_jobs=n_jobs).fit(x_train.to_numpy(), data.y_train)
    return ModelResult(
        cv=cv,
        params=params,
        p_val=model.predict_proba(x_val.to_numpy()),
        extras={
            "n_features": x_train.shape[1],
            "final_model_not_converged": model.n_not_converged_,
            "feature_config": asdict(feature_config),
            "featurizer": featurizer,
            "model": model,
            "feature_names": list(x_train.columns),
        },
    )


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
    band_grid: tuple[float, ...] = DEFAULT_BAND_GRID,
    strength_grid: tuple[float, ...] = DEFAULT_PRIOR_STRENGTH_GRID,
    n_folds: int = 5,
    n_boot: int = 1000,
    n_jobs: int = -1,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    data, targets, descriptions = load_data(dataset_dir)
    codes = data.codes
    log.info(
        "Train %d patients, val %d patients, %d targets", len(data.train), len(data.val), len(codes)
    )
    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(data.train))

    results: dict[str, ModelResult] = {}
    for name, at_risk_only in INCIDENCE_MODELS.items():
        log.info("Model '%s': cross-validating on train", name)
        results[name] = run_incidence(data, at_risk_only, folds, band_grid, strength_grid)
    for name, feature_config in LOGREG_MODELS.items():
        log.info("Model '%s': cross-validating on train", name)
        results[name] = run_logreg(data, feature_config, folds, c_grid, n_jobs)

    metrics: dict[str, Any] = {
        "protocol": {
            "c_grid": list(c_grid),
            "band_years_grid": list(band_grid),
            "prior_strength_grid": list(strength_grid),
            "n_folds": n_folds,
            "n_boot": n_boot,
            "seed": seed,
            "selection_metric": "mean cross-validated macro AUROC on train",
            "reference_model": REFERENCE_MODEL,
        },
        "models": {},
        "comparisons": {},
    }
    per_label_tables, cv_tables = [], []
    for name, result in results.items():
        summary, table = evaluate(data.y_val, result.p_val, data.prior_val, codes, n_boot, seed)
        best = result.cv
        for param, value in result.params.items():
            best = best[best[param] == value]
        summary.update(
            {
                "params": result.params,
                "cv_macro_auroc": float(best["macro_auroc"].mean()),
                "cv_mAP": float(best["mAP"].mean()),
                **{
                    k: v
                    for k, v in result.extras.items()
                    if k not in ("featurizer", "model", "feature_names")
                },
            }
        )
        metrics["models"][name] = summary
        per_label_tables.append(table.add_prefix(f"{name}_"))
        cv_tables.append(result.cv.assign(model=name))
        ci = summary["bootstrap_95ci"]
        log.info(
            "  %-26s val macro AUROC %.3f [%.3f, %.3f]  mAP %.3f [%.3f, %.3f]  params %s",
            name,
            summary["macro_auroc"],
            ci["macro_auroc"]["ci_low"],
            ci["macro_auroc"]["ci_high"],
            summary["mAP"],
            ci["mAP"]["ci_low"],
            ci["mAP"]["ci_high"],
            result.params,
        )

    reference = results[REFERENCE_MODEL].p_val
    for name, result in results.items():
        if name == REFERENCE_MODEL:
            continue
        diff = paired_bootstrap_difference(
            data.y_val, reference, result.p_val, n_boot=n_boot, seed=seed
        )
        metrics["comparisons"][f"{REFERENCE_MODEL}_minus_{name}"] = diff
        log.info(
            "  %s - %s: macro AUROC %+.3f [%+.3f, %+.3f]  mAP %+.3f [%+.3f, %+.3f]",
            REFERENCE_MODEL,
            name,
            diff["macro_auroc"]["difference"],
            diff["macro_auroc"]["ci_low"],
            diff["macro_auroc"]["ci_high"],
            diff["mAP"]["difference"],
            diff["mAP"]["ci_low"],
            diff["mAP"]["ci_high"],
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    extras = results[REFERENCE_MODEL].extras
    joblib.dump(
        {
            "featurizer": extras["featurizer"],
            "model": extras["model"],
            "feature_names": extras["feature_names"],
            "codes": codes,
        },
        models_dir / f"baseline_{REFERENCE_MODEL}.joblib",
    )
    top_features(extras["model"], extras["feature_names"], codes, descriptions).to_csv(
        reports_dir / f"top_features_{REFERENCE_MODEL}.csv", index=False
    )
    per_label = pd.concat(per_label_tables, axis=1)
    per_label.insert(0, "description", targets.set_index("CODE")["DESCRIPTION"])
    per_label.insert(1, "train_positives", data.y_train.sum(axis=0))
    per_label.to_csv(reports_dir / "per_condition_val.csv", index_label="code")
    pd.concat(cv_tables, ignore_index=True).to_csv(reports_dir / "cv_results.csv", index=False)
    for name, result in results.items():
        pd.DataFrame(result.p_val, index=data.val.index, columns=codes).to_csv(
            reports_dir / f"val_predictions_{name}.csv", index_label="patient_id"
        )
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    suspicious = per_label.index[per_label[f"{REFERENCE_MODEL}_auroc"] > 0.98].tolist()
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
