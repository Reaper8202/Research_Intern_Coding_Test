"""E11: LightGBM comparison baselines on the five training folds (model plan section 7.2).

Feature sets, all refitted inside each fold on its fit patients:

* ``counts``: ``BagOfCodes`` code counts, age, age bands, and sex.
* ``counts_values``: counts plus, for each numeric observation code and unit in at least five fit
  patients, the latest value before the anchor (median of distinct values at that latest time),
  a missingness flag, and log(1 + days since the measurement). Missing values take fit-patient
  medians. Values stay continuous; no clinical ranges or outside code maps are used.
* ``counts_binned``: ``counts_values`` with each latest value replaced by its quantile-bin index
  (``numeric.fit_bin_edges`` on all fit-patient values of the group, as in the fold tokenizer).
  This is the training-only value-bin information check; it is never submitted.

One binary classifier per target, trained on at-risk fit patients only (no prior condition).
Grid: ``num_leaves`` {4, 8, 16} x ``min_child_samples`` {20, 50} x ``n_estimators``
{100, 300, 600}, one setting per feature set shared by all targets, selected by mean CV masked
macro AUROC. Boosting adds trees in sequence with a fixed seed, so the first 100 or 300 trees of a
600-tree model are the 100- or 300-tree model; one fit covers the three tree counts.

Writes ``reports/lightgbm/``: the grid, the selection with grid-edge flags, pooled out-of-fold
metrics, paired patient BCa against the transformer and logistic regression, and the value-bin
check. Out-of-fold probabilities go to ``models/lightgbm/``. Official val is not read.

Usage::

    python -m ehrforecast.lightgbm_cv --dataset data/processed --models models \
        --reports reports/lightgbm --compare-config configs/multimodal.toml
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
import tomllib
import warnings
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd

from ehrforecast.contracts import load_training_data
from ehrforecast.evaluation import bootstrap_comparison, evaluate_predictions
from ehrforecast.features import BagOfCodes
from ehrforecast.folds import N_FOLDS, fold_patients, load_folds
from ehrforecast.logging_utils import setup_logging
from ehrforecast.numeric import assign_bins, fit_bin_edges, numeric_observations
from ehrforecast.provenance import write_json

log = logging.getLogger("ehrforecast.lightgbm_cv")
# LightGBM names numpy columns itself; scikit-learn then warns on every numpy prediction.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

FEATURE_SETS = ("counts", "counts_values", "counts_binned")
NUM_LEAVES = (4, 8, 16)
MIN_CHILD_SAMPLES = (20, 50)
N_ESTIMATORS = (100, 300, 600)
MIN_VALUE_PATIENTS = 5
MATERIAL = 0.005
SEED = 20261028
THREADS = 4
FIXED: dict[str, Any] = {
    "learning_rate": 0.05,
    "colsample_bytree": 0.5,  # feature fraction
    "subsample": 0.8,  # bagging fraction
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": THREADS,
    "random_state": SEED,
    "verbose": -1,
}


# This module must not import PyTorch: PyTorch and LightGBM each link a ``libomp.dylib`` on macOS,
# and two OpenMP runtimes in one process can crash. The transformer predictions are read directly.
def transformer_oof(
    models_dir: Path, config_path: Path, assignments: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    """Name and pooled held-out F probabilities of a transformer configuration."""
    name = str(tomllib.loads(config_path.read_text())["name"])
    parts = []
    for fold in range(N_FOLDS):
        path = (
            models_dir
            / "transformer"
            / name
            / f"fold{fold}"
            / "F"
            / "held_predictions_best.parquet"
        )
        _, held = fold_patients(assignments, fold)
        frame = pd.read_parquet(path)
        if set(frame.index) != set(held):
            raise ValueError(f"{path} does not hold exactly fold {fold}'s held-out patients")
        parts.append(frame)
    return name, pd.concat(parts)


# ---------------------------------------------------------------------- value features
def latest_values(events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per patient and ``code\\tunit``: latest value and log(1 + days before anchor)."""
    obs = numeric_observations(events)
    obs["key"] = obs["code"].astype(str) + "\t" + obs["unit"].mask(obs["unit"].eq(""), "nounit")
    last = obs.groupby(["patient_id", "key"])["time"].transform("max")
    latest = obs.loc[obs["time"].eq(last)]
    out = latest.groupby(["patient_id", "key"], as_index=False).agg(
        value=("value_num", "median"), time=("time", "first")
    )
    anchor = out["patient_id"].map(cohort["anchor_date"])
    days = (anchor - out["time"]).dt.total_seconds() / 86_400
    if (days < 0).any():
        raise ValueError("an observation is after its patient's anchor")
    out["log_days"] = np.log1p(days)
    return out.drop(columns="time")


def all_values(events: pd.DataFrame) -> pd.DataFrame:
    """Every numeric value with its ``code\\tunit`` key (for fit-patient bin edges)."""
    obs = numeric_observations(events)
    obs["key"] = obs["code"].astype(str) + "\t" + obs["unit"].mask(obs["unit"].eq(""), "nounit")
    return obs[["patient_id", "key", "value_num"]]


def value_features(
    latest: pd.DataFrame,
    values: pd.DataFrame,
    fit: pd.Index,
    patients: pd.Index,
    binned: bool,
) -> pd.DataFrame:
    """Latest-value, missingness, and recency columns; every statistic from ``fit`` only."""
    fit_latest = latest.loc[latest["patient_id"].isin(fit)]
    carriers = fit_latest.groupby("key")["patient_id"].nunique()
    keys = sorted(carriers.index[carriers >= MIN_VALUE_PATIENTS])
    rows = latest.loc[latest["key"].isin(keys) & latest["patient_id"].isin(patients)].copy()
    fit_rows = fit_latest.loc[fit_latest["key"].isin(keys)].copy()
    if binned:
        fit_values = values.loc[values["patient_id"].isin(fit) & values["key"].isin(keys)]
        edges = {k: fit_bin_edges(g.to_numpy()) for k, g in fit_values.groupby("key")["value_num"]}
        for frame in (rows, fit_rows):
            frame["value"] = [
                float(assign_bins(np.array([v]), edges[k])[0])
                for k, v in zip(frame["key"], frame["value"], strict=True)
            ]
    wide = rows.set_index(["patient_id", "key"])[["value", "log_days"]]
    value = wide["value"].unstack()  # noqa: PD010 - one row per patient and key
    days = wide["log_days"].unstack()  # noqa: PD010
    value = value.reindex(index=patients, columns=keys)
    days = days.reindex(index=patients, columns=keys)
    missing = value.isna().astype("float64")
    value_median = fit_rows.groupby("key")["value"].median().reindex(keys)
    days_median = fit_rows.groupby("key")["log_days"].median().reindex(keys)
    value = value.fillna(value_median)
    days = days.fillna(days_median)
    return pd.concat(
        [value.add_prefix("value:"), missing.add_prefix("missing:"), days.add_prefix("days:")],
        axis=1,
    ).astype("float32")


def fold_features(
    data: Any,
    latest: pd.DataFrame,
    values: pd.DataFrame,
    fit: pd.Index,
    held: pd.Index,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    events, cohort = data.inputs.events, data.inputs.cohort
    counts = BagOfCodes().fit(events, cohort.loc[fit])
    x_fit = counts.transform(events, cohort.loc[fit])
    x_held = counts.transform(events, cohort.loc[held])
    if feature_set != "counts":
        binned = feature_set == "counts_binned"
        x_fit = pd.concat([x_fit, value_features(latest, values, fit, fit, binned)], axis=1)
        x_held = pd.concat([x_held, value_features(latest, values, fit, held, binned)], axis=1)
    return x_fit.to_numpy("float32"), x_held.to_numpy("float32")


# ---------------------------------------------------------------------- fitting
def fit_target(
    x_fit: np.ndarray, y: np.ndarray, x_held: np.ndarray, num_leaves: int, min_child: int
) -> dict[int, np.ndarray]:
    """Held-out probabilities after each tree count in ``N_ESTIMATORS``."""
    if len(np.unique(y)) < 2:  # a single class among at-risk fit patients
        return {n: np.full(len(x_held), float(y.mean()) if len(y) else 0.0) for n in N_ESTIMATORS}
    model = lightgbm.LGBMClassifier(
        n_estimators=max(N_ESTIMATORS),
        num_leaves=num_leaves,
        min_child_samples=min_child,
        **FIXED,
    )
    model.fit(x_fit, y)
    return {n: model.predict_proba(x_held, num_iteration=n)[:, 1] for n in N_ESTIMATORS}


def run_feature_set(
    data: Any, assignments: pd.DataFrame, latest: pd.DataFrame, values: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], pd.DataFrame]]:
    """Grid scores per fold, and out-of-fold probabilities per setting."""
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    settings = list(itertools.product(NUM_LEAVES, MIN_CHILD_SAMPLES, N_ESTIMATORS))
    oof = {s: pd.DataFrame(np.nan, index=labels.index, columns=codes) for s in settings}
    rows = []
    for fold in range(N_FOLDS):
        started = time.perf_counter()
        fit, held = fold_patients(assignments, fold)
        x_fit, x_held = fold_features(data, latest, values, fit, held, name)
        for target in codes:
            at_risk = prior.loc[fit, target].to_numpy() == 0
            y = labels.loc[fit, target].to_numpy()[at_risk]
            for leaves, child in itertools.product(NUM_LEAVES, MIN_CHILD_SAMPLES):
                for n, p in fit_target(x_fit[at_risk], y, x_held, leaves, child).items():
                    oof[(leaves, child, n)].loc[held, target] = p
        for setting in settings:
            summary = evaluate_predictions(
                labels.loc[held], oof[setting].loc[held], codes, prior=prior.loc[held]
            ).summary
            rows.append(
                {
                    "feature_set": name,
                    "fold": fold,
                    "num_leaves": setting[0],
                    "min_child_samples": setting[1],
                    "n_estimators": setting[2],
                    "n_features": x_fit.shape[1],
                    "macro_auroc": summary["macro_auroc"],
                    "mAP": summary["mAP"],
                }
            )
        log.info(
            "%s fold %d: %d features, %.0f s",
            name,
            fold,
            x_fit.shape[1],
            time.perf_counter() - started,
        )
    return pd.DataFrame(rows), oof


def select_setting(grid: pd.DataFrame) -> dict[str, Any]:
    keys = ["num_leaves", "min_child_samples", "n_estimators"]
    means = grid.groupby(keys)[["macro_auroc", "mAP"]].mean().reset_index()
    best = means.sort_values(["macro_auroc", "mAP"], ascending=False).iloc[0]
    edges = {
        "num_leaves": NUM_LEAVES,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "n_estimators": N_ESTIMATORS,
    }
    return {
        **{k: int(best[k]) for k in keys},
        "mean_cv_macro_auroc": float(best["macro_auroc"]),
        "mean_cv_mAP": float(best["mAP"]),
        "grid_edge": [k for k in keys if int(best[k]) in (min(edges[k]), max(edges[k]))],
    }


def run(
    dataset_dir: Path, models_dir: Path, reports_dir: Path, compare: Path | None, n_boot: int
) -> dict[str, Any]:
    data = load_training_data(dataset_dir)
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    assignments = load_folds(dataset_dir)
    latest = latest_values(data.inputs.events, data.inputs.cohort)
    values = all_values(data.inputs.events)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lightgbm").mkdir(parents=True, exist_ok=True)
    grids, selected, oof = [], {}, {}
    for name in FEATURE_SETS:
        grid, probabilities = run_feature_set(data, assignments, latest, values, name)
        grids.append(grid)
        choice = select_setting(grid)
        selected[name] = choice
        key = (choice["num_leaves"], choice["min_child_samples"], choice["n_estimators"])
        oof[name] = probabilities[key]
        oof[name].to_parquet(models_dir / "lightgbm" / f"oof_{name}.parquet")
        selected[name]["fold_scores"] = (
            grid.loc[
                grid["num_leaves"].eq(key[0])
                & grid["min_child_samples"].eq(key[1])
                & grid["n_estimators"].eq(key[2]),
                ["fold", "macro_auroc", "mAP"],
            ]
            .sort_values("fold")
            .to_dict("records")
        )
    pd.concat(grids).to_csv(reports_dir / "grid.csv", index=False)

    folds_values = pd.DataFrame(selected["counts_values"]["fold_scores"]).set_index("fold")
    folds_binned = pd.DataFrame(selected["counts_binned"]["fold_scores"]).set_index("fold")
    difference = folds_binned - folds_values
    auroc_difference = float(difference["macro_auroc"].mean())
    map_difference = float(difference["mAP"].mean())
    value_bin_check: dict[str, Any] = {
        "binned_minus_values_per_fold": difference.reset_index().to_dict("records"),
        "mean_auroc_difference": auroc_difference,
        "mean_map_difference": map_difference,
        "material_loss": min(auroc_difference, map_difference) <= -MATERIAL,
    }

    summary: dict[str, Any] = {
        "scope": "training-split cross-validation; official val not read",
        "fixed_settings": FIXED,
        "selected": selected,
        "value_bin_check": value_bin_check,
        "pooled_oof": {
            name: evaluate_predictions(labels, p.loc[labels.index], codes, prior=prior).summary
            for name, p in oof.items()
        },
        "paired_bca_oof": {},
    }
    reference = pd.read_parquet(dataset_dir / "p3a_logreg_oof.parquet").loc[labels.index]
    others = {"logreg_historical": reference}
    if compare is not None:
        name, transformer = transformer_oof(models_dir, compare, assignments)
        others = {f"{name}_F": transformer.loc[labels.index], **others}
    for name in ("counts", "counts_values"):
        for other_name, other in others.items():
            label = (
                f"{other_name}_minus_lightgbm_{name}"
                if other_name.endswith("_F")
                else f"lightgbm_{name}_minus_{other_name}"
            )
            a, b = (other, oof[name]) if other_name.endswith("_F") else (oof[name], other)
            summary["paired_bca_oof"][label] = bootstrap_comparison(
                labels,
                a.loc[labels.index],
                codes,
                other=b.loc[labels.index],
                prior=prior,
                n_boot=n_boot,
            )
    write_json(reports_dir / "cv_summary.json", summary)
    return summary


def compare_saved(
    dataset_dir: Path, models_dir: Path, reports_dir: Path, configs: list[Path], n_boot: int
) -> dict[str, Any]:
    """Paired BCa of transformer F configurations against the saved LightGBM predictions."""
    data = load_training_data(dataset_dir)
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    assignments = load_folds(dataset_dir)
    out: dict[str, Any] = {"scope": "training-split cross-validation; official val not read"}
    for config in configs:
        name, transformer = transformer_oof(models_dir, config, assignments)
        transformer = transformer.loc[labels.index]
        out[name] = {
            "pooled": evaluate_predictions(labels, transformer, codes, prior=prior).summary
        }
        for feature_set in ("counts", "counts_values"):
            saved = pd.read_parquet(models_dir / "lightgbm" / f"oof_{feature_set}.parquet")
            out[name][f"minus_lightgbm_{feature_set}"] = bootstrap_comparison(
                labels,
                transformer,
                codes,
                other=saved.loc[labels.index],
                prior=prior,
                n_boot=n_boot,
            )
    write_json(reports_dir / "transformer_comparison.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--compare-config", type=Path, default=None)
    parser.add_argument(
        "--compare-only",
        type=Path,
        nargs="+",
        default=None,
        help="transformer configs to compare with the saved LightGBM predictions (no refit)",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    if args.compare_only:
        compare_saved(args.dataset, args.models, args.reports, args.compare_only, args.n_boot)
        return
    summary = run(args.dataset, args.models, args.reports, args.compare_config, args.n_boot)
    for name, choice in summary["selected"].items():
        log.info(
            "%s: AUROC %.4f mAP %.4f (leaves %d, min child %d, trees %d, edge %s)",
            name,
            choice["mean_cv_macro_auroc"],
            choice["mean_cv_mAP"],
            choice["num_leaves"],
            choice["min_child_samples"],
            choice["n_estimators"],
            choice["grid_edge"] or "none",
        )


if __name__ == "__main__":
    main()
