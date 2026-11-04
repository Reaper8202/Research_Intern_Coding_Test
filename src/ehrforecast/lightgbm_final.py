"""Final LightGBM models: fit on all training patients, save boosters and a feature spec, write
val and test probabilities (round 3; they are the stack's base models and the val comparisons).

Runs in its own process: this module must not import PyTorch (see ``lightgbm_cv``). The settings
are the ones training CV selected in ``reports/lightgbm_clean/cv_summary.json``; nothing is
tuned here. Each target's classifier fits on at-risk training patients, exactly as in E11. The
fit uses the E11 code path (``fold_features``, ``fit_target``), and a self-contained feature
spec (code vocabulary, value keys, medians, column order) is written beside the boosters so the
public package can rebuild the features without this module's fitting code. The spec path is
checked against the E11 path on every row before anything is written. No val or test label is
read.

Usage::

    python -m ehrforecast.lightgbm_final fit --dataset data/processed \
        --settings reports/lightgbm_clean/cv_summary.json --feature-sets counts counts_values \
        --models models/lightgbm_final --reports reports/lightgbm_final
    python -m ehrforecast.lightgbm_final predict --dataset data/processed \
        --models models/hf/lightgbm/counts --split test --out test_lightgbm_counts.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm
import numpy as np
import pandas as pd

from ehrforecast.contracts import (
    PredictionInputs,
    load_prediction_inputs,
    load_splits,
    load_training_data,
)
from ehrforecast.features import BagOfCodes, FeatureConfig
from ehrforecast.lightgbm_cv import (
    FIXED,
    MIN_VALUE_PATIENTS,
    N_ESTIMATORS,
    all_values,
    fold_features,
    latest_values,
)
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.provenance import file_hash, write_json
from ehrforecast.tokenizer import drop_observations

log = logging.getLogger("ehrforecast.lightgbm_final")

SPEC_VERSION = "lightgbm-feature-spec-v1"


# ---------------------------------------------------------------------- feature spec
def selected_settings(path: Path) -> tuple[dict[str, dict[str, int]], tuple[str, ...]]:
    """The CV-selected setting per feature set and the observation codes E11 excluded."""
    summary = json.loads(path.read_text())
    settings = {
        name: {k: int(choice[k]) for k in ("num_leaves", "min_child_samples", "n_estimators")}
        for name, choice in summary["selected"].items()
    }
    return settings, tuple(str(c) for c in summary.get("excluded_observations", []))


def fit_value_spec(latest: pd.DataFrame, fit: pd.Index) -> dict[str, Any]:
    """Keys with enough fit carriers and the fit medians that fill missing values."""
    fit_latest = latest.loc[latest["patient_id"].isin(fit)]
    carriers = fit_latest.groupby("key")["patient_id"].nunique()
    keys = sorted(carriers.index[carriers >= MIN_VALUE_PATIENTS])
    fit_rows = fit_latest.loc[fit_latest["key"].isin(keys)]
    return {
        "keys": keys,
        "value_median": fit_rows.groupby("key")["value"].median().reindex(keys).to_dict(),
        "days_median": fit_rows.groupby("key")["log_days"].median().reindex(keys).to_dict(),
    }


def value_features_from_spec(
    latest: pd.DataFrame, spec: dict[str, Any], patients: pd.Index
) -> pd.DataFrame:
    keys = list(spec["keys"])
    rows = latest.loc[latest["key"].isin(keys) & latest["patient_id"].isin(patients)]
    wide = rows.set_index(["patient_id", "key"])[["value", "log_days"]]
    value = wide["value"].unstack().reindex(index=patients, columns=keys)  # noqa: PD010
    days = wide["log_days"].unstack().reindex(index=patients, columns=keys)  # noqa: PD010
    missing = value.isna().astype("float64")
    value = value.fillna(pd.Series(spec["value_median"]).reindex(keys))
    days = days.fillna(pd.Series(spec["days_median"]).reindex(keys))
    return pd.concat(
        [value.add_prefix("value:"), missing.add_prefix("missing:"), days.add_prefix("days:")],
        axis=1,
    ).astype("float32")


def features_from_spec(
    spec: dict[str, Any], events: pd.DataFrame, cohort: pd.DataFrame, patients: pd.Index
) -> pd.DataFrame:
    """Rebuild the feature matrix for ``patients`` from the saved spec alone."""
    bag = BagOfCodes(FeatureConfig(**spec["feature_config"]), list(spec["vocabulary"]))
    frame = bag.transform(events, cohort.loc[patients])
    if spec.get("values") is not None:
        latest = latest_values(events.loc[events["patient_id"].isin(patients)], cohort)
        frame = pd.concat(
            [frame, value_features_from_spec(latest, spec["values"], patients)], axis=1
        )
    if list(frame.columns) != list(spec["columns"]):
        raise ValueError("rebuilt feature columns differ from the saved spec")
    return frame


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if spec.get("version") != SPEC_VERSION:
        raise ValueError(f"{path}: unknown feature spec version")
    return spec


def predict_from_models(
    model_dir: Path, inputs: PredictionInputs, exclude_observations: tuple[str, ...]
) -> pd.DataFrame:
    """Probabilities for ``inputs`` from a saved feature set directory (spec and boosters)."""
    spec = load_spec(model_dir / "feature_spec.json")
    events = drop_observations(inputs.events, frozenset(exclude_observations))
    x = features_from_spec(spec, events, inputs.cohort, inputs.cohort.index).to_numpy("float32")
    probabilities = pd.DataFrame(np.nan, index=inputs.cohort.index, columns=inputs.codes)
    for code in inputs.codes:
        target = spec["targets"][code]
        if "constant" in target:
            probabilities[code] = float(target["constant"])
        else:
            booster = lightgbm.Booster(model_file=str(model_dir / target["booster"]))
            probabilities[code] = booster.predict(x)
    return apply_prior_mask(probabilities, inputs.prior, inputs.codes)


# ---------------------------------------------------------------------- fitting
def run(
    dataset_dir: Path,
    settings_path: Path,
    feature_sets: tuple[str, ...],
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    settings, exclude_observations = selected_settings(settings_path)
    data = load_training_data(dataset_dir)
    splits = load_splits(dataset_dir)
    others = {
        split: load_prediction_inputs(dataset_dir, splits.index[splits.eq(split)])
        for split in ("val", "test")
    }
    if exclude_observations:  # D36: the same removal as the transformer's tokenizer
        dropped = frozenset(exclude_observations)
        data = replace(
            data, inputs=replace(data.inputs, events=drop_observations(data.inputs.events, dropped))
        )
        others = {
            s: replace(v, events=drop_observations(v.events, dropped)) for s, v in others.items()
        }
    train = data.inputs
    codes = train.codes
    for inputs in others.values():
        if inputs.codes != codes:
            raise ValueError("target lists differ between splits")
    frames = [train, *others.values()]
    combined = replace(
        train,
        cohort=pd.concat([f.cohort for f in frames]),
        events=pd.concat([f.events for f in frames], ignore_index=True),
        prior=pd.concat([f.prior for f in frames]),
    )
    latest = latest_values(combined.events, combined.cohort)
    values = all_values(combined.events)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "scope": "fitted on all training patients; val and test used as inputs only",
        "settings_sha256": file_hash(settings_path),
        "excluded_observations": list(exclude_observations),
        "n_train": len(train.cohort),
        "n_val": len(others["val"].cohort),
        "n_test": len(others["test"].cohort),
        "feature_sets": {},
    }
    for name in feature_sets:
        if name not in settings:
            raise ValueError(f"no CV-selected setting for feature set {name!r}")
        choice = settings[name]
        out_dir = models_dir / name
        out_dir.mkdir(exist_ok=True)
        # E11 code path for the fit matrix and the two prediction matrices.
        x_fit, x_val = fold_features(
            SimpleNamespace(inputs=combined),
            latest,
            values,
            train.cohort.index,
            others["val"].cohort.index,
            name,
        )
        _, x_test = fold_features(
            SimpleNamespace(inputs=combined),
            latest,
            values,
            train.cohort.index,
            others["test"].cohort.index,
            name,
        )
        # Self-contained spec, checked against the E11 matrices row by row.
        bag = BagOfCodes().fit(combined.events, train.cohort)
        spec: dict[str, Any] = {
            "version": SPEC_VERSION,
            "feature_set": name,
            "feature_config": asdict(bag.config),
            "vocabulary": list(bag.vocabulary_ or []),
            "values": None if name == "counts" else fit_value_spec(latest, train.cohort.index),
            "setting": choice,
            "excluded_observations": list(exclude_observations),
            "targets": {},
        }
        rebuilt = {
            split: bag.transform(combined.events, inputs.cohort)
            for split, inputs in (("train", train), *others.items())
        }
        if spec["values"] is not None:
            for split, inputs in (("train", train), *others.items()):
                rebuilt[split] = pd.concat(
                    [
                        rebuilt[split],
                        value_features_from_spec(latest, spec["values"], inputs.cohort.index),
                    ],
                    axis=1,
                )
        spec["columns"] = list(rebuilt["train"].columns)
        for split, reference in (("train", x_fit), ("val", x_val), ("test", x_test)):
            if not np.array_equal(rebuilt[split].to_numpy("float32"), reference, equal_nan=True):
                raise ValueError(f"feature spec does not reproduce the E11 {split} matrix ({name})")
        predictions = {
            split: pd.DataFrame(np.nan, index=inputs.cohort.index, columns=codes)
            for split, inputs in others.items()
        }
        for code in codes:
            at_risk = train.prior[code].to_numpy() == 0
            y = data.labels[code].to_numpy()[at_risk]
            if len(np.unique(y)) < 2:
                constant = float(y.mean()) if len(y) else 0.0
                spec["targets"][code] = {"constant": constant, "n_fit": int(at_risk.sum())}
                for split in others:
                    predictions[split][code] = constant
                continue
            model = lightgbm.LGBMClassifier(
                n_estimators=max(N_ESTIMATORS),
                num_leaves=choice["num_leaves"],
                min_child_samples=choice["min_child_samples"],
                **FIXED,
            )
            model.fit(x_fit[at_risk], y)
            n = choice["n_estimators"]
            path = out_dir / f"{code}.txt"
            model.booster_.save_model(str(path), num_iteration=n)
            booster = lightgbm.Booster(model_file=str(path))
            for split, x in (("val", x_val), ("test", x_test)):
                expected = model.predict_proba(x, num_iteration=n)[:, 1]
                reloaded = booster.predict(x)
                if not np.allclose(expected, reloaded, rtol=0, atol=1e-9):
                    raise ValueError(f"saved booster for {code} does not reproduce {split}")
                predictions[split][code] = reloaded
            spec["targets"][code] = {
                "booster": path.name,
                "n_fit": int(at_risk.sum()),
                "n_positive": int(y.sum()),
            }
        write_json(out_dir / "feature_spec.json", spec)
        for split, inputs in others.items():
            masked = apply_prior_mask(predictions[split], inputs.prior, codes)
            masked.to_csv(
                reports_dir / f"{split}_predictions_lightgbm_{name}.csv", index_label="patient_id"
            )
            check = predict_from_models(out_dir, inputs, exclude_observations)
            if not np.allclose(check.to_numpy(), masked.to_numpy(), rtol=0, atol=1e-9):
                raise ValueError(
                    f"predict_from_models does not reproduce the {split} file ({name})"
                )
        record["feature_sets"][name] = {
            **choice,
            "n_features": int(x_fit.shape[1]),
            "boosters": sum("booster" in t for t in spec["targets"].values()),
            "constants": sum("constant" in t for t in spec["targets"].values()),
            "model_dir": str(out_dir),
        }
        log.info(
            "%s: %d features, %d boosters saved to %s",
            name,
            x_fit.shape[1],
            record["feature_sets"][name]["boosters"],
            out_dir,
        )
    write_json(reports_dir / "lightgbm_final.json", record)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit on all training patients; save boosters and files")
    fit.add_argument("--settings", type=Path, required=True, help="E11 cv_summary.json")
    fit.add_argument("--feature-sets", nargs="+", default=["counts", "counts_values"])
    fit.add_argument("--models", type=Path, required=True)
    fit.add_argument("--reports", type=Path, required=True)
    predict = sub.add_parser("predict", help="probabilities from one saved feature-set directory")
    predict.add_argument("--models", type=Path, required=True, help="dir with feature_spec.json")
    predict.add_argument("--split", choices=["val", "test"], required=True)
    predict.add_argument("--out", type=Path, required=True)
    for p in (fit, predict):
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    if args.command == "fit":
        run(args.dataset, args.settings, tuple(args.feature_sets), args.models, args.reports)
        return
    spec = load_spec(args.models / "feature_spec.json")
    splits = load_splits(args.dataset)
    inputs = load_prediction_inputs(args.dataset, splits.index[splits.eq(args.split)])
    probabilities = predict_from_models(
        args.models, inputs, tuple(str(c) for c in spec["excluded_observations"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(args.out, index_label="patient_id")
    log.info("Wrote %d x %d LightGBM probabilities to %s", *probabilities.shape, args.out)


if __name__ == "__main__":
    main()
