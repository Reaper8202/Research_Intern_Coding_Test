"""LightGBM comparison baselines fitted on all training patients and scored on official val.

Runs only inside the P4e bundle (``ehrforecast.val_score``), in its own process: this module
must not import PyTorch (see ``lightgbm_cv``). The settings are the ones training CV selected
in ``reports/lightgbm_clean/cv_summary.json``; nothing is tuned here. Each target's classifier
fits on at-risk training patients, as in E11. Val labels are never read.

Usage::

    python -m ehrforecast.lightgbm_val --dataset data/processed \
        --settings reports/lightgbm_clean/cv_summary.json --feature-sets counts counts_values \
        --out reports/val
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.contracts import load_prediction_inputs, load_splits, load_training_data
from ehrforecast.lightgbm_cv import all_values, fit_target, fold_features, latest_values
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.provenance import write_json
from ehrforecast.tokenizer import drop_observations

log = logging.getLogger("ehrforecast.lightgbm_val")


def selected_settings(path: Path) -> tuple[dict[str, dict[str, int]], tuple[str, ...]]:
    """The CV-selected setting per feature set and the observation codes E11 excluded."""
    summary = json.loads(path.read_text())
    settings = {
        name: {k: int(choice[k]) for k in ("num_leaves", "min_child_samples", "n_estimators")}
        for name, choice in summary["selected"].items()
    }
    return settings, tuple(str(c) for c in summary.get("excluded_observations", []))


def fit_val(
    dataset_dir: Path,
    settings: dict[str, dict[str, int]],
    feature_sets: tuple[str, ...],
    exclude_observations: tuple[str, ...],
    out_dir: Path,
) -> dict[str, Any]:
    data = load_training_data(dataset_dir)
    splits = load_splits(dataset_dir)
    val = load_prediction_inputs(dataset_dir, splits.index[splits.eq("val")])
    if exclude_observations:
        dropped = frozenset(exclude_observations)
        data = replace(
            data, inputs=replace(data.inputs, events=drop_observations(data.inputs.events, dropped))
        )
        val = replace(val, events=drop_observations(val.events, dropped))
    train, codes = data.inputs, data.inputs.codes
    if val.codes != codes:
        raise ValueError("val and training target lists differ")
    combined = SimpleNamespace(
        inputs=replace(
            train,
            cohort=pd.concat([train.cohort, val.cohort]),
            events=pd.concat([train.events, val.events], ignore_index=True),
            prior=pd.concat([train.prior, val.prior]),
        )
    )
    latest = latest_values(combined.inputs.events, combined.inputs.cohort)
    values = all_values(combined.inputs.events)
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "scope": "fitted on all training patients; scored on official val; no labels read",
        "excluded_observations": list(exclude_observations),
        "n_train": len(train.cohort),
        "n_val": len(val.cohort),
        "feature_sets": {},
    }
    for name in feature_sets:
        if name not in settings:
            raise ValueError(f"no CV-selected setting for feature set {name!r}")
        choice = settings[name]
        x_fit, x_val = fold_features(
            combined, latest, values, train.cohort.index, val.cohort.index, name
        )
        probabilities = pd.DataFrame(np.nan, index=val.cohort.index, columns=codes)
        for target in codes:
            at_risk = train.prior[target].to_numpy() == 0
            y = data.labels[target].to_numpy()[at_risk]
            fitted = fit_target(
                x_fit[at_risk], y, x_val, choice["num_leaves"], choice["min_child_samples"]
            )
            probabilities[target] = fitted[choice["n_estimators"]]
        masked = apply_prior_mask(probabilities, val.prior, codes)
        path = out_dir / f"val_predictions_lightgbm_{name}.csv"
        masked.to_csv(path, index_label="patient_id")
        record["feature_sets"][name] = {
            **choice,
            "n_features": int(x_fit.shape[1]),
            "file": path.name,
        }
        log.info("%s: %d features, wrote %s", name, x_fit.shape[1], path)
    write_json(out_dir / "lightgbm_val.json", record)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True, help="E11 cv_summary.json")
    parser.add_argument("--feature-sets", nargs="+", default=["counts", "counts_values"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    settings, excluded = selected_settings(args.settings)
    fit_val(args.dataset, settings, tuple(args.feature_sets), excluded, args.out)


if __name__ == "__main__":
    main()
