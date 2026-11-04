"""Build the submitted probabilities: a fixed blend of base models, then Platt scaling (round 3).

The recipe lives in ``configs/final.toml`` under ``[stack]``: the base models, the stacker
(``logit_mean`` or ``prob_mean``; both have no fitted parameter), and the per-target Platt
weights fitted on training out-of-fold predictions (``ehrforecast.calibration``). The same
function builds the val probabilities inside the frozen bundle and the test ``predictions.csv``,
and ``hf_package verify`` rebuilds the test file from the public package through it. No labels
are read here.

Usage::

    python -m ehrforecast.submission --config configs/final.toml --dataset data/processed \
        --transformer reports/final/test_predictions_transformer.csv \
        --lightgbm-dir reports/lightgbm_final --split test --out predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.calibration import apply_platt
from ehrforecast.contracts import align_targets, load_prediction_inputs, load_splits
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.stacking import logit, sigmoid

log = logging.getLogger("ehrforecast.submission")

STACKERS = ("logit_mean", "prob_mean")


def read_predictions(path: Path, patients: pd.Index, codes: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    frame.columns = [str(c) for c in frame.columns]
    return align_targets(frame, patients, codes, path.name)


def blend(bases: list[pd.DataFrame], stacker: str) -> pd.DataFrame:
    if stacker not in STACKERS:
        raise ValueError(f"stacker must be one of {STACKERS}")
    first = bases[0]
    arrays = [b.loc[first.index, first.columns].to_numpy("float64") for b in bases]
    if stacker == "prob_mean":
        out = np.mean(arrays, axis=0)
    else:
        out = sigmoid(np.mean([logit(a) for a in arrays], axis=0))
    return pd.DataFrame(out, index=first.index, columns=first.columns)


def build(
    stack: dict[str, Any],
    base_frames: dict[str, pd.DataFrame],
    prior: pd.DataFrame,
    codes: list[str],
    root: Path,
) -> pd.DataFrame:
    """Blend the named bases in order, apply Platt weights when configured, then the prior mask."""
    names = [str(n) for n in stack["bases"]]
    missing = [n for n in names if n not in base_frames]
    if missing:
        raise ValueError(f"base predictions missing: {missing}")
    blended = blend([base_frames[n] for n in names], str(stack["stacker"]))
    if stack.get("platt_weights"):
        weights = json.loads((root / str(stack["platt_weights"])).read_text())
        blended = apply_platt(weights, blended, prior, codes)
    return apply_prior_mask(blended, prior.loc[blended.index], codes)


def main(argv: list[str] | None = None) -> None:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/final.toml"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--transformer", type=Path, required=True, help="transformer CSV")
    parser.add_argument("--lightgbm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    final = tomllib.loads(args.config.read_text())
    stack = final["stack"]
    splits = load_splits(args.dataset)
    patients = pd.Index(splits.index[splits.eq(args.split)], name="patient_id")
    inputs = load_prediction_inputs(args.dataset, patients)
    bases = {"transformer": read_predictions(args.transformer, inputs.cohort.index, inputs.codes)}
    for name in stack["bases"]:
        if str(name).startswith("lightgbm_"):
            path = args.lightgbm_dir / f"{args.split}_predictions_{name}.csv"
            bases[str(name)] = read_predictions(path, inputs.cohort.index, inputs.codes)
    probabilities = build(stack, bases, inputs.prior, inputs.codes, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(args.out, index_label="patient_id")
    log.info("Wrote %d x %d stacked probabilities to %s", *probabilities.shape, args.out)


if __name__ == "__main__":
    main()
