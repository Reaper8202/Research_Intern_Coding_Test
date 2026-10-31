"""Label-free prediction from saved fold artifacts (the final prediction interface).

Each run directory holds ``best.pt`` (or a named checkpoint) and sits inside a fold directory
with that fold's ``tokenizer.json``. Patients are encoded only with the saved tokenizer. With
several run directories, probabilities are averaged (D22). Prior-condition cells are set to zero.
This command reads no labels and no future tables. It refuses official val: val scoring belongs
to the frozen P4e bundle.

Usage::

    # mock test patients: the held-out training patients of fold 0
    python -m ehrforecast.predict --dataset data/processed \
        --run-dir models/transformer/small-v1/fold0/F --heldout-fold 0 --out preds.csv
    # test patients, five-fold ensemble
    python -m ehrforecast.predict --dataset data/processed --split test \
        --run-dir models/transformer/small-v1/fold{0,1,2,3,4}/F --out predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import PredictionInputs, load_prediction_inputs, load_splits
from ehrforecast.finetune import predict_samples
from ehrforecast.folds import fold_patients, load_folds
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.sequences import build_histories
from ehrforecast.tokenizer import FoldTokenizer
from ehrforecast.training import CHECKPOINT_VERSION, model_config, sequence_config

log = logging.getLogger("ehrforecast.predict")


@dataclass
class FoldModel:
    tokenizer: FoldTokenizer
    model: EHRTransformer
    config: dict[str, Any]
    provenance: dict[str, Any]


def load_fold_model(run_dir: Path, checkpoint: str = "best.pt") -> FoldModel:
    """Load our own checkpoint and the tokenizer it was trained with; verify they match."""
    tokenizer = FoldTokenizer.load(run_dir.parent / "tokenizer.json")
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    if state.get("version") != CHECKPOINT_VERSION or not state.get("complete"):
        raise ValueError(f"{run_dir / checkpoint} is incomplete or has an unknown version")
    provenance = state["provenance"]
    if not str(provenance.get("stage", "")).startswith("finetune_"):
        raise ValueError(f"{run_dir} is not a supervised run")
    if provenance["tokenizer_sha256"] != tokenizer.sha256:
        raise ValueError(f"{run_dir} was trained with a different tokenizer")
    config = state["config"]
    model = EHRTransformer(
        model_config(config, tokenizer, int(state["model"]["classifier.bias"].shape[0])),
        torch.from_numpy(tokenizer.input_only_ids()),
    )
    model.load_state_dict(state["model"])
    model.eval()
    return FoldModel(tokenizer, model, config, provenance)


def predict_inputs(
    fold_models: list[FoldModel],
    inputs: PredictionInputs,
    device: torch.device,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Average fold-model probabilities, then set prior-condition cells to zero."""
    if not fold_models:
        raise ValueError("at least one fold model is required")
    total = np.zeros((len(inputs.cohort), len(inputs.codes)))
    for fm in fold_models:
        if fm.model.config.n_targets != len(inputs.codes):
            raise ValueError("model outputs do not match the target list")
        histories = build_histories(
            fm.tokenizer.encode(inputs.events), inputs.cohort, fm.tokenizer, inputs.codes
        )
        sequence = sequence_config(fm.config)
        samples = [histories.prediction_sample(i, sequence) for i in range(len(histories))]
        total += predict_samples(fm.model.to(device), samples, device, batch_size)
    probabilities = pd.DataFrame(
        total / len(fold_models), index=inputs.cohort.index, columns=inputs.codes
    )
    return apply_prior_mask(probabilities, inputs.prior, inputs.codes)


def model_config_of(fm: FoldModel) -> ModelConfig:
    return fm.model.config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["test"])
    group.add_argument("--heldout-fold", type=int, help="mock test: held-out training patients")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    if args.split == "test":
        splits = load_splits(args.dataset)
        patients = splits.index[splits.eq("test")]
    else:
        _, patients = fold_patients(load_folds(args.dataset), args.heldout_fold)
    inputs = load_prediction_inputs(args.dataset, pd.Index(patients, name="patient_id"))
    fold_models = [load_fold_model(d, args.checkpoint) for d in args.run_dir]
    probabilities = predict_inputs(fold_models, inputs, torch.device(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    probabilities.to_csv(args.out, index_label="patient_id")
    log.info("Wrote %d x %d probabilities to %s", *probabilities.shape, args.out)


if __name__ == "__main__":
    main()
