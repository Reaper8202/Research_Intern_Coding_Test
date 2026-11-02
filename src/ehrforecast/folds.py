"""Persist and verify the exact training-patient folds used in step 2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ehrforecast.config import RANDOM_SEED
from ehrforecast.contracts import load_splits, validate_ids
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import file_hash, object_hash, write_json

FOLD_VERSION = "patient-kfold-v1"
N_FOLDS = 5
log = logging.getLogger(__name__)


def make_folds(splits: pd.Series) -> pd.DataFrame:
    validate_ids(splits.index)
    if not splits.isin(["train", "val", "test"]).all():
        raise ValueError("unknown or missing official split")
    patients = splits.index[splits.eq("train")]
    if len(patients) < N_FOLDS:
        raise ValueError("at least five training patients are required")
    assignments = np.full(len(patients), -1, dtype="int8")
    for fold, (_, held) in enumerate(
        KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(patients)
    ):
        assignments[held] = fold
    return pd.DataFrame({"patient_id": patients.to_numpy(), "fold": assignments})


def fold_patients(assignments: pd.DataFrame, fold: int) -> tuple[pd.Index, pd.Index]:
    if fold not in range(N_FOLDS):
        raise ValueError("fold must be in [0, 4]")
    validate_ids(pd.Index(assignments["patient_id"]))
    values = assignments["fold"]
    if not values.isin(range(N_FOLDS)).all() or set(values) != set(range(N_FOLDS)):
        raise ValueError("fold assignments must cover folds 0 through 4")
    fit = pd.Index(assignments.loc[values.ne(fold), "patient_id"], name="patient_id")
    held = pd.Index(assignments.loc[values.eq(fold), "patient_id"], name="patient_id")
    return fit, held


def save_folds(dataset_dir: Path) -> pd.DataFrame:
    splits = load_splits(dataset_dir)
    assignments = make_folds(splits)
    path, manifest_path = dataset_dir / "folds.parquet", dataset_dir / "folds.manifest.json"
    expected = {
        "version": FOLD_VERSION,
        "seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "cohort_sha256": file_hash(dataset_dir / "cohort.parquet"),
        "training_patient_order_sha256": object_hash(assignments["patient_id"].tolist()),
        "official_splits_sha256": object_hash(list(zip(splits.index, splits, strict=True))),
    }
    if path.exists() or manifest_path.exists():
        load_folds(dataset_dir)
        return assignments
    assignments.to_parquet(path, index=False)
    expected["folds_sha256"] = file_hash(path)
    write_json(manifest_path, expected)
    return assignments


def load_folds(dataset_dir: Path) -> pd.DataFrame:
    path, manifest_path = dataset_dir / "folds.parquet", dataset_dir / "folds.manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("fold file or fold manifest is missing; run the folds command")
    manifest = json.loads(manifest_path.read_text())
    splits = load_splits(dataset_dir)
    expected = make_folds(splits)
    actual = pd.read_parquet(path)
    required = {
        "version": FOLD_VERSION,
        "seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "cohort_sha256": file_hash(dataset_dir / "cohort.parquet"),
        "training_patient_order_sha256": object_hash(expected["patient_id"].tolist()),
        "official_splits_sha256": object_hash(list(zip(splits.index, splits, strict=True))),
        "folds_sha256": file_hash(path),
    }
    if manifest != required or not actual.equals(expected):
        raise ValueError(
            "saved folds or provenance do not match the current official training split"
        )
    return actual


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    assignments = save_folds(args.dataset)
    log.info("Verified %d training patients in five folds", len(assignments))


if __name__ == "__main__":
    main()
