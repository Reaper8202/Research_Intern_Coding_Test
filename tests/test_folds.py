from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import KFold

from ehrforecast.config import RANDOM_SEED
from ehrforecast.folds import fold_patients, load_folds, make_folds, save_folds


def _splits() -> pd.Series:
    return pd.Series(
        ["train"] * 20 + ["val", "test"],
        index=pd.Index([f"p{i}" for i in range(22)], name="patient_id"),
        name="split",
    )


def test_folds_match_step2_order_and_never_include_val_test() -> None:
    splits = _splits()
    assignments = make_folds(splits)
    patients = splits.index[splits.eq("train")]
    for fold, (fit_index, held_index) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(patients)
    ):
        fit, held = fold_patients(assignments, fold)
        assert fit.tolist() == patients[fit_index].tolist()
        assert held.tolist() == patients[held_index].tolist()
        assert set(fit).isdisjoint(held)
        assert set(fit) | set(held) == set(patients)
    assert set(assignments["patient_id"]).isdisjoint({"p20", "p21"})


def test_saved_folds_are_verified_and_reuse_is_idempotent(tmp_path: Path) -> None:
    _splits().to_frame().to_parquet(tmp_path / "cohort.parquet")
    first = save_folds(tmp_path)
    original = (tmp_path / "folds.parquet").read_bytes()
    pd.testing.assert_frame_equal(first, load_folds(tmp_path))
    pd.testing.assert_frame_equal(first, save_folds(tmp_path))
    assert (tmp_path / "folds.parquet").read_bytes() == original


def test_changed_split_or_fold_artifact_is_rejected(tmp_path: Path) -> None:
    _splits().to_frame().to_parquet(tmp_path / "cohort.parquet")
    assignments = save_folds(tmp_path)
    assignments["fold"] = (assignments["fold"] + 1) % 5
    assignments.to_parquet(tmp_path / "folds.parquet", index=False)
    with pytest.raises(ValueError, match="do not match"):
        load_folds(tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        save_folds(tmp_path)


def test_reordered_cohort_is_not_silently_reassigned(tmp_path: Path) -> None:
    splits = _splits()
    splits.to_frame().to_parquet(tmp_path / "cohort.parquet")
    save_folds(tmp_path)
    splits.iloc[::-1].to_frame().to_parquet(tmp_path / "cohort.parquet")
    with pytest.raises(ValueError, match="do not match"):
        load_folds(tmp_path)


def test_incomplete_manifest_fails(tmp_path: Path) -> None:
    _splits().to_frame().to_parquet(tmp_path / "cohort.parquet")
    save_folds(tmp_path)
    (tmp_path / "folds.manifest.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        save_folds(tmp_path)
