"""Project-wide constants and dataset layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The anchor is this many years before the last encounter; labels cover the same span after it.
HORIZON_YEARS = 5

SPLITS = ("train", "val", "test")
TRAIN_VAL_DIR = "train_val"
TEST_DIR = "test"

RANDOM_SEED = 20261028


@dataclass(frozen=True)
class DataLayout:
    """Locations of the files shipped with the task, relative to one root folder."""

    root: Path

    @property
    def splits_file(self) -> Path:
        return self.root / "patient_splits.csv"

    @property
    def targets_file(self) -> Path:
        return self.root / "target_conditions.csv"

    @property
    def test_anchors_file(self) -> Path:
        return self.root / "test_anchors.csv"

    @property
    def train_val_dir(self) -> Path:
        return self.root / TRAIN_VAL_DIR

    @property
    def test_dir(self) -> Path:
        return self.root / TEST_DIR

    def validate(self) -> None:
        """Fail early with a clear message when the dataset folder is incomplete."""
        required = [
            self.splits_file,
            self.targets_file,
            self.test_anchors_file,
            self.train_val_dir,
            self.test_dir,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Dataset folder is missing: {', '.join(missing)}")
