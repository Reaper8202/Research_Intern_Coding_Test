"""Patient splits, anchor dates, and the per-patient cohort table.

Anchor convention (checked against the released test files, see docs/DECISIONS.md):

* ``last_encounter`` is the latest ``STOP`` over the patient's encounters, with ``START`` used for
  any encounter whose ``STOP`` is blank.
* ``anchor_date`` is the calendar day of ``last_encounter - 5 years``. History is every event
  dated strictly before that day.
* ``window_end`` is ``anchor_date + 5 years``. The label window includes both ends.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ehrforecast.config import HORIZON_YEARS, SPLITS
from ehrforecast.io import parse_timestamps, read_csv_as_strings

HORIZON = pd.DateOffset(years=HORIZON_YEARS)


def load_splits(path: Path) -> pd.Series:
    """Return a Series that maps patient Id to its split name."""
    df = read_csv_as_strings(path, ("Id", "split"))
    if df["Id"].duplicated().any():
        raise ValueError("patient_splits.csv lists a patient more than once")
    unknown = set(df["split"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"patient_splits.csv has unknown split names: {sorted(unknown)}")
    return df.set_index("Id")["split"].rename("split")


def load_targets(path: Path) -> pd.DataFrame:
    """Return the scored condition codes, in file order, as strings."""
    df = read_csv_as_strings(path, ("CODE", "DESCRIPTION"))
    if df["CODE"].duplicated().any():
        raise ValueError("target_conditions.csv lists a code more than once")
    return df.reset_index(drop=True)


def load_test_anchors(path: Path) -> pd.Series:
    df = read_csv_as_strings(path, ("Id", "anchor_date"))
    if df["Id"].duplicated().any():
        raise ValueError("test_anchors.csv lists a patient more than once")
    anchors = parse_timestamps(df["anchor_date"])
    if (anchors != anchors.dt.normalize()).any():
        raise ValueError("test_anchors.csv must contain calendar dates")
    return pd.Series(anchors.to_numpy(), index=df["Id"], name="anchor_date")


def anchor_from_last_encounter(last_encounter: pd.Series) -> pd.Series:
    """Map last-encounter timestamps to anchor dates (midnight of the anchor day)."""
    return (last_encounter - HORIZON).dt.normalize()


def compute_anchors(encounters: pd.DataFrame) -> pd.DataFrame:
    """Compute last encounter, anchor date, and label-window end for every patient."""
    encounter_end = encounters["STOP"].fillna(encounters["START"])
    last = encounter_end.groupby(encounters["PATIENT"]).max().rename("last_encounter")
    out = last.to_frame()
    out["anchor_date"] = anchor_from_last_encounter(out["last_encounter"])
    out["window_end"] = out["anchor_date"] + HORIZON
    out.index.name = "patient_id"
    return out


def validate_split_membership(
    splits: pd.Series, train_val_ids: pd.Index, test_ids: pd.Index, test_anchor_ids: pd.Index
) -> None:
    """Check that each folder holds exactly the patients that the split file assigns to it."""
    expected_tv = set(splits.index[splits.isin(["train", "val"])])
    expected_test = set(splits.index[splits == "test"])
    problems = []
    if set(train_val_ids) != expected_tv:
        problems.append("train_val/patients.csv does not match the train+val split")
    if set(test_ids) != expected_test:
        problems.append("test/patients.csv does not match the test split")
    if set(test_anchor_ids) != expected_test:
        problems.append("test_anchors.csv does not match the test split")
    if problems:
        raise ValueError("; ".join(problems))


def build_cohort(
    splits: pd.Series,
    patients: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    """Join splits, demographics, and anchors into one row per patient.

    ``anchors`` must have ``anchor_date`` and ``window_end`` for every patient; ``last_encounter``
    is known only for train and val patients.
    """
    demo = patients.set_index("Id").rename(columns=str.lower)
    cohort = pd.DataFrame({"split": splits}).join(demo, how="left").join(anchors, how="left")
    cohort.index.name = "patient_id"
    if cohort[["birthdate", "gender", "anchor_date", "window_end"]].isna().any().any():
        raise ValueError("Some patients lack demographics or an anchor date")
    if (cohort["birthdate"] >= cohort["anchor_date"]).any():
        raise ValueError("Some patients are born on or after their anchor date")
    age_days = (cohort["anchor_date"] - cohort["birthdate"]).dt.days
    cohort["age_at_anchor"] = age_days / 365.25
    return cohort.sort_index()
