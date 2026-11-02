from __future__ import annotations

import pandas as pd

from ehrforecast.labels import build_labels, build_prior_flags
from tests.helpers import day, ts

CODES = ["X", "Y", "Z", "W"]


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {"anchor_date": [day("2015-06-15")] * 2, "window_end": [day("2020-06-15")] * 2},
        index=pd.Index(["p", "empty"], name="patient_id"),
    )


def _conditions() -> pd.DataFrame:
    rows = [
        ("p", "2010-01-01", "X"),  # had it before the anchor ...
        ("p", "2016-01-01", "X"),  # ... so a repeat in the window does not count
        ("p", "2015-06-15", "Y"),  # first diagnosis on the anchor day counts
        ("p", "2020-06-15", "Z"),  # first diagnosis on the window end counts
        ("p", "2020-06-16", "W"),  # one day after the window does not count
        ("p", "2016-01-01", "NOT_A_TARGET"),
    ]
    patient, start, code = zip(*rows, strict=True)
    return pd.DataFrame({"PATIENT": list(patient), "START": ts(*start), "CODE": list(code)})


def test_labels_count_only_first_ever_diagnosis_inside_the_window() -> None:
    labels = build_labels(_conditions(), _cohort(), CODES)
    assert list(labels.columns) == CODES
    assert labels.loc["p"].to_numpy().tolist() == [0, 1, 1, 0]
    assert labels.dtypes.unique().tolist() == ["int8"]


def test_patient_without_conditions_gets_an_all_zero_row() -> None:
    labels = build_labels(_conditions(), _cohort(), CODES)
    assert labels.loc["empty"].to_numpy().tolist() == [0, 0, 0, 0]


def test_column_order_follows_target_order() -> None:
    labels = build_labels(_conditions(), _cohort(), ["Z", "Y"])
    assert list(labels.columns) == ["Z", "Y"]


def test_prior_flags_come_from_history_only() -> None:
    conditions = _conditions()
    history = conditions[conditions["START"] < day("2015-06-15")]
    prior = build_prior_flags(history, _cohort().index, CODES)
    assert prior.loc["p"].to_numpy().tolist() == [1, 0, 0, 0]
    assert prior.loc["empty"].sum() == 0


def test_calendar_horizon_includes_feb_28_but_not_march_1_after_leap_anchor() -> None:
    cohort = pd.DataFrame(
        {"anchor_date": [day("2020-02-29")], "window_end": [day("2025-02-28")]},
        index=pd.Index(["p"], name="patient_id"),
    )
    conditions = pd.DataFrame(
        {
            "PATIENT": ["p"] * 5,
            "CODE": ["prior", "prior", "anchor", "last", "late"],
            "START": ts("2020-02-28", "2021-01-01", "2020-02-29", "2025-02-28", "2025-03-01"),
        }
    )
    labels = build_labels(conditions, cohort, ["prior", "anchor", "last", "late"])
    assert labels.loc["p"].to_numpy().tolist() == [0, 1, 1, 0]
