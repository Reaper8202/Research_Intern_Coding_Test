from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ehrforecast.cohort import (
    anchor_from_last_encounter,
    build_cohort,
    compute_anchors,
    load_splits,
    validate_split_membership,
)
from tests.helpers import day, ts


def _encounters(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    patient, start, stop = zip(*rows, strict=True)
    return pd.DataFrame({"PATIENT": list(patient), "START": ts(*start), "STOP": ts(*stop)})


def test_anchor_uses_latest_stop_with_start_fallback() -> None:
    enc = _encounters(
        [
            ("a", "2020-01-01T10:00:00Z", "2020-01-01T11:00:00Z"),
            ("a", "2020-06-15T23:30:00Z", None),  # blank STOP: START counts
            ("b", "2019-01-01T00:00:00Z", "2019-03-01T08:00:00Z"),  # long stay ends last
            ("b", "2019-02-01T00:00:00Z", "2019-02-02T00:00:00Z"),
        ]
    )
    out = compute_anchors(enc)
    assert out.loc["a", "last_encounter"] == pd.Timestamp("2020-06-15 23:30:00")
    assert out.loc["a", "anchor_date"] == day("2015-06-15")  # time of day dropped
    assert out.loc["a", "window_end"] == day("2020-06-15")
    assert out.loc["b", "anchor_date"] == day("2014-03-01")


def test_leap_day_anchor_rolls_back_to_feb_28() -> None:
    last = pd.Series([pd.Timestamp("2020-02-29 12:00")])
    assert anchor_from_last_encounter(last).iloc[0] == day("2015-02-28")


def test_load_splits_rejects_duplicates_and_unknown_names(tmp_path: Path) -> None:
    dup = tmp_path / "dup.csv"
    dup.write_text("Id,split\na,train\na,val\n")
    with pytest.raises(ValueError, match="more than once"):
        load_splits(dup)
    bad = tmp_path / "bad.csv"
    bad.write_text("Id,split\na,holdout\n")
    with pytest.raises(ValueError, match="unknown split"):
        load_splits(bad)


def test_split_membership_mismatch_raises() -> None:
    splits = pd.Series({"a": "train", "b": "val", "c": "test"})
    validate_split_membership(splits, pd.Index(["a", "b"]), pd.Index(["c"]), pd.Index(["c"]))
    with pytest.raises(ValueError, match="test split"):
        validate_split_membership(
            splits, pd.Index(["a", "b"]), pd.Index(["c", "a"]), pd.Index(["c"])
        )


def _patients(birth: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": ["a"],
            "BIRTHDATE": ts(birth),
            "GENDER": ["F"],
            "RACE": ["white"],
            "ETHNICITY": ["nonhispanic"],
        }
    )


def test_build_cohort_computes_age_at_anchor() -> None:
    anchors = pd.DataFrame({"anchor_date": [day("2010-01-01")], "window_end": [day("2015-01-01")]})
    anchors.index = pd.Index(["a"])
    cohort = build_cohort(pd.Series({"a": "train"}), _patients("1980-01-01"), anchors)
    assert cohort.loc["a", "age_at_anchor"] == pytest.approx(30.0, abs=0.01)


def test_build_cohort_rejects_birth_after_anchor() -> None:
    anchors = pd.DataFrame({"anchor_date": [day("2010-01-01")], "window_end": [day("2015-01-01")]})
    anchors.index = pd.Index(["a"])
    with pytest.raises(ValueError, match="born on or after"):
        build_cohort(pd.Series({"a": "train"}), _patients("2011-01-01"), anchors)
