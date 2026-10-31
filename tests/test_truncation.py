from __future__ import annotations

import pandas as pd
import pytest

from ehrforecast.schema import EVENT_TABLES
from ehrforecast.truncation import find_post_anchor_values, omit_invalid_stops, truncate_table
from tests.helpers import day, ts

ANCHORS = pd.Series({"p": day("2015-06-15")})


def test_point_events_on_the_anchor_day_are_removed() -> None:
    df = pd.DataFrame(
        {
            "PATIENT": ["p", "p", "p"],
            "DATE": ts("2015-06-14T23:59:59Z", "2015-06-15T00:00:00Z", "2015-06-15T10:00:00Z"),
            "CODE": ["keep", "drop", "drop"],
        }
    )
    out = truncate_table(df, EVENT_TABLES["observations"], ANCHORS)
    assert out["CODE"].tolist() == ["keep"]


def test_stop_on_or_after_anchor_is_blanked_but_row_is_kept() -> None:
    df = pd.DataFrame(
        {
            "PATIENT": ["p", "p", "p", "p"],
            "START": ts("2010-01-01", "2010-01-01", "2010-01-01", "2016-01-01"),
            "STOP": ts("2015-06-14", "2015-06-15", None, "2016-02-01"),
            "CODE": ["ended", "spans", "open", "future"],
        }
    )
    out = truncate_table(df, EVENT_TABLES["conditions"], ANCHORS)
    assert out["CODE"].tolist() == ["ended", "spans", "open"]
    assert out["STOP"].isna().tolist() == [False, True, True]
    assert find_post_anchor_values(out, EVENT_TABLES["conditions"], ANCHORS) == 0
    assert find_post_anchor_values(df, EVENT_TABLES["conditions"], ANCHORS) == 3


def test_truncation_is_idempotent() -> None:
    df = pd.DataFrame(
        {
            "PATIENT": ["p", "p"],
            "START": ts("2014-01-01T08:00:00Z", "2015-06-14T08:00:00Z"),
            "STOP": ts("2015-07-01T08:00:00Z", "2015-06-14T09:00:00Z"),
        }
    )
    spec = EVENT_TABLES["encounters"]
    once = truncate_table(df, spec, ANCHORS)
    twice = truncate_table(once, spec, ANCHORS)
    pd.testing.assert_frame_equal(once, twice)


def test_patient_without_anchor_raises() -> None:
    df = pd.DataFrame({"PATIENT": ["q"], "DATE": ts("2000-01-01"), "CODE": ["x"]})
    with pytest.raises(ValueError, match="no anchor"):
        truncate_table(df, EVENT_TABLES["procedures"], ANCHORS)


def test_stop_before_its_start_is_omitted_and_every_start_is_kept() -> None:
    # Shape of a real training case: an invalid row inside a valid row of the same drug.
    df = pd.DataFrame(
        {
            "PATIENT": ["p", "p", "p", "p"],
            "START": ts(
                "2001-02-22T05:07:06Z",
                "2001-02-28T05:04:06Z",
                "2002-03-06T05:04:06Z",
                "2003-01-01T08:00:00Z",
            ),
            "STOP": ts(
                "2002-03-06T05:04:06Z", "2001-02-22T05:07:06Z", None, "2003-01-01T08:00:00Z"
            ),
            "CODE": ["855332", "855332", "855332", "other"],
        }
    )
    out, omitted = omit_invalid_stops(df, EVENT_TABLES["medications"])
    pd.testing.assert_series_equal(out["START"], df["START"])
    # Only the invalid row loses its stop. The overlapping valid row and a zero-length row keep it.
    assert out["STOP"].isna().tolist() == [False, True, True, False]
    assert out.loc[0, "STOP"] == df.loc[0, "STOP"]
    assert omitted.to_dict("records") == [
        {
            "patient_id": "p",
            "source": "medications",
            "code": "855332",
            "start": pd.Timestamp("2001-02-28 05:04:06"),
            "stop": pd.Timestamp("2001-02-22 05:07:06"),
        }
    ]
    again, none = omit_invalid_stops(out, EVENT_TABLES["medications"])
    pd.testing.assert_frame_equal(again, out)
    assert none.empty


def test_tables_without_stops_are_unchanged_by_the_interval_repair() -> None:
    df = pd.DataFrame({"PATIENT": ["p"], "DATE": ts("2000-01-01"), "CODE": ["x"]})
    out, omitted = omit_invalid_stops(df, EVENT_TABLES["procedures"])
    pd.testing.assert_frame_equal(out, df)
    assert omitted.empty


def test_encounter_stop_before_start_fails() -> None:
    df = pd.DataFrame({"PATIENT": ["p"], "START": ts("2010-01-02"), "STOP": ts("2010-01-01")})
    with pytest.raises(ValueError, match="encounters"):
        omit_invalid_stops(df, EVENT_TABLES["encounters"])
