from __future__ import annotations

import pandas as pd
import pytest

from ehrforecast.schema import EVENT_TABLES
from ehrforecast.truncation import find_post_anchor_values, truncate_table
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
