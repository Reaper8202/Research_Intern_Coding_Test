from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ehrforecast.io import parse_timestamps, read_csv_as_strings
from tests.helpers import ts


def test_date_and_utc_timestamp_land_on_the_same_day() -> None:
    parsed = ts("2015-10-29", "2015-10-29T23:59:59Z")
    assert parsed.dt.normalize().iloc[0] == parsed.dt.normalize().iloc[1]
    assert parsed.iloc[0] == pd.Timestamp("2015-10-29 00:00:00")
    assert parsed.iloc[1] == pd.Timestamp("2015-10-29 23:59:59")
    assert parsed.dt.tz is None


def test_blank_becomes_nat() -> None:
    assert ts("2015-10-29", None).isna().tolist() == [False, True]


def test_malformed_timestamp_raises() -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_timestamps(pd.Series(["29/10/2015"]))


def test_codes_stay_strings(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("CODE,VALUE,OTHER\n08,1.50,x\n8302-2,,y\n")
    df = read_csv_as_strings(path, ("CODE", "VALUE"))
    assert df["CODE"].tolist() == ["08", "8302-2"]
    assert df["VALUE"].iloc[0] == "1.50"
    assert pd.isna(df["VALUE"].iloc[1])
    assert list(df.columns) == ["CODE", "VALUE"]


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("CODE\n1\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        read_csv_as_strings(path, ("CODE", "STOP"))
