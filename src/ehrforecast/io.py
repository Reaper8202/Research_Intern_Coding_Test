"""Reading the raw CSV tables with one consistent parser for codes and timestamps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ehrforecast.schema import EVENT_TABLES, PATIENT_COLUMNS, TableSpec


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse Synthea date or timestamp strings into timezone-naive UTC datetimes.

    The tables mix two formats: bare dates (``2015-10-29``) and UTC timestamps
    (``2015-10-29T14:00:00Z``). Both go through the same ISO-8601 parser, so a bare date becomes
    midnight UTC of that day. Blank values become ``NaT``; malformed values raise.
    """
    parsed = pd.to_datetime(values, format="ISO8601", utc=True)
    return parsed.dt.tz_localize(None)


def read_csv_as_strings(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read selected columns as strings, so codes keep leading zeros and never become floats."""
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return pd.read_csv(
        path,
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )[list(columns)]


def read_event_table(split_dir: Path, spec: TableSpec) -> pd.DataFrame:
    """Read one event table and parse its time columns."""
    df = read_csv_as_strings(split_dir / f"{spec.name}.csv", spec.columns)
    df[spec.time_col] = parse_timestamps(df[spec.time_col])
    if df[spec.time_col].isna().any():
        raise ValueError(f"{spec.name}.{spec.time_col} has blank values")
    if spec.stop_col is not None:
        df[spec.stop_col] = parse_timestamps(df[spec.stop_col])
    return df


def read_all_event_tables(split_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: read_event_table(split_dir, spec) for name, spec in EVENT_TABLES.items()}


def read_patients(split_dir: Path) -> pd.DataFrame:
    df = read_csv_as_strings(split_dir / "patients.csv", PATIENT_COLUMNS)
    df["BIRTHDATE"] = parse_timestamps(df["BIRTHDATE"])
    if df["Id"].duplicated().any():
        raise ValueError("patients.csv has duplicate patient Ids")
    return df
