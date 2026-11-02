"""Convert truncated EHR tables into one long, time-ordered event table.

One row is one clinical event:

========== ===============================================================================
column     meaning
========== ===============================================================================
patient_id patient Id
time       event time (timezone-naive UTC)
source     source table, for example ``conditions``
kind       ``point`` (dated event), ``start`` or ``stop`` (interval tables)
code       clinical code; imaging uses ``<modality>|<body site>``
value_num  numeric observation value; encounter length in hours for encounters
value_text text observation value; encounter class for encounters
unit       observation unit
interval_start  ``START`` of the source row for interval tables (start and stop rows); empty
           for point events. It links a stop to its own start for refill compression.
========== ===============================================================================

Interval tables (conditions, medications, care plans, allergies, devices) give a ``start`` row,
and a ``stop`` row when the ``STOP`` is known at the anchor. Encounters give one ``start`` row
with their length, because an encounter end carries no separate clinical meaning. A ``STOP``
before its own ``START`` must be removed first (``truncation.omit_invalid_stops``).

``events.manifest.json`` records the history rule version and the event file hash. Loaders use
it to reject an event table from an older build.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ehrforecast.provenance import file_hash, write_json
from ehrforecast.schema import EVENT_TABLES, PATIENT_COL, TableSpec
from ehrforecast.truncation import HISTORY_RULE_VERSION

log = logging.getLogger(__name__)

EVENT_COLUMNS = (
    "patient_id",
    "time",
    "source",
    "kind",
    "code",
    "value_num",
    "value_text",
    "unit",
    "interval_start",
)
# v2 adds ``interval_start``.
EVENT_SCHEMA_VERSION = "events-v2"
SOURCE_ORDER = tuple(EVENT_TABLES)
KIND_ORDER = ("start", "point", "stop")
EVENTS_FILE = "events.parquet"
EVENTS_MANIFEST = "events.manifest.json"


def _codes(df: pd.DataFrame, spec: TableSpec) -> pd.Series:
    if len(spec.code_cols) == 1:
        return df[spec.code_cols[0]]
    return df[list(spec.code_cols)].astype(str).agg("|".join, axis=1)


def _values(df: pd.DataFrame, spec: TableSpec) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(df)
    value_num = pd.Series(np.nan, index=df.index, dtype="float64")
    value_text = pd.Series(pd.NA, index=df.index, dtype="string")
    unit = pd.Series(pd.NA, index=df.index, dtype="string")
    if spec.name == "observations":
        numeric = df["TYPE"].eq("numeric")
        value_num[numeric] = pd.to_numeric(df.loc[numeric, "VALUE"], errors="coerce")
        bad = int(value_num[numeric].isna().sum())
        if bad:
            log.warning("observations: %d of %d numeric values could not be parsed", bad, n)
        value_text[~numeric] = df.loc[~numeric, "VALUE"]
        unit[:] = df["UNITS"]
    elif spec.name == "encounters":
        hours = (df["STOP"] - df["START"]).dt.total_seconds() / 3600.0
        value_num[:] = hours
        value_text[:] = df["ENCOUNTERCLASS"]
    return value_num, value_text, unit


def table_to_events(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Convert one (already truncated) table into rows of the event format."""
    code = _codes(df, spec)
    value_num, value_text, unit = _values(df, spec)
    starts = pd.DataFrame(
        {
            "patient_id": df[PATIENT_COL],
            "time": df[spec.time_col],
            "source": spec.name,
            "kind": "start" if spec.is_interval else "point",
            "code": code,
            "value_num": value_num,
            "value_text": value_text,
            "unit": unit,
            "interval_start": df[spec.time_col] if spec.is_interval else pd.NaT,
        }
    )
    frames = [starts]
    if spec.stop_col is not None and spec.name != "encounters":
        has_stop = df[spec.stop_col].notna()
        if (df.loc[has_stop, spec.stop_col] < df.loc[has_stop, spec.time_col]).any():
            raise ValueError(f"{spec.name}: a STOP precedes its START; apply omit_invalid_stops")
        stops = starts.loc[has_stop].copy()
        stops["time"] = df.loc[has_stop, spec.stop_col]
        stops["kind"] = "stop"
        frames.append(stops)
    return pd.concat(frames, ignore_index=True)


def build_events(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all tables into one event table sorted by patient and time."""
    unknown = set(tables) - set(EVENT_TABLES)
    if unknown:
        raise ValueError(f"Unknown tables: {sorted(unknown)}")
    events = pd.concat(
        [table_to_events(tables[name], EVENT_TABLES[name]) for name in tables],
        ignore_index=True,
    )
    events["patient_id"] = events["patient_id"].astype("string")
    events["code"] = events["code"].astype("string")
    events["source"] = pd.Categorical(events["source"], categories=SOURCE_ORDER)
    events["kind"] = pd.Categorical(events["kind"], categories=KIND_ORDER)
    events = events.sort_values(
        ["patient_id", "time", "source", "kind", "code"], kind="stable"
    ).reset_index(drop=True)
    return events[list(EVENT_COLUMNS)]


def save_events(events: pd.DataFrame, out_dir: Path, omitted_stops: dict[str, int]) -> None:
    """Write the event table and the manifest that ties it to the current history rule."""
    path = out_dir / EVENTS_FILE
    events.to_parquet(path, index=False)
    manifest = {
        "history_rule": HISTORY_RULE_VERSION,
        "event_schema": EVENT_SCHEMA_VERSION,
        "n_events": len(events),
        "omitted_stops_by_split": omitted_stops,
        "events_sha256": file_hash(path),
    }
    write_json(out_dir / EVENTS_MANIFEST, manifest)


def verify_events(dataset_dir: Path) -> None:
    """Reject an event table that predates the current history rule or changed after its build."""
    path, manifest_path = dataset_dir / EVENTS_FILE, dataset_dir / EVENTS_MANIFEST
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("event table or event manifest is missing; run `make dataset`")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("history_rule") != HISTORY_RULE_VERSION:
        raise ValueError("event table uses an older history rule; run `make dataset`")
    if manifest.get("event_schema") != EVENT_SCHEMA_VERSION:
        raise ValueError("event table uses an older column schema; run `make dataset`")
    if manifest.get("events_sha256") != file_hash(path):
        raise ValueError("event table does not match its manifest; run `make dataset`")


def build_code_descriptions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the most frequent description for each ``(source, code)`` pair."""
    frames = []
    for name, df in tables.items():
        spec = EVENT_TABLES[name]
        if name == "imaging_studies":
            desc = df["MODALITY_DESCRIPTION"] + " | " + df["BODYSITE_DESCRIPTION"]
        else:
            desc = df["DESCRIPTION"]
        frames.append(pd.DataFrame({"source": name, "code": _codes(df, spec), "description": desc}))
    allrows = pd.concat(frames, ignore_index=True).dropna()
    counts = allrows.value_counts(["source", "code", "description"]).reset_index(name="n")
    best = counts.sort_values("n", ascending=False, kind="stable").drop_duplicates(
        ["source", "code"]
    )
    return best.drop(columns="n").sort_values(["source", "code"]).reset_index(drop=True)
