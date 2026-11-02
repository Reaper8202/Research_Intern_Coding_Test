from __future__ import annotations

import pandas as pd
import pytest

from ehrforecast.events import EVENT_COLUMNS, build_code_descriptions, build_events
from tests.helpers import ts


def _tables() -> dict[str, pd.DataFrame]:
    conditions = pd.DataFrame(
        {
            "START": ts("2010-01-01", "2012-01-01"),
            "STOP": ts("2010-02-01", None),
            "PATIENT": ["p", "p"],
            "ENCOUNTER": ["e1", "e2"],
            "CODE": ["111", "222"],
            "DESCRIPTION": ["Cold", "Diabetes"],
        }
    )
    encounters = pd.DataFrame(
        {
            "Id": ["e1"],
            "START": ts("2010-01-01T08:00:00Z"),
            "STOP": ts("2010-01-01T10:30:00Z"),
            "PATIENT": ["p"],
            "ENCOUNTERCLASS": ["ambulatory"],
            "CODE": ["185345009"],
            "DESCRIPTION": ["Encounter for symptom"],
        }
    )
    observations = pd.DataFrame(
        {
            "DATE": ts("2011-01-01T08:00:00Z", "2011-01-01T08:00:00Z"),
            "PATIENT": ["p", "p"],
            "ENCOUNTER": ["e1", "e1"],
            "CODE": ["8302-2", "72166-2"],
            "DESCRIPTION": ["Body Height", "Tobacco smoking status"],
            "VALUE": ["176.7", "Never smoker"],
            "UNITS": ["cm", None],
            "TYPE": ["numeric", "text"],
        }
    )
    imaging = pd.DataFrame(
        {
            "Id": ["i1"],
            "DATE": ts("2009-05-05T00:00:00Z"),
            "PATIENT": ["q"],
            "ENCOUNTER": ["e9"],
            "BODYSITE_CODE": ["40983000"],
            "BODYSITE_DESCRIPTION": ["Arm"],
            "MODALITY_CODE": ["DX"],
            "MODALITY_DESCRIPTION": ["Digital Radiography"],
        }
    )
    return {
        "conditions": conditions,
        "encounters": encounters,
        "observations": observations,
        "imaging_studies": imaging,
    }


def test_event_table_shape_and_order() -> None:
    events = build_events(_tables())
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert events["patient_id"].is_monotonic_increasing
    for _, group in events.groupby("patient_id"):
        assert group["time"].is_monotonic_increasing


def test_interval_rows_give_start_and_known_stop_only() -> None:
    events = build_events(_tables())
    cond = events[events["source"] == "conditions"]
    assert sorted(zip(cond["code"], cond["kind"], strict=True)) == [
        ("111", "start"),
        ("111", "stop"),
        ("222", "start"),  # blank STOP: no stop row
    ]


def test_encounter_gets_length_and_class() -> None:
    events = build_events(_tables())
    enc = events[events["source"] == "encounters"].iloc[0]
    assert enc["kind"] == "start"
    assert enc["value_num"] == 2.5
    assert enc["value_text"] == "ambulatory"


def test_observation_values_are_split_by_type() -> None:
    events = build_events(_tables()).set_index("code")
    assert events.loc["8302-2", "value_num"] == 176.7
    assert events.loc["8302-2", "unit"] == "cm"
    assert pd.isna(events.loc["72166-2", "value_num"])
    assert events.loc["72166-2", "value_text"] == "Never smoker"


def test_imaging_code_combines_modality_and_body_site() -> None:
    events = build_events(_tables())
    assert events.loc[events["source"] == "imaging_studies", "code"].tolist() == ["DX|40983000"]
    desc = build_code_descriptions(_tables()).set_index(["source", "code"])
    assert (
        desc.loc[("imaging_studies", "DX|40983000"), "description"] == "Digital Radiography | Arm"
    )


def test_stop_before_start_must_be_repaired_before_event_construction() -> None:
    tables = _tables()
    tables["conditions"].loc[0, "STOP"] = pd.Timestamp("2009-12-01")
    with pytest.raises(ValueError, match="omit_invalid_stops"):
        build_events(tables)
