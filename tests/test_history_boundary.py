from __future__ import annotations

import numpy as np
import pandas as pd

from ehrforecast.contracts import prediction_cohort, prediction_events, prior_from_history
from ehrforecast.events import build_events
from ehrforecast.features import BagOfCodes, FeatureConfig
from ehrforecast.schema import EVENT_TABLES
from ehrforecast.truncation import omit_invalid_stops, truncate_table
from tests.helpers import ts
from tests.test_contracts import cohort_fixture


def test_future_changes_leave_history_prior_and_baseline_inputs_unchanged() -> None:
    cohort = prediction_cohort(cohort_fixture().loc[["a"]])
    anchor = cohort["anchor_date"]
    conditions = pd.DataFrame(
        {
            "PATIENT": ["a", "a"],
            "START": ts("2014-01-01", "2016-01-01"),
            "STOP": ts("2017-01-01", None),
            "CODE": ["prior", "future"],
        }
    )
    observations = pd.DataFrame(
        {
            "PATIENT": ["a", "a"],
            "DATE": ts("2014-06-01", "2015-01-01"),
            "CODE": ["lab", "future_lab"],
            "VALUE": ["10", "99999"],
            "TYPE": ["numeric", "numeric"],
            "UNITS": ["mg/L", "mg/L"],
        }
    )

    def permitted(c: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
        tables = {
            "conditions": truncate_table(c, EVENT_TABLES["conditions"], anchor),
            "observations": truncate_table(o, EVENT_TABLES["observations"], anchor),
        }
        return prediction_events(build_events(tables), cohort)

    original = permitted(conditions, observations)
    changed_conditions = conditions.copy()
    changed_conditions.loc[0, "STOP"] = pd.NaT
    changed_conditions.loc[1, "CODE"] = "different_future"
    changed_observations = observations.iloc[:1].copy()
    changed = permitted(changed_conditions, changed_observations)
    pd.testing.assert_frame_equal(original, changed)
    pd.testing.assert_frame_equal(
        prior_from_history(original, cohort.index, ["prior", "future"]),
        prior_from_history(changed, cohort.index, ["prior", "future"]),
    )
    features = BagOfCodes(FeatureConfig(min_patients=1)).fit(original, cohort)
    np.testing.assert_array_equal(
        features.transform(original, cohort).to_numpy(),
        features.transform(changed, cohort).to_numpy(),
    )


MEDICATIONS = EVENT_TABLES["medications"]


def _medications(starts: tuple[str, ...], stops: tuple[str | None, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {"PATIENT": ["a"] * len(starts), "START": ts(*starts), "STOP": ts(*stops), "CODE": "drug"}
    )


def test_invalid_stop_repair_after_the_cut_ignores_future_records() -> None:
    cohort = prediction_cohort(cohort_fixture().loc[["a"]])
    anchor = cohort["anchor_date"]
    medications = _medications(
        ("2010-01-01", "2010-01-07", "2014-06-01"),
        ("2011-01-01", "2010-01-01", "2016-01-01"),
    )

    def permitted(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        repaired, omitted = omit_invalid_stops(
            truncate_table(table, MEDICATIONS, anchor), MEDICATIONS
        )
        return prediction_events(build_events({"medications": repaired}), cohort), omitted

    original, omitted = permitted(medications)
    assert omitted["start"].tolist() == [pd.Timestamp("2010-01-07")]
    assert list(zip(original["time"].dt.strftime("%Y-%m-%d"), original["kind"], strict=True)) == [
        ("2010-01-01", "start"),
        ("2010-01-07", "start"),
        ("2011-01-01", "stop"),
        ("2014-06-01", "start"),
    ]
    # Later records: invalid rows that start after the anchor (one stops before it), and a
    # changed stop that is not visible at the anchor.
    future = pd.concat(
        [medications, _medications(("2016-01-07", "2015-03-01"), ("2016-01-01", "2014-12-01"))],
        ignore_index=True,
    )
    future.loc[2, "STOP"] = pd.Timestamp("2017-01-01")
    changed, changed_omitted = permitted(future)
    pd.testing.assert_frame_equal(original, changed)
    pd.testing.assert_frame_equal(omitted, changed_omitted)
    # The repair reads only each row's own dates, so applying it before the cut gives the same rows.
    repaired_first, _ = omit_invalid_stops(future, MEDICATIONS)
    pd.testing.assert_frame_equal(
        truncate_table(repaired_first, MEDICATIONS, anchor),
        omit_invalid_stops(truncate_table(future, MEDICATIONS, anchor), MEDICATIONS)[0],
    )


def test_omitted_stops_leave_baseline_features_unchanged() -> None:
    # The historical baseline counts start and point events only, so D31 cannot change it.
    cohort = prediction_cohort(cohort_fixture().loc[["a"]])
    medications = _medications(("2010-01-01", "2010-01-07"), ("2011-01-01", "2010-01-01"))
    repaired, _ = omit_invalid_stops(medications, MEDICATIONS)
    corrected = build_events({"medications": repaired})
    stop = corrected.iloc[[1]].assign(time=pd.Timestamp("2010-01-01"), kind="stop")
    uncorrected = pd.concat([corrected, stop], ignore_index=True)
    features = BagOfCodes(FeatureConfig(min_patients=1)).fit(corrected, cohort)
    pd.testing.assert_frame_equal(
        features.transform(corrected, cohort), features.transform(uncorrected, cohort)
    )
