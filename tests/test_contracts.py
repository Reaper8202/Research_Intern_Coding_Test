from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehrforecast.contracts import (
    align_targets,
    load_prediction_inputs,
    load_training_data,
    prediction_cohort,
    prediction_events,
    prior_from_history,
)
from ehrforecast.events import EVENTS_MANIFEST, save_events
from ehrforecast.masking import apply_prior_mask
from tests.helpers import ts


def cohort_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "val"],
            "birthdate": ts("1980-01-01", "1981-01-01").to_numpy(),
            "gender": ["F", "M"],
            "anchor_date": ts("2015-01-01", "2015-01-01").to_numpy(),
            "last_encounter": ts("2020-01-01", "2020-01-01").to_numpy(),
            "window_end": ts("2020-01-01", "2020-01-01").to_numpy(),
            "age_at_anchor": [-999.0, -999.0],
        },
        index=pd.Index(["a", "b"], name="patient_id"),
    )


def event_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "time": ts("2014-01-01", "2014-01-01", "2014-06-01"),
            "source": ["encounters", "observations", "conditions"],
            "kind": ["start", "point", "start"],
            "code": ["visit", "lab", "X"],
            "value_num": [24.0, 10.0, np.nan],
            "value_text": pd.Series(["ambulatory", None, "excluded"], dtype="string"),
            "unit": pd.Series(["hours", "mg/L", None], dtype="string"),
            "interval_start": ts("2014-01-01", None, "2014-06-01"),
        }
    )


def write_processed(path: Path) -> None:
    cohort = cohort_fixture()
    cohort.to_parquet(path / "cohort.parquet")
    save_events(event_fixture(), path, {"train": 0, "val": 0, "test": 0})
    pd.DataFrame({"CODE": ["X"], "DESCRIPTION": ["Example"]}).to_csv(
        path / "targets.csv", index=False
    )
    # Invalid val labels must not enter the training loader.
    pd.DataFrame({"X": [0, 99]}, index=cohort.index).to_parquet(path / "labels.parquet")
    pd.DataFrame({"X": [0, 1]}, index=cohort.index).to_parquet(path / "prior_conditions.parquet")


def test_prediction_view_excludes_future_fields_and_recomputes_age() -> None:
    cohort = cohort_fixture()
    first = prediction_cohort(cohort)
    assert list(first.columns) == ["birthdate", "gender", "anchor_date", "age_at_anchor"]
    assert first.loc["a", "age_at_anchor"] == pytest.approx(35.0, abs=0.01)
    cohort["last_encounter"] = pd.Timestamp("2099-01-01")
    cohort["window_end"] = pd.Timestamp("2099-01-01")
    cohort["age_at_anchor"] = 9000
    pd.testing.assert_frame_equal(first, prediction_cohort(cohort))


def test_only_observations_supply_numeric_inputs() -> None:
    cohort, events = prediction_cohort(cohort_fixture()), event_fixture()
    first = prediction_events(events, cohort)
    assert pd.isna(first.loc[0, "value_num"])
    assert first.loc[1, "value_num"] == 10
    assert pd.isna(first.loc[0, "unit"])
    assert pd.isna(first.loc[2, "value_text"])
    events.loc[0, "value_num"] = np.inf
    events.loc[2, "value_text"] = "future information"
    pd.testing.assert_frame_equal(first, prediction_events(events, cohort))


@pytest.mark.parametrize("date", [None, "2015-01-01", "1979-01-01"])
def test_missing_or_out_of_history_event_dates_fail(date: str | None) -> None:
    events = event_fixture()
    events.loc[0, "time"] = pd.NaT if date is None else pd.Timestamp(date)
    with pytest.raises(ValueError, match="date"):
        prediction_events(events, prediction_cohort(cohort_fixture()))


def test_unknown_patient_and_nonfinite_observation_fail() -> None:
    events, cohort = event_fixture(), prediction_cohort(cohort_fixture())
    events.loc[0, "patient_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown patient"):
        prediction_events(events, cohort)
    events = event_fixture()
    events.loc[1, "value_num"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        prediction_events(events, cohort)


def test_training_loader_excludes_val_labels_and_events(tmp_path: Path) -> None:
    write_processed(tmp_path)
    data = load_training_data(tmp_path)
    assert data.inputs.cohort.index.tolist() == ["a"]
    assert data.labels["X"].tolist() == [0]
    assert set(data.inputs.events["patient_id"]) == {"a"}
    assert "X" not in set(data.inputs.events["code"])
    assert data.inputs.prior.loc["a", "X"] == 0


def test_prediction_loader_needs_no_labels_or_stored_prior(tmp_path: Path) -> None:
    write_processed(tmp_path)
    (tmp_path / "labels.parquet").unlink()
    (tmp_path / "prior_conditions.parquet").unlink()
    result = load_prediction_inputs(tmp_path, pd.Index(["b", "a"]))
    assert result.cohort.index.tolist() == ["b", "a"]
    assert result.prior["X"].tolist() == [1, 0]


def test_training_loader_rejects_stale_prior(tmp_path: Path) -> None:
    write_processed(tmp_path)
    prior = pd.read_parquet(tmp_path / "prior_conditions.parquet")
    prior.loc["a", "X"] = 1
    prior.to_parquet(tmp_path / "prior_conditions.parquet")
    with pytest.raises(ValueError, match="disagrees"):
        load_training_data(tmp_path)


def test_duplicate_ids_and_incomplete_targets_fail() -> None:
    cohort = cohort_fixture()
    cohort.index = pd.Index(["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        prediction_cohort(cohort)
    with pytest.raises(ValueError, match="do not match"):
        align_targets(pd.DataFrame({"X": [0]}, index=["a"]), pd.Index(["a"]), ["Y"], "labels")


def test_prior_mask_aligns_patients_and_codes_and_does_not_mutate_scores() -> None:
    probabilities = pd.DataFrame({"X": [0.3, 0.5], "Y": [0.6, 0.9]}, index=["a", "b"])
    original = probabilities.copy()
    prior = pd.DataFrame({"Y": [0, 1], "X": [1, 0]}, index=["b", "a"])
    masked = apply_prior_mask(probabilities, prior, ["Y", "X"])
    expected = pd.DataFrame({"Y": [0.0, 0.9], "X": [0.3, 0.0]}, index=["a", "b"])
    pd.testing.assert_frame_equal(masked, expected)
    pd.testing.assert_frame_equal(probabilities, original)
    prior.loc["a", "Y"] = 2
    with pytest.raises(ValueError, match="binary"):
        apply_prior_mask(probabilities, prior, ["Y", "X"])


def test_prior_uses_condition_starts_even_without_model_vocabulary() -> None:
    events = event_fixture()
    prior = prior_from_history(events, pd.Index(["b", "a", "empty"]), ["absent", "X"])
    assert prior.loc["b"].to_numpy().tolist() == [0, 1]
    assert prior.loc["empty"].to_numpy().tolist() == [0, 0]


def test_loader_rejects_missing_older_or_changed_event_table(tmp_path: Path) -> None:
    patients = pd.Index(["a"])
    write_processed(tmp_path)
    manifest_path = tmp_path / EVENTS_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "history_rule": "none"}))
    with pytest.raises(ValueError, match="older history rule"):
        load_prediction_inputs(tmp_path, patients)
    write_processed(tmp_path)
    event_fixture().iloc[:2].to_parquet(tmp_path / "events.parquet", index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_prediction_inputs(tmp_path, patients)
    write_processed(tmp_path)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_prediction_inputs(tmp_path, patients)
