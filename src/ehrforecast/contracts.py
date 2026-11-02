"""Validated prediction inputs, separate from labels and end-of-record fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ehrforecast.events import EVENT_COLUMNS, EVENTS_FILE, KIND_ORDER, verify_events
from ehrforecast.schema import EVENT_TABLES

COHORT_COLUMNS = ("birthdate", "gender", "anchor_date")
# v2: event histories omit a STOP that precedes its own START (D31).
# v3: interval events carry ``interval_start`` for refill compression.
INPUT_SCHEMA_VERSION = "prediction-inputs-v3"


def validate_ids(ids: pd.Index, name: str = "patient IDs") -> None:
    if ids.has_duplicates or np.asarray(ids.isna()).any():
        raise ValueError(f"{name} must be unique and nonmissing")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{name} must be nonblank strings")


def validate_codes(codes: list[str]) -> None:
    if not codes:
        raise ValueError("target codes must not be empty")
    validate_ids(pd.Index(codes), "target codes")


def align_targets(
    frame: pd.DataFrame, patients: pd.Index, codes: list[str], name: str
) -> pd.DataFrame:
    """Align by identifiers; reject missing, extra, or duplicate rows and columns."""
    validate_ids(patients)
    validate_ids(frame.index, f"{name} patient IDs")
    validate_codes(codes)
    validate_ids(frame.columns, f"{name} target codes")
    if set(frame.index) != set(patients) or set(frame.columns) != set(codes):
        raise ValueError(f"{name} patient IDs or target codes do not match")
    return frame.loc[patients, codes].copy()


def validate_binary(frame: pd.DataFrame, name: str) -> None:
    if not frame.isin([0, 1]).to_numpy().all():
        raise ValueError(f"{name} must contain only finite binary values")


def validate_probabilities(frame: pd.DataFrame) -> None:
    try:
        values = frame.to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must be numeric") from error
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")


def prediction_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Copy the input allowlist and recompute age from permitted dates."""
    validate_ids(cohort.index)
    missing = set(COHORT_COLUMNS) - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort lacks input columns: {sorted(missing)}")
    out = cohort.loc[:, list(COHORT_COLUMNS)].copy()
    for name in ("birthdate", "anchor_date"):
        dates = out[name]
        if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
            raise ValueError(f"cohort.{name} must contain valid dates")
        if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError(f"cohort.{name} must contain timezone-naive calendar days")
    if not out["gender"].isin(["F", "M"]).all():
        raise ValueError("cohort.gender must use the supplied F/M encoding")
    if (out["birthdate"] >= out["anchor_date"]).any():
        raise ValueError("patients must be born before their anchors")
    out["age_at_anchor"] = (out["anchor_date"] - out["birthdate"]).dt.days / 365.25
    out.index.name = "patient_id"
    return out


def prediction_events(events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Validate pre-cut histories and remove non-observation numeric payloads."""
    missing = set(EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"events lack input columns: {sorted(missing)}")
    out = events.loc[:, list(EVENT_COLUMNS)].copy()
    for name in ("patient_id", "source", "kind", "code"):
        if out[name].isna().any() or out[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"events.{name} contains missing or blank values")
    if not out["patient_id"].isin(cohort.index).all():
        raise ValueError("events contain unknown patient IDs")
    if not out["source"].isin(EVENT_TABLES).all() or not out["kind"].isin(KIND_ORDER).all():
        raise ValueError("events contain an unknown source or kind")
    dates = out["time"]
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
        raise ValueError("events.time must contain valid dates")
    if dates.dt.tz is not None:
        raise ValueError("events.time must use timezone-naive UTC")
    anchors = out["patient_id"].map(cohort["anchor_date"])
    births = out["patient_id"].map(cohort["birthdate"])
    if (dates >= anchors).any() or (dates < births).any():
        raise ValueError("event dates must be on/after birth and strictly before the anchor")
    interval = out["kind"].isin(["start", "stop"])
    starts = out["interval_start"]
    if not pd.api.types.is_datetime64_any_dtype(starts) or starts[~interval].notna().any():
        raise ValueError("events.interval_start must be empty for point events")
    if starts[interval].isna().any() or (starts[interval] > dates[interval]).any():
        raise ValueError("events.interval_start must not follow the event for interval rows")
    # Step 1 stores encounter durations. They must never become numeric model inputs.
    observation = out["source"].eq("observations")
    out.loc[~observation, "value_num"] = np.nan
    values = out.loc[observation, "value_num"].dropna().to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("numeric observations must be finite or explicitly missing")
    out.loc[~observation, "unit"] = pd.NA
    out.loc[~out["source"].isin(["observations", "encounters"]), "value_text"] = pd.NA
    return out.reset_index(drop=True)


def prior_from_history(events: pd.DataFrame, patients: pd.Index, codes: list[str]) -> pd.DataFrame:
    """Use the full permitted diagnosis history, before model token truncation."""
    validate_ids(patients)
    validate_codes(codes)
    if not events["patient_id"].isin(patients).all():
        raise ValueError("prior history contains unknown patient IDs")
    hits = events.loc[
        events["source"].eq("conditions") & events["kind"].eq("start") & events["code"].isin(codes),
        ["patient_id", "code"],
    ].drop_duplicates()
    out = pd.DataFrame(0, index=patients, columns=codes, dtype="int8")
    for code, group in hits.groupby("code", observed=True):
        out.loc[group["patient_id"], code] = 1
    out.index.name = "patient_id"
    return out


@dataclass(frozen=True)
class PredictionInputs:
    cohort: pd.DataFrame
    events: pd.DataFrame
    prior: pd.DataFrame
    codes: list[str]


@dataclass(frozen=True)
class TrainingData:
    inputs: PredictionInputs
    labels: pd.DataFrame


def load_splits(dataset_dir: Path) -> pd.Series:
    """Read split metadata only, in the exact step-2 cohort order."""
    splits = pd.read_parquet(dataset_dir / "cohort.parquet", columns=["split"])["split"]
    validate_ids(splits.index)
    if not splits.isin(["train", "val", "test"]).all():
        raise ValueError("cohort contains unknown or missing splits")
    return splits


def load_prediction_inputs(dataset_dir: Path, patients: pd.Index) -> PredictionInputs:
    """Load no labels, descriptions, future tables, or prohibited cohort columns."""
    validate_ids(patients)
    codes = pd.read_csv(dataset_dir / "targets.csv", dtype=str, keep_default_na=False)[
        "CODE"
    ].tolist()
    validate_codes(codes)
    cohort = pd.read_parquet(dataset_dir / "cohort.parquet", columns=list(COHORT_COLUMNS))
    validate_ids(cohort.index)
    if not patients.isin(cohort.index).all():
        raise ValueError("requested patient IDs are missing from the cohort")
    cohort = prediction_cohort(cohort.loc[patients])
    verify_events(dataset_dir)
    events = pd.read_parquet(
        dataset_dir / EVENTS_FILE,
        columns=list(EVENT_COLUMNS),
        filters=[("patient_id", "in", patients.tolist())],
    )
    events = prediction_events(events, cohort)
    prior = prior_from_history(events, cohort.index, codes)
    return PredictionInputs(cohort, events, prior, codes)


def load_training_data(dataset_dir: Path) -> TrainingData:
    splits = load_splits(dataset_dir)
    patients = splits.index[splits.eq("train")]
    if patients.empty:
        raise ValueError("training split is empty")
    inputs = load_prediction_inputs(dataset_dir, patients)
    # Arrow row filtering prevents official val labels from entering the returned frame.
    labels = pd.read_parquet(
        dataset_dir / "labels.parquet", filters=[("patient_id", "in", patients.tolist())]
    )
    labels = align_targets(labels, patients, inputs.codes, "training labels")
    validate_binary(labels, "training labels")
    if np.logical_and(labels.to_numpy(), inputs.prior.to_numpy()).any():
        raise ValueError("a prior condition cannot have a positive first-diagnosis label")
    stored_prior = pd.read_parquet(
        dataset_dir / "prior_conditions.parquet",
        filters=[("patient_id", "in", patients.tolist())],
    )
    stored_prior = align_targets(stored_prior, patients, inputs.codes, "stored prior mask")
    validate_binary(stored_prior, "stored prior mask")
    if not np.array_equal(stored_prior.to_numpy(), inputs.prior.to_numpy()):
        raise ValueError("stored prior mask disagrees with the permitted condition history")
    return TrainingData(inputs, labels)
