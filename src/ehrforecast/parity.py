"""Summary statistics that compare the truncated train/val history with the released test files.

If the train/val truncation copies the test truncation correctly, these statistics agree up to
sampling noise. A large gap means that the model would see inputs in training that it never
sees at test time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ehrforecast.schema import EVENT_TABLES


def blank_stop_fraction(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Fraction of rows with a blank ``STOP``, for each interval table."""
    out = {}
    for name, df in tables.items():
        stop = EVENT_TABLES[name].stop_col
        if stop is not None and len(df):
            out[name] = round(float(df[stop].isna().mean()), 4)
    return out


def history_summary(events: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Any]:
    """Describe the length and recency of the visible history for one group of patients."""
    ev = events[events["patient_id"].isin(cohort.index)]
    per_patient = ev.groupby("patient_id", observed=True)["time"].agg(["size", "min", "max"])
    per_patient = per_patient.reindex(cohort.index)
    anchor = cohort["anchor_date"]
    gap_days = (anchor - per_patient["max"].dt.normalize()).dt.days
    history_years = (anchor - per_patient["min"]).dt.days / 365.25
    n = len(cohort)
    return {
        "n_patients": n,
        "patients_without_history": int(per_patient["size"].isna().sum()),
        "events_per_patient_median": float(per_patient["size"].median()),
        "events_per_patient_p90": float(per_patient["size"].quantile(0.9)),
        "history_years_median": round(float(history_years.median()), 2),
        "age_at_anchor_median": round(float(cohort["age_at_anchor"].median()), 2),
        "female_fraction": round(float((cohort["gender"] == "F").mean()), 4),
        "days_last_event_to_anchor_median": float(gap_days.median()),
        "days_last_event_to_anchor_share": {
            str(k): round(float((gap_days == k).mean()), 4) for k in range(0, 4)
        },
    }


def label_prevalence(labels: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Positive count and prevalence per target, for the train and val splits."""
    rows = {}
    for split in ("train", "val"):
        y = labels.loc[cohort.index[cohort["split"] == split].intersection(labels.index)]
        rows[f"{split}_positives"] = y.sum()
        rows[f"{split}_prevalence"] = y.mean().round(4)
    return pd.DataFrame(rows)
