"""Label construction: target conditions newly diagnosed in the five years after the anchor.

A condition counts only if its first-ever diagnosis falls inside
``[anchor_date, window_end]`` (both ends included). First-ever diagnosis dates come from the full,
untruncated timeline, so a condition that the patient had before the anchor never counts, even
when it is recorded again inside the window.
"""

from __future__ import annotations

import pandas as pd


def first_diagnosis(conditions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (patient, code) with the date of the first diagnosis."""
    return (
        conditions.groupby(["PATIENT", "CODE"], observed=True)["START"]
        .min()
        .rename("first_date")
        .reset_index()
    )


def _to_wide(pairs: pd.DataFrame, patients: pd.Index, codes: list[str]) -> pd.DataFrame:
    wide = pd.DataFrame(0, index=patients, columns=codes, dtype="int8")
    if len(pairs):
        hits = pd.crosstab(pairs["PATIENT"], pairs["CODE"]).clip(upper=1)
        hits = hits.reindex(index=patients, columns=codes, fill_value=0)
        wide = hits.astype("int8")
    wide.index.name = "patient_id"
    wide.columns.name = None
    return wide


def build_labels(conditions: pd.DataFrame, cohort: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Build the multi-label target matrix.

    Args:
        conditions: the full, untruncated conditions table for the labelled patients.
        cohort: rows for the labelled patients, with ``anchor_date`` and ``window_end``.
        codes: target condition codes, in submission column order.
    """
    if cohort[["anchor_date", "window_end"]].isna().any().any():
        raise ValueError("Labels need an anchor date and window end for every patient")
    first = first_diagnosis(conditions[conditions["PATIENT"].isin(cohort.index)])
    first = first[first["CODE"].isin(codes)]
    anchor = first["PATIENT"].map(cohort["anchor_date"])
    window_end = first["PATIENT"].map(cohort["window_end"])
    in_window = (first["first_date"] >= anchor) & (first["first_date"] <= window_end)
    return _to_wide(first.loc[in_window], cohort.index, codes)


def build_prior_flags(
    history_conditions: pd.DataFrame, patients: pd.Index, codes: list[str]
) -> pd.DataFrame:
    """Flag target conditions already diagnosed before the anchor (from history only)."""
    seen = history_conditions.loc[history_conditions["CODE"].isin(codes), ["PATIENT", "CODE"]]
    return _to_wide(seen.drop_duplicates(), patients, codes)
