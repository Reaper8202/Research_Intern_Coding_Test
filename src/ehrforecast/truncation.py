"""Cut train and val timelines at the anchor exactly the way the released test files are cut.

Rules, each checked against the test files (see docs/decisions.md):

1. Keep a row only when its start or event date is before the anchor day.
2. Keep the row when its ``STOP`` is on or after the anchor day, but blank the ``STOP``. The test
   files keep ongoing conditions, medications, care plans and spanning encounters, with no
   ``STOP``.

After the cut, :func:`omit_invalid_stops` repairs intervals whose ``STOP`` precedes their own
``START`` (decision D31). It applies to every split, including the released test files.
"""

from __future__ import annotations

import pandas as pd

from ehrforecast.schema import PATIENT_COL, TableSpec

HISTORY_RULE_VERSION = "omit-stop-before-start-v1"
OMITTED_STOP_COLUMNS = ("patient_id", "source", "code", "start", "stop")


def _anchor_per_row(df: pd.DataFrame, anchors: pd.Series, table: str) -> pd.Series:
    anchor = df[PATIENT_COL].map(anchors)
    if anchor.isna().any():
        missing = df.loc[anchor.isna(), PATIENT_COL].nunique()
        raise ValueError(f"{table}: {missing} patients have no anchor date")
    return anchor


def truncate_table(df: pd.DataFrame, spec: TableSpec, anchors: pd.Series) -> pd.DataFrame:
    """Return the part of ``df`` that is visible at each patient's anchor."""
    anchor = _anchor_per_row(df, anchors, spec.name)
    keep = df[spec.time_col] < anchor
    out = df.loc[keep].copy()
    if spec.stop_col is not None:
        after = out[spec.stop_col] >= anchor[keep]
        out.loc[after, spec.stop_col] = pd.NaT
    return out.reset_index(drop=True)


def find_post_anchor_values(df: pd.DataFrame, spec: TableSpec, anchors: pd.Series) -> int:
    """Count time values on or after the anchor. A correctly cut table returns zero."""
    anchor = _anchor_per_row(df, anchors, spec.name)
    count = int((df[spec.time_col] >= anchor).sum())
    if spec.stop_col is not None:
        count += int((df[spec.stop_col] >= anchor).sum())
    return count


def omit_invalid_stops(df: pd.DataFrame, spec: TableSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the recorded start and blank a ``STOP`` that precedes its own ``START``.

    Apply this to an already cut table. The rule reads only the row's own two dates, so it links
    each stop to its start without looking at other prescriptions of the same code. Returns the
    repaired table and one record per blanked ``STOP``.
    """
    if spec.stop_col is None:
        return df, pd.DataFrame(columns=list(OMITTED_STOP_COLUMNS))
    invalid = df[spec.stop_col].notna() & (df[spec.stop_col] < df[spec.time_col])
    if spec.name == "encounters" and invalid.any():
        # Encounter ends define the anchors. No repair rule is approved for them.
        raise ValueError(f"encounters: {int(invalid.sum())} rows stop before they start")
    (code_col,) = spec.code_cols  # every interval table has one code column
    omitted = pd.DataFrame(
        {
            "patient_id": df.loc[invalid, PATIENT_COL],
            "source": spec.name,
            "code": df.loc[invalid, code_col],
            "start": df.loc[invalid, spec.time_col],
            "stop": df.loc[invalid, spec.stop_col],
        }
    ).reset_index(drop=True)
    out = df.copy()
    out.loc[invalid, spec.stop_col] = pd.NaT
    return out, omitted


def combine_omitted(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate omission records. No records gives an empty table with the same columns."""
    nonempty = [part for part in parts if len(part)]
    if not nonempty:
        return pd.DataFrame(columns=list(OMITTED_STOP_COLUMNS))
    return pd.concat(nonempty, ignore_index=True)
