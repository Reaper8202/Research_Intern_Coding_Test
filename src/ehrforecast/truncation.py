"""Cut train and val timelines at the anchor exactly the way the released test files are cut.

Rules, each checked against the test files (see docs/decisions.md):

1. Keep a row only when its start or event date is before the anchor day.
2. Keep the row when its ``STOP`` is on or after the anchor day, but blank the ``STOP``. The test
   files keep ongoing conditions, medications, care plans and spanning encounters, with no
   ``STOP``.
"""

from __future__ import annotations

import pandas as pd

from ehrforecast.schema import PATIENT_COL, TableSpec


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
