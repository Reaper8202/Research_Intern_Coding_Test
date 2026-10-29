from __future__ import annotations

import pandas as pd

from ehrforecast.io import parse_timestamps


def ts(*values: str | None) -> pd.Series:
    """Parse literal timestamps with the production parser."""
    return parse_timestamps(pd.Series(list(values), dtype="string"))


def day(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)
