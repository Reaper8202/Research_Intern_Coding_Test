"""Training observation audits and shared quantile boundaries; no clinical thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIT_RULE_VERSION = "separate-literal-units-v1"


def numeric_observations(events: pd.DataFrame) -> pd.DataFrame:
    out = events.loc[
        events["source"].eq("observations") & events["value_num"].notna(),
        ["patient_id", "time", "code", "value_num", "unit"],
    ].copy()
    if not np.isfinite(out["value_num"].to_numpy(dtype="float64")).all():
        raise ValueError("numeric observations contain non-finite values")
    # Unit case can carry meaning. Strip surrounding whitespace only. Empty means missing.
    out["unit"] = out["unit"].fillna("").astype(str).str.strip()
    return out


def unit_audit(events: pd.DataFrame) -> pd.DataFrame:
    observations = numeric_observations(events)
    table = observations.groupby(["code", "unit"], observed=True).agg(
        n_values=("value_num", "size"),
        n_patients=("patient_id", "nunique"),
        minimum=("value_num", "min"),
        median=("value_num", "median"),
        maximum=("value_num", "max"),
    )
    return table.reset_index().sort_values(["code", "unit"]).reset_index(drop=True)


def fit_bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("quantile fitting needs a nonempty finite value vector")
    n_bins = max(1, min(10, len(values) // 20))
    return np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1), method="linear"))


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return zero-based bins. Constant groups have one bin; outside values clip."""
    values, edges = np.asarray(values, dtype="float64"), np.asarray(edges, dtype="float64")
    if edges.ndim != 1 or not len(edges) or not np.isfinite(edges).all():
        raise ValueError("bin boundaries must be a nonempty finite vector")
    if (np.diff(edges) <= 0).any():
        raise ValueError("bin boundaries must increase strictly")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bin assignment needs finite values; handle missing values separately")
    return np.searchsorted(edges[1:-1], values, side="right")
