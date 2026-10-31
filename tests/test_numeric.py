from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehrforecast.numeric import assign_bins, fit_bin_edges, numeric_observations, unit_audit
from tests.test_contracts import event_fixture


def test_bin_count_boundaries_and_clipping() -> None:
    edges = fit_bin_edges(np.arange(40, dtype=float))
    np.testing.assert_array_equal(edges, [0, 19.5, 39])
    np.testing.assert_array_equal(
        assign_bins(np.array([-9, 0, 19.49, 19.5, 39, 90]), edges), [0, 0, 0, 1, 1, 1]
    )
    assert len(fit_bin_edges(np.arange(400))) == 11
    assert len(fit_bin_edges(np.arange(19))) == 2


def test_duplicate_boundaries_and_constant_groups() -> None:
    edges = fit_bin_edges(np.ones(200))
    np.testing.assert_array_equal(edges, [1])
    np.testing.assert_array_equal(assign_bins(np.array([-1, 1, 9]), edges), [0, 0, 0])
    duplicated = fit_bin_edges(np.repeat([0.0, 1.0], 100))
    assert np.all(np.diff(duplicated) > 0)


@pytest.mark.parametrize("bad", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_invalid_fit_values_fail(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_bin_edges(bad)


def test_numeric_audit_keeps_units_and_excludes_encounter_duration() -> None:
    events = event_fixture()
    extra = events.iloc[[1]].copy()
    extra["unit"] = " MG/L "
    extra["value_num"] = 1000
    events = pd.concat([events, extra], ignore_index=True)
    observed = numeric_observations(events)
    assert observed["unit"].tolist() == ["mg/L", "MG/L"]
    table = unit_audit(events)
    assert len(table) == 2
    assert table["n_values"].tolist() == [1, 1]
    assert set(table["code"]) == {"lab"}
