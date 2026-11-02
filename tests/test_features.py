from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehrforecast.features import BagOfCodes, FeatureConfig
from tests.helpers import day, ts


def _cohort(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_date": [day("2015-01-01")] * len(ids),
            "age_at_anchor": [40.0] * len(ids),
            "gender": ["M"] * len(ids),
        },
        index=pd.Index(ids, name="patient_id"),
    )


def _events(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    patient, time, kind, code = zip(*rows, strict=True)
    return pd.DataFrame(
        {
            "patient_id": list(patient),
            "time": ts(*time),
            "source": "conditions",
            "kind": list(kind),
            "code": list(code),
        }
    )


EVENTS = _events(
    [
        ("a", "2000-01-01", "start", "common"),
        ("a", "2014-06-01", "start", "common"),  # recent (within 2 years)
        ("a", "2014-07-01", "stop", "common"),  # stop rows are not counted
        ("b", "2001-01-01", "start", "common"),
        ("b", "2001-01-01", "start", "rare"),
        ("c", "2001-01-01", "start", "only_in_c"),  # c is not used for fitting
    ]
)


def test_vocabulary_comes_from_fit_patients_and_min_count() -> None:
    bag = BagOfCodes(FeatureConfig(min_patients=2)).fit(EVENTS, _cohort(["a", "b"]))
    assert bag.vocabulary_ == ["conditions:common"]


def test_counts_are_log1p_and_unknown_codes_are_ignored() -> None:
    bag = BagOfCodes(FeatureConfig(min_patients=2)).fit(EVENTS, _cohort(["a", "b"]))
    x = bag.transform(EVENTS, _cohort(["a", "b", "c"]))
    assert x.loc["a", "count:conditions:common"] == pytest.approx(np.log1p(2))
    assert x.loc["a", "recent:conditions:common"] == pytest.approx(np.log1p(1))
    assert x.loc["b", "recent:conditions:common"] == 0
    assert x.loc["c"].filter(like="conditions").sum() == 0
    assert list(x.index) == ["a", "b", "c"]
    assert (x.dtypes == "float32").all()


def test_demographics_only_has_no_code_columns() -> None:
    bag = BagOfCodes(FeatureConfig(include_codes=False)).fit(EVENTS, _cohort(["a"]))
    x = bag.transform(EVENTS, _cohort(["a"]))
    assert not any(":" in c for c in x.columns)
    assert x.loc["a", "sex_male"] == 1.0
    assert x.loc["a", "age_band_35_50"] == 1.0
    assert x.filter(like="age_band").sum(axis=1).loc["a"] == 1.0


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        BagOfCodes().transform(EVENTS, _cohort(["a"]))
