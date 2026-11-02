from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehrforecast.incidence import StratifiedIncidence


def _cohort(rows: list[tuple[str, float]]) -> pd.DataFrame:
    sex, age = zip(*rows, strict=True)
    return pd.DataFrame({"gender": list(sex), "age_at_anchor": list(age)})


TRAIN = _cohort([("F", 30), ("F", 35), ("F", 72), ("M", 31), ("M", 33), ("M", 38)])
Y = np.array([[1], [0], [0], [0], [0], [1]])


def test_unsmoothed_rate_is_the_stratum_proportion() -> None:
    model = StratifiedIncidence(band_years=10, prior_strength=0).fit(TRAIN, Y)
    p = model.predict_proba(_cohort([("F", 39.9), ("F", 70), ("M", 30)]))
    np.testing.assert_allclose(p[:, 0], [1 / 2, 0.0, 1 / 3])


def test_smoothing_pulls_towards_the_sex_rate() -> None:
    model = StratifiedIncidence(band_years=10, prior_strength=2).fit(TRAIN, Y)
    p = model.predict_proba(_cohort([("F", 30)]))
    female_rate = 1 / 3
    assert p[0, 0] == pytest.approx((1 + 2 * female_rate) / (2 + 2))


def test_empty_band_falls_back_to_the_sex_rate() -> None:
    model = StratifiedIncidence(band_years=10, prior_strength=0).fit(TRAIN, Y)
    p = model.predict_proba(_cohort([("F", 55), ("M", 80)]))
    np.testing.assert_allclose(p[:, 0], [1 / 3, 1 / 3])


def test_ages_above_the_top_band_share_one_band() -> None:
    model = StratifiedIncidence(band_years=10, top_band_age=90)
    strata = model._strata(_cohort([("F", 90.5), ("F", 104), ("F", 89)]))
    assert strata["band"].tolist() == [9, 9, 8]


def test_at_risk_rate_excludes_prior_cases_and_scores_them_zero() -> None:
    prior = np.array([[0], [1], [0], [0], [0], [0]])  # patient 2 already had the condition
    model = StratifiedIncidence(band_years=10, prior_strength=0, at_risk_only=True)
    model.fit(TRAIN, Y, prior)
    p = model.predict_proba(_cohort([("F", 30), ("F", 31)]), prior=np.array([[0], [1]]))
    np.testing.assert_allclose(p[:, 0], [1.0, 0.0])  # 1 case among 1 at-risk woman in 30-39


def test_stratum_with_nobody_at_risk_gets_the_sex_rate() -> None:
    prior = np.array([[0], [0], [1], [0], [0], [0]])  # the only woman aged 70-79 is a prior case
    model = StratifiedIncidence(band_years=10, prior_strength=0, at_risk_only=True)
    model.fit(TRAIN, Y, prior)
    p = model.predict_proba(_cohort([("F", 75)]), prior=np.array([[0]]))
    assert p[0, 0] == pytest.approx(1 / 2)  # 1 case among 2 at-risk women


def test_at_risk_without_prior_flags_raises() -> None:
    with pytest.raises(ValueError, match="prior-condition flags"):
        StratifiedIncidence(at_risk_only=True).fit(TRAIN, Y)
