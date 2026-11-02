"""Sex- and age-stratified incidence: the epidemiological reference baseline.

The predicted probability for a target is the observed five-year rate of first diagnosis among
training patients of the same sex and age band at the anchor:

    rate(stratum) = (cases + m * parent_rate) / (patients + m)

``parent_rate`` is the rate for the same sex over all ages, and ``m`` (``prior_strength``) is a
pseudo-count that pulls small strata towards it. An age band that has no training patients gets
the parent rate.

With ``at_risk_only=True`` the denominator counts only patients who did not already have the
condition before the anchor (the epidemiological definition of incidence), and a patient who
already has the condition gets probability 0, because a prior condition cannot be newly
diagnosed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StratifiedIncidence:
    band_years: float = 10.0
    prior_strength: float = 10.0
    at_risk_only: bool = False
    top_band_age: float = 90.0

    def _strata(self, cohort: pd.DataFrame) -> pd.DataFrame:
        age = cohort["age_at_anchor"].clip(lower=0, upper=self.top_band_age)
        return pd.DataFrame(
            {
                "sex": cohort["gender"].to_numpy(),
                "band": np.floor(age.to_numpy() / self.band_years).astype(int),
            }
        )

    def _at_risk(self, y: np.ndarray, prior: np.ndarray | None) -> np.ndarray:
        if not self.at_risk_only:
            return np.ones_like(y, dtype="float64")
        if prior is None:
            raise ValueError("at_risk_only=True needs the prior-condition flags")
        return (prior == 0).astype("float64")

    def fit(
        self, cohort: pd.DataFrame, y: np.ndarray, prior: np.ndarray | None = None
    ) -> StratifiedIncidence:
        if len(cohort) != len(y):
            raise ValueError("cohort and y must have the same number of rows")
        risk = self._at_risk(y, prior)
        cases = pd.DataFrame(y * risk)
        at_risk = pd.DataFrame(risk)
        strata = self._strata(cohort)

        overall = cases.sum() / at_risk.sum().clip(lower=1)
        sex_cases = cases.groupby(strata["sex"]).sum()
        sex_n = at_risk.groupby(strata["sex"]).sum()
        self.parent_rate_ = (sex_cases / sex_n.clip(lower=1)).to_numpy()
        self.sexes_ = list(sex_cases.index)

        keys = [strata["sex"], strata["band"]]
        band_cases = cases.groupby(keys).sum()
        band_n = at_risk.groupby(keys).sum()
        parent = self.parent_rate_[
            [self.sexes_.index(s) for s in band_cases.index.get_level_values(0)]
        ]
        m = self.prior_strength
        denominator = band_n.to_numpy() + m
        smoothed = (band_cases.to_numpy() + m * parent) / np.where(denominator > 0, denominator, 1)
        # A stratum with nobody at risk (possible only when m = 0) gets the sex rate.
        self.rates_ = np.where(denominator > 0, smoothed, parent)
        self.strata_ = list(band_cases.index)
        self.overall_rate_ = overall.to_numpy()
        return self

    def predict_proba(self, cohort: pd.DataFrame, prior: np.ndarray | None = None) -> np.ndarray:
        """Return the stratum rate for every patient and target, shape (n, k)."""
        strata = self._strata(cohort)
        out = np.tile(self.overall_rate_, (len(cohort), 1))
        sex_row = {s: i for i, s in enumerate(self.sexes_)}
        stratum_row = {key: i for i, key in enumerate(self.strata_)}
        for i, (sex, band) in enumerate(zip(strata["sex"], strata["band"], strict=True)):
            if (sex, band) in stratum_row:
                out[i] = self.rates_[stratum_row[(sex, band)]]
            elif sex in sex_row:
                out[i] = self.parent_rate_[sex_row[sex]]
        if self.at_risk_only:
            if prior is None:
                raise ValueError("at_risk_only=True needs the prior-condition flags")
            out = np.where(prior == 1, 0.0, out)
        return out
