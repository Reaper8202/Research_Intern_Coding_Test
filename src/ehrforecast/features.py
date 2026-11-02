"""Bag-of-codes features for the baseline model.

Each patient becomes one fixed-length vector:

* ``count:<source>:<code>``: log(1 + number of times the code occurs in the whole history).
* ``recent:<source>:<code>``: the same count over the last ``recent_years`` before the anchor.
* ``age_years``, one-hot age bands, and ``sex_male``.

Only ``start`` and ``point`` events are counted, so an interval event counts once. The
vocabulary comes only from the patients passed to :meth:`BagOfCodes.fit` (the training split, or
the training part of a cross-validation fold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEMOGRAPHIC_PREFIXES = ("age_", "sex_")


@dataclass(frozen=True)
class FeatureConfig:
    min_patients: int = 10
    recent_years: float | None = 2.0
    kinds: tuple[str, ...] = ("start", "point")
    age_band_edges: tuple[float, ...] = (18.0, 35.0, 50.0, 65.0, 80.0)
    include_codes: bool = True


def demographic_features(cohort: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    age = cohort["age_at_anchor"].astype("float64")
    out = pd.DataFrame(index=cohort.index)
    out["age_years"] = age
    edges = [-np.inf, *config.age_band_edges, np.inf]
    band = pd.cut(age, bins=edges, right=False, labels=False)
    for i in range(len(edges) - 1):
        lo = "min" if np.isinf(edges[i]) else f"{edges[i]:g}"
        hi = "max" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:g}"
        out[f"age_band_{lo}_{hi}"] = (band == i).astype("float64")
    out["sex_male"] = (cohort["gender"] == "M").astype("float64")
    return out


@dataclass
class BagOfCodes:
    """Fit a code vocabulary on training patients, then map any patient to a feature vector."""

    config: FeatureConfig = field(default_factory=FeatureConfig)
    vocabulary_: list[str] | None = None

    def _tokens(self, events: pd.DataFrame, patients: pd.Index) -> pd.DataFrame:
        ev = events.loc[
            events["patient_id"].isin(patients) & events["kind"].isin(self.config.kinds),
            ["patient_id", "time", "source", "code"],
        ]
        token = ev["source"].astype("string") + ":" + ev["code"].astype("string")
        return pd.DataFrame({"patient_id": ev["patient_id"], "time": ev["time"], "token": token})

    def fit(self, events: pd.DataFrame, cohort: pd.DataFrame) -> BagOfCodes:
        if not self.config.include_codes:
            self.vocabulary_ = []
            return self
        tokens = self._tokens(events, cohort.index)
        n_patients = tokens.drop_duplicates(["patient_id", "token"])["token"].value_counts()
        kept = n_patients[n_patients >= self.config.min_patients].index
        self.vocabulary_ = sorted(kept.astype(str))
        return self

    def _count_matrix(self, tokens: pd.DataFrame, patients: pd.Index, prefix: str) -> pd.DataFrame:
        assert self.vocabulary_ is not None
        known = tokens[tokens["token"].isin(self.vocabulary_)]
        # unstack (not pivot_table) because it also handles an empty selection.
        counts = known.groupby(["patient_id", "token"]).size().unstack(fill_value=0)  # noqa: PD010
        counts = counts.reindex(index=patients, columns=self.vocabulary_, fill_value=0)
        return pd.DataFrame(
            np.log1p(counts.to_numpy(dtype="float64")),
            index=patients,
            columns=[f"{prefix}:{token}" for token in self.vocabulary_],
        )

    def transform(self, events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        """Return a float32 feature frame indexed like ``cohort``."""
        if self.vocabulary_ is None:
            raise RuntimeError("BagOfCodes.transform called before fit")
        parts = [demographic_features(cohort, self.config)]
        if self.config.include_codes:
            tokens = self._tokens(events, cohort.index)
            parts.append(self._count_matrix(tokens, cohort.index, "count"))
            if self.config.recent_years is not None:
                window = pd.Timedelta(days=365.25 * self.config.recent_years)
                start = tokens["patient_id"].map(cohort["anchor_date"]) - window
                recent = tokens[tokens["time"] >= start]
                parts.append(self._count_matrix(recent, cohort.index, "recent"))
        return pd.concat(parts, axis=1).astype("float32")
