"""One identifier-aligned prior-condition rule for all new model predictions."""

from __future__ import annotations

import pandas as pd

from ehrforecast.contracts import align_targets, validate_binary, validate_probabilities


def apply_prior_mask(
    probabilities: pd.DataFrame, prior: pd.DataFrame, codes: list[str]
) -> pd.DataFrame:
    probabilities = align_targets(probabilities, probabilities.index, codes, "probabilities")
    prior = align_targets(prior, probabilities.index, codes, "prior mask")
    validate_probabilities(probabilities)
    validate_binary(prior, "prior mask")
    return probabilities.mask(prior.astype(bool), 0.0)
