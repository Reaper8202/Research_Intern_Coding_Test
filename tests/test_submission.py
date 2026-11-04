"""Submission recipe and the LightGBM feature spec: exact blends, Platt scaling, spec round trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehrforecast.calibration import apply_platt, fit_platt, nested_platt
from ehrforecast.lightgbm_cv import all_values, latest_values, value_features
from ehrforecast.lightgbm_final import fit_value_spec, value_features_from_spec
from ehrforecast.stacking import logit, sigmoid
from ehrforecast.submission import blend, build
from tests.transformer_fixtures import TARGETS, fold_data

CODES = ["A", "B"]


def _frames(n: int = 40, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    index = pd.Index([f"p{i:03d}" for i in range(n)], name="patient_id")
    bases = {
        "transformer": pd.DataFrame(rng.random((n, 2)), index=index, columns=CODES),
        "lightgbm_counts": pd.DataFrame(rng.random((n, 2)), index=index, columns=CODES),
    }
    prior = pd.DataFrame(0, index=index, columns=CODES)
    prior.iloc[:3, 1] = 1
    return bases, prior


def test_blend_is_the_exact_logit_or_probability_mean() -> None:
    bases, _ = _frames()
    frames = list(bases.values())
    z = np.mean([logit(f.to_numpy()) for f in frames], axis=0)
    np.testing.assert_allclose(blend(frames, "logit_mean").to_numpy(), sigmoid(z))
    np.testing.assert_allclose(
        blend(frames, "prob_mean").to_numpy(), np.mean([f.to_numpy() for f in frames], axis=0)
    )
    with pytest.raises(ValueError):
        blend(frames, "logistic")


def test_build_applies_platt_weights_and_the_prior_mask(tmp_path: Path) -> None:
    bases, prior = _frames()
    weights = {"A": {"a": 2.0, "b": -0.5, "fitted": 1.0}, "B": {"a": 1.0, "b": 0.0, "fitted": 0.0}}
    (tmp_path / "platt.json").write_text(json.dumps(weights))
    stack = {
        "bases": ["transformer", "lightgbm_counts"],
        "stacker": "logit_mean",
        "platt_weights": "platt.json",
    }
    out = build(stack, bases, prior, CODES, tmp_path)
    blended = blend([bases["transformer"], bases["lightgbm_counts"]], "logit_mean")
    expected_a = sigmoid(2.0 * logit(blended["A"].to_numpy()) - 0.5)
    np.testing.assert_allclose(out["A"].to_numpy(), expected_a)
    assert (out.to_numpy()[prior.to_numpy() == 1] == 0).all()
    unmasked = prior["B"].to_numpy() == 0
    np.testing.assert_allclose(out["B"].to_numpy()[unmasked], blended["B"].to_numpy()[unmasked])
    with pytest.raises(ValueError, match="missing"):
        build({**stack, "bases": ["transformer", "other"]}, bases, prior, CODES, tmp_path)


def test_platt_is_monotone_within_a_target_and_nested_by_fold() -> None:
    bases, prior = _frames(n=60)
    rng = np.random.default_rng(1)
    probabilities = bases["transformer"]
    # labels drawn from the probabilities, so the model carries signal and the slope is positive
    labels = pd.DataFrame(
        (rng.random((60, 2)) < probabilities.to_numpy()).astype(int),
        index=prior.index,
        columns=CODES,
    )
    labels[prior == 1] = 0
    weights = fit_platt(probabilities, labels, prior, CODES)
    assert set(weights) == set(CODES) and weights["A"]["fitted"] == 1.0 and weights["A"]["a"] > 0
    # a target whose labels oppose the probabilities gets the identity, never a reversed ranking
    opposed = labels.copy()
    opposed["A"] = (rng.random(60) < 1 - probabilities["A"].to_numpy()).astype(int)
    opposed[prior == 1] = 0
    assert fit_platt(probabilities, opposed, prior, CODES)["A"] == {
        "a": 1.0,
        "b": 0.0,
        "fitted": 0.0,
    }
    calibrated = apply_platt(weights, probabilities, prior, CODES)
    for code in CODES:
        rows = prior[code].to_numpy() == 0
        order = np.argsort(probabilities[code].to_numpy()[rows])
        assert np.all(np.diff(calibrated[code].to_numpy()[rows][order]) >= 0)
    assignments = pd.DataFrame({"patient_id": prior.index, "fold": np.arange(60) % 5})
    nested = nested_platt(probabilities, labels, prior, CODES, assignments)
    assert (
        nested.shape == probabilities.shape
        and (nested.to_numpy()[prior.to_numpy() == 1] == 0).all()
    )
    # fold 0 must equal weights fitted on folds 1-4 only
    held = assignments.loc[assignments["fold"] == 0, "patient_id"]
    fit = assignments.loc[assignments["fold"] != 0, "patient_id"]
    w0 = fit_platt(probabilities.loc[fit], labels.loc[fit], prior.loc[fit], CODES)
    pd.testing.assert_frame_equal(
        nested.loc[held], apply_platt(w0, probabilities.loc[held], prior.loc[held], CODES)
    )


def test_value_feature_spec_reproduces_the_e11_matrix() -> None:
    fold = fold_data()
    events, cohort = fold.data.inputs.events, fold.data.inputs.cohort
    latest = latest_values(events, cohort)
    values = all_values(events)
    reference_fit = value_features(latest, values, fold.fit, fold.fit, binned=False)
    reference_held = value_features(latest, values, fold.fit, fold.held, binned=False)
    spec = fit_value_spec(latest, fold.fit)
    assert spec["keys"] and set(spec["value_median"]) == set(spec["keys"])
    rebuilt_fit = value_features_from_spec(latest, spec, fold.fit)
    rebuilt_held = value_features_from_spec(latest, spec, fold.held)
    pd.testing.assert_frame_equal(rebuilt_fit, reference_fit)
    pd.testing.assert_frame_equal(rebuilt_held, reference_held)
    # the spec survives JSON, so the package can carry it
    again = json.loads(json.dumps(spec))
    pd.testing.assert_frame_equal(
        value_features_from_spec(latest, again, fold.held), reference_held
    )
    assert len(TARGETS) == 3
