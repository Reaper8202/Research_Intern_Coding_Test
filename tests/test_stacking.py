"""Stacking: nested out-of-fold weights, mask preserved, parameter-free stackers exact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ehrforecast.stacking import logit, nested_stack, sigmoid, stack_target

CODES = ["A", "B"]


def _data(n: int = 60, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    index = pd.Index([f"p{i:03d}" for i in range(n)], name="patient_id")
    labels = pd.DataFrame(rng.integers(0, 2, (n, 2)), index=index, columns=CODES)
    prior = pd.DataFrame(0, index=index, columns=CODES)
    prior.iloc[:5, 0] = 1
    labels[prior == 1] = 0
    bases = {
        "m1": pd.DataFrame(rng.random((n, 2)), index=index, columns=CODES),
        "m2": pd.DataFrame(rng.random((n, 2)), index=index, columns=CODES),
    }
    assignments = pd.DataFrame({"patient_id": index, "fold": np.arange(n) % 5})
    return bases, labels, prior, assignments


def test_parameter_free_stackers_are_exact_and_keep_the_mask() -> None:
    bases, labels, prior, assignments = _data()
    probs, weights = nested_stack(bases, labels, prior, CODES, assignments, "prob_mean")
    expected = (bases["m1"] + bases["m2"]) / 2
    expected[prior == 1] = 0.0
    pd.testing.assert_frame_equal(probs, expected)
    assert weights == {}
    probs, _ = nested_stack(bases, labels, prior, CODES, assignments, "logit_mean")
    z = (logit(bases["m1"].to_numpy()) + logit(bases["m2"].to_numpy())) / 2
    expected = pd.DataFrame(sigmoid(z), index=labels.index, columns=CODES)
    expected[prior == 1] = 0.0
    pd.testing.assert_frame_equal(probs, expected)


def test_logistic_weights_are_fitted_without_the_held_out_fold() -> None:
    bases, labels, prior, assignments = _data()
    probs, weights = nested_stack(bases, labels, prior, CODES, assignments, "logistic")
    assert set(weights) == set(CODES) and set(weights["A"]) == {f"fold{k}" for k in range(5)}
    assert (probs.to_numpy()[prior.to_numpy() == 1] == 0).all()
    # recompute fold 0, target B by hand: fit on folds 1-4, apply to fold 0
    held = assignments.loc[assignments["fold"] == 0, "patient_id"]
    fit = assignments.loc[assignments["fold"] != 0, "patient_id"]
    pos = pd.Series(np.arange(len(labels)), index=labels.index)
    base = [bases["m1"]["B"].to_numpy(), bases["m2"]["B"].to_numpy()]
    manual = stack_target(
        base, labels.loc[fit, "B"].to_numpy(), pos[fit].to_numpy(), pos[held].to_numpy(), "logistic"
    )
    np.testing.assert_allclose(probs.loc[held, "B"].to_numpy(), manual, rtol=1e-9)
    # a stronger base model gets the larger weight
    strong = {"good": labels.astype(float) * 0.6 + 0.2, "noise": bases["m2"]}
    _, w = nested_stack(strong, labels, prior, CODES, assignments, "logistic")
    assert all(f["good"] > f["noise"] for f in w["B"].values())


def test_single_class_target_falls_back_to_logit_mean() -> None:
    bases, labels, prior, assignments = _data()
    labels["A"] = 0
    probs, weights = nested_stack(bases, labels, prior, CODES, assignments, "logistic")
    assert "A" not in weights and "B" in weights
    z = (logit(bases["m1"]["A"].to_numpy()) + logit(bases["m2"]["A"].to_numpy())) / 2
    unmasked = prior["A"].to_numpy() == 0
    np.testing.assert_allclose(probs["A"].to_numpy()[unmasked], sigmoid(z)[unmasked])
    with pytest.raises(ValueError):
        nested_stack(bases, labels, prior, CODES, assignments, "unknown")
