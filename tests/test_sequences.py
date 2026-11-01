from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ehrforecast.contracts import prediction_cohort
from ehrforecast.sequences import (
    DAYS_PER_YEAR,
    SequenceConfig,
    build_histories,
    load_histories,
    save_histories,
)
from ehrforecast.tokenizer import FoldTokenizer
from tests.transformer_fixtures import TARGETS, permitted_events, raw_tables


def _histories(n: int = 12, seed: int = 0, shift_days: int = 0):  # type: ignore[no-untyped-def]
    cohort, tables = raw_tables(n, seed)
    if shift_days:
        delta = pd.Timedelta(days=shift_days)
        cohort = cohort.assign(
            birthdate=cohort["birthdate"] + delta, anchor_date=cohort["anchor_date"] + delta
        )
        for table in tables.values():
            for column in ("START", "STOP", "DATE"):
                if column in table:
                    table[column] = table[column] + delta
    events = permitted_events(cohort, tables)
    tokenizer = FoldTokenizer.fit(events, cohort.index[: n - 3], min_patients=1)
    return build_histories(
        tokenizer.encode(events), prediction_cohort(cohort), tokenizer, TARGETS
    ), tokenizer


@pytest.mark.parametrize("shift", [-3652, 3652])
def test_common_date_shift_preserves_tokens_ages_and_head_inputs(shift: int) -> None:
    base, _ = _histories()
    shifted, _ = _histories(shift_days=shift)
    config = SequenceConfig(context_length=40, max_summary=4)
    for i in range(len(base)):
        a, b = base.prediction_sample(i, config), shifted.prediction_sample(i, config)
        np.testing.assert_array_equal(a.tokens, b.tokens)
        np.testing.assert_array_equal(a.ages, b.ages)
        np.testing.assert_array_equal(a.head, b.head)


def test_prediction_sequence_fits_and_keeps_the_most_recent_complete_days() -> None:
    histories, tokenizer = _histories(n=16, seed=5)
    config = SequenceConfig(context_length=24, max_summary=4)
    sep = tokenizer.token_id("[SEP]")
    for i in range(len(histories)):
        sample = histories.prediction_sample(i, config)
        assert len(sample.tokens) <= config.context_length
        assert sample.tokens[sample.n_prefix - 1] == sep
        assert sample.readout == len(sample.tokens) - 1
        n = histories.n_tokens(i)
        a = histories.token_offsets[i]
        # Retained events are exactly the last tokens of the history.
        np.testing.assert_array_equal(
            sample.tokens[sample.n_prefix :], histories.tokens[a + n - sample.n_events : a + n]
        )
        if not sample.partial_day:
            starts, _ = histories._days(i)
            assert n - sample.n_events in set(starts.tolist())  # starts at a day boundary


def test_summary_uses_only_conditions_first_diagnosed_before_the_window() -> None:
    histories, tokenizer = _histories(n=20, seed=8)
    config = SequenceConfig(context_length=16, max_summary=3)
    checked = 0
    for i in range(len(histories)):
        sample = histories.prediction_sample(i, config)
        if sample.n_events == histories.n_tokens(i):
            assert sample.n_summary == 0
            continue
        hx_tokens, hx_days, hx_target = histories._hx(i)
        _, day_values = histories._days(i)
        window_day = day_values[sample.first_day]
        summary = sample.tokens[1 : sample.n_prefix - 1]
        first_day = dict(zip(hx_tokens.tolist(), hx_days.tolist(), strict=True))
        candidates = {int(t) for t, d in zip(hx_tokens, hx_days, strict=True) if d < window_day}
        assert set(summary.tolist()) <= candidates
        assert len(summary) == min(config.max_summary, len(candidates))
        targets = {
            int(t)
            for t, d, g in zip(hx_tokens, hx_days, hx_target, strict=True)
            if g and d < window_day
        }
        assert targets <= set(summary.tolist()) or len(summary) == config.max_summary
        np.testing.assert_allclose(
            sample.ages[1 : sample.n_prefix - 1],
            [first_day[int(t)] / DAYS_PER_YEAR for t in summary],
            rtol=1e-6,
        )
        assert all(tokenizer.vocabulary[t].startswith("hx:") for t in summary)
        checked += 1
    assert checked


def test_oversized_day_and_empty_history_still_give_valid_samples() -> None:
    histories, _ = _histories(n=6, seed=1)
    tiny = SequenceConfig(context_length=4, max_summary=0)
    for j in range(len(histories)):
        sample = histories.prediction_sample(j, tiny)
        assert len(sample.tokens) <= 4 and sample.n_events >= 1
        starts, _ = histories._days(j)
        kept_from = histories.n_tokens(j) - sample.n_events
        assert sample.partial_day == (kept_from not in set(starts.tolist()))
    assert any(histories.prediction_sample(j, tiny).partial_day for j in range(len(histories)))
    # A patient without events: zero-state readout, missing-history flag, no pretraining window.
    cohort, tables = raw_tables(2, seed=0)
    tables = {k: v[v["PATIENT"].ne("p01")] for k, v in tables.items()}
    events = permitted_events(cohort, tables)
    tok = FoldTokenizer.fit(events, cohort.index[:1], min_patients=1)
    h = build_histories(tok.encode(events), prediction_cohort(cohort), tok, TARGETS)
    sample = h.prediction_sample(1, SequenceConfig(32, 4))
    assert sample.readout == -1 and sample.head[3] == 1.0 and sample.head[2] == 0.0
    assert h.window_starts(1, SequenceConfig(32, 4)) == 0


def test_windows_recompute_the_summary_and_seeded_windows_repeat() -> None:
    histories, _ = _histories(n=16, seed=5)
    config = SequenceConfig(context_length=20, max_summary=3)
    for i in range(len(histories)):
        for first in range(histories.window_starts(i, config)):
            sample = histories.window_sample(i, first, config)
            assert len(sample.tokens) <= config.context_length
            _, day_values = histories._days(i)
            hx_tokens, hx_days, _ = histories._hx(i)
            allowed = {
                int(t) for t, d in zip(hx_tokens, hx_days, strict=True) if d < day_values[first]
            }
            assert set(sample.tokens[1 : sample.n_prefix - 1].tolist()) <= allowed

    def draw(seed: int) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        return [
            histories.window_sample(
                i, int(rng.integers(histories.window_starts(i, config))), config
            ).tokens.tolist()
            for i in range(len(histories))
        ]

    assert draw(3) == draw(3)


def test_histories_round_trip_and_reject_stale_provenance(tmp_path: Path) -> None:
    histories, _ = _histories()
    path = tmp_path / "histories.npz"
    save_histories(histories, path, {"tokenizer_sha256": "abc"})
    loaded = load_histories(path, {"tokenizer_sha256": "abc"})
    config = SequenceConfig(40, 4)
    for i in range(len(histories)):
        np.testing.assert_array_equal(
            loaded.prediction_sample(i, config).tokens,
            histories.prediction_sample(i, config).tokens,
        )
    with pytest.raises(ValueError, match="stale"):
        load_histories(path, {"tokenizer_sha256": "other"})


def test_all_unknown_history_gives_a_valid_finite_prediction() -> None:
    import torch

    from ehrforecast.finetune import predict_samples
    from ehrforecast.model import EHRTransformer, ModelConfig

    cohort, tables = raw_tables(4, seed=6)
    events = permitted_events(cohort, tables)
    tokenizer = FoldTokenizer.fit(events, cohort.index[:2], min_patients=99)  # nothing is eligible
    h = build_histories(tokenizer.encode(events), prediction_cohort(cohort), tokenizer, TARGETS)
    sample = h.prediction_sample(3, SequenceConfig(32, 4))
    assert all(tokenizer.vocabulary[t].startswith("[") for t in sample.tokens)
    model = EHRTransformer(
        ModelConfig(vocab_size=tokenizer.size, d_model=16, n_heads=2, n_targets=3)
    )
    p = predict_samples(model, [sample], torch.device("cpu"), 1)
    assert p.shape == (1, 3) and np.isfinite(p).all()


def test_age_sex_head_features_match_the_baseline_age_bands() -> None:
    import numpy as np

    from ehrforecast.features import FeatureConfig, demographic_features
    from ehrforecast.sequences import N_AGE_SEX_FEATURES, N_HEAD_FEATURES, SequenceConfig
    from ehrforecast.training import model_config
    from tests.transformer_fixtures import config as fixture_config
    from tests.transformer_fixtures import fold_data

    fold = fold_data()
    histories = fold.histories
    plain = SequenceConfig(48, 6)
    extended = SequenceConfig(48, 6, age_sex_head=True)
    assert extended.n_head_features == N_HEAD_FEATURES + N_AGE_SEX_FEATURES == 16
    cohort = fold.data.inputs.cohort.assign(age_at_anchor=histories.age_at_anchor)
    bands = demographic_features(cohort, FeatureConfig()).filter(like="age_band").to_numpy()
    for i in range(len(histories)):
        a = histories.prediction_sample(i, plain).head
        b = histories.prediction_sample(i, extended).head
        assert len(a) == 4 and np.array_equal(a, b[:4])
        assert b[4:10].sum() == 1 and np.array_equal(b[4:10], bands[i])
        assert np.array_equal(b[10:], b[4:10] * histories.male[i])
    on = fixture_config(model={"age_sex_head": True})
    assert model_config(on, fold.tokenizer, 3).n_head_features == 16
    assert model_config(fixture_config(), fold.tokenizer, 3).n_head_features == 4
