"""D36: simulator-score exclusion and the last_mean readout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ehrforecast.contracts import prediction_cohort
from ehrforecast.finetune import batch_logits, load_parent_body, with_landmarks
from ehrforecast.full_timeline import full_events
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.sequences import (
    LANDMARK_HORIZON_DAYS,
    LANDMARK_SPAN_DAYS,
    Sample,
    SequenceConfig,
    build_histories,
)
from ehrforecast.tokenizer import FoldTokenizer, drop_observations
from ehrforecast.training import (
    CHECKPOINT_VERSION,
    FoldData,
    collate,
    model_config,
    next_token_loss_sum,
    target_loss_weights,
)
from tests.transformer_fixtures import TARGETS, config, fold_data, permitted_events, raw_tables


def _events() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort, tables = raw_tables()
    return prediction_cohort(cohort), permitted_events(cohort, tables)


def test_excluded_observations_leave_fitting_and_encoding_only(tmp_path: Path) -> None:
    prepared, events = _events()
    fit = prepared.index[:16]
    plain = FoldTokenizer.fit(events, fit, min_patients=1)
    clean = FoldTokenizer.fit(events, fit, min_patients=1, exclude_observations=["O1"])
    assert any(t.startswith("obs:O1:") for t in plain.vocabulary)
    assert not any(t.startswith("obs:O1:") for t in clean.vocabulary)
    assert any(t.startswith("obs:O2:") for t in clean.vocabulary)
    kept = drop_observations(events, frozenset({"O1"}))
    assert not kept["code"].astype(str).eq("O1").any()
    assert kept["source"].astype(str).eq("conditions").sum() == (
        events["source"].astype(str).eq("conditions").sum()
    )
    # A saved tokenizer removes the codes from any split it encodes, including unseen patients.
    clean.save(tmp_path / "t.json")
    loaded = FoldTokenizer.load(tmp_path / "t.json")
    assert loaded.excluded_observations == frozenset({"O1"})
    assert loaded.sha256 == clean.sha256
    a, b = loaded.encode(events), loaded.encode(kept)
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))


def test_an_empty_exclusion_keeps_the_earlier_tokenizer_hash() -> None:
    prepared, events = _events()
    fit = prepared.index[:16]
    a = FoldTokenizer.fit(events, fit, min_patients=1)
    b = FoldTokenizer.fit(events, fit, min_patients=1, exclude_observations=())
    assert a.sha256 == b.sha256 and "excluded_observations" not in a.to_dict()


def _model(fold: FoldData, readout: str) -> EHRTransformer:
    torch.manual_seed(0)
    cfg = ModelConfig(fold.tokenizer.size, 1, 16, 2, 2, 0.0, 0.0, 3, readout=readout)
    return EHRTransformer(cfg, torch.from_numpy(fold.tokenizer.input_only_ids())).eval()


def test_last_mean_averages_event_states_only() -> None:
    fold = fold_data()
    seq = SequenceConfig(48, 6)
    samples = [fold.histories.prediction_sample(i, seq) for i in range(4)]
    b = collate(samples, torch.device("cpu"))
    model = _model(fold, "last_mean")
    states = model.hidden(b["tokens"], b["ages"])
    logits = model.classify(states, b["readout"], b["head"], b["event_mask"])
    for row, s in enumerate(samples):
        mean = states[row, s.n_prefix : len(s.tokens)].mean(dim=0)
        joined = torch.cat([states[row, s.readout], mean, b["head"][row]])
        assert torch.allclose(model.classifier(joined), logits[row], atol=1e-5)
    with pytest.raises(ValueError, match="event mask"):
        model.classify(states, b["readout"], b["head"])


def test_last_mean_is_independent_of_padding() -> None:
    fold = fold_data()
    seq = SequenceConfig(48, 6)
    samples = [fold.histories.prediction_sample(i, seq) for i in range(len(fold.histories))]
    samples = sorted(samples, key=lambda s: len(s.tokens))
    short, long = samples[0], samples[-1]
    assert len(short.tokens) < len(long.tokens)
    model = _model(fold, "last_mean")
    alone = collate([short], torch.device("cpu"))
    pair = collate([short, long], torch.device("cpu"))

    def run(b: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(b["tokens"], b["ages"], b["readout"], b["head"], None, b["event_mask"])

    assert torch.allclose(run(alone)[0], run(pair)[0], atol=1e-5)


def test_the_default_readout_and_head_size_are_unchanged() -> None:
    fold = fold_data()
    base = model_config(config(), fold.tokenizer, 3)
    assert base.readout == "last"
    head = EHRTransformer(base).classifier
    assert head.in_features == base.d_model + 4
    pooled = model_config(config(model={"readout": "last_mean"}), fold.tokenizer, 3)
    assert EHRTransformer(pooled).classifier.in_features == 2 * base.d_model + 4


def test_excluded_codes_do_not_reach_a_patient_history() -> None:
    prepared, events = _events()
    tok = FoldTokenizer.fit(
        events, prepared.index[:16], min_patients=1, exclude_observations=["O1"]
    )
    hist = build_histories(tok.encode(events), prepared, tok, TARGETS)
    ids = {t for t, name in enumerate(tok.vocabulary) if name.startswith("obs:O1:")}
    assert not ids.intersection(np.unique(hist.tokens).tolist())


def _landmark_case() -> tuple[FoldData, int, int, dict[int, int]]:
    fold = fold_data(n_patients=24)
    hist = fold.histories
    cohort = fold.data.inputs.cohort
    days = (cohort["anchor_date"] - cohort["birthdate"]).dt.days
    columns = {
        fold.tokenizer.token_id(f"hx:{c}"): j
        for j, c in enumerate(TARGETS)
        if f"hx:{c}" in fold.tokenizer.vocabulary
    }
    for i in range(len(hist)):
        anchor = int(days[str(hist.patients[i])])
        allowed = hist.landmark_range(i, anchor)
        if allowed is not None and allowed[1] - allowed[0] > 200:
            return fold, i, anchor, columns
    raise AssertionError("no patient has a usable landmark range")


def test_a_landmark_sees_only_events_before_its_cut_day() -> None:
    fold, i, anchor, _ = _landmark_case()
    hist, seq = fold.histories, SequenceConfig(48, 6)
    low, high = hist.landmark_range(i, anchor) or (0, 0)
    assert high + LANDMARK_HORIZON_DAYS < anchor  # the whole window ends before the real anchor
    base = int(hist.token_offsets[i])
    event_days = hist.token_days[base : base + hist.n_tokens(i)]
    for cut in (low, (low + high) // 2, high):
        sample = hist.landmark_sample(i, cut, seq)
        n_events = len(sample.tokens) - sample.n_prefix
        visible = int(np.searchsorted(event_days, cut, side="left"))
        kept = hist.tokens[base + visible - n_events : base + visible]
        assert np.array_equal(sample.tokens[sample.n_prefix :], kept)
        assert len(sample.tokens) <= seq.context_length
        assert sample.ages[sample.readout] * 365.25 < cut
        assert sample.head[0] == pytest.approx(cut / 365.25 / 100, abs=1e-5)
        assert sample.head[2] > 0  # recency is measured from the cut day
    flagged = SequenceConfig(48, 6, landmark_flag=True)
    assert flagged.n_head_features == 5
    assert hist.landmark_sample(i, high, flagged).head[-1] == 1.0  # an extra anchor
    assert hist.prediction_sample(i, flagged).head[-1] == 0.0  # a real anchor
    assert len(hist.prediction_sample(i, seq).head) == 4  # the default is unchanged
    assert high - low <= LANDMARK_SPAN_DAYS  # cut days stay near the real anchor


def test_later_events_cannot_change_a_landmark_input() -> None:
    fold, i, anchor, _ = _landmark_case()
    hist, seq = fold.histories, SequenceConfig(48, 6)
    low, high = hist.landmark_range(i, anchor) or (0, 0)
    cut = (low + high) // 2
    before = hist.landmark_sample(i, cut, seq)
    base = int(hist.token_offsets[i])
    event_days = hist.token_days[base : base + hist.n_tokens(i)]
    later = base + int(np.searchsorted(event_days, cut, side="left"))
    saved = hist.tokens[later:].copy()
    hist.tokens[later : base + hist.n_tokens(i)] = 0
    try:
        after = hist.landmark_sample(i, cut, seq)
    finally:
        hist.tokens[later:] = saved
    assert np.array_equal(before.tokens, after.tokens) and np.array_equal(before.ages, after.ages)


def test_landmark_labels_follow_first_diagnosis_days() -> None:
    fold, i, _, columns = _landmark_case()
    hist = fold.histories
    token = next(iter(columns))
    tokens, days, target = hist._hx(i)
    hist_copy = {
        "hx_offsets": hist.hx_offsets,
        "hx_tokens": hist.hx_tokens,
        "hx_days": hist.hx_days,
        "hx_target": hist.hx_target,
    }
    a = int(hist.hx_offsets[i])
    hist.hx_tokens = np.insert(hist.hx_tokens, a, token)
    hist.hx_days = np.insert(hist.hx_days, a, 5000)
    hist.hx_target = np.insert(hist.hx_target, a, True)
    hist.hx_offsets = hist.hx_offsets + (np.arange(len(hist.hx_offsets)) > i)
    try:
        col = columns[token]
        prior = hist.landmark_targets(i, 5001, {token: col}, 3)
        start = hist.landmark_targets(i, 5000, {token: col}, 3)
        inside = hist.landmark_targets(i, 5000 - LANDMARK_HORIZON_DAYS, {token: col}, 3)
        outside = hist.landmark_targets(i, 4999 - LANDMARK_HORIZON_DAYS, {token: col}, 3)
    finally:
        for name, value in hist_copy.items():
            setattr(hist, name, value)
    assert (prior[0][col], prior[1][col]) == (0.0, 0.0)  # already diagnosed: masked
    assert (start[0][col], start[1][col]) == (1.0, 1.0)  # on the cut day: inside the window
    assert (inside[0][col], inside[1][col]) == (1.0, 1.0)  # on the last window day
    assert (outside[0][col], outside[1][col]) == (0.0, 1.0)  # after the window: a known 0
    other = next(j for j in range(3) if j != col)
    assert prior[1][other] == 0.0  # a target outside the vocabulary map stays masked
    assert len(tokens) == len(days) == len(target)


def test_extra_anchors_use_fit_patients_only_and_repeat_for_a_seed() -> None:
    fold = fold_data(n_patients=24)
    seq = SequenceConfig(48, 6)
    base = [fold.histories.prediction_sample(int(r), seq) for r in fold.fit_rows]
    y = fold.data.labels.loc[fold.fit, TARGETS].to_numpy("float32")
    mask = 1.0 - fold.data.inputs.prior.loc[fold.fit, TARGETS].to_numpy("float32")
    calls: list[int] = []
    real = fold.histories.landmark_sample

    def spy(i: int, cut: int, config: SequenceConfig) -> Sample:
        calls.append(i)
        return real(i, cut, config)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fold.histories, "landmark_sample", spy)
        a = with_landmarks(fold, seq, base, y, mask, 1, 7, 3)
    assert calls and set(calls) <= {int(r) for r in fold.fit_rows}
    assert not set(calls) & {int(r) for r in fold.held_rows}
    b = with_landmarks(fold, seq, base, y, mask, 1, 7, 3)
    c = with_landmarks(fold, seq, base, y, mask, 1, 7, 4)
    assert len(a[0]) == len(base) + len(calls) == a[1].shape[0] == a[2].shape[0]
    assert a[0][: len(base)] == base  # the real anchors stay first and unchanged
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])  # same seed and epoch
    tokens_a = [x.tokens.tolist() for x in a[0][len(base) :]]
    tokens_b = [x.tokens.tolist() for x in b[0][len(base) :]]
    tokens_c = [x.tokens.tolist() for x in c[0][len(base) :]]
    assert tokens_a == tokens_b and tokens_a != tokens_c


def test_full_timeline_events_hold_only_the_given_patients_and_pass_the_anchor() -> None:
    cohort, tables = raw_tables()
    prepared = prediction_cohort(cohort)
    fit = prepared.index[:16]
    uncut = full_events(tables, fit)
    assert set(uncut["patient_id"]) <= set(fit)
    anchors = uncut["patient_id"].map(prepared["anchor_date"])
    assert (uncut["time"] >= anchors).any()  # later events are present for fit patients
    cut = permitted_events(cohort, tables)
    assert (cut["time"] < cut["patient_id"].map(prepared["anchor_date"])).all()  # inputs stay cut
    encounters = uncut["source"].astype(str).eq("encounters")
    assert uncut.loc[encounters, "value_num"].isna().all()  # no encounter durations


def test_full_timeline_pretraining_never_reads_held_out_patients() -> None:
    cohort, tables = raw_tables()
    prepared = prediction_cohort(cohort)
    fit, held = prepared.index[:16], prepared.index[16:]
    events = permitted_events(cohort, tables)
    tok = FoldTokenizer.fit(events, fit, min_patients=1)
    uncut = full_events(tables, fit)
    hist = build_histories(tok.encode(uncut), prepared, tok, TARGETS)
    rows = hist.index_of(held)
    assert all(hist.n_tokens(int(r)) == 0 for r in rows)
    base = build_histories(tok.encode(events), prepared, tok, TARGETS)
    longer = [hist.n_tokens(int(r)) > base.n_tokens(int(r)) for r in hist.index_of(fit)]
    assert any(longer)
    # A saved tokenizer fitted before the anchor stays the only vocabulary source.
    assert FoldTokenizer.fit(events, fit, min_patients=1).sha256 == tok.sha256


def test_counts_cover_the_visible_history_and_nothing_later() -> None:
    fold, i, anchor, _ = _landmark_case()
    hist = fold.histories
    full = SequenceConfig(48, 6, count_branch=True)
    base = int(hist.token_offsets[i])
    tokens = hist.tokens[base : base + hist.n_tokens(i)]
    sample = hist.prediction_sample(i, full)
    assert sample.count_ids is not None and sample.count_weights is not None
    ids, counts = np.unique(tokens, return_counts=True)
    assert np.array_equal(sample.count_ids, ids)
    assert np.allclose(sample.count_weights, np.log1p(counts))
    low, high = hist.landmark_range(i, anchor) or (0, 0)
    cut = (low + high) // 2
    days = hist.token_days[base : base + hist.n_tokens(i)]
    visible = tokens[: int(np.searchsorted(days, cut, side="left"))]
    early = hist.landmark_sample(i, cut, full)
    assert early.count_ids is not None
    assert set(early.count_ids.tolist()) == set(visible.tolist())
    assert hist.prediction_sample(i, SequenceConfig(48, 6)).count_ids is None


def test_a_short_context_still_counts_events_older_than_the_window() -> None:
    fold = fold_data()
    hist = fold.histories
    short = SequenceConfig(16, 2, count_branch=True)
    long_rows = [i for i in range(len(hist)) if hist.n_tokens(i) > 20]
    assert long_rows
    sample = hist.prediction_sample(long_rows[0], short)
    assert sample.count_weights is not None
    assert len(sample.tokens) <= 16
    base = int(hist.token_offsets[long_rows[0]])
    all_tokens = hist.tokens[base : base + hist.n_tokens(long_rows[0])]
    assert np.exp(sample.count_weights).sum() - len(sample.count_weights) == pytest.approx(
        len(all_tokens), abs=1e-3
    )


def test_the_count_branch_trains_from_a_parent_and_ignores_padding() -> None:
    fold = fold_data()
    seq = SequenceConfig(48, 6, count_branch=True)
    torch.manual_seed(0)
    make = lambda on: EHRTransformer(  # noqa: E731
        ModelConfig(fold.tokenizer.size, 1, 16, 2, 2, 0.0, 0.0, 3, count_branch=on),
        torch.from_numpy(fold.tokenizer.input_only_ids()),
    ).eval()
    model = make(True)
    assert model.count_mlp is not None
    assert all(any(p is q for q in model.head_parameters()) for p in model.count_mlp.parameters())
    samples = sorted(
        [fold.histories.prediction_sample(i, seq) for i in range(8)], key=lambda s: len(s.tokens)
    )
    alone = collate(samples[:1], torch.device("cpu"))
    pair = collate([samples[0], samples[-1]], torch.device("cpu"))
    assert torch.allclose(batch_logits(model, alone)[0], batch_logits(model, pair)[0], atol=1e-5)
    plain = collate(samples[:1], torch.device("cpu"))
    plain.pop("count_ids")
    with pytest.raises(ValueError, match="count inputs"):
        batch_logits(model, plain)


def test_parent_loading_allows_only_head_keys_to_be_missing(tmp_path: Path) -> None:
    fold = fold_data()
    body = EHRTransformer(ModelConfig(fold.tokenizer.size, 1, 16, 2, 2, 0.0, 0.0, 3))
    state = {
        "version": CHECKPOINT_VERSION,
        "complete": True,
        "model": body.state_dict(),
        "provenance": {**fold.provenance, "stage": "pretrain"},
    }
    path = tmp_path / "parent.pt"
    torch.save(state, path)
    child = EHRTransformer(
        ModelConfig(fold.tokenizer.size, 1, 16, 2, 2, 0.0, 0.0, 3, count_branch=True)
    )
    load_parent_body(child, path, fold)
    assert torch.equal(child.token_embedding.weight, body.token_embedding.weight)
    smaller = EHRTransformer(ModelConfig(fold.tokenizer.size, 2, 16, 2, 2, 0.0, 0.0, 3))
    with pytest.raises(ValueError, match="does not match"):
        load_parent_body(smaller, path, fold)


def test_target_weights_favour_rare_targets_with_mean_one() -> None:
    labels = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype="float32")
    unmasked = np.ones_like(labels)
    weights = target_loss_weights(labels, unmasked)
    assert weights[1] > weights[0]
    assert weights[2] == weights.max()  # no positive: the largest weight in use
    assert weights.mean() == pytest.approx(1.0)


def test_token_weights_scale_only_the_named_targets() -> None:
    fold = fold_data()
    seq = SequenceConfig(48, 6)
    samples = [fold.histories.prediction_sample(i, seq) for i in range(6)]
    batch = collate(samples, torch.device("cpu"))
    model = _model(fold, "last")
    plain, count = next_token_loss_sum(model, batch)
    ones = torch.ones(fold.tokenizer.size)
    same, _ = next_token_loss_sum(model, batch, ones)
    assert torch.allclose(plain, same)
    target = batch["targets"][batch["targets"] >= 0][0]
    weights = ones.clone()
    weights[target] = 3.0
    raised, count2 = next_token_loss_sum(model, batch, weights)
    assert raised > plain and count == count2
