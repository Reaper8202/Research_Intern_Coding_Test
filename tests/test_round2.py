"""D36: simulator-score exclusion and the last_mean readout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ehrforecast.contracts import prediction_cohort
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.sequences import SequenceConfig, build_histories
from ehrforecast.tokenizer import FoldTokenizer, drop_observations
from ehrforecast.training import FoldData, collate, model_config
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
