from __future__ import annotations

import numpy as np
import torch

from ehrforecast.finetune import predict_samples
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.sequences import Sample
from ehrforecast.tokenizer import age_features
from ehrforecast.training import IGNORE, collate, next_token_loss_sum, supervised_loss_sum


def _model(
    dropout: float = 0.0, vocab: int = 20, input_only: list[int] | None = None
) -> EHRTransformer:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=vocab,
        n_layers=2,
        d_model=16,
        n_heads=2,
        dropout=dropout,
        head_dropout=dropout,
        n_targets=3,
    )
    return EHRTransformer(config, torch.tensor(input_only or [0, 1], dtype=torch.int64))


def _sample(tokens: list[int], n_prefix: int = 2, readout: int | None = None) -> Sample:
    t = np.array(tokens, dtype="int64")
    return Sample(
        tokens=t,
        ages=np.linspace(30, 40, len(t)).astype("float32"),
        sources=np.r_[np.full(n_prefix, -1), np.zeros(len(t) - n_prefix)].astype("int64"),
        n_prefix=n_prefix,
        readout=len(t) - 1 if readout is None else readout,
        head=np.array([0.4, 1.0, 0.2, 0.0], dtype="float32"),
    )


def test_later_tokens_do_not_change_earlier_outputs_in_eval_mode() -> None:
    model = _model(dropout=0.1).eval()
    tokens = torch.randint(2, 20, (1, 12))
    ages = torch.from_numpy(age_features(np.linspace(20, 30, 12)[None]))
    changed = tokens.clone()
    changed[0, 8:] = torch.randint(2, 20, (4,))
    with torch.no_grad():
        a, b = model.hidden(tokens, ages), model.hidden(changed, ages)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_padding_and_batch_composition_do_not_change_predictions() -> None:
    model = _model(dropout=0.1)
    samples = [_sample([2, 1, 5, 6, 7]), _sample([3, 1, 9]), _sample([2, 1, *range(4, 18)])]
    alone = np.vstack([predict_samples(model, [s], torch.device("cpu"), 1) for s in samples])
    together = predict_samples(model, samples, torch.device("cpu"), 3)
    np.testing.assert_allclose(alone, together, rtol=1e-5, atol=1e-6)


def test_empty_history_uses_a_zero_state_and_structure_is_never_a_target() -> None:
    model = _model().eval()
    empty_a, empty_b = _sample([2, 1], readout=-1), _sample([3, 1], readout=-1)
    p = predict_samples(model, [empty_a, empty_b], torch.device("cpu"), 2)
    np.testing.assert_allclose(p[0], p[1])  # only the head features enter
    batch = collate([_sample([2, 4, 1, 5, 6], n_prefix=3)], torch.device("cpu"))
    # Prefix [SEX, hx, SEP]: position 2 (SEP) predicts 5, position 3 predicts 6.
    assert batch["targets"].tolist() == [[IGNORE, IGNORE, 5, 6, IGNORE]]
    logits = model.next_token_logits(torch.randn(3, 16))
    assert torch.isinf(logits[:, [0, 1]]).all() and torch.isfinite(logits[:, 2:]).all()


def test_masked_cells_have_zero_gradient() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    loss, count = supervised_loss_sum(logits, labels, mask)
    (loss / count).backward()
    assert count == 4
    assert logits.grad is not None
    assert (logits.grad[mask == 0] == 0).all() and (logits.grad[mask == 1] != 0).all()


def test_model_fits_eight_distinct_samples() -> None:
    model = _model()
    rng = np.random.default_rng(0)
    samples = [_sample([2 + i % 2, 1, *rng.integers(2, 20, 6).tolist()]) for i in range(8)]
    labels = torch.from_numpy((rng.random((8, 3)) < 0.5).astype("float32"))
    batch = collate(samples, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(300):
        optimizer.zero_grad()
        logits = model(batch["tokens"], batch["ages"], batch["readout"], batch["head"])
        loss, count = supervised_loss_sum(logits, labels, torch.ones_like(labels))
        (loss / count).backward()
        optimizer.step()
    assert float(loss.detach() / count) < 0.05


def test_language_model_learns_an_unambiguous_sequence() -> None:
    model = _model()
    samples = [_sample([2, 1, *range(3, 15)]) for _ in range(4)]
    batch = collate(samples, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad()
        loss, count = next_token_loss_sum(model, batch)
        (loss / count).backward()
        optimizer.step()
    assert float(loss.detach() / count) < 0.05
