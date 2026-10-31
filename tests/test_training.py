from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ehrforecast.finetune import run_finetuning
from ehrforecast.pretrain import run_pretraining
from ehrforecast.sequences import Sample
from ehrforecast.training import (
    EarlyStopping,
    accumulate_step,
    collate,
    load_checkpoint,
    next_token_loss_sum,
    save_checkpoint,
    supervised_loss_sum,
    warmup_cosine,
)
from tests.test_model import _model, _sample
from tests.transformer_fixtures import config, fold_data

CPU = torch.device("cpu")


def _grads(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()
    ]


def test_unequal_microbatches_match_the_combined_batch_for_both_objectives() -> None:
    samples: list[Sample] = [
        _sample([2, 1, 4, 5]),
        _sample([3, 1, *range(4, 12)]),
        _sample([2, 1, 7]),
    ]
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    def supervised(idx: list[int]):  # type: ignore[no-untyped-def]
        def fn() -> tuple[torch.Tensor, int]:
            b = collate([samples[i] for i in idx], CPU)
            return supervised_loss_sum(
                model(b["tokens"], b["ages"], b["readout"], b["head"]), labels[idx], mask[idx]
            )

        return fn, int(mask[idx].sum())

    def language(idx: list[int]):  # type: ignore[no-untyped-def]
        chosen = [samples[i] for i in idx]
        return (lambda: next_token_loss_sum(model, collate(chosen, CPU))), sum(
            s.n_events for s in chosen
        )

    for objective in (supervised, language):
        results = []
        for split in ([[0, 1, 2]], [[0], [1, 2]], [[0, 1], [2]]):
            model = _model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            loss, count, _ = accumulate_step(
                [objective(s) for s in split], model, optimizer, None, 1e9
            )
            results.append((loss / count, _grads(model)))
        for loss, grads in results[1:]:
            assert loss == pytest.approx(results[0][0], rel=1e-6)
            for a, b in zip(grads, results[0][1], strict=True):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_batch_without_eligible_targets_makes_no_update() -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    before = [p.detach().clone() for p in model.parameters()]
    loss, count, norm = accumulate_step(
        [(lambda: (torch.tensor(0.0), 0), 0)], model, optimizer, scheduler, 1.0
    )
    assert (loss, count, norm) == (0.0, 0, None)
    assert scheduler.last_epoch == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_schedule_and_early_stopping_rules() -> None:
    factor = warmup_cosine(100, 0.1, 0.1)
    assert factor(0) == pytest.approx(0.1) and factor(9) == pytest.approx(1.0)
    assert factor(100) == pytest.approx(0.1) and factor(55) == pytest.approx(0.55)
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(0, 0.7, 0.2)
    assert stopper.update(1, 0.7, 0.3)  # exact AUROC tie: higher mAP wins
    assert not stopper.update(2, 0.7, 0.3)  # full tie: the earlier epoch stays
    assert stopper.best_epoch == 1 and not stopper.stop
    assert not stopper.update(3, 0.6)
    assert stopper.stop


def test_checkpoints_reject_other_runs_and_corrupt_files(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "a.pt", {"provenance": {"fold": 0, "config_sha256": "x"}})
    assert load_checkpoint(tmp_path / "a.pt", {"fold": 0})["provenance"]["config_sha256"] == "x"
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(tmp_path / "a.pt", {"fold": 1})
    (tmp_path / "b.pt").write_bytes((tmp_path / "a.pt").read_bytes()[:50])
    with pytest.raises(ValueError, match="cannot be read"):
        load_checkpoint(tmp_path / "b.pt", {})


def _state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _metrics(run_dir: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_pretraining_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    fold = fold_data()
    full = run_pretraining(config(), fold, tmp_path / "full", device=CPU)
    assert full["status"] == "completed" and full["epochs_completed"] == 3
    # Interrupt during the second epoch: same configuration, stopped by a simulated crash.
    short = config()
    first = tmp_path / "split"
    _run_epochs(short, fold, first, epochs=1)
    with pytest.raises(ValueError, match="pass --resume"):
        run_pretraining(short, fold, first, device=CPU)
    run_pretraining(short, fold, first, resume=True, device=CPU)
    for key, value in _state(tmp_path / "full" / "last.pt").items():
        torch.testing.assert_close(value, _state(first / "last.pt")[key], rtol=1e-5, atol=1e-6)
    # The crashed attempt logged some steps twice; the resumed values replace them.
    lr_full = {m["step"]: m["lr"] for m in _metrics(tmp_path / "full") if "lr" in m}
    lr_split = {m["step"]: m["lr"] for m in _metrics(first) if "lr" in m}
    assert lr_full == pytest.approx(lr_split)
    with pytest.raises(ValueError, match="config_sha256"):
        run_pretraining(config(pretrain={"lr": 0.01}), fold, first, resume=True, device=CPU)


def _run_epochs(cfg: dict, fold, run_dir: Path, epochs: int) -> None:  # type: ignore[no-untyped-def]
    """Run ``epochs`` epochs of a longer configuration, then stop as a crash would."""
    import ehrforecast.pretrain as pretrain

    original = pretrain.evaluate_windows
    calls = {"n": 0}

    def crash_after(*args, **kwargs):  # type: ignore[no-untyped-def]
        if calls["n"] == epochs:
            raise KeyboardInterrupt
        calls["n"] += 1
        return original(*args, **kwargs)

    pretrain.evaluate_windows = crash_after
    try:
        with pytest.raises(KeyboardInterrupt):
            pretrain.run_pretraining(cfg, fold, run_dir, device=CPU)
    finally:
        pretrain.evaluate_windows = original


def test_supervised_methods_run_and_the_fixed_body_stays_unchanged(tmp_path: Path) -> None:
    fold = fold_data()
    run_pretraining(config(), fold, tmp_path / "pretrain", device=CPU)
    parent = tmp_path / "pretrain" / "best.pt"
    results = {
        m: run_finetuning(
            config(), fold, m, tmp_path / m, parent=None if m == "S" else parent, device=CPU
        )
        for m in ("S", "H", "F")
    }
    assert all(r["status"] == "completed" for r in results.values())
    assert results["H"]["body_unchanged"] is True
    assert results["H"]["trainable_parameters"] < results["F"]["trainable_parameters"]
    parent_body = {k: v for k, v in _state(parent).items() if not k.startswith("classifier.")}
    trained = _state(tmp_path / "H" / "best.pt")
    assert all(torch.equal(v, trained[k]) for k, v in parent_body.items())
    assert len(list((tmp_path / "F" / "epochs").glob("epoch*.pt"))) == 3
    with pytest.raises(ValueError, match="pretraining parent"):
        run_finetuning(config(), fold, "S", tmp_path / "bad", parent=parent, device=CPU)


def test_supervised_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    fold = fold_data()
    run_finetuning(config(), fold, "S", tmp_path / "full", device=CPU)
    import ehrforecast.finetune as finetune

    original, calls = finetune.evaluate_predictions, {"n": 0}

    def crash_after(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:  # held-out and training evaluation of epoch 0 done; crash in epoch 1
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    finetune.evaluate_predictions = crash_after
    try:
        with pytest.raises(KeyboardInterrupt):
            run_finetuning(config(), fold, "S", tmp_path / "split", device=CPU)
    finally:
        finetune.evaluate_predictions = original
    run_finetuning(config(), fold, "S", tmp_path / "split", resume=True, device=CPU)
    for key, value in _state(tmp_path / "full" / "last.pt").items():
        torch.testing.assert_close(
            value, _state(tmp_path / "split" / "last.pt")[key], rtol=1e-5, atol=1e-6
        )
    np.testing.assert_allclose(
        [m["held_macro_auroc"] for m in _metrics(tmp_path / "full") if "held_macro_auroc" in m],
        [m["held_macro_auroc"] for m in _metrics(tmp_path / "split") if "held_macro_auroc" in m][
            -3:
        ],
    )


def test_session_limit_pauses_and_resume_completes(tmp_path: Path) -> None:
    fold = fold_data()
    paused = run_pretraining(config(), fold, tmp_path / "p", max_session_epochs=1, device=CPU)
    assert (paused["status"], paused["epochs_completed"]) == ("paused", 1)
    done = run_pretraining(config(), fold, tmp_path / "p", resume=True, device=CPU)
    assert (done["status"], done["epochs_completed"]) == ("completed", 3)
