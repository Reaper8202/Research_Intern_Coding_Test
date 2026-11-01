"""P4b: self-pretraining on fit patients' histories before their real anchors (section 9.2).

Each epoch takes one window per fit patient, starting at a day boundary chosen with
``(seed, epoch)``. Held-out training patients supply fixed evaluation windows (``eval_seed``).
Next-token loss covers clinical events only. Early stopping keeps the best held-out epoch
(``best.pt``); ``last.pt`` supports resume at completed epochs.

Usage::

    python -m ehrforecast.pretrain --config configs/transformer.toml --dataset data/processed \
        --models models/transformer --fold 0 [--resume] [--run-name pretrain]
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ehrforecast.events import SOURCE_ORDER
from ehrforecast.logging_utils import setup_logging
from ehrforecast.model import EHRTransformer
from ehrforecast.provenance import object_hash
from ehrforecast.sequences import Sample, SequenceConfig
from ehrforecast.training import (
    IGNORE,
    EarlyStopping,
    FoldData,
    RunLogger,
    accumulate_step,
    batch_order,
    build_model,
    collate,
    early_stopping_dict,
    epoch_rng,
    fold_dir,
    load_checkpoint,
    load_config,
    load_fold,
    micro_batches,
    next_token_loss_sum,
    param_groups,
    release_cached_memory,
    rng_state,
    run_record,
    save_checkpoint,
    seed_everything,
    select_device,
    sequence_config,
    set_rng_state,
    warmup_cosine,
)

log = logging.getLogger("ehrforecast.pretrain")


def epoch_windows(
    fold: FoldData, rows: list[int], seed: int, epoch: int, config: SequenceConfig
) -> tuple[list[Sample], np.random.Generator]:
    rng = epoch_rng(seed, epoch)
    hist = fold.histories
    samples = [
        hist.window_sample(r, int(rng.integers(hist.window_starts(r, config))), config)
        for r in rows
    ]
    return samples, rng


def fixed_windows(
    fold: FoldData, rows: list[int], seed: int, config: SequenceConfig
) -> list[Sample]:
    rng = np.random.default_rng(seed)
    hist = fold.histories
    return [
        hist.window_sample(r, int(rng.integers(hist.window_starts(r, config))), config)
        for r in rows
    ]


@torch.no_grad()
def evaluate_windows(
    model: EHRTransformer,
    samples: list[Sample],
    device: torch.device,
    batch_size: int,
    target_start_ids: torch.Tensor,
) -> dict[str, Any]:
    """Held-out next-token loss overall, by source, and for target-condition start tokens."""
    model.eval()
    n_sources = len(SOURCE_ORDER)
    loss_by_source = torch.zeros(n_sources, dtype=torch.float64)
    count_by_source = torch.zeros(n_sources, dtype=torch.float64)
    target_loss, target_count = 0.0, 0
    order = np.argsort([len(s.tokens) for s in samples], kind="stable")
    for i in range(0, len(order), batch_size):
        batch = collate([samples[k] for k in order[i : i + batch_size]], device)
        eligible = batch["targets"] != IGNORE
        if not eligible.any():
            continue
        states = model.hidden(batch["tokens"], batch["ages"], batch.get("images"))[eligible]
        targets = batch["targets"][eligible]
        losses = (
            torch.nn.functional.cross_entropy(
                model.next_token_logits(states),
                targets,
                reduction="none",
            )
            .cpu()
            .double()  # MPS has no float64
        )
        sources = batch["target_sources"][eligible].cpu()
        loss_by_source += torch.bincount(sources, weights=losses, minlength=n_sources)
        count_by_source += torch.bincount(sources, minlength=n_sources).double()
        is_target = torch.isin(targets.cpu(), target_start_ids)
        target_loss += float(losses[is_target].sum())
        target_count += int(is_target.sum())
    release_cached_memory(device)
    total = float(count_by_source.sum())
    if total == 0:
        raise ValueError("held-out pretraining windows contain no target tokens")
    return {
        "loss": float(loss_by_source.sum()) / total,
        "tokens": int(total),
        "loss_by_source": {
            s: float(loss_by_source[k] / count_by_source[k])
            for k, s in enumerate(SOURCE_ORDER)
            if count_by_source[k]
        },
        "tokens_by_source": {s: int(count_by_source[k]) for k, s in enumerate(SOURCE_ORDER)},
        "target_start_loss": target_loss / target_count if target_count else None,
        "target_start_support": target_count,
    }


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type == "mps":
        return float(torch.mps.driver_allocated_memory()) / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS


def run_pretraining(
    config: dict[str, Any],
    fold: FoldData,
    run_dir: Path,
    *,
    resume: bool = False,
    max_session_epochs: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    pc, runtime = config["pretrain"], config["runtime"]
    seed, sequence = int(config["seed"]), sequence_config(config)
    device = device or select_device(str(runtime["device"]))
    hist = fold.histories
    fit_rows = [int(r) for r in fold.fit_rows if hist.n_tokens(int(r))]
    held_rows = [int(r) for r in fold.held_rows if hist.n_tokens(int(r))]
    eval_samples = fixed_windows(fold, held_rows, int(pc["eval_seed"]), sequence)
    codes = fold.data.inputs.codes
    target_start_ids = torch.tensor(
        [
            fold.tokenizer.token_id(f"cond:start:{c}")
            for c in codes
            if f"cond:start:{c}" in fold.tokenizer.vocabulary
        ],
        dtype=torch.int64,
    )

    seed_everything(seed)
    model = build_model(config, fold.tokenizer, len(codes)).to(device)
    optimizer = torch.optim.AdamW(
        param_groups(list(model.parameters()), float(pc["lr"]), float(pc["weight_decay"])),
        betas=tuple(pc["betas"]),
    )
    batch_size, micro = int(pc["batch_size"]), int(pc["micro_batch_size"])
    steps_per_epoch = math.ceil(len(fit_rows) / batch_size)
    if int(runtime["max_steps_per_epoch"]):
        steps_per_epoch = min(steps_per_epoch, int(runtime["max_steps_per_epoch"]))
    total_steps = int(pc["max_epochs"]) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(total_steps, float(pc["warmup_fraction"]), float(pc["final_lr_fraction"])),
    )
    stopper = EarlyStopping(int(pc["patience"]), "min")
    provenance = {**fold.provenance, "stage": "pretrain", "config_sha256": object_hash(config)}
    run_id = object_hash({**provenance, "run_dir": run_dir.name})[:16]
    start_epoch, global_step = 0, 0
    last = run_dir / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError(
                f"{run_dir} already has a checkpoint; pass --resume or choose a new run name"
            )
        state = load_checkpoint(last, provenance)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stopper = EarlyStopping(**state["early_stopping"])
        set_rng_state(state["rng"])
        start_epoch, global_step = int(state["epoch"]) + 1, int(state["global_step"])
        log.info("Resumed %s after epoch %d", run_dir, state["epoch"])

    n_params = sum(p.numel() for p in model.parameters())
    logger = RunLogger(
        run_dir,
        f"{config['name']}-fold{fold.provenance['fold']}-{run_dir.name}",
        config,
        str(runtime["wandb"]),
        run_id,
    )
    started, trained_tokens, padded_tokens, skipped = time.perf_counter(), 0, 0, 0
    epoch = done = start_epoch - 1  # ``done``: last completed epoch
    try:
        for epoch in range(start_epoch, int(pc["max_epochs"])):
            if stopper.stop or (
                max_session_epochs is not None and epoch - start_epoch >= max_session_epochs
            ):
                break
            samples, rng = epoch_windows(fold, fit_rows, seed, epoch, sequence)
            batches = batch_order(np.array([len(s.tokens) for s in samples]), batch_size, rng)
            batches = batches[:steps_per_epoch]
            model.train()
            epoch_loss, epoch_count = 0.0, 0
            for batch in batches:
                parts = []
                for mb in micro_batches(batch, micro):
                    chosen = [samples[k] for k in mb]
                    count = sum(s.n_events for s in chosen)
                    parts.append(
                        (lambda c=chosen: next_token_loss_sum(model, collate(c, device)), count)
                    )
                    padded_tokens += len(chosen) * max(len(s.tokens) for s in chosen)
                loss_sum, count, norm = accumulate_step(
                    parts, model, optimizer, scheduler, float(pc["grad_clip"])
                )
                release_cached_memory(device)
                if norm is None:
                    skipped += 1
                    logger.log(
                        {"event": "skipped_batch", "epoch": epoch, "reason": "no target tokens"}
                    )
                    continue
                global_step += 1
                trained_tokens += count
                epoch_loss, epoch_count = epoch_loss + loss_sum, epoch_count + count
                logger.log(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": loss_sum / count,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": norm,
                        "target_tokens": count,
                    }
                )
            held = evaluate_windows(
                model, eval_samples, device, int(runtime["eval_batch_size"]), target_start_ids
            )
            improved = stopper.update(epoch, held["loss"])
            record = {
                "epoch": epoch,
                "step": global_step,
                "train_loss_epoch": epoch_loss / max(1, epoch_count),
                "held_loss": held["loss"],
                "held_target_start_loss": held["target_start_loss"],
                "held_target_start_support": held["target_start_support"],
                "improved": improved,
            }
            logger.log(
                {**record, **{f"held_loss_{k}": v for k, v in held["loss_by_source"].items()}}
            )
            if improved:
                save_checkpoint(
                    run_dir / "best.pt",
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "provenance": provenance,
                        "config": config,
                        "held": held,
                    },
                )
            save_checkpoint(
                last,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "early_stopping": early_stopping_dict(stopper),
                    "rng": rng_state(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "sampler": {"seed": seed, "next_epoch": epoch + 1},
                    "provenance": provenance,
                    "config": config,
                },
            )
            done = epoch
            log.info(
                "Epoch %d: train %.4f held %.4f%s",
                epoch,
                record["train_loss_epoch"],
                held["loss"],
                " *" if improved else "",
            )
    except Exception as error:
        run_record(
            run_dir,
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "epoch": epoch,
                "provenance": provenance,
            },
        )
        raise
    finally:
        logger.close()
    seconds = time.perf_counter() - started
    result = {
        "status": "completed" if stopper.stop or done + 1 >= int(pc["max_epochs"]) else "paused",
        "stage": "pretrain",
        "run_id": run_id,
        "provenance": provenance,
        "parameters": n_params,
        "device": str(device),
        "epochs_completed": done + 1,
        "best_epoch": stopper.best_epoch,
        "best_held_loss": stopper.best,
        "stopped_early": stopper.stop,
        "skipped_batches": skipped,
        "fit_patients_with_history": len(fit_rows),
        "held_patients_with_history": len(held_rows),
        "seconds_this_session": seconds,
        "target_tokens_per_second": trained_tokens / seconds if seconds else None,
        "padded_tokens_per_second": padded_tokens / seconds if seconds else None,
        "peak_memory_gb": peak_memory_gb(device),
    }
    run_record(run_dir, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-name", default="pretrain")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-session-epochs",
        type=int,
        default=None,
        help="stop after this many epochs; resume later",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    config = load_config(args.config)
    fold = load_fold(args.dataset, args.models, config, args.fold)
    result = run_pretraining(
        config,
        fold,
        fold_dir(args.models, config, args.fold) / args.run_name,
        resume=args.resume,
        max_session_epochs=args.max_session_epochs,
    )
    log.info(
        "Pretraining finished: best epoch %s, held loss %s",
        result["best_epoch"],
        result["best_held_loss"],
    )


if __name__ == "__main__":
    main()
