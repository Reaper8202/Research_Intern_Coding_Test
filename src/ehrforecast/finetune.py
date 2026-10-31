"""P4a/P4c: supervised training methods S, H, and F (model plan section 9.3).

=====  ================================  ==========================
method initial body                      parameters updated
=====  ================================  ==========================
S      random                            body and head
H      self-pretrained checkpoint        head only (body in eval mode)
F      the same self-pretrained one      body and head
=====  ================================  ==========================

Loss: unweighted binary cross-entropy over cells that are not prior conditions, divided by their
count. Early stopping uses held-out masked macro AUROC (patience 5); mAP breaks exact ties, then
the earlier epoch wins. Every epoch's weights are kept for the later validation curves.

Usage::

    python -m ehrforecast.finetune --config configs/transformer.toml --dataset data/processed \
        --models models/transformer --fold 0 --method F [--parent pretrain] [--resume]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast.evaluation import evaluate_predictions
from ehrforecast.logging_utils import setup_logging
from ehrforecast.model import EHRTransformer
from ehrforecast.provenance import file_hash, object_hash
from ehrforecast.sequences import Sample
from ehrforecast.training import (
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
    param_groups,
    rng_state,
    run_record,
    save_checkpoint,
    seed_everything,
    select_device,
    sequence_config,
    set_rng_state,
    supervised_loss_sum,
)

log = logging.getLogger("ehrforecast.finetune")
METHODS = ("S", "H", "F")


@torch.no_grad()
def predict_samples(
    model: EHRTransformer, samples: list[Sample], device: torch.device, batch_size: int
) -> np.ndarray:
    """Probabilities in sample order. Length bucketing does not change any row (right padding)."""
    model.eval()
    out = np.zeros((len(samples), model.config.n_targets), dtype="float64")
    order = np.argsort([len(s.tokens) for s in samples], kind="stable")
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        batch = collate([samples[k] for k in idx], device)
        logits = model(batch["tokens"], batch["ages"], batch["readout"], batch["head"])
        probabilities = torch.sigmoid(logits.cpu().double()).numpy()
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("non-finite predicted probabilities")
        out[idx] = probabilities
    return out


def masked_bce(
    probabilities: pd.DataFrame, labels: pd.DataFrame, prior: pd.DataFrame, codes: list[str]
) -> float:
    """Mean binary cross-entropy over cells that are not prior conditions."""
    keep = prior.loc[probabilities.index, codes].to_numpy() == 0
    y = labels.loc[probabilities.index, codes].to_numpy("float64")[keep]
    p = np.clip(probabilities[codes].to_numpy("float64")[keep], 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def set_train_mode(model: EHRTransformer, method: str) -> None:
    model.train()
    if method == "H":  # fixed body: no dropout in the body; the head keeps its dropout
        for module in model.body_modules():
            module.eval()


def load_parent_body(model: EHRTransformer, parent: Path, fold: FoldData) -> str:
    """Copy body weights from our own pretraining checkpoint; the head stays newly initialized."""
    expected = {k: fold.provenance[k] for k in ("fold", "tokenizer_sha256", "histories_sha256")}
    state = load_checkpoint(parent, {**expected, "stage": "pretrain"})
    body = {k: v for k, v in state["model"].items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(body, strict=False)
    if unexpected or set(missing) != {"classifier.weight", "classifier.bias"}:
        raise ValueError("parent checkpoint does not match the model body")
    return file_hash(parent)


def body_snapshot(model: EHRTransformer) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.body_parameters()]


def run_finetuning(
    config: dict[str, Any],
    fold: FoldData,
    method: str,
    run_dir: Path,
    *,
    parent: Path | None = None,
    resume: bool = False,
    max_session_epochs: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if (method == "S") != (parent is None):
        raise ValueError("H and F need a pretraining parent; S must not have one")
    fc, runtime = config["finetune"], config["runtime"]
    seed, sequence = int(config["seed"]), sequence_config(config)
    device = device or select_device(str(runtime["device"]))
    codes = fold.data.inputs.codes
    hist = fold.histories
    labels, prior = fold.data.labels, fold.data.inputs.prior
    fit_samples = [hist.prediction_sample(int(r), sequence) for r in fold.fit_rows]
    held_samples = [hist.prediction_sample(int(r), sequence) for r in fold.held_rows]
    y_fit = labels.loc[fold.fit, codes].to_numpy("float32")
    unmasked_fit = 1.0 - prior.loc[fold.fit, codes].to_numpy("float32")

    seed_everything(seed)
    model = build_model(config, fold.tokenizer, len(codes)).to(device)
    parent_sha = load_parent_body(model, parent, fold) if parent is not None else None
    if method == "H":
        for p in model.body_parameters():
            p.requires_grad_(False)
    groups = param_groups(model.head_parameters(), float(fc["head_lr"]), float(fc["weight_decay"]))
    if method != "H":
        groups += param_groups(
            model.body_parameters(), float(fc[f"body_lr_{method}"]), float(fc["weight_decay"])
        )
    optimizer = torch.optim.AdamW(groups)
    stopper = EarlyStopping(int(fc["patience"]), "max")
    provenance = {
        **fold.provenance,
        "stage": f"finetune_{method}",
        "config_sha256": object_hash(config),
        "parent_sha256": parent_sha,
    }
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
        stopper = EarlyStopping(**state["early_stopping"])
        set_rng_state(state["rng"])
        start_epoch, global_step = int(state["epoch"]) + 1, int(state["global_step"])
        log.info("Resumed %s after epoch %d", run_dir, state["epoch"])
    initial_body = body_snapshot(model) if method == "H" else None

    batch_size, micro = int(fc["batch_size"]), int(fc["micro_batch_size"])
    cap = int(runtime["max_steps_per_epoch"])
    eval_batch = int(runtime["eval_batch_size"])
    logger = RunLogger(
        run_dir,
        f"{config['name']}-fold{fold.provenance['fold']}-{run_dir.name}",
        config,
        str(runtime["wandb"]),
        run_id,
    )
    started, skipped = time.perf_counter(), 0
    epoch = done = start_epoch - 1  # ``done``: last completed epoch
    try:
        for epoch in range(start_epoch, int(fc["max_epochs"])):
            if stopper.stop or (
                max_session_epochs is not None and epoch - start_epoch >= max_session_epochs
            ):
                break
            rng = epoch_rng(seed, epoch)
            batches = batch_order(np.array([len(s.tokens) for s in fit_samples]), batch_size, rng)
            if cap:
                batches = batches[:cap]
            set_train_mode(model, method)
            epoch_loss, epoch_count = 0.0, 0
            for batch in batches:
                parts = []
                for mb in micro_batches(batch, micro):
                    y = torch.from_numpy(y_fit[mb]).to(device)
                    mask = torch.from_numpy(unmasked_fit[mb]).to(device)
                    chosen = [fit_samples[k] for k in mb]

                    def loss_fn(
                        c: list[Sample] = chosen, y: torch.Tensor = y, m: torch.Tensor = mask
                    ) -> tuple[torch.Tensor, int]:
                        b = collate(c, device)
                        return supervised_loss_sum(
                            model(b["tokens"], b["ages"], b["readout"], b["head"]), y, m
                        )

                    parts.append((loss_fn, int(unmasked_fit[mb].sum())))
                loss_sum, count, norm = accumulate_step(
                    parts, model, optimizer, None, float(fc["grad_clip"])
                )
                if norm is None:
                    skipped += 1
                    logger.log(
                        {"event": "skipped_batch", "epoch": epoch, "reason": "all cells masked"}
                    )
                    continue
                global_step += 1
                epoch_loss, epoch_count = epoch_loss + loss_sum, epoch_count + count
                logger.log(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": loss_sum / count,
                        "grad_norm": norm,
                        **{
                            f"lr_group{g}": group["lr"]
                            for g, group in enumerate(optimizer.param_groups)
                        },
                    }
                )
            probabilities = pd.DataFrame(
                predict_samples(model, held_samples, device, eval_batch),
                index=fold.held,
                columns=codes,
            )
            held_eval = evaluate_predictions(
                labels.loc[fold.held], probabilities, codes, prior=prior.loc[fold.held]
            )
            summary = held_eval.summary
            if summary["macro_auroc"] is None:
                raise ValueError("held-out macro AUROC is undefined; check the held-out labels")
            held_loss = masked_bce(
                probabilities, labels.loc[fold.held], prior.loc[fold.held], codes
            )
            train_probabilities = pd.DataFrame(
                predict_samples(model, fit_samples, device, eval_batch),
                index=fold.fit,
                columns=codes,
            )
            train_summary = evaluate_predictions(
                labels.loc[fold.fit], train_probabilities, codes, prior=prior.loc[fold.fit]
            ).summary
            improved = stopper.update(epoch, float(summary["macro_auroc"]), float(summary["mAP"]))
            record = {
                "epoch": epoch,
                "step": global_step,
                "train_loss_epoch": epoch_loss / max(1, epoch_count),
                "held_loss": held_loss,
                "held_macro_auroc": summary["macro_auroc"],
                "held_mAP": summary["mAP"],
                "held_macro_brier": summary["macro_brier"],
                "train_macro_auroc_eval_mode": train_summary["macro_auroc"],
                "train_mAP_eval_mode": train_summary["mAP"],
                "improved": improved,
            }
            logger.log(record)
            weights = {
                "model": model.state_dict(),
                "epoch": epoch,
                "provenance": provenance,
                "config": config,
            }
            save_checkpoint(run_dir / "epochs" / f"epoch{epoch:02d}.pt", weights)
            if improved:
                save_checkpoint(run_dir / "best.pt", weights)
                held_eval.per_target.to_csv(run_dir / "held_per_target_best.csv")
                probabilities.to_parquet(run_dir / "held_predictions_best.parquet")
            save_checkpoint(
                last,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
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
                "%s epoch %d: train %.4f held AUROC %.4f mAP %.4f%s",
                method,
                epoch,
                record["train_loss_epoch"],
                summary["macro_auroc"],
                summary["mAP"],
                " *" if improved else "",
            )
        if initial_body is not None and not all(
            torch.equal(a, b) for a, b in zip(initial_body, body_snapshot(model), strict=True)
        ):
            raise AssertionError("fixed-body training changed body parameters")
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
    result = {
        "status": "completed" if stopper.stop or done + 1 >= int(fc["max_epochs"]) else "paused",
        "stage": f"finetune_{method}",
        "run_id": run_id,
        "provenance": provenance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device),
        "epochs_completed": done + 1,
        "best_epoch": stopper.best_epoch,
        "best_held_macro_auroc": stopper.best,
        "best_held_mAP": stopper.best_tiebreak,
        "stopped_early": stopper.stop,
        "skipped_batches": skipped,
        "body_unchanged": True if method == "H" else None,
        "seconds_this_session": time.perf_counter() - started,
        "note": "Held-out training-fold metrics. Not official val results.",
    }
    run_record(run_dir, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--parent", default="pretrain", help="pretraining run name (H and F)")
    parser.add_argument("--run-name", default=None, help="defaults to the method letter")
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
    directory = fold_dir(args.models, config, args.fold)
    parent = None if args.method == "S" else directory / args.parent / "best.pt"
    result = run_finetuning(
        config,
        fold,
        args.method,
        directory / (args.run_name or args.method),
        parent=parent,
        resume=args.resume,
        max_session_epochs=args.max_session_epochs,
    )
    log.info(
        "Finished %s: best epoch %s, held AUROC %s",
        args.method,
        result["best_epoch"],
        result["best_held_macro_auroc"],
    )


if __name__ == "__main__":
    main()
