"""Shared training machinery for pretraining and supervised methods (model plan section 9.5).

* Length-bucketed optimizer batches. Each epoch's order depends only on ``(seed, epoch)``, so a
  resumed run creates the same batches as an uninterrupted one.
* Loss accounting: every optimizer update divides the summed loss by the eligible count of the
  whole optimizer batch, across all of its microbatches. A batch without eligible targets makes
  no optimizer or schedule update.
* Checkpoints at completed epochs hold the model, optimizer, schedule, early-stopping, and random
  states. They are written to a temporary file and then renamed. Loading checks provenance.
"""

from __future__ import annotations

import json
import math
import os
import time
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import TrainingData, load_training_data
from ehrforecast.events import EVENTS_MANIFEST
from ehrforecast.folds import fold_patients, load_folds
from ehrforecast.modalities import Modalities, load_modalities
from ehrforecast.model import EHRTransformer, ModelConfig
from ehrforecast.provenance import file_hash, object_hash, write_json
from ehrforecast.sequences import (
    Histories,
    Sample,
    SequenceConfig,
    histories_provenance,
    load_histories,
)
from ehrforecast.tokenizer import IMG, FoldTokenizer, age_features

CHECKPOINT_VERSION = "transformer-checkpoint-v1"
IGNORE = -100


# ---------------------------------------------------------------------- configuration
def load_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text())
    for section in ("representation", "model", "pretrain", "finetune", "runtime"):
        if section not in config:
            raise ValueError(f"{path} lacks the [{section}] section")
    return config


def modality_settings(config: dict[str, Any]) -> tuple[bool, bool, int]:
    """``[modalities]`` (D34): genomics on, imaging on, image size. Absent means CSV only."""
    section = config.get("modalities", {})
    return (
        bool(section.get("genomics", False)),
        bool(section.get("imaging", False)),
        int(section.get("image_size", 64)),
    )


def config_modalities(dataset_dir: Path, config: dict[str, Any]) -> Modalities | None:
    genomics, imaging, size = modality_settings(config)
    return load_modalities(dataset_dir, size) if genomics or imaging else None


def sequence_config(config: dict[str, Any]) -> SequenceConfig:
    rep = config["representation"]
    return SequenceConfig(int(rep["context_length"]), int(rep["max_summary_tokens"]))


def model_config(config: dict[str, Any], tokenizer: FoldTokenizer, n_targets: int) -> ModelConfig:
    m = config["model"]
    return ModelConfig(
        vocab_size=tokenizer.size,
        n_layers=int(m["n_layers"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        ff_mult=int(m["ff_mult"]),
        dropout=float(m["dropout"]),
        head_dropout=float(m["head_dropout"]),
        n_targets=n_targets,
        image_token_id=tokenizer.token_id(IMG) if tokenizer.imaging else -1,
    )


def build_model(config: dict[str, Any], tokenizer: FoldTokenizer, n_targets: int) -> EHRTransformer:
    return EHRTransformer(
        model_config(config, tokenizer, n_targets), torch.from_numpy(tokenizer.input_only_ids())
    )


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(name)


def release_cached_memory(device: torch.device) -> None:
    """Return unused cached MPS blocks to the system.

    Sequence lengths change from batch to batch, so the MPS cache fragments and keeps growing.
    One fold-4 pretraining process passed 20 GB on a 16 GB Mac and filled the disk with swap.
    The call frees only blocks that no tensor uses, so it changes no result.
    """
    if device.type == "mps":
        torch.mps.empty_cache()


def fold_dir(models_dir: Path, config: dict[str, Any], fold: int) -> Path:
    """Where this configuration's runs for one fold are written."""
    return models_dir / str(config["name"]) / f"fold{fold}"


def representation_dir(models_dir: Path, config: dict[str, Any], fold: int) -> Path:
    """Where the fold tokenizer and histories are read.

    An experiment variant can set ``representation_from`` to another configuration's name and
    reuse its tokenizers and histories. ``load_fold`` then checks that the tokenizer settings and
    modalities agree, so only training and sampling settings (for example the context length,
    which applies when a sample is built) can differ.
    """
    return models_dir / str(config.get("representation_from", config["name"])) / f"fold{fold}"


def pretrain_dir(models_dir: Path, config: dict[str, Any], fold: int) -> Path:
    """The self-pretraining run that H and F start from (``pretrain_from`` reuses another's)."""
    return models_dir / str(config.get("pretrain_from", config["name"])) / f"fold{fold}"


# ---------------------------------------------------------------------- fold data
@dataclass
class FoldData:
    """Training-split inputs for one fold. Official val and test never enter this object."""

    data: TrainingData
    tokenizer: FoldTokenizer
    histories: Histories
    fit: pd.Index
    held: pd.Index
    fit_rows: np.ndarray
    held_rows: np.ndarray
    provenance: dict[str, Any]


def _capped(patients: pd.Index, cap: int) -> pd.Index:
    return patients if cap <= 0 else patients[:cap]


def load_fold(dataset_dir: Path, models_dir: Path, config: dict[str, Any], fold: int) -> FoldData:
    data = load_training_data(dataset_dir)
    fit, held = fold_patients(load_folds(dataset_dir), fold)
    directory = representation_dir(models_dir, config, fold)
    tokenizer = FoldTokenizer.load(directory / "tokenizer.json")
    if tokenizer.fit_patients_sha256 != object_hash(sorted(fit.tolist())):
        raise ValueError("tokenizer was fitted on different patients than this fold's fit set")
    rep = config["representation"]
    if (tokenizer.min_patients, tokenizer.refill_gap_days) != (
        int(rep["min_patients"]),
        float(rep["refill_gap_days"]),
    ):
        raise ValueError("the fold tokenizer and the configuration use different token rules")
    genomics, imaging, _ = modality_settings(config)
    if (tokenizer.genomics, tokenizer.imaging) != (genomics, imaging):
        raise ValueError("the fold tokenizer and the configuration use different modalities")
    modalities = config_modalities(dataset_dir, config)
    expected = histories_provenance(
        tokenizer,
        file_hash(dataset_dir / EVENTS_MANIFEST),
        data.inputs.cohort.index,
        None if modalities is None else modalities.manifest_sha256,
    )
    histories = load_histories(directory / "histories.npz", expected)
    if modalities is not None:
        histories.attach_volumes(modalities)
    runtime = config["runtime"]
    fit = _capped(fit, int(runtime.get("max_fit_patients", 0)))
    held = _capped(held, int(runtime.get("max_held_patients", 0)))
    provenance = {
        "fold": fold,
        "fit_patients_sha256": object_hash(sorted(fit.tolist())),
        "held_patients_sha256": object_hash(sorted(held.tolist())),
        "tokenizer_sha256": tokenizer.sha256,
        "histories_sha256": file_hash(directory / "histories.npz"),
        "events_manifest_sha256": expected["events_manifest_sha256"],
    }
    if modalities is not None:
        provenance["modalities_manifest_sha256"] = modalities.manifest_sha256
    return FoldData(
        data,
        tokenizer,
        histories,
        fit,
        held,
        histories.index_of(fit),
        histories.index_of(held),
        provenance,
    )


# ---------------------------------------------------------------------- batching
def batch_order(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Group similar lengths into batches (random tie-breaks), then shuffle the batch order."""
    order = np.lexsort((rng.random(len(lengths)), lengths))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return [batches[k] for k in rng.permutation(len(batches))]


def micro_batches(batch: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for i in range(0, len(batch), size):
        yield batch[i : i + size]


def collate(samples: Sequence[Sample], device: torch.device) -> dict[str, torch.Tensor]:
    """Right-pad samples. Also build next-token targets: clinical events after the prefix.

    ``images`` (present only when a sample has ``[IMG]``) stacks the volumes in sample order.
    """
    n, width = len(samples), max(len(s.tokens) for s in samples)
    tokens = np.zeros((n, width), dtype="int64")
    ages = np.zeros((n, width), dtype="float32")
    targets = np.full((n, width), IGNORE, dtype="int64")
    sources = np.full((n, width), -1, dtype="int64")
    for b, s in enumerate(samples):
        length = len(s.tokens)
        tokens[b, :length], ages[b, :length] = s.tokens, s.ages
        # Position j predicts token j + 1 when token j + 1 is a clinical event.
        targets[b, s.n_prefix - 1 : length - 1] = s.tokens[s.n_prefix :]
        sources[b, s.n_prefix - 1 : length - 1] = s.sources[s.n_prefix :]
    batch = {
        "tokens": torch.from_numpy(tokens).to(device),
        "ages": torch.from_numpy(age_features(ages)).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "target_sources": torch.from_numpy(sources).to(device),
        "readout": torch.tensor([s.readout for s in samples], device=device),
        "head": torch.from_numpy(np.stack([s.head for s in samples])).to(device),
    }
    images = [s.image for s in samples if s.image is not None]
    if images:
        stacked = np.stack(images).astype("float32")[:, None]
        batch["images"] = torch.from_numpy(stacked).to(device)
    return batch


# ---------------------------------------------------------------------- losses
def next_token_loss_sum(
    model: EHRTransformer, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, int]:
    """Summed cross-entropy over eligible target positions and their count."""
    eligible = batch["targets"] != IGNORE
    count = int(eligible.sum())
    states = model.hidden(batch["tokens"], batch["ages"], batch.get("images"))[eligible]
    if count == 0:
        return states.sum() * 0.0, 0
    logits = model.next_token_logits(states)
    loss = torch.nn.functional.cross_entropy(logits, batch["targets"][eligible], reduction="sum")
    return loss, count


def supervised_loss_sum(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Summed binary cross-entropy over unmasked patient-target cells and their count."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * mask).sum(), int(mask.sum())


def accumulate_step(
    micro: Sequence[tuple[Callable[[], tuple[torch.Tensor, int]], int]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    grad_clip: float,
) -> tuple[float, int, float | None]:
    """One optimizer update from microbatches ``(loss_fn, eligible_count)``.

    Each microbatch's summed loss is divided by the total eligible count of the whole batch.
    Returns the summed loss, the count, and the gradient norm (``None`` if the batch is skipped).
    """
    total = sum(count for _, count in micro)
    optimizer.zero_grad(set_to_none=True)
    if total == 0:
        return 0.0, 0, None
    loss_sum = 0.0
    for loss_fn, count in micro:
        if count == 0:
            continue
        loss, _ = loss_fn()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        (loss / total).backward()
        loss_sum += float(loss.detach())
    params = [p for p in model.parameters() if p.grad is not None]
    norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip, error_if_nonfinite=True))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss_sum, total, norm


# ---------------------------------------------------------------------- optimizer and schedule
def param_groups(
    params: Sequence[torch.nn.Parameter], lr: float, weight_decay: float
) -> list[dict[str, Any]]:
    """Weight decay on weight matrices only; biases and LayerNorm parameters are not decayed."""
    decay = [p for p in params if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in params if p.requires_grad and p.ndim < 2]
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def warmup_cosine(
    total_steps: int, warmup_fraction: float, final_fraction: float
) -> Callable[[int], float]:
    warmup = max(1, round(warmup_fraction * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


# ---------------------------------------------------------------------- early stopping
@dataclass
class EarlyStopping:
    """Track the best epoch. ``mode='min'`` for losses, ``'max'`` for AUROC (tie-break: mAP)."""

    patience: int
    mode: str
    best: float | None = None
    best_tiebreak: float | None = None
    best_epoch: int = -1
    bad_epochs: int = 0

    def update(self, epoch: int, score: float, tiebreak: float | None = None) -> bool:
        if not math.isfinite(score):
            raise FloatingPointError("non-finite held-out score")
        sign = 1.0 if self.mode == "max" else -1.0
        improved = self.best is None or sign * (score - self.best) > 0
        if not improved and score == self.best and tiebreak is not None:
            improved = self.best_tiebreak is None or tiebreak > self.best_tiebreak
        if improved:
            self.best, self.best_tiebreak, self.best_epoch, self.bad_epochs = (
                score,
                tiebreak,
                epoch,
                0,
            )
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stop(self) -> bool:
        return self.bad_epochs >= self.patience


# ---------------------------------------------------------------------- random states
# Sampling uses numpy Generators seeded by (seed, epoch). Torch state controls dropout.
def rng_state() -> dict[str, Any]:
    state = {"torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


def epoch_rng(seed: int, epoch: int) -> np.random.Generator:
    return np.random.default_rng([seed, epoch])


# ---------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"version": CHECKPOINT_VERSION, "complete": True, **state}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Load a local checkpoint and reject a different configuration, fold, or tokenizer."""
    if not path.is_file():
        raise ValueError(f"checkpoint {path} does not exist")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # truncated or corrupt file
        raise ValueError(f"checkpoint {path} cannot be read: {error}") from error
    if state.get("version") != CHECKPOINT_VERSION or not state.get("complete"):
        raise ValueError(f"checkpoint {path} is incomplete or has an unknown version")
    saved = state.get("provenance", {})
    different = sorted(k for k, v in provenance.items() if saved.get(k) != v)
    if different:
        raise ValueError(f"checkpoint {path} does not match this run: {', '.join(different)}")
    return state


# ---------------------------------------------------------------------- logging
class RunLogger:
    """Metrics to ``metrics.jsonl``; also to wandb (offline by default) when enabled."""

    def __init__(
        self, run_dir: Path, name: str, config: dict[str, Any], mode: str, run_id: str
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "metrics.jsonl"
        self.wandb: Any = None
        if mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                project="ehr-forecast",
                name=name,
                id=run_id,
                resume="allow",
                dir=str(run_dir),
                config=config,
                mode=mode,  # type: ignore[arg-type]
            )

    def log(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in record.items() if isinstance(v, int | float)})

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def run_record(run_dir: Path, record: dict[str, Any]) -> None:
    write_json(run_dir / "run.json", record)


def early_stopping_dict(stopper: EarlyStopping) -> dict[str, Any]:
    return asdict(stopper)
