"""P4e: the frozen official validation bundle (model plan section 11).

Two commands::

    python -m ehrforecast.val_score freeze --config configs/final.toml \
        --dataset data/processed --models models --manifest reports/final/manifest.json
    python -m ehrforecast.val_score score --config configs/final.toml \
        --dataset data/processed --models models --manifest reports/final/manifest.json \
        --reports reports/val

``freeze`` records every artifact the bundle reads: the fold checkpoints and their saved epoch
checkpoints, the fold tokenizers, the dataset manifests, the target list, the project code and
lock file, the step-2 val prediction files, and the LightGBM settings that training CV selected.
``score`` refuses a tracked working-tree change, a hash that differs from the manifest, or a
bundle that already completed. It writes a start record before any metric and a completion or
failure record afterwards. Official val labels are read once, here, through ``load_val_labels``.
The val curves are the per-epoch checkpoints scored on val after training; they do not select
anything (the best epoch was fixed by each run's held-out training fold).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import (
    PredictionInputs,
    align_targets,
    load_prediction_inputs,
    load_splits,
    validate_binary,
)
from ehrforecast.cv_summary import modality_patients
from ehrforecast.evaluation import bootstrap_comparison, evaluate_predictions
from ehrforecast.finetune import predict_samples
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.modalities import Modalities
from ehrforecast.predict import FoldModel, load_fold_model, predict_inputs
from ehrforecast.provenance import code_hashes, file_hash, object_hash, write_json
from ehrforecast.sequences import Sample, build_histories
from ehrforecast.submission import build as build_submission
from ehrforecast.training import (
    config_modalities,
    fold_dir,
    load_config,
    modality_settings,
    representation_dir,
    sequence_config,
)

log = logging.getLogger("ehrforecast.val_score")

MANIFEST_VERSION = "val-bundle-manifest-v1"
DATASET_FILES = (
    "events.manifest.json",
    "folds.manifest.json",
    "modalities.manifest.json",
    "targets.csv",
    "cohort.parquet",
    "labels.parquet",
    "prior_conditions.parquet",
)


@dataclass(frozen=True)
class Member:
    """One fold model of the ensemble: a supervised run directory and its tokenizer."""

    config_name: str
    fold: int
    run_dir: Path  # relative to the models directory
    tokenizer: Path  # relative to the models directory
    best_epoch: int
    run_id: str


# ---------------------------------------------------------------------- git state
def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def tracked_changes(root: Path) -> list[str]:
    """Tracked files that differ from the index or HEAD; untracked files are not changes."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------- freeze
def load_final(path: Path) -> dict[str, Any]:
    import tomllib

    final = tomllib.loads(path.read_text())
    for key in ("members", "folds", "method", "checkpoint", "evaluation", "comparisons"):
        if key not in final:
            raise ValueError(f"{path} lacks {key!r}")
    return final


def members_of(
    final: dict[str, Any], configs: dict[str, dict[str, Any]], models_dir: Path
) -> list[Member]:
    transformer = models_dir / "transformer"
    members = []
    for path, config in configs.items():
        for fold in final["folds"]:
            run_dir = fold_dir(transformer, config, int(fold)) / str(final["method"])
            record = json.loads((run_dir / "run.json").read_text())
            if record.get("status") != "completed":
                raise ValueError(f"{run_dir} did not complete ({record.get('status')})")
            if record["provenance"]["config_sha256"] != object_hash(config):
                raise ValueError(f"{run_dir} was trained with a different {path}")
            if record["provenance"]["fold"] != int(fold):
                raise ValueError(f"{run_dir} belongs to another fold")
            tokenizer = representation_dir(transformer, config, int(fold)) / "tokenizer.json"
            members.append(
                Member(
                    str(config["name"]),
                    int(fold),
                    run_dir.relative_to(models_dir),
                    tokenizer.relative_to(models_dir),
                    int(record["best_epoch"]),
                    str(record["run_id"]),
                )
            )
    return members


def build_manifest(
    root: Path,
    final: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    dataset_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    """Every hash the bundle will verify, plus the ensemble and mask rules in words."""
    members = members_of(final, configs, models_dir)
    codes = pd.read_csv(dataset_dir / "targets.csv", dtype=str, keep_default_na=False)[
        "CODE"
    ].tolist()
    model_hashes: dict[str, str] = {}
    member_records = []
    for m in members:
        run_dir = models_dir / m.run_dir
        files = [m.run_dir / str(final["checkpoint"]), m.run_dir / "run.json", m.tokenizer]
        epochs = sorted(p.relative_to(models_dir) for p in (run_dir / "epochs").glob("epoch*.pt"))
        if bool(final["evaluation"].get("curves", True)):
            files += epochs
        for relative in files:
            model_hashes[str(relative)] = file_hash(models_dir / relative)
        member_records.append(
            {
                "config": m.config_name,
                "fold": m.fold,
                "run_dir": str(m.run_dir),
                "tokenizer": str(m.tokenizer),
                "best_epoch": m.best_epoch,
                "training_run_id": m.run_id,
                "epoch_checkpoints": [str(p) for p in epochs],
            }
        )
    comparison_hashes = frozen_inputs(root, final)
    return {
        "version": MANIFEST_VERSION,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze_commit": git_commit(root),
        "final_config_sha256": object_hash(final),
        "member_config_sha256": {path: object_hash(c) for path, c in configs.items()},
        "members": member_records,
        "ensemble": (
            f"mean of {len(members)} fold-model probabilities ({final['checkpoint']}); "
            "prior-condition cells set to 0 after averaging"
        ),
        "prior_mask": "prior_from_history on the permitted pre-anchor condition events",
        "targets": codes,
        "model_hashes": model_hashes,
        "dataset_hashes": {
            name: file_hash(dataset_dir / name)
            for name in DATASET_FILES
            if (dataset_dir / name).is_file()
        },
        "code_hashes": code_hashes(root),
        "comparison_hashes": comparison_hashes,
        "evaluation": final["evaluation"],
    }


def frozen_inputs(root: Path, final: dict[str, Any]) -> dict[str, str]:
    """Hashes of every file the bundle reads besides the members: step-2 val predictions, the
    LightGBM settings, the final LightGBM boosters and prediction files, and the stack inputs."""
    comparisons = final["comparisons"]
    paths: list[Path] = [
        Path(str(comparisons[key]))
        for key in comparisons
        if key.endswith(("_codes", "_at_risk", "_settings"))
    ]
    for key in ("lightgbm_dir", "lightgbm_models"):
        if key in comparisons:
            directory = root / str(comparisons[key])
            if not directory.is_dir():
                raise ValueError(f"{directory} is missing; run make lightgbm-final first")
            paths += sorted(q.relative_to(root) for q in directory.rglob("*") if q.is_file())
    stack = final.get("stack")
    if stack:
        paths += [Path(str(stack[k])) for k in ("platt_weights", "oof_source") if stack.get(k)]
    return {str(q): file_hash(root / q) for q in paths}


def verify_manifest(
    root: Path, manifest: dict[str, Any], dataset_dir: Path, models_dir: Path
) -> None:
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unknown manifest version")
    mismatches = []
    for base, hashes in (
        (models_dir, manifest["model_hashes"]),
        (dataset_dir, manifest["dataset_hashes"]),
        (root, manifest["code_hashes"]),
        (root, manifest["comparison_hashes"]),
    ):
        for relative, expected in hashes.items():
            path = base / relative
            if not path.is_file() or file_hash(path) != expected:
                mismatches.append(str(relative))
    if mismatches:
        raise ValueError("artifact hash mismatch: " + ", ".join(sorted(mismatches)))


# ---------------------------------------------------------------------- val data
def load_val_labels(dataset_dir: Path, patients: pd.Index, codes: list[str]) -> pd.DataFrame:
    """The only reader of official val labels. Arrow filtering keeps other splits out."""
    labels = pd.read_parquet(
        dataset_dir / "labels.parquet", filters=[("patient_id", "in", patients.tolist())]
    )
    labels = align_targets(labels, patients, codes, "val labels")
    validate_binary(labels, "val labels")
    return labels


def read_comparison(path: Path, patients: pd.Index, codes: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    frame.columns = [str(c) for c in frame.columns]
    return align_targets(frame, patients, codes, path.name)


# ---------------------------------------------------------------------- curves
class SampleCache:
    """Prediction samples per tokenizer, built once; epoch checkpoints share them."""

    def __init__(self, inputs: PredictionInputs, modalities: Modalities | None) -> None:
        self.inputs, self.modalities = inputs, modalities
        self._cache: dict[tuple[str, str], list[Sample]] = {}

    def samples(self, fm: FoldModel) -> list[Sample]:
        sequence = sequence_config(fm.config)
        key = (fm.tokenizer.sha256, object_hash(asdict(sequence)))
        if key not in self._cache:
            multimodal = fm.tokenizer.genomics or fm.tokenizer.imaging
            histories = build_histories(
                fm.tokenizer.encode(self.inputs.events),
                self.inputs.cohort,
                fm.tokenizer,
                self.inputs.codes,
                self.modalities if multimodal else None,
            )
            self._cache[key] = [
                histories.prediction_sample(i, sequence) for i in range(len(histories))
            ]
        return self._cache[key]


def checkpoint_probabilities(
    fm: FoldModel, cache: SampleCache, device: torch.device, batch_size: int
) -> pd.DataFrame:
    raw = predict_samples(fm.model.to(device), cache.samples(fm), device, batch_size)
    frame = pd.DataFrame(raw, index=cache.inputs.cohort.index, columns=cache.inputs.codes)
    return apply_prior_mask(frame, cache.inputs.prior, cache.inputs.codes)


def val_curves(
    members: list[dict[str, Any]],
    models_dir: Path,
    labels: pd.DataFrame,
    cache: SampleCache,
    device: torch.device,
    batch_size: int,
    wandb_run: Callable[[dict[str, Any]], Any] | None,
) -> pd.DataFrame:
    """Score every saved epoch checkpoint of every member on val; log one wandb run each."""
    rows = []
    for m in members:
        run = wandb_run(m) if wandb_run is not None else None
        for relative in m["epoch_checkpoints"]:
            path = models_dir / relative
            fm = load_fold_model(path.parent.parent, str(path.relative_to(path.parent.parent)))
            epoch = int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
            probabilities = checkpoint_probabilities(fm, cache, device, batch_size)
            summary = evaluate_predictions(
                labels, probabilities, cache.inputs.codes, prior=cache.inputs.prior
            ).summary
            row = {
                "config": m["config"],
                "fold": m["fold"],
                "epoch": epoch,
                "is_best": epoch == int(m["best_epoch"]),
                "val_macro_auroc": summary["macro_auroc"],
                "val_mAP": summary["mAP"],
                "val_macro_brier": summary["macro_brier"],
            }
            rows.append(row)
            if run is not None:
                run.log({k: v for k, v in row.items() if isinstance(v, int | float)})
        if run is not None:
            run.finish()
        log.info(
            "Val curve: %s fold %d, %d checkpoints",
            m["config"],
            m["fold"],
            len(m["epoch_checkpoints"]),
        )
    return pd.DataFrame(rows).sort_values(["config", "fold", "epoch"]).reset_index(drop=True)


# ---------------------------------------------------------------------- scoring
def score_bundle(
    inputs: PredictionInputs,
    labels: pd.DataFrame,
    fold_models: list[FoldModel],
    comparisons: dict[str, pd.DataFrame],
    modalities: Modalities | None,
    *,
    clock_year: int,
    n_boot: int,
    seed: int,
    device: torch.device,
    batch_size: int = 64,
    stack: dict[str, Any] | None = None,
    root: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Final probabilities, metrics, predeclared paired comparisons, and descriptive subgroups.

    Without ``stack`` the final model is the transformer ensemble. With it, the final model is
    the configured blend of the ensemble and the named comparison models (``submission.build``),
    and the ensemble itself becomes one more comparison. Returns the summary, the final
    probabilities, the per-target table, and the transformer ensemble probabilities.
    """
    codes, prior = inputs.codes, inputs.prior
    if np.logical_and(labels.to_numpy(), prior.to_numpy()).any():
        raise ValueError("positive val labels overlap prior conditions")
    transformer = predict_inputs(fold_models, inputs, device, batch_size, modalities)
    if stack is None:
        final = transformer
        submission = f"transformer ensemble of {len(fold_models)} fold models"
    else:
        if root is None:
            raise ValueError("a stack needs the project root for its weight files")
        final = build_submission(
            stack, {"transformer": transformer, **comparisons}, prior, codes, root
        )
        comparisons = {"transformer": transformer, **comparisons}
        submission = f"{stack['stacker']} of {list(stack['bases'])}" + (
            " with Platt scaling" if stack.get("platt_weights") else ""
        )
    models = {"final": final, **comparisons}
    masked = evaluate_predictions(labels, final, codes, prior=prior)
    at_risk = evaluate_predictions(labels, final, codes, prior=prior, at_risk=True)
    summary: dict[str, Any] = {
        "scope": "official validation split, scored once by the frozen bundle",
        "n_patients": len(labels),
        "n_positive_labels": int(labels.to_numpy().sum()),
        "ensemble_members": len(fold_models),
        "submission": submission,
        "final": {"masked": masked.summary, "at_risk": at_risk.summary},
        "final_bca": bootstrap_comparison(
            labels, final, codes, prior=prior, n_boot=n_boot, seed=seed
        ),
        "comparison_models": {},
        "paired_bca": {},
        "subgroups": {},
        "selection_note": (
            "The final configuration was chosen on training CV; this is its first and only val "
            "score. Val informed the step-2 baselines' reporting (not their settings) earlier."
        ),
    }
    per_target = [masked.per_target.assign(model="final")]
    for name, probabilities in comparisons.items():
        evaluation = evaluate_predictions(labels, probabilities, codes, prior=prior)
        summary["comparison_models"][name] = {
            "masked": evaluation.summary,
            "at_risk": evaluate_predictions(
                labels, probabilities, codes, prior=prior, at_risk=True
            ).summary,
        }
        summary["paired_bca"][f"final_minus_{name}"] = bootstrap_comparison(
            labels, final, codes, other=probabilities, prior=prior, n_boot=n_boot, seed=seed
        )
        per_target.append(evaluation.per_target.assign(model=name))
    anchor_year = inputs.cohort["anchor_date"].dt.year
    groups = {
        f"anchor_before_{clock_year}": labels.index[(anchor_year < clock_year).to_numpy()],
        f"anchor_from_{clock_year}": labels.index[(anchor_year >= clock_year).to_numpy()],
    }
    if modalities is not None:
        groups["modality_input"] = labels.index.intersection(
            modality_patients(modalities, inputs.cohort)
        )
    for group, rows in groups.items():
        entry: dict[str, Any] = {"n_patients": len(rows)}
        if len(rows) >= 2:
            for name, probabilities in models.items():
                entry[name] = evaluate_predictions(
                    labels.loc[rows], probabilities.loc[rows], codes, prior=prior.loc[rows]
                ).summary
        summary["subgroups"][group] = entry
    return summary, final, pd.concat(per_target), transformer


# ---------------------------------------------------------------------- command
def _wandb_factory(
    mode: str, entity: str | None, out_dir: Path, bundle_id: str
) -> Callable[[dict[str, Any]], Any] | None:
    if mode == "disabled":
        return None
    import wandb

    def start(member: dict[str, Any]) -> Any:
        return wandb.init(
            project="ehr-forecast",
            entity=entity,
            name=f"val-curve-{member['config']}-fold{member['fold']}",
            group=f"val-bundle-{bundle_id}",
            job_type="val-curve",
            dir=str(out_dir),
            config={**member, "bundle_id": bundle_id},
            mode=mode,  # type: ignore[arg-type]
        )

    return start


def run(
    root: Path,
    config_path: Path,
    dataset_dir: Path,
    models_dir: Path,
    manifest_path: Path,
    reports_dir: Path,
    *,
    device: torch.device,
    wandb_mode: str = "online",
    wandb_entity: str | None = None,
    replacement_reason: str | None = None,
    check_tree: bool = True,
) -> dict[str, Any]:
    final = load_final(config_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["final_config_sha256"] != object_hash(final):
        raise ValueError("configs/final.toml changed after the freeze")
    if check_tree:
        changes = tracked_changes(root)
        if changes:
            raise ValueError("tracked working tree has changes: " + "; ".join(changes))
    verify_manifest(root, manifest, dataset_dir, models_dir)
    bundle_path = reports_dir / "bundle.json"
    if bundle_path.exists():
        previous = json.loads(bundle_path.read_text())
        if replacement_reason is None:
            raise ValueError(
                f"{bundle_path} exists with status {previous.get('status')!r}; "
                "a replacement needs --replacement with the documented reason"
            )
        attempts = reports_dir / "attempts" / str(previous.get("bundle_id", "unknown"))
        attempts.mkdir(parents=True, exist_ok=True)
        for item in list(reports_dir.iterdir()):
            if item.name != "attempts":
                item.replace(attempts / item.name)
        log.warning("Previous bundle moved to %s; reason: %s", attempts, replacement_reason)
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation = final["evaluation"]
    bundle_id = object_hash({"manifest": file_hash(manifest_path), "time": time.time()})[:12]
    record: dict[str, Any] = {
        "status": "started",
        "bundle_id": bundle_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "score_commit": git_commit(root),
        "freeze_commit": manifest.get("freeze_commit"),
        "manifest_sha256": file_hash(manifest_path),
        "final_config_sha256": manifest["final_config_sha256"],
        "replacement_reason": replacement_reason,
        "device": str(device),
        "wandb_mode": wandb_mode,
        "members": [m["run_dir"] for m in manifest["members"]],
    }
    write_json(bundle_path, record)
    try:
        comparisons_cfg = final["comparisons"]
        comparison_files = {
            "logreg_codes": root / str(comparisons_cfg["logreg_codes"]),
            "incidence_at_risk": root / str(comparisons_cfg["incidence_at_risk"]),
        }
        for name in comparisons_cfg.get("lightgbm_feature_sets", []):
            comparison_files[f"lightgbm_{name}"] = (
                root / str(comparisons_cfg["lightgbm_dir"]) / f"val_predictions_lightgbm_{name}.csv"
            )
        splits = load_splits(dataset_dir)
        patients = pd.Index(splits.index[splits.eq("val")], name="patient_id")
        inputs = load_prediction_inputs(dataset_dir, patients)
        if inputs.codes != manifest["targets"]:
            raise ValueError("target order differs from the manifest")
        labels = load_val_labels(dataset_dir, inputs.cohort.index, inputs.codes)
        comparisons = {
            name: read_comparison(path, labels.index, inputs.codes)
            for name, path in comparison_files.items()
        }
        fold_models = [
            load_fold_model(models_dir / m["run_dir"], str(final["checkpoint"]))
            for m in manifest["members"]
        ]
        if len({modality_settings(fm.config) for fm in fold_models}) > 1:
            raise ValueError("the fold models use different modalities")
        modalities = config_modalities(dataset_dir, fold_models[0].config)
        summary, final_probabilities, per_target, transformer = score_bundle(
            inputs,
            labels,
            fold_models,
            comparisons,
            modalities,
            clock_year=int(evaluation["clock_year"]),
            n_boot=int(evaluation["n_boot"]),
            seed=int(evaluation["bootstrap_seed"]),
            device=device,
            stack=final.get("stack"),
            root=root,
        )
        final_probabilities.to_csv(
            reports_dir / "val_predictions_final.csv", index_label="patient_id"
        )
        transformer.to_csv(
            reports_dir / "val_predictions_transformer.csv", index_label="patient_id"
        )
        per_target.to_csv(reports_dir / "val_per_target.csv")
        summary["bundle_id"] = bundle_id
        write_json(reports_dir / "val_summary.json", summary)
        if bool(evaluation.get("curves", True)):
            cache = SampleCache(inputs, modalities)
            curves = val_curves(
                manifest["members"],
                models_dir,
                labels,
                cache,
                device,
                64,
                _wandb_factory(wandb_mode, wandb_entity, reports_dir, bundle_id),
            )
            curves.to_csv(reports_dir / "val_curves.csv", index=False)
            best = curves[curves["is_best"]]
            record["curves"] = {
                "checkpoints_scored": len(curves),
                "mean_best_epoch_val_macro_auroc": float(best["val_macro_auroc"].mean()),
                "mean_best_epoch_val_mAP": float(best["val_mAP"].mean()),
            }
        if wandb_mode != "disabled":
            import wandb

            final_run = wandb.init(
                project="ehr-forecast",
                entity=wandb_entity,
                name="val-bundle-final",
                group=f"val-bundle-{bundle_id}",
                job_type="val-bundle",
                dir=str(reports_dir),
                config={
                    **record,
                    "ensemble": manifest["ensemble"],
                    "submission": summary["submission"],
                },
                mode=wandb_mode,  # type: ignore[arg-type]
            )
            final_run.log(
                {
                    "val_macro_auroc": summary["final"]["masked"]["macro_auroc"],
                    "val_mAP": summary["final"]["masked"]["mAP"],
                    "val_macro_brier": summary["final"]["masked"]["macro_brier"],
                    "val_at_risk_macro_auroc": summary["final"]["at_risk"]["macro_auroc"],
                }
            )
            final_run.log({"per_target": wandb.Table(dataframe=per_target.reset_index())})
            record["wandb_final_run"] = {"id": final_run.id, "url": final_run.url}
            final_run.finish()
        record.update(
            status="completed",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            val_macro_auroc=summary["final"]["masked"]["macro_auroc"],
            val_mAP=summary["final"]["masked"]["mAP"],
        )
    except BaseException as error:
        record.update(
            status="failed",
            failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            error=f"{type(error).__name__}: {error}",
        )
        write_json(bundle_path, record)
        raise
    write_json(bundle_path, record)
    log.info(
        "Val bundle %s completed: macro AUROC %.4f, mAP %.4f",
        bundle_id,
        record["val_macro_auroc"],
        record["val_mAP"],
    )
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "score"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("."))
        p.add_argument("--config", type=Path, default=Path("configs/final.toml"))
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--models", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--log-file", type=Path, default=None)
    score = sub.choices["score"]
    score.add_argument("--reports", type=Path, required=True)
    score.add_argument("--device", default="cpu")
    score.add_argument("--wandb", choices=["online", "offline", "disabled"], default="online")
    score.add_argument("--wandb-entity", default=None)
    score.add_argument("--replacement", default=None, help="reason for a replacement bundle")
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    root = args.root.resolve()
    final = load_final(args.config)
    if args.command == "freeze":
        configs = {m: load_config(root / m) for m in final["members"]}
        manifest = build_manifest(root, final, configs, args.dataset, args.models)
        write_json(args.manifest, manifest)
        log.info(
            "Froze %d members and %d hashes to %s",
            len(manifest["members"]),
            sum(len(manifest[k]) for k in ("model_hashes", "dataset_hashes", "code_hashes")),
            args.manifest,
        )
        return
    run(
        root,
        args.config,
        args.dataset,
        args.models,
        args.manifest,
        args.reports,
        device=torch.device(args.device),
        wandb_mode=args.wandb,
        wandb_entity=args.wandb_entity,
        replacement_reason=args.replacement,
    )


if __name__ == "__main__":
    main()
