"""Build and verify the public model package for the Hugging Face Hub (model plan section 17).

The package mirrors the ``models/transformer`` layout that ``ehrforecast.predict`` reads, so
the same prediction command works from the downloaded package without any training cache::

    python -m ehrforecast.hf_package build --manifest reports/final/manifest.json \
        --models models --dataset data/processed --out models/hf --repo Reaper8202/ehr-forecast
    python -m ehrforecast.hf_package verify --package models/hf --dataset data/processed \
        --predictions predictions.csv

``build`` copies each ensemble member's ``best.pt`` and ``run.json``, every fold tokenizer, the
member configurations, and ``package.json`` (target order, prior-mask rule, input schema, file
hashes). It refuses any file whose hash differs from the frozen manifest. ``verify`` checks
every hash, then predicts the test split from the package on CPU and compares the result with
``predictions.csv`` (G4: clean reload through the final prediction command).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import (
    INPUT_SCHEMA_VERSION,
    PredictionInputs,
    load_prediction_inputs,
    load_splits,
)
from ehrforecast.logging_utils import setup_logging
from ehrforecast.predict import load_fold_model, predict_inputs
from ehrforecast.provenance import file_hash, write_json
from ehrforecast.training import config_modalities, modality_settings

log = logging.getLogger("ehrforecast.hf_package")

PACKAGE_VERSION = "ehr-forecast-package-v1"
TOLERANCE = 1e-6


def build_package(
    manifest_path: Path,
    models_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    config_paths: list[Path],
    repo: str | None,
    val_summary: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; remove it to rebuild the package")
    out_dir.mkdir(parents=True)
    hashes: dict[str, str] = {}
    members = []
    for m in manifest["members"]:
        files = [
            Path(m["run_dir"]) / "best.pt",
            Path(m["run_dir"]) / "run.json",
            Path(m["tokenizer"]),
        ]
        for relative in files:
            expected = manifest["model_hashes"].get(str(relative))
            if expected is not None and file_hash(models_dir / relative) != expected:
                raise ValueError(f"{relative} differs from the frozen manifest")
            target = out_dir / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(models_dir / relative, target)
                hashes[str(relative)] = file_hash(target)
        members.append(
            {
                "config": m["config"],
                "fold": m["fold"],
                "run_dir": m["run_dir"],
                "tokenizer": m["tokenizer"],
                "best_epoch": m["best_epoch"],
                "training_run_id": m["training_run_id"],
            }
        )
    (out_dir / "configs").mkdir()
    for path in config_paths:
        shutil.copyfile(path, out_dir / "configs" / path.name)
        hashes[f"configs/{path.name}"] = file_hash(out_dir / "configs" / path.name)
    events_manifest = json.loads((dataset_dir / "events.manifest.json").read_text())
    fm = load_fold_model(out_dir / members[0]["run_dir"])
    genomics, imaging, image_size = modality_settings(fm.config)
    package: dict[str, Any] = {
        "version": PACKAGE_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": repo,
        "manifest_sha256": file_hash(manifest_path),
        "freeze_commit": manifest.get("freeze_commit"),
        "members": members,
        "ensemble": manifest["ensemble"],
        "prior_mask": manifest["prior_mask"],
        "targets": manifest["targets"],
        "input_schema": INPUT_SCHEMA_VERSION,
        "event_schema": events_manifest.get("event_schema"),
        "history_rule": events_manifest.get("history_rule"),
        "modalities": {"genomics": genomics, "imaging": imaging, "image_size": image_size},
        "prediction_command": (
            "python -m ehrforecast.predict --dataset data/processed --split test "
            "--run-dir <package>/"
            + " <package>/".join(m["run_dir"] for m in members)
            + " --out predictions.csv"
        ),
        "file_hashes": hashes,
    }
    if val_summary is not None and val_summary.is_file():
        summary = json.loads(val_summary.read_text())
        package["val_result"] = {
            "n_patients": summary["n_patients"],
            "macro_auroc": summary["final"]["masked"]["macro_auroc"],
            "mAP": summary["final"]["masked"]["mAP"],
            "bundle_id": summary.get("bundle_id"),
        }
    write_json(out_dir / "package.json", package)
    log.info("Built package with %d members and %d files at %s", len(members), len(hashes), out_dir)
    return package


def verify_files(package_dir: Path) -> dict[str, Any]:
    package = json.loads((package_dir / "package.json").read_text())
    if package.get("version") != PACKAGE_VERSION:
        raise ValueError("unknown package version")
    bad = [r for r, h in package["file_hashes"].items() if file_hash(package_dir / r) != h]
    if bad:
        raise ValueError("package file hash mismatch: " + ", ".join(sorted(bad)))
    return package


def package_predictions(
    package_dir: Path, inputs: PredictionInputs, modalities: Any, device: torch.device
) -> pd.DataFrame:
    package = verify_files(package_dir)
    fold_models = [load_fold_model(package_dir / m["run_dir"]) for m in package["members"]]
    if len({modality_settings(fm.config) for fm in fold_models}) > 1:
        raise ValueError("the package members use different modalities")
    if inputs.codes != package["targets"]:
        raise ValueError("target order differs from the package")
    return predict_inputs(fold_models, inputs, device, modalities=modalities)


def compare_predictions(ours: pd.DataFrame, submitted: pd.DataFrame) -> dict[str, Any]:
    submitted = submitted.reindex(index=ours.index, columns=ours.columns)
    if submitted.isna().any().any():
        raise ValueError("predictions.csv lacks patients or targets of the package output")
    difference = float(np.abs(ours.to_numpy() - submitted.to_numpy()).max())
    return {"max_abs_difference": difference, "agrees": difference <= TOLERANCE}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--models", type=Path, required=True)
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--configs", type=Path, nargs="*", default=[])
    build.add_argument("--repo", default=None)
    build.add_argument("--val-summary", type=Path, default=None)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--predictions", type=Path, required=True)
    verify.add_argument("--device", default="cpu")
    verify.add_argument("--out", type=Path, default=None, help="where to write the record")
    for p in (build, verify):
        p.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    if args.command == "build":
        build_package(
            args.manifest,
            args.models,
            args.dataset,
            args.out,
            args.configs,
            args.repo,
            args.val_summary,
        )
        return
    splits = load_splits(args.dataset)
    inputs = load_prediction_inputs(args.dataset, pd.Index(splits.index[splits.eq("test")]))
    package = verify_files(args.package)
    fm = load_fold_model(args.package / package["members"][0]["run_dir"])
    ours = package_predictions(
        args.package, inputs, config_modalities(args.dataset, fm.config), torch.device(args.device)
    )
    submitted = pd.read_csv(args.predictions, dtype={"patient_id": str}).set_index("patient_id")
    submitted.columns = [str(c) for c in submitted.columns]
    result = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package": str(args.package),
        "device": args.device,
        "torch": torch.__version__,
        "n_patients": len(ours),
        "n_targets": ours.shape[1],
        **compare_predictions(ours, submitted),
    }
    if args.out is not None:
        write_json(args.out, result)
    log.info(
        "Package reload: max |difference| %.2e, agrees=%s",
        result["max_abs_difference"],
        result["agrees"],
    )
    if not result["agrees"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
