"""Build and verify the public model package for the Hugging Face Hub (model plan section 17).

The package holds everything the submission needs: the transformer members in the
``models/transformer`` layout that ``ehrforecast.predict`` reads, the final LightGBM boosters
with their self-contained feature spec, the Platt weights, the stack recipe, and
``package.json`` (target order, prior-mask rule, input schema, every file hash)::

    python -m ehrforecast.hf_package build --final configs/final.toml \\
        --manifest reports/final/manifest.json --models models --dataset data/processed \\
        --out models/hf --repo Reaper8202/ehr-forecast
    python -m ehrforecast.hf_package verify --package models/hf --dataset data/processed \\
        --predictions predictions.csv

``build`` refuses any file whose hash differs from the frozen manifest. ``verify`` checks every
hash, then rebuilds the test probabilities from the package alone on CPU (transformer in this
process, LightGBM in its own process, then the blend and the scaling) and compares them with
``predictions.csv`` (G4: clean reload through the final prediction path).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
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
from ehrforecast.submission import build as build_submission
from ehrforecast.submission import read_predictions
from ehrforecast.training import config_modalities, modality_settings

log = logging.getLogger("ehrforecast.hf_package")

PACKAGE_VERSION = "ehr-forecast-package-v2"
TOLERANCE = 1e-6


def _copy(source: Path, target: Path, hashes: dict[str, str], relative: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    hashes[relative] = file_hash(target)


def build_package(
    manifest_path: Path,
    models_dir: Path,
    dataset_dir: Path,
    out_dir: Path,
    config_paths: list[Path],
    repo: str | None,
    val_summary: Path | None = None,
    final: dict[str, Any] | None = None,
    root: Path | None = None,
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
            if not (out_dir / relative).exists():
                _copy(models_dir / relative, out_dir / relative, hashes, str(relative))
        members.append({k: m[k] for k in ("config", "fold", "run_dir", "tokenizer", "best_epoch")})
        members[-1]["training_run_id"] = m["training_run_id"]
    (out_dir / "configs").mkdir()
    for path in config_paths:
        _copy(path, out_dir / "configs" / path.name, hashes, f"configs/{path.name}")
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
        "stack": None,
        "lightgbm": {},
    }
    stack = (final or {}).get("stack")
    if stack:
        if root is None:
            raise ValueError("a stack needs the project root to copy its weight files")
        comparisons = (final or {})["comparisons"]
        lightgbm_models = root / str(comparisons["lightgbm_models"])
        for name in stack["bases"]:
            if not str(name).startswith("lightgbm_"):
                continue
            feature_set = str(name)[len("lightgbm_") :]
            source_dir = lightgbm_models / feature_set
            copied = 0
            for source in sorted(q for q in source_dir.rglob("*") if q.is_file()):
                target_name = f"lightgbm/{feature_set}/{source.relative_to(source_dir)}"
                expected = manifest["comparison_hashes"].get(str(source.relative_to(root)))
                if expected is not None and file_hash(source) != expected:
                    raise ValueError(f"{source} differs from the frozen manifest")
                _copy(source, out_dir / target_name, hashes, target_name)
                copied += 1
            package["lightgbm"][str(name)] = {"dir": f"lightgbm/{feature_set}", "files": copied}
        recipe = {"bases": list(stack["bases"]), "stacker": str(stack["stacker"])}
        if stack.get("platt_weights"):
            source = root / str(stack["platt_weights"])
            expected = manifest["comparison_hashes"].get(str(stack["platt_weights"]))
            if expected is not None and file_hash(source) != expected:
                raise ValueError(f"{source} differs from the frozen manifest")
            _copy(
                source,
                out_dir / "calibration" / "platt_weights.json",
                hashes,
                "calibration/platt_weights.json",
            )
            recipe["platt_weights"] = "calibration/platt_weights.json"
        package["stack"] = recipe
    package["prediction_command"] = (
        "make dataset && make modalities, then python -m ehrforecast.hf_package verify "
        "--package <package> --dataset data/processed --predictions predictions.csv"
    )
    package["file_hashes"] = hashes
    if val_summary is not None and val_summary.is_file():
        summary = json.loads(val_summary.read_text())
        package["val_result"] = {
            "n_patients": summary["n_patients"],
            "submission": summary.get("submission"),
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


def transformer_predictions(
    package_dir: Path, inputs: PredictionInputs, modalities: Any, device: torch.device
) -> pd.DataFrame:
    package = verify_files(package_dir)
    fold_models = [load_fold_model(package_dir / m["run_dir"]) for m in package["members"]]
    if len({modality_settings(fm.config) for fm in fold_models}) > 1:
        raise ValueError("the package members use different modalities")
    if inputs.codes != package["targets"]:
        raise ValueError("target order differs from the package")
    return predict_inputs(fold_models, inputs, device, modalities=modalities)


def lightgbm_predictions(
    package_dir: Path, package: dict[str, Any], dataset_dir: Path, split: str, work: Path
) -> dict[str, pd.DataFrame]:
    """Each LightGBM base through its own process (LightGBM and PyTorch must not share one)."""
    out: dict[str, pd.DataFrame] = {}
    for name, entry in package["lightgbm"].items():
        path = work / f"{split}_{name}.csv"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ehrforecast.lightgbm_final",
                "predict",
                "--dataset",
                str(dataset_dir),
                "--models",
                str(package_dir / entry["dir"]),
                "--split",
                split,
                "--out",
                str(path),
            ],
            check=True,
        )
        out[name] = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
        out[name].columns = [str(c) for c in out[name].columns]
    return out


def package_predictions(
    package_dir: Path,
    inputs: PredictionInputs,
    modalities: Any,
    device: torch.device,
    lightgbm: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The submitted probabilities from the package: transformer, LightGBM bases, blend, scaling."""
    package = verify_files(package_dir)
    transformer = transformer_predictions(package_dir, inputs, modalities, device)
    if not package.get("stack"):
        return transformer
    bases = {"transformer": transformer}
    for name in package["stack"]["bases"]:
        if name == "transformer":
            continue
        if lightgbm is None or name not in lightgbm:
            raise ValueError(f"LightGBM predictions for {name} are required")
        bases[name] = lightgbm[name].reindex(index=inputs.cohort.index, columns=inputs.codes)
        if bases[name].isna().any().any():
            raise ValueError(f"{name} predictions do not cover every patient and target")
    return build_submission(package["stack"], bases, inputs.prior, inputs.codes, package_dir)


def compare_predictions(ours: pd.DataFrame, submitted: pd.DataFrame) -> dict[str, Any]:
    submitted = submitted.reindex(index=ours.index, columns=ours.columns)
    if submitted.isna().any().any():
        raise ValueError("predictions.csv lacks patients or targets of the package output")
    difference = float(np.abs(ours.to_numpy() - submitted.to_numpy()).max())
    return {"max_abs_difference": difference, "agrees": difference <= TOLERANCE}


def main(argv: list[str] | None = None) -> None:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", type=Path, default=Path("."))
    build.add_argument("--final", type=Path, default=None, help="configs/final.toml (stack)")
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
        final = tomllib.loads(args.final.read_text()) if args.final else None
        build_package(
            args.manifest,
            args.models,
            args.dataset,
            args.out,
            args.configs,
            args.repo,
            args.val_summary,
            final,
            args.root.resolve(),
        )
        return
    splits = load_splits(args.dataset)
    inputs = load_prediction_inputs(args.dataset, pd.Index(splits.index[splits.eq("test")]))
    package = verify_files(args.package)
    fm = load_fold_model(args.package / package["members"][0]["run_dir"])
    with tempfile.TemporaryDirectory() as work:
        lightgbm = lightgbm_predictions(args.package, package, args.dataset, "test", Path(work))
        ours = package_predictions(
            args.package,
            inputs,
            config_modalities(args.dataset, fm.config),
            torch.device(args.device),
            lightgbm,
        )
    submitted = read_predictions(args.predictions, ours.index, inputs.codes)
    result = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package": str(args.package),
        "device": args.device,
        "torch": torch.__version__,
        "submission": package.get("stack") or "transformer ensemble",
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
