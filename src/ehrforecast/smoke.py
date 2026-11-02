"""G2 pipeline smoke test: one fold, real training data, tiny settings, temporary outputs.

Steps:

1. Fit the fold-0 tokenizer and histories.
2. Self-pretrain from random weights. Stop after one epoch, then resume to the end.
3. Train H and F from that checkpoint, and S from random weights.
4. In a fresh process, reload F and predict 40 probabilities for the held-out training patients
   (mock test patients) through the final prediction command. No label file is read there.
5. Score those saved predictions in a separate step, joined by patient ID and target code.
6. If MPS is available, compare CPU and MPS predictions of the same weights.

This checks execution only. Its scores are not performance results, and official val is unused.

Usage::

    python -m ehrforecast.smoke --dataset data/processed --config configs/smoke.toml \
        --work models/smoke --report reports/transformer/pipeline_smoke.json
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ehrforecast import representation
from ehrforecast.evaluation import evaluate_predictions
from ehrforecast.finetune import predict_samples, run_finetuning
from ehrforecast.logging_utils import setup_logging
from ehrforecast.predict import load_fold_model
from ehrforecast.pretrain import run_pretraining
from ehrforecast.provenance import write_json
from ehrforecast.training import fold_dir, load_config, load_fold, sequence_config

log = logging.getLogger("ehrforecast.smoke")


def run(dataset_dir: Path, config_path: Path, work_dir: Path, report_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    models = work_dir / "models"
    config = load_config(config_path)
    representation.run(dataset_dir, config_path, models, work_dir / "reports", [0])
    fold = load_fold(dataset_dir, models, config, 0)
    directory = fold_dir(models, config, 0)
    cpu = torch.device("cpu")

    paused = run_pretraining(config, fold, directory / "pretrain", max_session_epochs=1, device=cpu)
    resumed = run_pretraining(config, fold, directory / "pretrain", resume=True, device=cpu)
    if paused["status"] != "paused" or resumed["status"] != "completed":
        raise AssertionError("pretraining did not pause and resume as expected")
    parent = directory / "pretrain" / "best.pt"
    methods = {
        m: run_finetuning(
            config, fold, m, directory / m, parent=None if m == "S" else parent, device=cpu
        )
        for m in ("H", "F", "S")
    }

    predictions_path = work_dir / "mock_test_predictions.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ehrforecast.predict",
            "--dataset",
            str(dataset_dir),
            "--run-dir",
            str(directory / "F"),
            "--heldout-fold",
            "0",
            "--out",
            str(predictions_path),
        ],
        check=True,
    )
    predictions = pd.read_csv(predictions_path, dtype={"patient_id": str}).set_index("patient_id")
    codes = fold.data.inputs.codes
    if list(predictions.columns) != codes or predictions.index.has_duplicates:
        raise AssertionError("prediction file has wrong target columns or duplicate patients")
    values = predictions.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError("prediction file contains invalid probabilities")
    held = predictions.index
    scored = evaluate_predictions(
        fold.data.labels.loc[held], predictions, codes, prior=fold.data.inputs.prior.loc[held]
    ).summary
    prior_cells = fold.data.inputs.prior.loc[held, codes].to_numpy() == 1
    if (values[prior_cells] != 0).any():
        raise AssertionError("prior-condition cells are not zero")

    device_check: dict[str, Any] = {"mps_available": torch.backends.mps.is_available()}
    if torch.backends.mps.is_available():
        fm = load_fold_model(directory / "F")
        sequence = sequence_config(config)
        samples = [fold.histories.prediction_sample(int(r), sequence) for r in fold.held_rows]
        on_cpu = predict_samples(fm.model.to(cpu), samples, cpu, 32)
        on_mps = predict_samples(fm.model.to(torch.device("mps")), samples, torch.device("mps"), 32)
        device_check["max_abs_probability_difference"] = float(np.abs(on_cpu - on_mps).max())
        device_check["tolerance"] = 1e-4
        if device_check["max_abs_probability_difference"] > 1e-4:
            raise AssertionError("CPU and MPS predictions differ beyond the declared tolerance")

    report = {
        "status": "passed",
        "gate": "G2 execution check on training data; not a performance result",
        "config": str(config_path),
        "official_val_used": False,
        "pretrain_paused_status": paused["status"],
        "pretrain_resumed": resumed["status"],
        "pretrain_epochs": resumed["epochs_completed"],
        "methods": {
            m: {k: r[k] for k in ("status", "best_epoch", "trainable_parameters")}
            for m, r in methods.items()
        },
        "fixed_body_unchanged": methods["H"]["body_unchanged"],
        "fresh_process_prediction": {
            "patients": len(predictions),
            "targets": len(codes),
            "prior_cells_zero": True,
            "separately_scored_masked_macro_auroc": scored["macro_auroc"],
        },
        "device_check": device_check,
        "seconds": time.perf_counter() - started,
    }
    write_json(report_path, report)
    log.info("Pipeline smoke test passed in %.0f s", report["seconds"])
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(args.dataset, args.config, args.work, args.report)


if __name__ == "__main__":
    main()
