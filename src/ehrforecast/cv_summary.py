"""P4c: training-CV results for the transformer methods and the section 5.3 method selection.

Reads each fold's ``pretrain/run.json`` and ``<method>/`` outputs. Writes, to
``reports/transformer/<config name>/``:

* ``fold_results.csv``: best epoch and held-out masked metrics per fold and method.
* ``oof_per_target.csv``: pooled out-of-fold per-target metrics per method.
* ``cv_summary.json``: mean fold metrics, pooled out-of-fold metrics, paired patient BCa
  differences, and the method selection (only when all five folds are complete).

All numbers are training-CV development evidence. Official val is not read.

Usage::

    python -m ehrforecast.cv_summary --config configs/transformer.toml --dataset data/processed \
        --models models/transformer --reports reports/transformer
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ehrforecast.contracts import load_training_data
from ehrforecast.evaluation import bootstrap_comparison, evaluate_predictions
from ehrforecast.folds import N_FOLDS, fold_patients, load_folds
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import write_json
from ehrforecast.training import fold_dir, load_config

log = logging.getLogger("ehrforecast.cv_summary")

METHODS = ("S", "H", "F")
TRAINING_WORK = {"S": 1, "H": 2, "F": 3}  # S: no pretraining; H: pretraining + head; F: both, full
MARGIN = 0.005


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def fold_table(models_dir: Path, config: dict[str, Any], folds: list[int]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        directory = fold_dir(models_dir, config, fold)
        pretrain = _read(directory / "pretrain" / "run.json") or {}
        for method in METHODS:
            run = _read(directory / method / "run.json")
            if run is None or run.get("status") != "completed":
                continue
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "best_epoch": run["best_epoch"],
                    "epochs_completed": run["epochs_completed"],
                    "stopped_early": run["stopped_early"],
                    "macro_auroc": run["best_held_macro_auroc"],
                    "mAP": run["best_held_mAP"],
                    "trainable_parameters": run["trainable_parameters"],
                    "parameters": run["parameters"],
                    "finetune_seconds": run["seconds_this_session"],
                    "pretrain_best_epoch": pretrain.get("best_epoch") if method != "S" else None,
                    "pretrain_held_loss": pretrain.get("best_held_loss") if method != "S" else None,
                }
            )
    return pd.DataFrame(rows)


def pretraining_table(models_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """P4a speed and memory record, and the held-out next-token loss of each fold."""
    keys = (
        "status",
        "best_epoch",
        "best_held_loss",
        "epochs_completed",
        "stopped_early",
        "seconds_this_session",
        "target_tokens_per_second",
        "padded_tokens_per_second",
        "peak_memory_gb",
        "device",
        "parameters",
    )
    out = {}
    for fold in range(N_FOLDS):
        run = _read(fold_dir(models_dir, config, fold) / "pretrain" / "run.json")
        if run is not None:
            out[f"fold{fold}"] = {k: run.get(k) for k in keys}
    return out


def out_of_fold(
    models_dir: Path, config: dict[str, Any], method: str, assignments: pd.DataFrame
) -> pd.DataFrame:
    parts = []
    for fold in range(N_FOLDS):
        path = fold_dir(models_dir, config, fold) / method / "held_predictions_best.parquet"
        _, held = fold_patients(assignments, fold)
        frame = pd.read_parquet(path)
        if set(frame.index) != set(held):
            raise ValueError(f"{path} does not hold exactly fold {fold}'s held-out patients")
        parts.append(frame)
    return pd.concat(parts)


def select_method(table: pd.DataFrame) -> dict[str, Any]:
    """Section 5.3: S is the reference; the consistency rule and simplification rules follow."""
    by_method = {m: g.set_index("fold").sort_index() for m, g in table.groupby("method")}
    if set(by_method) != set(METHODS) or any(len(g) != N_FOLDS for g in by_method.values()):
        return {"status": "pending", "reason": "all three methods need five completed folds"}
    means = {
        m: (float(g["macro_auroc"].mean()), float(g["mAP"].mean())) for m, g in by_method.items()
    }
    reference = by_method["S"]
    checks = {}
    for method in ("H", "F"):
        g = by_method[method]
        gain = means[method][0] - means["S"][0]
        folds_better = int((g["macro_auroc"] > reference["macro_auroc"]).sum())
        map_loss = means["S"][1] - means[method][1]
        checks[method] = {
            "mean_auroc_gain_vs_S": gain,
            "folds_with_higher_auroc": folds_better,
            "mean_map_loss_vs_S": map_loss,
            "passes_consistency_rule": gain >= MARGIN and folds_better >= 4 and map_loss <= MARGIN,
        }
    confirmed = ["S"] + [m for m, c in checks.items() if c["passes_consistency_rule"]]
    best = max(confirmed, key=lambda m: means[m])  # mean AUROC, then mean mAP
    close = [
        m
        for m in METHODS
        if means[m][0] >= means[best][0] - MARGIN and means[m][1] >= means[best][1] - MARGIN
    ]
    params = {m: int(by_method[m]["trainable_parameters"].iloc[0]) for m in METHODS}
    total = {m: int(by_method[m]["parameters"].iloc[0]) for m in METHODS}
    selected = min(close, key=lambda m: (params[m], total[m], TRAINING_WORK[m]))
    return {
        "status": "decided",
        "rule": "model plan section 5.3; margin 0.005; four of five folds",
        "mean_fold_metrics": {m: {"macro_auroc": a, "mAP": p} for m, (a, p) in means.items()},
        "consistency_checks_against_S": checks,
        "best_confirmed": best,
        "within_margin_of_best": close,
        "tie_order": (
            "fewer supervised parameters, then fewer total parameters, then less training work"
        ),
        "supervised_parameters": params,
        "selected_method": selected,
    }


def run(
    dataset_dir: Path, config_path: Path, models_dir: Path, reports_dir: Path, n_boot: int
) -> dict[str, Any]:
    config = load_config(config_path)
    out_dir = reports_dir / str(config["name"])
    data = load_training_data(dataset_dir)
    labels, prior, codes = data.labels, data.inputs.prior, data.inputs.codes
    assignments = load_folds(dataset_dir)
    table = fold_table(models_dir, config, list(range(N_FOLDS)))
    if table.empty:
        raise ValueError("no completed supervised runs were found")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "fold_results.csv", index=False)
    summary: dict[str, Any] = {
        "scope": "training-split cross-validation; official val not read",
        "config": config["name"],
        "folds_completed": {m: sorted(g["fold"].tolist()) for m, g in table.groupby("method")},
        "mean_fold_metrics": {
            m: {
                "macro_auroc": float(g["macro_auroc"].mean()),
                "mAP": float(g["mAP"].mean()),
                "folds": len(g),
            }
            for m, g in table.groupby("method")
        },
        "selection": select_method(table),
        "pretraining": pretraining_table(models_dir, config),
    }
    complete = [str(m) for m, g in table.groupby("method") if len(g) == N_FOLDS]
    if complete:
        oof = {
            m: out_of_fold(models_dir, config, m, assignments).loc[labels.index] for m in complete
        }
        reference = pd.read_parquet(dataset_dir / "p3a_logreg_oof.parquet").loc[labels.index]
        oof["logreg_historical"] = reference
        per_target = []
        summary["pooled_oof"] = {}
        for name, probabilities in oof.items():
            masked = evaluate_predictions(labels, probabilities, codes, prior=prior)
            at_risk = evaluate_predictions(labels, probabilities, codes, prior=prior, at_risk=True)
            summary["pooled_oof"][name] = {"masked": masked.summary, "at_risk": at_risk.summary}
            per_target.append(masked.per_target.assign(model=name))
        pd.concat(per_target).to_csv(out_dir / "oof_per_target.csv")
        summary["paired_bca_oof"] = {
            f"{m}_minus_logreg_historical": bootstrap_comparison(
                labels, oof[m], codes, other=reference, prior=prior, n_boot=n_boot
            )
            for m in complete
        }
        for a, b in (("F", "S"), ("H", "S"), ("F", "H")):
            if a in complete and b in complete:
                summary["paired_bca_oof"][f"{a}_minus_{b}"] = bootstrap_comparison(
                    labels, oof[a], codes, other=oof[b], prior=prior, n_boot=n_boot
                )
        summary["reference_note"] = (
            "logreg_historical is the step-2 code-count configuration (C=0.003) reproduced in "
            "P3a, scored with the common mask. The revised at-risk baselines run in P4c."
        )
    write_json(out_dir / "cv_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    summary = run(args.dataset, args.config, args.models, args.reports, args.n_boot)
    log.info(
        "Selection: %s", summary["selection"].get("selected_method", summary["selection"]["status"])
    )
    log.info("Mean fold metrics: %s", json.dumps(summary["mean_fold_metrics"]))


if __name__ == "__main__":
    main()
