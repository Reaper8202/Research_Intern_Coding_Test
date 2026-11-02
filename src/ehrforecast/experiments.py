"""Tuning-round runner and experiment register (model plan section 5.3, ``configs/budget.toml``).

``run``: for each experiment in budget order and each listed fold, run self-pretraining (unless
the configuration sets ``pretrain_from``) and then the budget's supervised method. Each stage runs
in a fresh process. Completed runs are skipped, and an interrupted run resumes from its last
completed epoch, so the command can be repeated after any stop.

``summarize``: compare every experiment with the incumbent fold by fold and apply the section 5.3
rule. An experiment may name its own ``reference`` configuration; the default is the incumbent.
With the E0 seeds present, the rule is also checked against the per-fold mean over seeds,
and the seed spread is reported. Writes ``experiments.csv`` and ``experiments.json``. All numbers
are training-CV; official val is not read.

Usage::

    python -m ehrforecast.experiments run --budget configs/budget.toml \
        --dataset data/processed --models models/transformer --logs logs/experiments
    python -m ehrforecast.experiments summarize --budget configs/budget.toml \
        --models models/transformer --reports reports/transformer
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from ehrforecast.cv_summary import MARGIN, fold_table
from ehrforecast.folds import N_FOLDS
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import write_json
from ehrforecast.training import fold_dir, load_config, pretrain_dir

log = logging.getLogger("ehrforecast.experiments")

SEED_PREFIX = "E0-"


def load_budget(path: Path) -> dict[str, Any]:
    budget = tomllib.loads(path.read_text())
    ids = [e["id"] for e in budget["experiment"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} repeats an experiment id")
    return budget


def run_status(run_dir: Path) -> str:
    """``completed``, ``partial`` (a checkpoint exists), or ``missing``."""
    record = run_dir / "run.json"
    if record.is_file() and json.loads(record.read_text()).get("status") == "completed":
        return "completed"
    return "partial" if (run_dir / "last.pt").is_file() else "missing"


def run_stage(command: list[str], run_dir: Path) -> None:
    status = run_status(run_dir)
    if status == "completed":
        log.info("Skipping %s (completed)", run_dir)
        return
    log.info("%s %s", "Resuming" if status == "partial" else "Starting", run_dir)
    subprocess.run([*command, *(["--resume"] if status == "partial" else [])], check=True)


def run(budget_path: Path, dataset_dir: Path, models_dir: Path, logs_dir: Path) -> None:
    budget = load_budget(budget_path)
    method = str(budget["method"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    for experiment in budget["experiment"]:
        config_path = Path(experiment["config"])
        config = load_config(config_path)
        for fold in experiment["folds"]:
            common = [
                "--config",
                str(config_path),
                "--dataset",
                str(dataset_dir),
                "--models",
                str(models_dir),
                "--fold",
                str(fold),
            ]
            stem = f"{experiment['id']}_fold{fold}"
            if "pretrain_from" in config:
                parent = pretrain_dir(models_dir, config, fold) / "pretrain"
                if run_status(parent) != "completed":
                    raise ValueError(f"{experiment['id']} needs the completed pretraining {parent}")
            else:
                run_stage(
                    [
                        sys.executable,
                        "-m",
                        "ehrforecast.pretrain",
                        *common,
                        "--log-file",
                        str(logs_dir / f"{stem}_pretrain.log"),
                    ],
                    fold_dir(models_dir, config, fold) / "pretrain",
                )
            run_stage(
                [
                    sys.executable,
                    "-m",
                    "ehrforecast.finetune",
                    *common,
                    "--method",
                    method,
                    "--log-file",
                    str(logs_dir / f"{stem}_{method}.log"),
                ],
                fold_dir(models_dir, config, fold) / method,
            )
        log.info("Experiment %s finished", experiment["id"])


# ---------------------------------------------------------------------- register
def method_scores(models_dir: Path, config: dict[str, Any], method: str) -> pd.DataFrame:
    """Completed folds of one method: ``macro_auroc`` and ``mAP`` indexed by fold."""
    table = fold_table(models_dir, config, list(range(N_FOLDS)))
    if table.empty:
        return pd.DataFrame(columns=["macro_auroc", "mAP"])
    rows = table.loc[table["method"].eq(method)].set_index("fold").sort_index()
    return rows[["macro_auroc", "mAP"]]


def compare(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Section 5.3 on the shared folds. Fewer than five shared folds give a screening result."""
    folds = candidate.index.intersection(reference.index)
    gain = candidate.loc[folds, "macro_auroc"] - reference.loc[folds, "macro_auroc"]
    map_loss = reference.loc[folds, "mAP"] - candidate.loc[folds, "mAP"]
    out: dict[str, Any] = {
        "folds": [int(f) for f in folds],
        "mean_auroc_gain": float(gain.mean()) if len(folds) else None,
        "folds_with_higher_auroc": int((gain > 0).sum()),
        "mean_map_loss": float(map_loss.mean()) if len(folds) else None,
    }
    clears_margin = bool(len(folds) and gain.mean() >= MARGIN and map_loss.mean() <= MARGIN)
    if len(folds) == N_FOLDS:
        out["passes_rule"] = clears_margin and out["folds_with_higher_auroc"] >= 4
    else:
        out["passes_screen"] = clears_margin
    return out


def summarize(budget_path: Path, models_dir: Path, reports_dir: Path) -> dict[str, Any]:
    budget = load_budget(budget_path)
    method = str(budget["method"])
    incumbent_config = load_config(Path(budget["incumbent"]))
    incumbent = method_scores(models_dir, incumbent_config, method)
    if len(incumbent) != N_FOLDS:
        raise ValueError("the incumbent needs five completed folds")
    scores = {
        e["id"]: method_scores(models_dir, load_config(Path(e["config"])), method)
        for e in budget["experiment"]
    }
    seeds = [incumbent] + [s for k, s in scores.items() if k.startswith(SEED_PREFIX)]
    complete_seeds = [s for s in seeds if len(s) == N_FOLDS]
    seed_mean: pd.DataFrame | None = None
    if len(complete_seeds) > 1:  # per-fold mean over seeds
        seed_mean = pd.concat(complete_seeds).groupby(level=0)[["macro_auroc", "mAP"]].mean()
    rows = [
        {
            "experiment": "incumbent",
            "config": incumbent_config["name"],
            "folds_completed": len(incumbent),
            "mean_macro_auroc": float(incumbent["macro_auroc"].mean()),
            "mean_mAP": float(incumbent["mAP"].mean()),
        }
    ]
    details: dict[str, Any] = {}
    for experiment in budget["experiment"]:
        key, frame = experiment["id"], scores[experiment["id"]]
        name = load_config(Path(experiment["config"]))["name"]
        row: dict[str, Any] = {"experiment": key, "config": name, "folds_completed": len(frame)}
        if len(frame):
            row["mean_macro_auroc"] = float(frame["macro_auroc"].mean())
            row["mean_mAP"] = float(frame["mAP"].mean())
            own = experiment.get("reference")
            reference = (
                incumbent
                if own is None
                else method_scores(models_dir, load_config(Path(own)), method)
            )
            versus = compare(frame, reference)
            row["reference"] = own or budget["incumbent"]
            row.update({f"vs_incumbent_{k}": v for k, v in versus.items() if k != "folds"})
            details[key] = {"vs_incumbent": versus}
            if seed_mean is not None and not key.startswith(SEED_PREFIX):
                versus_seeds = compare(frame, seed_mean)
                row.update(
                    {f"vs_seed_mean_{k}": v for k, v in versus_seeds.items() if k != "folds"}
                )
                details[key]["vs_seed_mean"] = versus_seeds
        rows.append(row)
    table = pd.DataFrame(rows)
    seed_report: dict[str, Any] | None = None
    if len(complete_seeds) > 1:
        aurocs = pd.concat([s["macro_auroc"] for s in complete_seeds], axis=1)
        maps = pd.concat([s["mAP"] for s in complete_seeds], axis=1)
        seed_report = {
            "seeds": len(complete_seeds),
            "mean_auroc_per_seed": [float(v) for v in aurocs.mean()],
            "mean_map_per_seed": [float(v) for v in maps.mean()],
            "per_fold_auroc_range": [float(v) for v in aurocs.max(axis=1) - aurocs.min(axis=1)],
            "note": "spread is descriptive; it is not a significance threshold (section 5.3)",
        }
    out_dir = reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "experiments.csv", index=False)
    summary = {
        "scope": "training-split cross-validation; official val not read",
        "budget": str(budget_path),
        "method": method,
        "rule": "mean AUROC gain >= 0.005, >= 4 of 5 folds higher, mean mAP loss <= 0.005",
        "incumbent": {
            "config": incumbent_config["name"],
            "fold_scores": incumbent.reset_index().to_dict("records"),
        },
        "experiments": details,
        "seed_stability": seed_report,
        "passing": [
            k
            for k, v in details.items()
            if v["vs_incumbent"].get("passes_rule") and not k.startswith(SEED_PREFIX)
        ],
    }
    write_json(out_dir / "experiments.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "summarize"])
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("logs/experiments"))
    parser.add_argument("--reports", type=Path, default=Path("reports/transformer"))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    if args.command == "run":
        run(args.budget, args.dataset, args.models, args.logs)
    else:
        summary = summarize(args.budget, args.models, args.reports)
        log.info("Passing the section 5.3 rule: %s", summary["passing"] or "none")


if __name__ == "__main__":
    main()
