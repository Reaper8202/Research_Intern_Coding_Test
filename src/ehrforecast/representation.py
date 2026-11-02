"""P3b-P3c command: fit fold tokenizers, build patient histories, and write retention reports.

For each fold, the tokenizer sees only the fold's fit patients. All official training patients
are then encoded with that tokenizer. Val and test histories are encoded later, only through the
saved tokenizers (``ehrforecast.predict``).

Outputs per fold, in ``models/transformer/<config name>/fold<k>/``: ``tokenizer.json``,
``histories.npz``, and ``histories.manifest.json``. Reports: ``reports/representation/<config
name>/fold<k>.json`` (M1, M3 to M5, M7, M8) and ``summary.csv``.

Usage::

    python -m ehrforecast.representation --dataset data/processed \
        --config configs/transformer.toml --models models/transformer \
        --reports reports/representation --log-file logs/tokenize.log
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.contracts import load_training_data
from ehrforecast.events import EVENTS_MANIFEST
from ehrforecast.folds import N_FOLDS, fold_patients, load_folds
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import file_hash, write_json
from ehrforecast.sequences import (
    Histories,
    build_histories,
    histories_provenance,
    retention_report,
    save_histories,
)
from ehrforecast.tokenizer import FoldTokenizer, compress_refills
from ehrforecast.training import (
    config_modalities,
    fold_dir,
    load_config,
    modality_settings,
    sequence_config,
)

log = logging.getLogger("ehrforecast.representation")

GAP_BUCKETS = (0.0, 1.0, 2.0, 7.0, 30.0, 365.0, math.inf)


def refill_gap_audit(events: pd.DataFrame) -> dict[str, int]:
    """M1: days from each medication stop to the next start of the same drug (training only).

    Zero-length prescriptions (stop equal to their own start) are counted separately. Their next
    start at or after the stop would often be their own start, which is not a restart.
    """
    med = events.loc[events["source"].astype(str).eq("medications")]
    kind = med["kind"].astype(str)
    stops = med.loc[kind.eq("stop"), ["patient_id", "code", "time", "interval_start"]]
    zero_length = stops["time"].eq(stops["interval_start"])
    stops = stops.loc[~zero_length, ["patient_id", "code", "time"]].sort_values("time")
    starts = med.loc[kind.eq("start"), ["patient_id", "code", "time"]].sort_values("time")
    matched = pd.merge_asof(
        stops.rename(columns={"time": "stop"}),
        starts.rename(columns={"time": "start"}),
        left_on="stop",
        right_on="start",
        by=["patient_id", "code"],
        direction="forward",
    )
    gap = (matched["start"] - matched["stop"]).dt.total_seconds() / 86_400
    out = {
        "zero_length_stops": int(zero_length.sum()),
        "other_stops": len(gap),
        "no_later_start": int(gap.isna().sum()),
        "exactly_0": int(gap.eq(0).sum()),
    }
    for low, high in itertools.pairwise(GAP_BUCKETS):
        out[f"over_{low:g}_to_{high:g}"] = int(((gap > low) & (gap <= high)).sum())
    return out


def vocabulary_report(tokenizer: FoldTokenizer) -> dict[str, Any]:
    tokens = pd.Series(tokenizer.vocabulary)
    group = tokens.str.extract(r"^(\[[A-Z]+|[a-z_]+)", expand=False)
    numeric = tokenizer.numeric_bins.values()
    return {
        "size": tokenizer.size,
        "tokens_by_kind": {str(k): int(v) for k, v in group.value_counts().sort_index().items()},
        "eligible_base_codes": len(tokenizer.eligible),
        "numeric_groups": len(tokenizer.numeric_bins),
        "numeric_groups_all_fallback": sum(
            all(t.endswith("other_value") for t in rule["tokens"]) for rule in numeric
        ),
        "numeric_bins_fallback": sum(
            sum(t.endswith("other_value") for t in rule["tokens"]) for rule in numeric
        ),
        "numeric_bins_total": sum(len(rule["tokens"]) for rule in numeric),
        "text_values": sum(len(v) for v in tokenizer.text_values.values()),
    }


def modality_report(histories: Histories, rows: np.ndarray, sequence: Any) -> dict[str, Any]:
    """D34 coverage: patients with genotype tokens, and with a pre-anchor scan in the input."""
    samples = [histories.prediction_sample(int(r), sequence) for r in rows]
    genotype = [int(histories.static_offsets[r + 1] - histories.static_offsets[r]) for r in rows]
    return {
        "patients": len(rows),
        "with_genotype_tokens": int(sum(n > 0 for n in genotype)),
        "genotype_tokens_max": max(genotype, default=0),
        "with_prescan_image": int(sum(s.image is not None for s in samples)),
    }


def run(
    dataset_dir: Path, config_path: Path, models_dir: Path, reports_dir: Path, folds: list[int]
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    if "representation_from" in config:
        raise ValueError(
            f"{config_path} reuses the tokenizers of {config['representation_from']}; "
            "tokenize that configuration instead"
        )
    rep, sequence = config["representation"], sequence_config(config)
    data = load_training_data(dataset_dir)
    events, cohort, codes = data.inputs.events, data.inputs.cohort, data.inputs.codes
    events_manifest = file_hash(dataset_dir / EVENTS_MANIFEST)
    genomics_on, imaging_on, _ = modality_settings(config)
    modalities = config_modalities(dataset_dir, config)
    modalities_sha = None if modalities is None else modalities.manifest_sha256
    assignments = load_folds(dataset_dir)
    out_dir = reports_dir / str(config["name"])
    batch_size = int(config["pretrain"]["batch_size"])
    rows = []
    for fold in folds:
        fit, held = fold_patients(assignments, fold)
        tokenizer = FoldTokenizer.fit(
            events,
            fit,
            min_patients=int(rep["min_patients"]),
            refill_gap_days=float(rep["refill_gap_days"]),
            input_sha256=events_manifest,
            genomics=modalities.genomics if modalities is not None and genomics_on else None,
            imaging=imaging_on,
        )
        histories = build_histories(tokenizer.encode(events), cohort, tokenizer, codes, modalities)
        directory = fold_dir(models_dir, config, fold)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save(directory / "tokenizer.json")
        save_histories(
            histories,
            directory / "histories.npz",
            histories_provenance(tokenizer, events_manifest, cohort.index, modalities_sha),
        )
        fit_rows, held_rows = histories.index_of(fit), histories.index_of(held)
        fit_events = events.loc[events["patient_id"].isin(fit)]
        _, merged = compress_refills(fit_events, float(rep["refill_gap_days"]))
        pretrain = [int(r) for r in fit_rows if histories.n_tokens(int(r))]
        starts = np.array([histories.window_starts(r, sequence) for r in pretrain])
        report = {
            "fold": fold,
            "config": config["name"],
            "tokenizer_sha256": tokenizer.sha256,
            "fit_patients": len(fit),
            "held_patients": len(held),
            "vocabulary": vocabulary_report(tokenizer),
            "M1_refill_gaps_fit": refill_gap_audit(fit_events),
            "refill_pairs_merged_fit": merged,
            "fit": retention_report(histories, fit_rows, tokenizer, sequence, data.labels),
            "held": retention_report(histories, held_rows, tokenizer, sequence, data.labels),
            "M8_pretraining": {
                "patients_with_history": len(pretrain),
                "history_tokens": int(sum(histories.n_tokens(r) for r in pretrain)),
                "windows_per_epoch": len(pretrain),
                "optimizer_steps_per_epoch": math.ceil(len(pretrain) / batch_size),
                "patients_with_several_window_starts": int((starts > 1).sum()),
                "window_starts_max": int(starts.max()) if len(starts) else 0,
            },
        }
        if modalities is not None:
            report["modalities"] = {
                "genotype_vocabulary": sum(t.startswith("dna:") for t in tokenizer.vocabulary),
                "fit": modality_report(histories, fit_rows, sequence),
                "held": modality_report(histories, held_rows, sequence),
            }
        write_json(out_dir / f"fold{fold}.json", report)
        log.info(
            "Fold %d: vocabulary %d, fit truncated %.1f%%, held truncated %.1f%%",
            fold,
            tokenizer.size,
            100 * report["fit"]["truncated_fraction"],
            100 * report["held"]["truncated_fraction"],
        )
        rows.append(
            {
                "fold": fold,
                "vocabulary": tokenizer.size,
                "fit_truncated_fraction": report["fit"]["truncated_fraction"],
                "held_truncated_fraction": report["held"]["truncated_fraction"],
                "fit_tokens_lost_fraction": report["fit"]["event_tokens_lost_fraction"],
                "held_unknown_rate_conditions": report["held"]["unknown_rate_by_source"].get(
                    "conditions"
                ),
                "median_sequence_length_fit": report["fit"]["sequence_length"]["median"],
                "pretraining_tokens": report["M8_pretraining"]["history_tokens"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(args.dataset, args.config, args.models, args.reports, args.folds)


if __name__ == "__main__":
    main()
