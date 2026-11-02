"""Small synthetic raw tables that pass through the production history pipeline."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.contracts import (
    PredictionInputs,
    TrainingData,
    prediction_cohort,
    prediction_events,
    prior_from_history,
)
from ehrforecast.events import build_events
from ehrforecast.provenance import object_hash
from ehrforecast.schema import EVENT_TABLES
from ehrforecast.sequences import build_histories
from ehrforecast.tokenizer import FoldTokenizer
from ehrforecast.training import FoldData
from ehrforecast.truncation import omit_invalid_stops, truncate_table

TARGETS = ["T1", "T2", "T3"]
ANCHOR = pd.Timestamp("2015-01-01")

CONFIG: dict[str, Any] = {
    "name": "fixture",
    "seed": 7,
    "representation": {
        "context_length": 48,
        "max_summary_tokens": 6,
        "min_patients": 1,
        "refill_gap_days": 1.0,
    },
    "model": {
        "n_layers": 1,
        "d_model": 16,
        "n_heads": 2,
        "ff_mult": 2,
        "dropout": 0.1,
        "head_dropout": 0.1,
    },
    "pretrain": {
        "batch_size": 4,
        "micro_batch_size": 2,
        "max_epochs": 3,
        "patience": 3,
        "lr": 0.003,
        "warmup_fraction": 0.1,
        "final_lr_fraction": 0.1,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "eval_seed": 11,
    },
    "finetune": {
        "batch_size": 4,
        "micro_batch_size": 3,
        "max_epochs": 3,
        "patience": 5,
        "head_lr": 0.003,
        "body_lr_S": 0.001,
        "body_lr_F": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
    },
    "runtime": {
        "device": "cpu",
        "wandb": "disabled",
        "max_steps_per_epoch": 0,
        "max_fit_patients": 0,
        "max_held_patients": 0,
        "eval_batch_size": 8,
    },
}


def config(**overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(CONFIG)
    for section, values in overrides.items():
        out[section].update(values)
    return out


def raw_tables(n_patients: int = 24, seed: int = 0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Random cohort and raw tables, including rows after the anchor."""
    rng = np.random.default_rng(seed)
    patients = [f"p{i:02d}" for i in range(n_patients)]
    birth = pd.to_datetime(rng.integers(1950, 1985, n_patients).astype(str) + "-03-10")
    cohort = pd.DataFrame(
        {
            "birthdate": birth,
            "gender": rng.choice(["F", "M"], n_patients),
            "anchor_date": ANCHOR,
        },
        index=pd.Index(patients, name="patient_id"),
    )
    rows: dict[str, list[dict[str, Any]]] = {
        k: [] for k in ("conditions", "medications", "observations", "encounters", "procedures")
    }
    for p in patients:
        days = np.sort(
            rng.choice(np.arange(-3000, 400), size=int(rng.integers(3, 14)), replace=False)
        )
        for d in days:
            t = ANCHOR + pd.Timedelta(days=int(d)) + pd.Timedelta(hours=9)
            rows["encounters"].append(
                {
                    "PATIENT": p,
                    "START": t,
                    "STOP": t + pd.Timedelta(hours=1),
                    "ENCOUNTERCLASS": rng.choice(["wellness", "ambulatory"]),
                    "CODE": "E1",
                }
            )
            rows["observations"].append(
                {
                    "PATIENT": p,
                    "DATE": t,
                    "CODE": "O1",
                    "VALUE": f"{rng.normal(100, 15):.1f}",
                    "UNITS": "mg/dL",
                    "TYPE": "numeric",
                }
            )
            if rng.random() < 0.4:
                rows["observations"].append(
                    {
                        "PATIENT": p,
                        "DATE": t,
                        "CODE": "O2",
                        "VALUE": rng.choice(["Never smoker", "Former  smoker"]),
                        "UNITS": None,
                        "TYPE": "text",
                    }
                )
            if rng.random() < 0.5:
                code = str(rng.choice(["C1", "C2", "C3", *TARGETS]))
                stop = (
                    t + pd.Timedelta(days=int(rng.integers(5, 400)))
                    if rng.random() < 0.5
                    else pd.NaT
                )
                rows["conditions"].append({"PATIENT": p, "START": t, "STOP": stop, "CODE": code})
            if rng.random() < 0.4:
                drug, start = str(rng.choice(["M1", "M2"])), t
                for _ in range(int(rng.integers(1, 4))):  # a refill chain
                    stop = start + pd.Timedelta(days=30)
                    rows["medications"].append(
                        {"PATIENT": p, "START": start, "STOP": stop, "CODE": drug}
                    )
                    start = stop
            if rng.random() < 0.3:
                rows["procedures"].append({"PATIENT": p, "DATE": t, "CODE": "P1"})
    tables = {k: pd.DataFrame(v) for k, v in rows.items()}
    return cohort, tables


def permitted_events(cohort: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Production cut, D31 repair, event construction, and input contract."""
    prepared = prediction_cohort(cohort)
    history = {}
    for name, table in tables.items():
        spec = EVENT_TABLES[name]
        history[name], _ = omit_invalid_stops(
            truncate_table(table, spec, prepared["anchor_date"]), spec
        )
    return prediction_events(build_events(history), prepared)


def fold_data(n_patients: int = 24, seed: int = 0, n_fit: int = 16) -> FoldData:
    cohort, tables = raw_tables(n_patients, seed)
    prepared = prediction_cohort(cohort)
    events = permitted_events(cohort, tables)
    prior = prior_from_history(events, prepared.index, TARGETS)
    rng = np.random.default_rng(seed + 1)
    labels = pd.DataFrame(
        (rng.random((n_patients, len(TARGETS))) < 0.4).astype("int8"),
        index=prepared.index,
        columns=TARGETS,
    )
    labels = labels.mask(prior.astype(bool), 0).astype("int8")
    labels.iloc[n_fit : n_fit + 2] = 1 - prior.iloc[n_fit : n_fit + 2]  # both classes held out
    labels.iloc[n_fit + 2] = 0
    fit, held = prepared.index[:n_fit], prepared.index[n_fit:]
    tokenizer = FoldTokenizer.fit(events, fit, min_patients=1)
    histories = build_histories(tokenizer.encode(events), prepared, tokenizer, TARGETS)
    provenance = {
        "fold": 0,
        "fit_patients_sha256": object_hash(sorted(fit)),
        "held_patients_sha256": object_hash(sorted(held)),
        "tokenizer_sha256": tokenizer.sha256,
        "histories_sha256": "fixture",
        "events_manifest_sha256": "fixture",
    }
    data = TrainingData(PredictionInputs(prepared, events, prior, TARGETS), labels)
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
