"""Optional full-timeline events for self-pretraining (plan E2 test; see D16 and D36).

Self-pretraining normally reads fit patients' histories before their anchors. This module builds
the alternative: every event of the official training patients, with no anchor cut. Only fit
patients of a fold train on it. Held-out, val, and test patients never enter it, so held-out scores
stay fair. Prediction inputs always stay cut at the anchor. The D31 interval repair still applies.

Usage::

    python -m ehrforecast.full_timeline --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ehrforecast.cohort import load_splits
from ehrforecast.config import DataLayout
from ehrforecast.events import EVENT_COLUMNS, build_events
from ehrforecast.io import read_all_event_tables
from ehrforecast.logging_utils import setup_logging
from ehrforecast.provenance import file_hash, object_hash, write_json
from ehrforecast.schema import EVENT_TABLES, PATIENT_COL
from ehrforecast.truncation import omit_invalid_stops

log = logging.getLogger("ehrforecast.full_timeline")

FULL_EVENTS_FILE = "pretrain_events_full.parquet"
FULL_EVENTS_MANIFEST = "pretrain_events_full.manifest.json"


def full_events(tables: dict[str, pd.DataFrame], patients: pd.Index) -> pd.DataFrame:
    """Uncut events of ``patients`` only, after the D31 repair, with model-input columns."""
    kept = {}
    for name, table in tables.items():
        rows = table.loc[table[PATIENT_COL].isin(patients)]
        kept[name], _ = omit_invalid_stops(rows, EVENT_TABLES[name])
    events = build_events(kept).loc[:, list(EVENT_COLUMNS)].copy()
    # The same input contract as ``contracts.prediction_events``: no encounter durations.
    observation = events["source"].astype(str).eq("observations")
    events.loc[~observation, "value_num"] = np.nan
    events.loc[~observation, "unit"] = pd.NA
    has_text = events["source"].astype(str).isin(["observations", "encounters"])
    events.loc[~has_text, "value_text"] = pd.NA
    return events.reset_index(drop=True)


def build(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    layout = DataLayout(raw_dir)
    splits = load_splits(layout.splits_file)
    train = pd.Index(splits.index[splits.eq("train")])
    events = full_events(read_all_event_tables(layout.train_val_dir), train)
    path = out_dir / FULL_EVENTS_FILE
    events.to_parquet(path, index=False)
    manifest: dict[str, object] = {
        "train_patients_sha256": object_hash(sorted(train.tolist())),
        "rows": len(events),
        "patients": int(events["patient_id"].nunique()),
        "file_sha256": file_hash(path),
    }
    write_json(out_dir / FULL_EVENTS_MANIFEST, manifest)
    return manifest


def load_fit_events(dataset_dir: Path, fit: pd.Index, train: pd.Index) -> pd.DataFrame:
    """Uncut events of one fold's fit patients, after checking the file and patient set."""
    path, manifest_path = dataset_dir / FULL_EVENTS_FILE, dataset_dir / FULL_EVENTS_MANIFEST
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"{path} or its manifest is missing; run ehrforecast.full_timeline")
    manifest = json.loads(manifest_path.read_text())
    if manifest["file_sha256"] != file_hash(path):
        raise ValueError(f"{path} does not match its manifest")
    if manifest["train_patients_sha256"] != object_hash(sorted(train.tolist())):
        raise ValueError(f"{path} was built for a different training split")
    if not fit.isin(train).all():
        raise ValueError("fit patients must belong to the official training split")
    events = pd.read_parquet(path)
    return events.loc[events["patient_id"].isin(fit)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    log.info("Built full-timeline events: %s", build(args.raw, args.out))


if __name__ == "__main__":
    main()
