"""Step 1: compute anchors, cut timelines at the anchor without leakage, and build labels.

Outputs (in ``--out``):

* ``cohort.parquet``: one row per patient with split, demographics, and anchor dates.
* ``events.parquet``: all pre-anchor events for every split, in the long event format.
* ``labels.parquet``: train and val labels, one int8 column per target code.
* ``prior_conditions.parquet``: target conditions already present before the anchor (all splits).
* ``code_descriptions.parquet``: description for each (source, code) pair.
* ``targets.csv``: scored codes in submission column order.
* ``label_prevalence.csv`` and ``dataset_report.json``: checks and train/test parity statistics.

Usage::

    python -m ehrforecast.build_dataset --data-dir data --out artifacts/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ehrforecast import cohort as cohort_mod
from ehrforecast.config import DataLayout
from ehrforecast.events import build_code_descriptions, build_events
from ehrforecast.io import read_all_event_tables, read_patients
from ehrforecast.labels import build_labels, build_prior_flags
from ehrforecast.parity import blank_stop_fraction, history_summary, label_prevalence
from ehrforecast.schema import EVENT_TABLES
from ehrforecast.truncation import find_post_anchor_values, truncate_table

log = logging.getLogger("ehrforecast.build_dataset")


def _check_tables_belong_to(
    tables: dict[str, pd.DataFrame], patients: pd.Index, where: str
) -> None:
    allowed = set(patients)
    for name, df in tables.items():
        stray = set(df["PATIENT"]) - allowed
        if stray:
            raise ValueError(f"{where}/{name}.csv has {len(stray)} patients from another split")


def build(data_dir: Path, out_dir: Path) -> dict[str, object]:
    layout = DataLayout(data_dir)
    layout.validate()

    splits = cohort_mod.load_splits(layout.splits_file)
    targets = cohort_mod.load_targets(layout.targets_file)
    codes = targets["CODE"].tolist()
    test_anchor_dates = cohort_mod.load_test_anchors(layout.test_anchors_file)

    patients_tv = read_patients(layout.train_val_dir)
    patients_test = read_patients(layout.test_dir)
    cohort_mod.validate_split_membership(
        splits,
        pd.Index(patients_tv["Id"]),
        pd.Index(patients_test["Id"]),
        test_anchor_dates.index,
    )
    log.info(
        "Splits: %s",
        splits.value_counts().reindex(["train", "val", "test"]).to_dict(),
    )

    log.info("Reading train_val tables")
    full_tv = read_all_event_tables(layout.train_val_dir)
    log.info("Reading test tables")
    test_tables = read_all_event_tables(layout.test_dir)
    _check_tables_belong_to(full_tv, pd.Index(patients_tv["Id"]), "train_val")
    _check_tables_belong_to(test_tables, pd.Index(patients_test["Id"]), "test")

    # Anchors: computed for train/val from the full encounter table; given for test.
    anchors_tv = cohort_mod.compute_anchors(full_tv["encounters"])
    missing = set(patients_tv["Id"]) - set(anchors_tv.index)
    if missing:
        raise ValueError(f"{len(missing)} train/val patients have no encounters")
    anchors_test = pd.DataFrame(
        {
            "anchor_date": test_anchor_dates,
            "window_end": test_anchor_dates + cohort_mod.HORIZON,
        }
    )
    anchors = pd.concat([anchors_tv, anchors_test])
    cohort = cohort_mod.build_cohort(
        splits, pd.concat([patients_tv, patients_test], ignore_index=True), anchors
    )
    anchor_dates = cohort["anchor_date"]

    # Labels use the full train/val timeline; nothing from it is kept after this point.
    labelled = cohort[cohort["split"].isin(["train", "val"])]
    labels = build_labels(full_tv["conditions"], labelled, codes)

    # History: cut train/val exactly like the released test files.
    history_tv = {
        name: truncate_table(df, EVENT_TABLES[name], anchor_dates) for name, df in full_tv.items()
    }
    del full_tv

    # Hard invariants. The released test files must already be cut, and our cut must be clean.
    for name, df in history_tv.items():
        n = find_post_anchor_values(df, EVENT_TABLES[name], anchor_dates)
        if n:
            raise AssertionError(f"train_val/{name}: {n} values on or after the anchor remain")
    for name, df in test_tables.items():
        n = find_post_anchor_values(df, EVENT_TABLES[name], anchor_dates)
        if n:
            raise AssertionError(f"test/{name}: {n} values on or after the given anchor")

    history_all = {
        name: pd.concat([history_tv[name], test_tables[name]], ignore_index=True)
        for name in EVENT_TABLES
    }
    events = build_events(history_all)
    descriptions = build_code_descriptions(history_all)
    prior = build_prior_flags(history_all["conditions"], cohort.index, codes)

    overlap = int((labels.to_numpy() & prior.loc[labels.index].to_numpy()).sum())
    if overlap:
        raise AssertionError(f"{overlap} labels are positive although the condition is prior")

    by_split = {s: cohort[cohort["split"] == s] for s in ("train", "val", "test")}
    report: dict[str, object] = {
        "n_events": len(events),
        "n_events_by_source": events["source"].value_counts().sort_index().to_dict(),
        "history_summary": {s: history_summary(events, c) for s, c in by_split.items()},
        "blank_stop_fraction": {
            "train_val_truncated": blank_stop_fraction(history_tv),
            "test_released": blank_stop_fraction(test_tables),
        },
        "label_positives_total": int(labels.to_numpy().sum()),
        "prior_condition_rate_by_split": {
            s: round(float(prior.loc[c.index].to_numpy().mean()), 5) for s, c in by_split.items()
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_dir / "cohort.parquet")
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet")
    prior.to_parquet(out_dir / "prior_conditions.parquet")
    descriptions.to_parquet(out_dir / "code_descriptions.parquet", index=False)
    targets.to_csv(out_dir / "targets.csv", index=False)
    prevalence = label_prevalence(labels, cohort)
    prevalence.insert(0, "description", targets.set_index("CODE")["DESCRIPTION"])
    prevalence.to_csv(out_dir / "label_prevalence.csv", index_label="code")
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote dataset artifacts to %s", out_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="folder with the task files")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    report = build(args.data_dir, args.out)
    log.info("Report:\n%s", json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
