"""P3a training-only audits and one historical baseline reproduction.

This command never calls the legacy baseline runner, which would also score official val.
It checks training anchors, labels, and permitted histories against the raw supplied tables.
Permitted histories include the D31 repair: a ``STOP`` before its own ``START`` is omitted after
the cut. The audit checks the repaired rows against ``reports/dataset/omitted_stops.csv``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import platform
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ehrforecast.baseline import MultiLabelLogistic
from ehrforecast.cohort import compute_anchors, load_targets
from ehrforecast.cohort import load_splits as load_raw_splits
from ehrforecast.contracts import (
    INPUT_SCHEMA_VERSION,
    TrainingData,
    load_splits,
    load_training_data,
    prediction_events,
)
from ehrforecast.evaluation import METRIC_VERSION, evaluate_predictions
from ehrforecast.events import build_events
from ehrforecast.features import BagOfCodes
from ehrforecast.folds import fold_patients, save_folds
from ehrforecast.io import parse_timestamps, read_csv_as_strings
from ehrforecast.labels import build_labels, build_prior_flags
from ehrforecast.logging_utils import setup_logging
from ehrforecast.masking import apply_prior_mask
from ehrforecast.metrics import summarize as legacy_summary
from ehrforecast.numeric import UNIT_RULE_VERSION, unit_audit
from ehrforecast.provenance import code_hashes, file_hash, object_hash, write_json
from ehrforecast.schema import EVENT_TABLES, TableSpec
from ehrforecast.truncation import (
    OMITTED_STOP_COLUMNS,
    combine_omitted,
    omit_invalid_stops,
    truncate_table,
)

log = logging.getLogger(__name__)


def _raw_training_table(root: Path, spec: TableSpec, patients: pd.Index) -> pd.DataFrame:
    table = read_csv_as_strings(root / "train_val" / f"{spec.name}.csv", spec.columns)
    table = table.loc[table["PATIENT"].isin(patients)].copy()
    table[spec.time_col] = parse_timestamps(table[spec.time_col])
    if table[spec.time_col].isna().any():
        raise ValueError(f"training {spec.name} contains a missing event date")
    if spec.stop_col is not None:
        table[spec.stop_col] = parse_timestamps(table[spec.stop_col])
        # Encounter ends define anchors. D31 repairs only stops that become history events.
        if spec.name == "encounters" and _stop_before_start(table, spec).any():
            raise ValueError("training encounters contain a stop before its start")
    if table[list(spec.code_cols)].isna().any().any():
        raise ValueError(f"training {spec.name} contains a missing clinical code")
    return table


def _stop_before_start(table: pd.DataFrame, spec: TableSpec) -> pd.Series:
    assert spec.stop_col is not None
    return table[spec.stop_col].notna() & (table[spec.stop_col] < table[spec.time_col])


def _sorted_omissions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(OMITTED_STOP_COLUMNS)
    return frame[columns].sort_values(columns, kind="stable").reset_index(drop=True)


def audit_raw_training(
    raw_dir: Path, data: TrainingData, recorded_omissions: pd.DataFrame
) -> dict[str, Any]:
    """Rebuild training anchors, labels, and histories; fit no preprocessing."""
    patients, codes = data.inputs.cohort.index, data.inputs.codes
    demographics = read_csv_as_strings(
        raw_dir / "train_val" / "patients.csv",
        ("Id", "BIRTHDATE", "GENDER"),
    )
    demographics = demographics.loc[demographics["Id"].isin(patients)].copy()
    if demographics["Id"].duplicated().any() or set(demographics["Id"]) != set(patients):
        raise ValueError("raw training demographics do not match the training patient IDs")
    demographics["BIRTHDATE"] = parse_timestamps(demographics["BIRTHDATE"])
    demographics = demographics.set_index("Id").rename(
        columns={"BIRTHDATE": "birthdate", "GENDER": "gender"}
    )
    demographics.index.name = "patient_id"
    pd.testing.assert_frame_equal(
        demographics.loc[patients], data.inputs.cohort[["birthdate", "gender"]], check_dtype=False
    )
    encounters = _raw_training_table(raw_dir, EVENT_TABLES["encounters"], patients)
    anchors = compute_anchors(encounters)
    if set(anchors.index) != set(patients):
        raise ValueError("training patients and raw encounter anchors differ")
    pd.testing.assert_series_equal(
        anchors.loc[patients, "anchor_date"],
        data.inputs.cohort["anchor_date"],
        check_dtype=False,
    )
    history = {}
    label_checks = {}
    invalid_stops: dict[str, dict[str, int]] = {}
    omitted_parts = []
    for name, spec in EVENT_TABLES.items():
        raw = encounters if name == "encounters" else _raw_training_table(raw_dir, spec, patients)
        if name == "conditions":
            rebuilt = build_labels(raw, anchors.loc[patients], codes)
            pd.testing.assert_frame_equal(rebuilt, data.labels, check_dtype=False)
            label_checks = {"training_positives": int(rebuilt.to_numpy().sum())}
        cut = truncate_table(raw, spec, data.inputs.cohort["anchor_date"])
        history[name], omitted = omit_invalid_stops(cut, spec)
        omitted_parts.append(omitted)
        if spec.stop_col is not None:
            # An invalid row that survives the cut keeps both dates before the anchor.
            n_raw = int(_stop_before_start(raw, spec).sum())
            invalid_stops[name] = {
                "raw_rows": n_raw,
                "removed_by_cutoff": n_raw - len(omitted),
                "stop_omitted": len(omitted),
            }
    recorded = recorded_omissions.loc[recorded_omissions["split"].eq("train")]
    pd.testing.assert_frame_equal(
        _sorted_omissions(combine_omitted(omitted_parts)),
        _sorted_omissions(recorded),
        check_dtype=False,
    )
    rebuilt_prior = build_prior_flags(history["conditions"], patients, codes)
    pd.testing.assert_frame_equal(rebuilt_prior, data.inputs.prior, check_dtype=False)
    rebuilt_events = prediction_events(build_events(history), data.inputs.cohort)
    pd.testing.assert_frame_equal(rebuilt_events, data.inputs.events, check_dtype=False)
    return {
        "status": "passed",
        "scope": "official training patients only",
        "anchors_match": True,
        "demographics_match": True,
        "labels_match": True,
        "prior_masks_match": True,
        "permitted_events_match": True,
        "recorded_omissions_match": True,
        "stop_before_start": invalid_stops,
        "n_patients": len(patients),
        "n_events": len(rebuilt_events),
        **label_checks,
    }


def coverage_audit(data: TrainingData, assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fold in range(5):
        fit, held = fold_patients(assignments, fold)
        for code in data.inputs.codes:
            rows.append(
                {
                    "fold": fold,
                    "code": code,
                    "fit_preanchor_patients": int(data.inputs.prior.loc[fit, code].sum()),
                    "fit_label_positives": int(data.labels.loc[fit, code].sum()),
                    "held_label_positives": int(data.labels.loc[held, code].sum()),
                    "held_label_negatives": int((1 - data.labels.loc[held, code]).sum()),
                }
            )
    table = pd.DataFrame(rows)
    absent = [code for code in data.inputs.codes if data.inputs.prior[code].sum() == 0]
    if set(absent) != {"233604007", "314994000"}:
        raise ValueError("generation coverage differs from F11; review the data before proceeding")
    if (table[["held_label_positives", "held_label_negatives"]] == 0).any().any():
        raise ValueError("a full held-out fold lacks both classes for a required target")
    decision = {
        "status": "deferred",
        "reason": "Required target starts have no permitted pre-anchor training examples.",
        "generation_runs": 0,
        "supervised_outputs": len(data.inputs.codes),
        "unsupported_targets": [
            {
                "code": code,
                "preanchor_patients": 0,
                "supervised_positives": int(data.labels[code].sum()),
            }
            for code in absent
        ],
    }
    return table, decision


def reproduce_baseline(
    data: TrainingData,
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the one recorded C on five fit sets. Read no official val inputs or labels."""
    if config["maximum_configurations"] != 1:
        raise ValueError("P3a permits one historical baseline configuration")
    c = float(config["C"])
    expected = reference.loc[reference["model"].eq("logreg_codes") & reference["C"].eq(c)]
    if len(expected) != 5 or expected["fold"].duplicated().any():
        raise ValueError("the baseline reference must contain exactly one row per fold")
    rows, metric_rows, timing = [], [], []
    probabilities = pd.DataFrame(np.nan, index=data.inputs.cohort.index, columns=data.inputs.codes)
    for fold in range(5):
        started = time.perf_counter()
        fit, held = fold_patients(assignments, fold)
        features = BagOfCodes().fit(data.inputs.events, data.inputs.cohort.loc[fit])
        x_fit = features.transform(data.inputs.events, data.inputs.cohort.loc[fit]).to_numpy()
        x_held = features.transform(data.inputs.events, data.inputs.cohort.loc[held]).to_numpy()
        model = MultiLabelLogistic(C=c, n_jobs=int(config["n_jobs"])).fit(
            x_fit, data.labels.loc[fit].to_numpy()
        )
        scores = pd.DataFrame(model.predict_proba(x_held), index=held, columns=data.inputs.codes)
        observed = {
            "fold": fold,
            "C": c,
            "n_features": x_fit.shape[1],
            "not_converged": model.n_not_converged_,
            **legacy_summary(
                data.labels.loc[held].to_numpy(), scores.to_numpy(), data.inputs.codes
            ),
        }
        old = expected.loc[expected["fold"].eq(fold)].iloc[0]
        for metric in ("macro_auroc", "mAP", "chance_mAP", "n_targets_scored", "n_features"):
            if not np.isclose(
                observed[metric], old[metric], atol=config["score_tolerance"], rtol=0
            ):
                raise ValueError(f"historical baseline mismatch: fold {fold}, {metric}")
        if model.n_not_converged_:
            raise ValueError("historical baseline reproduction did not converge")
        rows.append(observed)
        probabilities.loc[held] = scores
        for mode, prior, at_risk in (
            ("unmasked", None, False),
            ("masked", data.inputs.prior.loc[held], False),
            ("at_risk", data.inputs.prior.loc[held], True),
        ):
            summary = evaluate_predictions(
                data.labels.loc[held], scores, data.inputs.codes, prior=prior, at_risk=at_risk
            ).summary
            metric_rows.append({"fold": fold, "mode": mode, "C": c, **summary})
        unmasked, masked = metric_rows[-3], metric_rows[-2]
        for metric in ("macro_auroc", "mAP"):
            if masked[metric] + 1e-12 < unmasked[metric]:
                raise AssertionError("prior masking reduced a known-negative ranking metric")
        duration = time.perf_counter() - started
        timing.append(duration)
        log.info("Reproduced baseline fold %d in %.2fs", fold, duration)
    # Validation also detects a missing fold assignment or an unfilled prediction row.
    apply_prior_mask(probabilities, data.inputs.prior, data.inputs.codes)
    budget = {
        "configuration_count": 1,
        "binary_classifier_fits": 5 * len(data.inputs.codes),
        "fold_seconds": timing,
        "total_seconds": sum(timing),
        "grid_search_permitted_before_G2": False,
        "next_budget_check": "Measure LightGBM and transformer costs during the later phases.",
    }
    return pd.DataFrame(rows), pd.DataFrame(metric_rows), probabilities, budget


def run(
    raw_dir: Path,
    dataset_dir: Path,
    reports_dir: Path,
    config_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    config = tomllib.loads(config_path.read_text())
    if config["units"]["policy"] != UNIT_RULE_VERSION:
        raise ValueError("unsupported unit policy")
    phase_dir = reports_dir / "p3a"
    phase_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {"status": "started", "started_at": started, "phase": "P3a"}
    write_json(phase_dir / "run.json", report)
    try:
        data = load_training_data(dataset_dir)
        if len(data.inputs.codes) != config["expected_targets"]:
            raise ValueError("target count does not match the assessment configuration")
        raw_splits, processed_splits = (
            load_raw_splits(raw_dir / "patient_splits.csv"),
            load_splits(dataset_dir),
        )
        if raw_splits.to_dict() != processed_splits.to_dict():
            raise ValueError("processed split assignments differ from the supplied split file")
        if load_targets(raw_dir / "target_conditions.csv")["CODE"].tolist() != data.inputs.codes:
            raise ValueError("processed target order differs from the supplied target file")
        assignments = save_folds(dataset_dir)
        log.info("Checking raw training anchors, labels, and permitted events")
        recorded_omissions = pd.read_csv(
            reports_dir / "dataset" / "omitted_stops.csv",
            dtype={"split": str, "patient_id": str, "source": str, "code": str},
            parse_dates=["start", "stop"],
        )
        raw_checks = audit_raw_training(raw_dir, data, recorded_omissions)
        support, decision = coverage_audit(data, assignments)
        support.to_csv(phase_dir / "target_fold_support.csv", index=False)
        write_json(reports_dir / "transformer" / "generation_gates.json", decision)
        units = unit_audit(data.inputs.events)
        units.to_csv(phase_dir / "numeric_units.csv", index=False)
        write_json(
            phase_dir / "unit_rules.json",
            {
                "version": UNIT_RULE_VERSION,
                "scope": "training histories before real anchors",
                "normalization": "Strip surrounding whitespace; preserve case.",
                "missing_unit": "empty string",
                "conversions": [],
                "policy": "Keep every code/unit group separate. Infer no equivalences.",
                "fitted_bins": "Not fitted here. P3b must fit each fold separately.",
            },
        )
        reference_path = reports_dir / "baseline" / "cv_results.csv"
        reproduced, metrics, oof, budget = reproduce_baseline(
            data, assignments, pd.read_csv(reference_path), config["baseline_reproduction"]
        )
        reproduced.to_csv(phase_dir / "baseline_reproduction.csv", index=False)
        metrics.to_csv(reports_dir / "baseline" / "masked_vs_unmasked_cv.csv", index=False)
        oof.to_parquet(dataset_dir / "p3a_logreg_oof.parquet")
        write_json(phase_dir / "baseline_budget.json", budget)
        raw_paths = [raw_dir / name for name in ("patient_splits.csv", "target_conditions.csv")]
        raw_paths += [raw_dir / "train_val" / f"{name}.csv" for name in ["patients", *EVENT_TABLES]]
        processed_names = (
            "cohort.parquet",
            "events.parquet",
            "events.manifest.json",
            "labels.parquet",
            "prior_conditions.parquet",
            "targets.csv",
            "folds.parquet",
            "folds.manifest.json",
        )
        manifest = {
            "input_schema_version": INPUT_SCHEMA_VERSION,
            "metric_version": METRIC_VERSION,
            "config_sha256": file_hash(config_path),
            "reference_cv_sha256": file_hash(reference_path),
            "raw_sha256": {str(path.relative_to(raw_dir)): file_hash(path) for path in raw_paths},
            "processed_sha256": {name: file_hash(dataset_dir / name) for name in processed_names},
            "code_sha256": code_hashes(project_root),
            "training_patient_order_sha256": object_hash(data.inputs.cohort.index.tolist()),
            "target_order": data.inputs.codes,
            "split_counts": {str(k): int(v) for k, v in processed_splits.value_counts().items()},
            "software": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")
            },
            "python": platform.python_version(),
            "git_commit_at_run": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
            ).strip(),
            "git_worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=project_root,
                    text=True,
                ).strip()
            ),
            "dirty_scope_note": (
                "Tracked files only. code_sha256 covers every source file, tracked or not."
            ),
            "hash_scope_note": (
                "File hashes include whole supplied files; fitting used training rows only."
            ),
        }
        write_json(phase_dir / "input_manifest.json", manifest)
        report.update(
            status="passed",
            completed_at=datetime.now(UTC).isoformat(),
            raw_checks=raw_checks,
            metric_version=METRIC_VERSION,
            baseline_reproduced=True,
            official_val_scored=False,
            generation_status=decision["status"],
            input_manifest_sha256=file_hash(phase_dir / "input_manifest.json"),
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(phase_dir / "run.json", report)
        raise
    write_json(phase_dir / "run.json", report)
    log.info("P3a training-only audits passed; official val was not scored")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/p3a.toml"))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    run(args.raw, args.dataset, args.reports, args.config, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
