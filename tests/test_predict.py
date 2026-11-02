from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ehrforecast.contracts import PredictionInputs, prediction_cohort, prior_from_history
from ehrforecast.finetune import predict_samples, run_finetuning
from ehrforecast.predict import load_fold_model, predict_inputs
from ehrforecast.training import sequence_config
from tests.transformer_fixtures import (
    ANCHOR,
    TARGETS,
    config,
    fold_data,
    permitted_events,
    raw_tables,
)

CPU = torch.device("cpu")


def _trained(tmp_path: Path):  # type: ignore[no-untyped-def]
    fold = fold_data()
    (tmp_path / "tokenizer.json").parent.mkdir(parents=True, exist_ok=True)
    fold.tokenizer.save(tmp_path / "tokenizer.json")
    run_finetuning(config(), fold, "S", tmp_path / "S", device=CPU)
    return fold, load_fold_model(tmp_path / "S")


def _inputs(cohort: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> PredictionInputs:
    prepared = prediction_cohort(cohort)
    events = permitted_events(cohort, tables)
    return PredictionInputs(
        prepared, events, prior_from_history(events, prepared.index, TARGETS), TARGETS
    )


def test_reloaded_model_reproduces_in_memory_predictions(tmp_path: Path) -> None:
    fold, fm = _trained(tmp_path)
    sequence = sequence_config(config())
    samples = [fold.histories.prediction_sample(int(r), sequence) for r in fold.held_rows]
    in_memory = predict_samples(fm.model, samples, CPU, 4)
    inputs = PredictionInputs(
        fold.data.inputs.cohort.loc[fold.held],
        fold.data.inputs.events[fold.data.inputs.events["patient_id"].isin(fold.held)],
        fold.data.inputs.prior.loc[fold.held],
        TARGETS,
    )
    reloaded = predict_inputs([fm], inputs, CPU)
    assert list(reloaded.columns) == TARGETS and list(reloaded.index) == list(fold.held)
    expected = np.where(inputs.prior.to_numpy() == 1, 0.0, in_memory)
    np.testing.assert_allclose(reloaded.to_numpy(), expected, rtol=1e-6, atol=1e-7)
    assert ((reloaded.to_numpy() >= 0) & (reloaded.to_numpy() <= 1)).all()


def test_future_records_and_labels_do_not_change_predictions(tmp_path: Path) -> None:
    _, fm = _trained(tmp_path)
    cohort, tables = raw_tables(24, seed=0)
    cohort, tables = (
        cohort.iloc[16:],
        {k: v[v["PATIENT"].isin(cohort.index[16:])].copy() for k, v in tables.items()},
    )
    original = predict_inputs([fm], _inputs(cohort, tables), CPU)
    future = ANCHOR + pd.Timedelta(days=30)
    changed = {k: v.copy() for k, v in tables.items()}
    for name, table in changed.items():
        time_column = "START" if "START" in table else "DATE"
        later = table[time_column] >= ANCHOR
        if "VALUE" in table:
            table.loc[later & table["TYPE"].eq("numeric"), "VALUE"] = "99999"
        if "STOP" in table:  # stops on or after the anchor change, but stay invisible
            table.loc[table["STOP"] >= ANCHOR, "STOP"] = future
        extra = table.iloc[:3].copy()
        extra[time_column] = future
        changed[name] = pd.concat([table, extra], ignore_index=True)
    np.testing.assert_array_equal(
        predict_inputs([fm], _inputs(cohort, changed), CPU).to_numpy(), original.to_numpy()
    )
