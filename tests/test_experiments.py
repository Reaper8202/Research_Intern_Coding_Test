"""Tuning-round register: the section 5.3 rule, run states, and artifact reuse."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ehrforecast.experiments import compare, load_budget, run_status
from ehrforecast.training import fold_dir, pretrain_dir, representation_dir


def _scores(aurocs: list[float], maps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"macro_auroc": aurocs, "mAP": maps}, index=range(len(aurocs)))


def test_rule_needs_margin_four_folds_and_no_map_loss() -> None:
    incumbent = _scores([0.75] * 5, [0.20] * 5)
    passing = compare(_scores([0.76, 0.76, 0.76, 0.76, 0.74], [0.20] * 5), incumbent)
    assert passing["passes_rule"] and passing["folds_with_higher_auroc"] == 4
    three_folds = compare(_scores([0.77, 0.77, 0.77, 0.74, 0.74], [0.20] * 5), incumbent)
    assert three_folds["mean_auroc_gain"] >= 0.005 and not three_folds["passes_rule"]
    map_loss = compare(_scores([0.76] * 5, [0.19] * 5), incumbent)
    assert not map_loss["passes_rule"]
    screen = compare(_scores([0.76], [0.20]), incumbent)
    assert screen["passes_screen"] and "passes_rule" not in screen


def test_run_status_distinguishes_completed_partial_and_missing(tmp_path: Path) -> None:
    assert run_status(tmp_path / "none") == "missing"
    (tmp_path / "last.pt").write_bytes(b"x")
    (tmp_path / "run.json").write_text(json.dumps({"status": "failed"}))
    assert run_status(tmp_path) == "partial"
    (tmp_path / "run.json").write_text(json.dumps({"status": "completed"}))
    assert run_status(tmp_path) == "completed"


def test_variants_reuse_artifacts_but_write_their_own_runs() -> None:
    models = Path("m")
    plain = {"name": "base"}
    variant = {"name": "v", "representation_from": "base", "pretrain_from": "base"}
    assert (
        representation_dir(models, plain, 1)
        == pretrain_dir(models, plain, 1)
        == models / "base/fold1"
    )
    assert representation_dir(models, variant, 1) == models / "base/fold1"
    assert pretrain_dir(models, variant, 1) == models / "base/fold1"
    assert fold_dir(models, variant, 1) == models / "v/fold1"


def test_budget_ids_must_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "budget.toml"
    path.write_text(
        'incumbent = "x"\nmethod = "F"\n'
        '[[experiment]]\nid = "a"\nconfig = "c"\nfolds = [0]\n'
        '[[experiment]]\nid = "a"\nconfig = "d"\nfolds = [0]\n'
    )
    with pytest.raises(ValueError, match="repeats"):
        load_budget(path)


def test_the_committed_budget_lists_existing_configs_once() -> None:
    budget = load_budget(Path("configs/budget.toml"))
    assert all(Path(e["config"]).is_file() for e in budget["experiment"])
    assert budget["method"] == "F" and Path(budget["incumbent"]).is_file()
