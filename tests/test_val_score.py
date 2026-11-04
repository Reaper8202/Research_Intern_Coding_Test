"""P4e bundle: freeze, hash verification, refusal paths, scoring, and val curves (G2/G4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from ehrforecast import val_score
from ehrforecast.contracts import PredictionInputs
from ehrforecast.finetune import run_finetuning
from ehrforecast.predict import load_fold_model, predict_inputs
from ehrforecast.provenance import object_hash, write_json
from ehrforecast.val_score import (
    SampleCache,
    build_manifest,
    score_bundle,
    val_curves,
    verify_manifest,
)
from tests.transformer_fixtures import TARGETS, config, fold_data

CPU = torch.device("cpu")
FINAL: dict[str, Any] = {
    "name": "final",
    "method": "S",
    "checkpoint": "best.pt",
    "folds": [0],
    "members": ["configs/fixture.toml"],
    "evaluation": {"n_boot": 20, "bootstrap_seed": 0, "clock_year": 2015, "curves": True},
    "comparisons": {
        "logreg_codes": "reports/ref_codes.csv",
        "incidence_at_risk": "reports/ref_at_risk.csv",
        "lightgbm_settings": "reports/lgbm.json",
        "lightgbm_feature_sets": ["counts"],
    },
}


def _project(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A tiny project tree: one trained S run, a dataset stub, comparison files, and code."""
    root = tmp_path / "root"
    fold = fold_data()
    cfg = config()
    fold_root = root / "models" / "transformer" / cfg["name"] / "fold0"
    fold_root.mkdir(parents=True)
    fold.tokenizer.save(fold_root / "tokenizer.json")
    run_finetuning(cfg, fold, "S", fold_root / "S", device=CPU)
    dataset = root / "data"
    dataset.mkdir()
    pd.DataFrame({"CODE": TARGETS, "DESCRIPTION": TARGETS}).to_csv(
        dataset / "targets.csv", index=False
    )
    (dataset / "events.manifest.json").write_text("{}")
    held = fold.held
    rng = np.random.default_rng(3)
    (root / "reports").mkdir()
    for name in ("ref_codes.csv", "ref_at_risk.csv"):
        pd.DataFrame(rng.random((len(held), len(TARGETS))), index=held, columns=TARGETS).to_csv(
            root / "reports" / name, index_label="patient_id"
        )
    (root / "reports" / "lgbm.json").write_text("{}")
    (root / "src").mkdir()
    (root / "src" / "m.py").write_text("x = 1\n")
    for name in ("pyproject.toml", "requirements-lock.txt", "Makefile"):
        (root / name).write_text(name + "\n")
    return root, fold, cfg


def test_freeze_records_members_and_verify_detects_a_changed_artifact(tmp_path: Path) -> None:
    root, _, cfg = _project(tmp_path)
    dataset, models = root / "data", root / "models"
    manifest = build_manifest(root, FINAL, {"configs/fixture.toml": cfg}, dataset, models)
    assert manifest["targets"] == TARGETS and len(manifest["members"]) == 1
    member = manifest["members"][0]
    assert member["config"] == cfg["name"] and member["fold"] == 0
    assert len(member["epoch_checkpoints"]) == cfg["finetune"]["max_epochs"]
    assert set(manifest["model_hashes"]) >= {member["run_dir"] + "/best.pt", member["tokenizer"]}
    assert "src/m.py" in manifest["code_hashes"]
    verify_manifest(root, manifest, dataset, models)
    # a retrained checkpoint, an edited module, or a changed comparison file all refuse
    (models / member["run_dir"] / "best.pt").write_bytes(b"changed")
    (root / "src" / "m.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match=r"hash mismatch: src/m\.py, .*best\.pt"):
        verify_manifest(root, manifest, dataset, models)


def test_freeze_rejects_a_run_trained_with_another_config(tmp_path: Path) -> None:
    root, _, cfg = _project(tmp_path)
    other = config(finetune={"head_lr": 0.001})
    with pytest.raises(ValueError, match="different"):
        build_manifest(root, FINAL, {"configs/fixture.toml": other}, root / "data", root / "models")
    assert object_hash(other) != object_hash(cfg)


def _val_inputs(fold):  # type: ignore[no-untyped-def]
    data = fold.data.inputs
    return PredictionInputs(
        data.cohort.loc[fold.held],
        data.events[data.events["patient_id"].isin(fold.held)],
        data.prior.loc[fold.held],
        TARGETS,
    )


def test_score_bundle_matches_the_prediction_command_and_reports_comparisons(
    tmp_path: Path,
) -> None:
    root, fold, cfg = _project(tmp_path)
    inputs = _val_inputs(fold)
    labels = fold.data.labels.loc[fold.held]
    fm = load_fold_model(root / "models" / "transformer" / cfg["name"] / "fold0" / "S")
    rng = np.random.default_rng(1)
    other = pd.DataFrame(
        rng.random((len(labels), len(TARGETS))), index=labels.index, columns=TARGETS
    )
    summary, final, per_target = score_bundle(
        inputs, labels, [fm], {"other": other}, None, clock_year=2015, n_boot=20, seed=0, device=CPU
    )
    expected = predict_inputs([fm], inputs, CPU)
    pd.testing.assert_frame_equal(final, expected)
    assert summary["n_patients"] == len(labels) and summary["ensemble_members"] == 1
    assert summary["final"]["masked"]["prior_mask_applied"]
    assert "final_minus_other" in summary["paired_bca"]
    assert summary["comparison_models"]["other"]["masked"]["n_patients"] == len(labels)
    assert set(per_target["model"]) == {"final", "other"}
    assert set(summary["subgroups"]) == {"anchor_before_2015", "anchor_from_2015"}
    assert summary["subgroups"]["anchor_from_2015"]["n_patients"] == len(labels)


def test_val_curves_score_every_epoch_and_mark_the_best(tmp_path: Path) -> None:
    root, fold, cfg = _project(tmp_path)
    models = root / "models"
    manifest = build_manifest(root, FINAL, {"configs/fixture.toml": cfg}, root / "data", models)
    inputs = _val_inputs(fold)
    labels = fold.data.labels.loc[fold.held]
    logged: list[dict[str, Any]] = []

    class Run:
        def log(self, record: dict[str, Any]) -> None:
            logged.append(record)

        def finish(self) -> None:
            logged.append({"finished": 1})

    curves = val_curves(
        manifest["members"], models, labels, SampleCache(inputs, None), CPU, 4, lambda m: Run()
    )
    assert len(curves) == cfg["finetune"]["max_epochs"] and curves["is_best"].sum() == 1
    assert curves["epoch"].tolist() == list(range(cfg["finetune"]["max_epochs"]))
    assert logged[-1] == {"finished": 1} and len(logged) == len(curves) + 1
    fm = load_fold_model(models / manifest["members"][0]["run_dir"])
    best = score_bundle(
        inputs, labels, [fm], {}, None, clock_year=2015, n_boot=20, seed=0, device=CPU
    )[0]["final"]["masked"]
    row = curves[curves["is_best"]].iloc[0]
    assert row["val_macro_auroc"] == pytest.approx(best["macro_auroc"], abs=1e-9)
    assert row["val_mAP"] == pytest.approx(best["mAP"], abs=1e-9)


def test_score_refuses_a_dirty_tree_a_changed_artifact_and_a_completed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, cfg = _project(tmp_path)
    dataset, models = root / "data", root / "models"
    (root / "configs").mkdir()
    final_path = root / "configs" / "final.toml"
    final_path.write_text(_toml(FINAL))
    manifest = build_manifest(
        root, val_score.load_final(final_path), {"configs/fixture.toml": cfg}, dataset, models
    )
    manifest_path = root / "models" / "final" / "manifest.json"
    write_json(manifest_path, manifest)
    reports = root / "reports" / "val"
    common = dict(device=CPU, wandb_mode="disabled", lightgbm=False)

    monkeypatch.setattr(val_score, "tracked_changes", lambda root: [" M src/x.py"])
    with pytest.raises(ValueError, match="working tree has changes"):
        val_score.run(root, final_path, dataset, models, manifest_path, reports, **common)
    assert not reports.exists()

    monkeypatch.setattr(val_score, "tracked_changes", lambda root: [])
    member = manifest["members"][0]
    (models / member["tokenizer"]).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        val_score.run(root, final_path, dataset, models, manifest_path, reports, **common)
    assert not reports.exists()

    manifest["model_hashes"][member["tokenizer"]] = val_score.file_hash(
        models / member["tokenizer"]
    )
    write_json(manifest_path, manifest)
    reports.mkdir(parents=True)
    write_json(reports / "bundle.json", {"status": "completed", "bundle_id": "abc"})
    with pytest.raises(ValueError, match="replacement"):
        val_score.run(root, final_path, dataset, models, manifest_path, reports, **common)
    assert json.loads((reports / "bundle.json").read_text())["status"] == "completed"

    final_path.write_text(_toml({**FINAL, "checkpoint": "last.pt"}))
    with pytest.raises(ValueError, match="changed after the freeze"):
        val_score.run(root, final_path, dataset, models, manifest_path, reports, **common)


def _toml(value: dict[str, Any]) -> str:
    def literal(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int | float):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(literal(x) for x in v) + "]"
        return json.dumps(str(v))

    lines = [f"{k} = {literal(v)}" for k, v in value.items() if not isinstance(v, dict)]
    for section, body in value.items():
        if isinstance(body, dict):
            lines += ["", f"[{section}]"] + [f"{k} = {literal(v)}" for k, v in body.items()]
    return "\n".join(lines) + "\n"
