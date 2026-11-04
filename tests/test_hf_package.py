"""Public package: copies match the frozen manifest, hashes are checked, reload reproduces."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ehrforecast.hf_package import (
    build_package,
    compare_predictions,
    package_predictions,
    verify_files,
)
from ehrforecast.predict import load_fold_model, predict_inputs
from ehrforecast.provenance import write_json
from ehrforecast.val_score import build_manifest
from tests.test_val_score import CPU, FINAL, _project, _val_inputs


def test_build_verify_and_reload_from_the_package(tmp_path: Path) -> None:
    root, fold, cfg = _project(tmp_path)
    dataset, models = root / "data", root / "models"
    manifest = build_manifest(root, FINAL, {"configs/fixture.toml": cfg}, dataset, models)
    manifest_path = root / "reports" / "final" / "manifest.json"
    write_json(manifest_path, manifest)
    config_file = root / "configs" / "fixture.toml"
    config_file.parent.mkdir()
    config_file.write_text("name = 'fixture'\n")
    out = root / "hf"
    package = build_package(manifest_path, models, dataset, out, [config_file], "x/y")
    member = package["members"][0]
    assert (out / member["run_dir"] / "best.pt").is_file()
    assert (out / member["tokenizer"]).is_file() and (out / "configs" / "fixture.toml").is_file()
    assert package["targets"] == manifest["targets"] and package["input_schema"].startswith(
        "prediction"
    )
    assert json.loads((out / "package.json").read_text())["version"] == package["version"]
    assert verify_files(out)["members"] == package["members"]

    inputs = _val_inputs(fold)
    ours = package_predictions(out, inputs, None, CPU)
    original = predict_inputs([load_fold_model(models / member["run_dir"])], inputs, CPU)
    pd.testing.assert_frame_equal(ours, original)
    assert compare_predictions(ours, original)["agrees"]
    shifted = original.copy()
    shifted.iloc[:1, :1] = 0.25 if original.to_numpy()[0, 0] > 0.5 else 0.75
    assert not compare_predictions(ours, shifted)["agrees"]

    (out / member["run_dir"] / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_files(out)


def test_build_refuses_an_artifact_that_changed_after_the_freeze(tmp_path: Path) -> None:
    root, _, cfg = _project(tmp_path)
    dataset, models = root / "data", root / "models"
    manifest = build_manifest(root, FINAL, {"configs/fixture.toml": cfg}, dataset, models)
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    (models / manifest["members"][0]["tokenizer"]).write_text("{}")
    with pytest.raises(ValueError, match="differs from the frozen manifest"):
        build_package(manifest_path, models, dataset, root / "hf", [], None)
