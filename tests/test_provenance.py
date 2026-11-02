from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehrforecast.provenance import file_hash, object_hash, verify_files, write_json


def test_hash_checks_detect_missing_or_changed_inputs(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("value\n1\n")
    expected = {path.name: file_hash(path)}
    verify_files(tmp_path, expected)
    path.write_text("value\n2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_files(tmp_path, expected)
    path.unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_files(tmp_path, expected)


def test_json_write_rejects_nonfinite_values_without_replacing_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_json(path, {"status": "passed"})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"status": "passed"}
    assert object_hash({"a": 1, "b": 2}) == object_hash({"b": 2, "a": 1})
