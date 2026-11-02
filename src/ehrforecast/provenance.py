"""Small, explicit fingerprints and atomic JSON files for experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Reject non-finite JSON and atomically replace only a complete file."""
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(name).replace(path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def verify_files(root: Path, hashes: dict[str, Any]) -> None:
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"artifact hash mismatch: {relative}")


def code_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "requirements-lock.txt", "Makefile")]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}
