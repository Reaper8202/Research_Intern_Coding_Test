"""Record the hashes of the files that leave the repository, and the commit they came from.

Run after the final PDF is built: ``make submission-record``. Writes one JSON file so that
``predictions.csv``, the report, the model package, and the frozen manifest can be matched to a
commit later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

CANDIDATES = (
    "predictions.csv",
    "docs/report.pdf",
    "reports/final/manifest.json",
    "models/hf/package.json",
    "requirements-lock.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = [Path(p) for p in CANDIDATES] + sorted(Path("docs").glob("*_results.pdf"))
    record = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no") or "",
        "files": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
            if path.is_file()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    for name, info in record["files"].items():
        print(f"{info['sha256'][:16]}  {info['bytes']:>10}  {name}")
    print(f"commit {record['commit']}  tracked changes: {'none' if not record['tracked_changes'] else 'YES'}")


if __name__ == "__main__":
    main()
