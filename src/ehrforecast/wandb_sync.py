"""Upload finished offline wandb runs to one entity and project.

A run directory is finished when its ``run.json`` (written at the end of every session) is newer
than its ``last.pt``. Runs that are still training are skipped. Sessions that carry the
``.wandb.synced`` marker that ``wandb sync`` writes after an upload are also skipped, so the
command can be repeated safely.

Usage::

    python -m ehrforecast.wandb_sync --models models/transformer \
        --entity derek-wu820-university-of-toronto
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from ehrforecast.logging_utils import setup_logging

log = logging.getLogger("ehrforecast.wandb_sync")


def finished_sessions(models_dir: Path) -> list[Path]:
    sessions = []
    for record in sorted(models_dir.rglob("run.json")):
        run_dir = record.parent
        last = run_dir / "last.pt"
        if not (run_dir / "wandb").is_dir() or not last.is_file():
            continue
        status = json.loads(record.read_text()).get("status")
        if (
            status in {"completed", "paused", "failed"}
            and record.stat().st_mtime >= last.stat().st_mtime
        ):
            sessions += [
                s
                for s in sorted((run_dir / "wandb").glob("offline-run-*"))
                if not any(s.glob("*.wandb.synced"))
            ]
    return sessions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="ehr-forecast")
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_file)
    sessions = finished_sessions(args.models)
    if not sessions:
        log.info("No finished offline runs to upload")
        return
    log.info("Uploading %d offline sessions to %s/%s", len(sessions), args.entity, args.project)
    wandb = Path(sys.executable).with_name("wandb")
    command = [str(wandb), "sync", "--entity", args.entity, "--project", args.project]
    subprocess.run([*command, *map(str, sessions)], check=True)


if __name__ == "__main__":
    main()
