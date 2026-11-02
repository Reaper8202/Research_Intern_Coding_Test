"""Shared logging setup for the command-line entry points."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Log to the console and, when ``log_file`` is given, also to that file (overwritten)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w"))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
