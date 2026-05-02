"""Standard `logging`-based logger factory.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from config import settings

_CONFIGURED = False


def _configure_root_logger(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        settings.logs_dir / "app.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger."""
    _configure_root_logger(settings.log_level)
    return logging.getLogger(name or "app")
