# src/logging_config.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from causalityiq_app.settings import settings


def setup_logging() -> None:
  log_cfg = settings.logging

  # Ensure log directory exists
  log_path = Path(log_cfg.file)
  log_path.parent.mkdir(parents=True, exist_ok=True)

  level = getattr(logging, log_cfg.level.upper(), logging.INFO)

  logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
      logging.StreamHandler(),
      _make_rotating_file_handler(log_path),
    ],
  )


def _make_rotating_file_handler(path: Path) -> RotatingFileHandler:
  rot = settings.logging.rotation
  handler = RotatingFileHandler(
    path,
    maxBytes=rot.max_bytes if rot.enabled else 0,
    backupCount=rot.backup_count if rot.enabled else 0,
    encoding=rot.encoding,
  )
  formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
  )
  handler.setFormatter(formatter)
  return handler
