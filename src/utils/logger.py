"""
Logging utility for hospital no-show prediction application.
Provides structured, standardized logging across pipeline and services.
"""
import logging
import os
import sys
from pathlib import Path

_LOGGERS = {}

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is in src/utils/logger.py -> parents[2] is root
    return Path(__file__).resolve().parents[2]

def get_logger(name: str = "hospital_no_show") -> logging.Logger:
    """
    Get or create a configured logger instance.
    Logs are written to standard output and to artifacts/execution.log.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler in artifacts/
        try:
            root_dir = get_project_root()
            log_dir = root_dir / "artifacts"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "execution.log"
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fall back to console only if file cannot be created
            pass

    _LOGGERS[name] = logger
    return logger
