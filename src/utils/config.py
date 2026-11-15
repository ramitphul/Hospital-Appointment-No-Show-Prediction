"""
Configuration loading and path resolution utilities.
"""
import os
from pathlib import Path
from typing import Any, Dict
import yaml

def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[2]

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    Resolves relative paths against the project root.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_resolved_path(relative_or_absolute_path: str) -> Path:
    """Convert relative path in config to absolute path from project root."""
    path = Path(relative_or_absolute_path)
    if path.is_absolute():
        return path
    return get_project_root() / path
