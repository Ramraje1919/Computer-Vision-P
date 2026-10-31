"""
Path utilities for MOT17 project.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Return path to data directory."""
    return get_project_root() / "data" / "MOT17"


def get_weights_dir() -> Path:
    """Return path to weights directory."""
    return get_project_root() / "weights"


def get_outputs_dir() -> Path:
    """Return path to outputs directory."""
    out = get_project_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_reports_dir() -> Path:
    """Return path to reports directory."""
    rep = get_project_root() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    return rep

