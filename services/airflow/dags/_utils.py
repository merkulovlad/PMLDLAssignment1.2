"""Shared helpers for Airflow DAG modules."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_root_on_path(current_file: Path, levels: int = 3) -> Path:
    """Insert the repository root into ``sys.path`` and return that path."""

    repo_root = current_file.resolve().parents[levels]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


__all__ = ["ensure_repo_root_on_path"]
