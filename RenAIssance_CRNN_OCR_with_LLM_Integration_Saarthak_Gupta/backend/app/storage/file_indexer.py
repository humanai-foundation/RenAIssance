"""Utilities for storage indexing, IDs, and metadata loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_numeric_id(parent: Path, prefix: str) -> str:
    """Return the next ID like `prefix_001` based on existing directories."""
    ensure_dir(parent)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_id = 0

    for child in parent.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_id = max(max_id, int(match.group(1)))

    return f"{prefix}_{max_id + 1:03d}"


def read_metadata(entry_dir: Path) -> dict[str, Any]:
    metadata_file = entry_dir / "metadata.json"
    if not metadata_file.exists():
        return {}

    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_rmtree(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return

    for child in path.iterdir():
        if child.is_dir():
            safe_rmtree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()
