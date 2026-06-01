"""Shared utilities for reading and writing data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, skipping blank lines.

    Args:
        path: Path to .jsonl file.
    """
    raise NotImplementedError


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed.

    Args:
        rows: List of JSON-serializable dicts.
        path: Destination file path.
    """
    raise NotImplementedError


def append_jsonl(row: dict, path: str) -> None:
    """Append a single dict as a JSON line to a file (creates file if absent).

    Useful for incremental writes so a partial run can be resumed.

    Args:
        row: JSON-serializable dict.
        path: Destination file path.
    """
    raise NotImplementedError


def load_json(path: str) -> dict | list:
    """Load a JSON file."""
    raise NotImplementedError


def save_json(obj: dict | list, path: str, indent: int = 2) -> None:
    """Save an object as pretty-printed JSON, creating parent dirs as needed."""
    raise NotImplementedError


def already_processed_ids(out_path: str) -> set[str]:
    """Return the set of "id" values already written to out_path.

    Used to resume a partial generation or filtering run without reprocessing.

    Args:
        out_path: Path to an existing (possibly partial) JSONL output file.

    Returns:
        Set of id strings, or empty set if the file does not exist.
    """
    raise NotImplementedError
