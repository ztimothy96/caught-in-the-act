"""Shared utilities for reading and writing data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
    """
    return model_name.split("/")[-1].lower()


def iter_jsonl(path: str) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, skipping blank lines.

    Args:
        path: Path to .jsonl file.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed.

    Args:
        rows: List of JSON-serializable dicts.
        path: Destination file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(row: dict, path: str) -> None:
    """Append a single dict as a JSON line to a file (creates file if absent).

    Useful for incremental writes so a partial run can be resumed.

    Args:
        row: JSON-serializable dict.
        path: Destination file path.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_json(path: str) -> dict | list:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict | list, path: str, indent: int = 2) -> None:
    """Save an object as pretty-printed JSON, creating parent dirs as needed."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def already_processed_ids(out_path: str) -> set[str]:
    """Return the set of "id" values already written to out_path.

    Used to resume a partial generation or filtering run without reprocessing.

    Args:
        out_path: Path to an existing (possibly partial) JSONL output file.

    Returns:
        Set of id strings, or empty set if the file does not exist.
    """
    if not Path(out_path).exists():
        return set()
    with open(out_path, "r", encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}
