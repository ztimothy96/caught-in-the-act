"""
Stage 1: Sample questions from MMLU and convert to binary-choice format.

Input:  config/experiment.yaml (mmlu.subjects, mmlu.samples_per_subject, mmlu.seed)
Output: data/raw/mmlu_binary.jsonl

Each output row:
    {
        "id":        str,   # "{subject}_{split}_{index}"
        "subject":   str,
        "question":  str,
        "choice_A":  str,   # the correct answer
        "choice_B":  str,   # one distractor
        "correct":   "A"    # always A — distractor placement is randomized later
    }

Usage:
    python -m src.dataset.sample_mmlu
    python -m src.dataset.sample_mmlu --config config/experiment.yaml --out data/raw/mmlu_binary.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


MMLU_HF_PATH = "cais/mmlu"
# MMLU answer index → letter
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_subject(
    subject: str,
    n: int | None,
    rng: random.Random,
) -> list[dict]:
    """Download one MMLU subject and return binary-choice rows.

    Args:
        subject: MMLU subject name matching cais/mmlu subset names.
        n: Number of questions to keep (None = all).
        rng: Seeded RNG for reproducible distractor selection and shuffling.

    Returns:
        List of dicts with keys: id, subject, question, choice_A, choice_B, correct.
    """
    raise NotImplementedError


def convert_to_binary(row: dict, subject: str, index: int, rng: random.Random) -> dict:
    """Convert a single MMLU row (4-choice) to a binary-choice dict.

    Picks the correct answer as one option and samples one distractor.
    Randomly assigns correct answer to choice_A or choice_B so the model
    cannot exploit position bias.

    Args:
        row: Raw HuggingFace MMLU row with keys: question, choices, answer.
        subject: Subject name for the id field.
        index: Row index within subject for the id field.
        rng: Seeded RNG.

    Returns:
        Binary-choice dict (see module docstring for schema).
    """
    raise NotImplementedError


def main(config_path: str, out_path: str) -> None:
    """Run stage 1: sample MMLU → write data/raw/mmlu_binary.jsonl."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample MMLU into binary-choice format.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--out", default="data/raw/mmlu_binary.jsonl")
    args = parser.parse_args()
    main(args.config, args.out)
