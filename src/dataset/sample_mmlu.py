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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def convert_to_binary(row: dict, subject: str, index: int,
                      rng: random.Random) -> dict:
    """Convert a single MMLU row (4-choice) to a binary-choice dict.

    Picks the correct answer as choice_A and samples one distractor as choice_B.
    correct is always "A"; presentation-order randomisation happens downstream.

    Args:
        row: Raw HuggingFace MMLU row with keys: question, choices, answer.
        subject: Subject name for the id field.
        index: Row index within subject for the id field.
        rng: Seeded RNG for reproducible distractor selection.

    Returns:
        Binary-choice dict (see module docstring for schema).
    """
    correct_idx = row["answer"]  # integer 0-3
    correct_text = row["choices"][correct_idx]

    distractor_indices = [i for i in range(4) if i != correct_idx]
    distractor_text = row["choices"][rng.choice(distractor_indices)]

    return {
        "id": f"{subject}_test_{index}",
        "subject": subject,
        "question": row["question"],
        "choice_A": correct_text,
        "choice_B": distractor_text,
        "correct": "A",
    }


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
    dataset = load_dataset(MMLU_HF_PATH, subject, split="test", streaming=True)
    if n is not None:
        # shuffle() requires an int seed, not a Random instance
        dataset = dataset.shuffle(seed=rng.randint(0, 2**31)).take(n)

    return [
        convert_to_binary(example, subject, i, rng)
        for i, example in enumerate(dataset)
    ]


def main(config_path: str, out_path: str) -> None:
    """Run stage 1: sample MMLU → write data/raw/mmlu_binary.jsonl."""
    cfg = load_config(config_path)
    mmlu_cfg = cfg["mmlu"]
    rng = random.Random(mmlu_cfg["seed"])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for subject in mmlu_cfg["subjects"]:
        rows = sample_subject(subject, mmlu_cfg["samples_per_subject"], rng)
        all_rows.extend(rows)
        print(f"  {subject}: {len(rows)} questions")

    with open(out, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample MMLU into binary-choice format.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--out", default="data/raw/mmlu_binary.jsonl")
    args = parser.parse_args()
    main(args.config, args.out)
