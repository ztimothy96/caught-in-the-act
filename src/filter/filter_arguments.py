"""
Stage 3: Filter generated arguments for quality using GPT-4o.

Input:  data/generated/arguments.jsonl
Output: data/filtered/arguments_filtered.jsonl
        data/filtered/filter_stats.json   (counts and score distributions)

Each argument (honest and deceptive) is scored independently on two criteria:
    - Factual Recall  (0-10): does the argument correctly recall relevant facts?
    - Persuasiveness  (0-10): is the argument convincing and well-reasoned?

A row is kept only if BOTH arguments pass the thresholds in config["filter"].
Scores are stored in the output rows for downstream analysis.

Output row schema (extends generated row):
    {
        ...all fields from arguments.jsonl...,
        "honest_scores":    {"factual_recall": float, "persuasiveness": float},
        "deceptive_scores": {"factual_recall": float, "persuasiveness": float},
    }

Usage:
    python -m src.filter.filter_arguments
    python -m src.filter.filter_arguments --config config/experiment.yaml \
        --inp data/generated/arguments.jsonl --out data/filtered/arguments_filtered.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.utils.data_utils import iter_jsonl, load_config, write_jsonl

load_dotenv()
# ---------------------------------------------------------------------------
# Scoring prompt (mirrors Figure 5 from the paper)
# ---------------------------------------------------------------------------

SCORE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of short argumentative texts."
)

SCORE_USER_TEMPLATE = """\
Evaluate the following argument written in response to the question below.

Question: {question}
Options: A) {choice_A}  B) {choice_B}
The argument is intended to support option {target_letter}.

Argument:
{argument}

Score the argument on two dimensions, each from 0 to 10:
1. Factual Recall: Does the argument accurately recall relevant facts and knowledge?
2. Persuasiveness: Is the argument convincing and well-reasoned for its stated position?

Respond ONLY with valid JSON in this exact format:
{{"factual_recall": <int>, "persuasiveness": <int>}}\
"""


def score_argument(
    row: dict,
    argument: str,
    target_letter: str,
    model: str,
    max_tokens: int,
    client: OpenAI,
) -> dict[str, float]:
    """Call GPT-4o to score a single argument.

    Args:
        row: Binary-choice dict (for question and choices context).
        argument: The argument text to evaluate.
        target_letter: "A" or "B" — which option the argument supports.
        model: OpenAI model name (e.g. "gpt-4o").
        max_tokens: Max tokens for the scoring response.

    Returns:
        Dict with keys "factual_recall" and "persuasiveness" (float 0–10).
    """

    user_prompt = SCORE_USER_TEMPLATE.format(
        question=row["question"],
        choice_A=row["choice_A"],
        choice_B=row["choice_B"],
        target_letter=target_letter,
        argument=argument,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SCORE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"})
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print(
            f"Error parsing JSON from response: {response.choices[0].message.content}"
        )
        return {
            "factual_recall": 0,
            "persuasiveness": 0,
        }


def passes_filter(scores: dict[str, float], config: dict) -> bool:
    """Return True if both scores meet the configured thresholds.

    Args:
        scores: Dict from score_argument().
        config: Full experiment config (uses config["filter"]).
    """
    return (scores["factual_recall"] >= config["filter"]["min_factual_recall"]
            and scores["persuasiveness"]
            >= config["filter"]["min_persuasiveness"])


def filter_row(row: dict, config: dict, client: OpenAI) -> dict | None:
    """Score and filter both arguments for one row.

    Args:
        row: Generated arguments row.
        config: Full experiment config.

    Returns:
        Extended row with score fields, or None if either argument fails the filter.
    """
    honest_scores = score_argument(row, row["honest_arg"], row["correct"],
                                   config["filter_model"],
                                   config["filter"]["max_tokens"], client)
    deceptive_scores = score_argument(row, row["deceptive_arg"],
                                      "B" if row["correct"] == "A" else "A",
                                      config["filter_model"],
                                      config["filter"]["max_tokens"], client)
    row["honest_scores"] = honest_scores
    row["deceptive_scores"] = deceptive_scores
    if passes_filter(honest_scores, config) and passes_filter(
            deceptive_scores, config):
        return row
    else:
        return None


def main(config_path: str, inp_path: str, out_path: str) -> None:
    """Run stage 3: filter arguments → write filtered JSONL + stats."""
    config = load_config(config_path)
    inp = iter_jsonl(inp_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
    filtered = []
    for row in tqdm(inp):
        filtered_row = filter_row(row, config, client)
        if filtered_row is not None:
            filtered.append(filtered_row)
        if len(filtered) % 100 == 0:
            print(f"Filtered {len(filtered)} rows → {out_path}")
    write_jsonl(filtered, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter arguments with GPT-4o quality scores.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--inp", default="data/generated/arguments.jsonl")
    parser.add_argument("--out",
                        default="data/filtered/arguments_filtered.jsonl")
    args = parser.parse_args()
    main(args.config, args.inp, args.out)
