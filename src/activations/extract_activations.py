"""
Stage 4: Extract hidden-state activations from the target model at every layer.

Input:  data/filtered/arguments_filtered.jsonl
Output: results/activations/{model_slug}_train.pt
        results/activations/{model_slug}_test.pt

Each .pt file is a dict:
    {
        "activations": Tensor[N, num_layers, hidden_dim],  # float32
        "labels":      Tensor[N],                          # 0=honest, 1=deceptive
        "ids":         list[str],                          # row["id"] + "_honest"/"_deceptive"
        "model":       str,
        "num_layers":  int,
        "hidden_dim":  int,
    }

The dataset is flattened: each filtered row contributes TWO examples
(honest_arg with label=0, deceptive_arg with label=1).

Activations are captured at the final non-padding token position of each layer,
which is the representation the model uses when completing the sequence.

Usage:
    python -m src.activations.extract_activations
    python -m src.activations.extract_activations \
        --config config/experiment.yaml \
        --inp data/filtered/arguments_filtered.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --out-dir results/activations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def load_model_and_tokenizer(model_name: str):
    """Load a HuggingFace causal LM and its tokenizer.

    Uses bfloat16 and device_map="auto" for multi-GPU support.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        (model, tokenizer) tuple. Model is in eval mode with grads disabled.
    """
    raise NotImplementedError


def register_hooks(model) -> tuple[dict, list]:
    """Attach forward hooks to every transformer hidden layer.

    Hooks store the output hidden state (before the next layer's input projection)
    for each layer in a shared dict keyed by layer index.

    Args:
        model: HuggingFace causal LM.

    Returns:
        (activation_store, hook_handles) — activation_store is populated after
        each forward pass; call handle.remove() on each handle when done.
    """
    raise NotImplementedError


def extract_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the hidden state at the final non-padding token position.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]  (1 = real token, 0 = padding)

    Returns:
        [batch, hidden_dim]
    """
    raise NotImplementedError


def build_flat_dataset(jsonl_path: str) -> list[dict]:
    """Read filtered JSONL and return a flat list of (text, label, id) dicts.

    Each row is expanded into two items:
        {"text": row["honest_arg"],    "label": 0, "id": row["id"] + "_honest"}
        {"text": row["deceptive_arg"], "label": 1, "id": row["id"] + "_deceptive"}

    Args:
        jsonl_path: Path to arguments_filtered.jsonl.

    Returns:
        List of dicts with keys: text, label, id.
    """
    raise NotImplementedError


def extract_activations(
    flat_dataset: list[dict],
    model,
    tokenizer,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run forward passes and collect layer activations for all examples.

    Args:
        flat_dataset: Output of build_flat_dataset().
        model: Loaded causal LM with registered hooks (call register_hooks first).
        tokenizer: Matching tokenizer.
        batch_size: Number of examples per forward pass.

    Returns:
        (activations, labels, ids)
        activations: float32 Tensor [N, num_layers, hidden_dim]
        labels:      int64 Tensor [N]
        ids:         list of str length N
    """
    raise NotImplementedError


def train_test_split(
    activations: torch.Tensor,
    labels: torch.Tensor,
    ids: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[dict, dict]:
    """Stratified split into train and test sets.

    Returns two dicts each with keys: activations, labels, ids.
    """
    raise NotImplementedError


def model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
    """
    return model_name.split("/")[-1].lower()


def main(config_path: str, inp_path: str, model_name: str, out_dir: str) -> None:
    """Run stage 4: extract activations → save train/test .pt files."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract layer activations from a target model.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--inp", default="data/filtered/arguments_filtered.jsonl")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--out-dir", default="results/activations")
    args = parser.parse_args()
    main(args.config, args.inp, args.model, args.out_dir)
