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

from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformer_lens.model_bridge import TransformerBridge

from src.utils.data_utils import iter_jsonl, load_config
from src.utils.model_utils import _get_residual_hooks, load_model


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
    flat_dataset = []
    for row in iter_jsonl(jsonl_path):
        flat_dataset.append({
            "text": row["honest_arg"],
            "label": 0,
            "id": row["id"] + "_honest",
        })
        flat_dataset.append({
            "text": row["deceptive_arg"],
            "label": 1,
            "id": row["id"] + "_deceptive",
        })
    return flat_dataset


def extract_activations(
    flat_dataset: list[dict],
    model: TransformerBridge,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run forward passes and collect layer activations for all examples.

    Args:
        flat_dataset: Output of build_flat_dataset().
        model: Loaded causal LM with registered hooks (call register_hooks first).
        batch_size: Number of examples per forward pass.

    Returns:
        (activations, labels, ids)
        activations: float32 Tensor [N, num_layers, hidden_dim]
        labels:      int64 Tensor [N]
        ids:         list of str length N
    """
    activations, labels, ids = [], [], []
    for i in tqdm(range(0, len(flat_dataset), batch_size)):
        batch = flat_dataset[i:i + batch_size]
        tokens = model.tokenizer([item['text'] for item in batch],
                                 return_tensors="pt",
                                 padding=True)
        residuals, hooks = _get_residual_hooks(model)
        _ = model.run_with_hooks(tokens["input_ids"], fwd_hooks=hooks)
        activations.append(torch.stack(residuals, dim=1))
        labels.extend([item["label"] for item in batch])
        ids.extend([item["id"] for item in batch])
    return torch.cat(activations, dim=0), torch.tensor(labels), ids


def split_dataset(
    activations: torch.Tensor,
    labels: torch.Tensor,
    ids: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[dict, dict]:
    """Stratified split into train and test sets.

    Returns two dicts each with keys: activations, labels, ids.
    """

    train_activations, test_activations, train_labels, test_labels, train_ids, test_ids = train_test_split(
        activations,
        labels,
        ids,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels)

    return {
        "activations": train_activations,
        "labels": train_labels,
        "ids": train_ids,
    }, {
        "activations": test_activations,
        "labels": test_labels,
        "ids": test_ids,
    }


def model_slug(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe slug.

    e.g. "Qwen/Qwen2.5-7B-Instruct" → "qwen2.5-7b-instruct"
    """
    return model_name.split("/")[-1].lower()


def main(config_path: str, inp_path: str, model_name: str,
         out_dir: str) -> None:
    """Run stage 4: extract activations → save train/test .pt files."""
    config = load_config(config_path)
    flat_dataset = build_flat_dataset(inp_path)
    model = load_model(model_name)
    activations, labels, ids = extract_activations(
        flat_dataset, model, config["activations"]["batch_size"])
    train_activations, test_activations = split_dataset(
        activations, labels, ids, config["activations"]["test_split"],
        config["activations"]["seed"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_name,
        "num_layers": model.cfg.n_layers,
        "hidden_dim": model.cfg.d_model,
    }
    train_activations.update(metadata)
    test_activations.update(metadata)
    torch.save(train_activations,
               Path(out_dir) / f"{model_slug(model_name)}_train.pt")
    torch.save(test_activations,
               Path(out_dir) / f"{model_slug(model_name)}_test.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract layer activations from a target model.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--inp",
                        default="data/filtered/arguments_filtered.jsonl")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--out-dir", default="results/activations")
    args = parser.parse_args()
    main(args.config, args.inp, args.model, args.out_dir)
