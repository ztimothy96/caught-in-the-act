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
from collections import defaultdict
from pathlib import Path
import random

from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from src.utils.data_utils import iter_jsonl, load_config, model_slug
from src.utils.model_utils import ModelWrapper, _get_residual_hooks, load_model

_PROBE_CONTEXT_TEMPLATE = """\
Question: {question}

Options:
A) {choice_A}
B) {choice_B}

Argument: {argument}\
"""


def build_flat_dataset(jsonl_path: str, seed: int = 42) -> list[dict]:
    """Read filtered JSONL and return a flat list of context-aware example dicts.

    Each row is expanded into two items — one honest (label=0) and one deceptive
    (label=1).  Each item stores a messages list with the question, both choices,
    and the argument (matching the black-box judge's input from Figure 8), so the
    probe model can form representations that distinguish honest from deceptive
    content without being given the generation direction as a shortcut.

    Args:
        jsonl_path: Path to arguments_filtered.jsonl.
        seed: RNG seed for reproducible A/B randomization.

    Returns:
        List of dicts with keys: messages (list), label (int), id (str).
    """
    rng = random.Random(seed)
    flat_dataset = []
    for row in iter_jsonl(jsonl_path):
        if rng.random() < 0.5:
            disp_A, disp_B = row["choice_B"], row["choice_A"]
        else:
            disp_A, disp_B = row["choice_A"], row["choice_B"]

        for arg, label, suffix in [
            (row["honest_arg"],    0, "_honest"),
            (row["deceptive_arg"], 1, "_deceptive"),
        ]:
            user_content = _PROBE_CONTEXT_TEMPLATE.format(
                question=row["question"],
                choice_A=disp_A,
                choice_B=disp_B,
                argument=arg,
            )
            flat_dataset.append({
                "messages": [
                    {"role": "user", "content": user_content},
                ],
                "label": label,
                "id": row["id"] + suffix,
            })
    return flat_dataset


def extract_activations(
    flat_dataset: list[dict],
    model: ModelWrapper,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run forward passes and collect layer activations for all examples.

    Applies the model's chat template to each example so that the instruct model
    processes the conversation with the correct special tokens, then captures the
    residual-stream vector at the final (non-padding) token position for every layer.

    Args:
        flat_dataset: Output of build_flat_dataset().
        model: Loaded causal LM wrapper.
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
        texts = [
            model.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in batch
        ]
        tokens = model.tokenizer(texts, return_tensors="pt", padding=True)
        residuals, hooks = _get_residual_hooks(model)
        _ = model.run_with_hooks(
            tokens["input_ids"],
            fwd_hooks=hooks,
            attention_mask=tokens.get("attention_mask"),
        )
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
    """Group-aware train/test split that keeps question pairs together.

    Returns two dicts each with keys: activations, labels, ids.
    """
    # Group example indices by base question ID (strip _honest / _deceptive)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, id_ in enumerate(ids):
        base = id_.rsplit("_", 1)[0]
        groups[base].append(i)

    group_keys = list(groups.keys())
    train_keys, test_keys = train_test_split(
        group_keys,
        test_size=test_fraction,
        random_state=seed,
    )

    def gather(keys: list[str]) -> dict:
        idx = [i for k in keys for i in groups[k]]
        return {
            "activations": activations[idx],
            "labels":      labels[idx],
            "ids":         [ids[i] for i in idx],
        }

    return gather(train_keys), gather(test_keys)


def main(config_path: str, inp_path: str, model_name: str,
         out_dir: str) -> None:
    """Run stage 4: extract activations → save train/test .pt files."""
    config = load_config(config_path)
    flat_dataset = build_flat_dataset(inp_path, seed=config["activations"]["seed"])
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
    slug = model_slug(model_name)
    torch.save(train_activations, Path(out_dir) / f"{slug}_train.pt")
    torch.save(test_activations, Path(out_dir) / f"{slug}_test.pt")


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
