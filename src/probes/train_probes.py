"""
Stage 5: Train a logistic regression probe on activations at each layer.

Input:  results/activations/{model_slug}_train.pt
Output: results/probes/{model_slug}_probes.pkl

The .pkl file is a dict keyed by layer index:
    {
        layer_idx (int): {
            "probe":     sklearn LogisticRegression,
            "train_acc": float,
            "cv_accs":   list[float] | None,   # per-fold accuracies if cv_folds set
        }
    }

Usage:
    python -m src.probes.train_probes --model Qwen/Qwen2.5-7B-Instruct
    python -m src.probes.train_probes \
        --config config/experiment.yaml \
        --model Qwen/Qwen2.5-7B-Instruct \
        --act-dir results/activations \
        --out-dir results/probes
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

from src.utils.data_utils import load_config, model_slug


def train_all_layers(
    activation_data: dict,
    config: dict,
) -> tuple[list[LogisticRegression], list[StandardScaler]]:
    """Train probes for all layers in the activation file.

    Args:
        activation_data: Output of load_activations().
        config: Full experiment config (uses config["probes"]).

    Returns:
        List of probes and scalers.
    """
    activations, labels, _, _, num_layers, _ = activation_data.values()
    probes = [None] * num_layers
    scalers = [None] * num_layers

    for layer in tqdm(range(num_layers)):
        scaler = StandardScaler()
        X = scaler.fit_transform(activations[:, layer, :])
        y = labels
        C = config["probes"]["C"]
        max_iter = config["probes"]["max_iter"]
        probe = LogisticRegression(C=C, max_iter=max_iter)
        probe.fit(X, y)
        probes[layer] = probe
        scalers[layer] = scaler
    return probes, scalers


def main(config_path: str, model_name: str, act_dir: str,
         out_dir: str) -> None:
    """Run stage 5: train probes for all layers → save .pkl."""
    slug = model_slug(model_name)
    config = load_config(config_path)
    # Dict with keys: activations (Tensor), labels (Tensor), ids, model, num_layers, hidden_dim.
    activation_data = torch.load(f"{act_dir}/{slug}_train.pt")
    probes, scalers = train_all_layers(activation_data, config)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{slug}_probes.pkl", "wb") as f:
        pickle.dump(probes, f)
    with open(Path(out_dir) / f"{slug}_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train per-layer deception probes.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--out-dir", default="results/probes")
    args = parser.parse_args()
    main(args.config, args.model, args.act_dir, args.out_dir)
