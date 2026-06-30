"""
Stage 6: Evaluate trained probes on the held-out test set and produce figures.

Input:  results/probes/{model_slug}_probes.pkl
        results/activations/{model_slug}_test.pt
Output: results/figures/{model_slug}_layer_accuracy.png
        results/probes/{model_slug}_eval.json

eval.json schema:
    {
        "model":       str,
        "best_layer":  int,
        "best_acc":    float,
        "layer_accs":  {layer_idx: float, ...},
        "cv_mean_accs": {layer_idx: float, ...} | null,
    }

Replicates Figure 1 from the paper: layer index on x-axis,
probe accuracy on y-axis, with a horizontal dashed line at 0.5 (chance).

Usage:
    python -m src.probes.evaluate_probes --model Qwen/Qwen2.5-7B-Instruct
    python -m src.probes.evaluate_probes \
        --config config/experiment.yaml \
        --model Qwen/Qwen2.5-7B-Instruct \
        --act-dir results/activations \
        --probe-dir results/probes \
        --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.utils.data_utils import model_slug


def plot_layer_accuracy(
    layer_accs: list[float],
    model_name: str,
    out_path: str,
) -> None:
    """Plot probe accuracy vs. layer index and save to file.

    Includes:
      - Test accuracy curve
      - CV mean ± std shading (if cv_mean_accs provided)
      - Horizontal dashed line at 0.5 (chance baseline)

    Args:
        layer_accs: {layer_idx: test_accuracy}.
        model_name: Used in the plot title.
        out_path: Destination .png path.
        cv_mean_accs: Optional {layer_idx: mean_cv_accuracy} for shading.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(layer_accs)), layer_accs, label="Test Accuracy")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Chance Level")
    plt.ylim(0, 1)
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Probe Accuracy")
    plt.legend()
    plt.savefig(out_path)


def main(
    model_name: str,
    act_dir: str,
    probe_dir: str,
    fig_dir: str,
) -> None:
    """Run stage 6: evaluate probes → save eval.json and layer-accuracy figure."""
    slug = model_slug(model_name)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    activation_data = torch.load(f"{act_dir}/{slug}_test.pt")
    probes = pickle.load(open(f"{probe_dir}/{slug}_probes.pkl", "rb"))
    scalers = pickle.load(open(f"{probe_dir}/{slug}_scalers.pkl", "rb"))
    layer_accs = [None] * len(probes)
    for layer in range(len(probes)):
        scaler = scalers[layer]
        X = scaler.transform(activation_data["activations"][:, layer, :])
        y = activation_data["labels"]
        layer_accs[layer] = probes[layer].score(X, y)
    plot_layer_accuracy(layer_accs, model_name,
                        f"{fig_dir}/{slug}_layer_accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate per-layer deception probes.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--probe-dir", default="results/probes")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()
    main(args.model, args.act_dir, args.probe_dir, args.fig_dir)
