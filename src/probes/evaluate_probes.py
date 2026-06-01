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
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score


def load_probes(path: str) -> dict[int, dict]:
    """Load probe dict from a .pkl file."""
    raise NotImplementedError


def evaluate_layer(probe, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Return accuracy of probe on (X_test, y_test).

    Args:
        probe: Fitted sklearn LogisticRegression.
        X_test: [N, hidden_dim] float32.
        y_test: [N] int.
    """
    raise NotImplementedError


def plot_layer_accuracy(
    layer_accs: dict[int, float],
    model_name: str,
    out_path: str,
    cv_mean_accs: dict[int, float] | None = None,
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
    raise NotImplementedError


def main(
    config_path: str,
    model_name: str,
    act_dir: str,
    probe_dir: str,
    fig_dir: str,
) -> None:
    """Run stage 6: evaluate probes → save eval.json and layer-accuracy figure."""
    raise NotImplementedError


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate per-layer deception probes.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--probe-dir", default="results/probes")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()
    main(args.config, args.model, args.act_dir, args.probe_dir, args.fig_dir)
