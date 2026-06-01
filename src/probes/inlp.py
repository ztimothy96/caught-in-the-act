"""
Iterative Nullspace Projection (INLP) analysis.

Finds and removes successive orthogonal "deception directions" from the activation
space to measure how robustly deception is encoded (Figure 2 from the paper).

Algorithm per round:
    1. Train a probe on current activations.
    2. Extract the probe's weight vector (the deception direction).
    3. Project activations onto the nullspace of that vector (remove the direction).
    4. Record accuracy of the new probe trained on projected activations.
    5. Repeat until max_rounds or accuracy drops to floor_accuracy.

Input:  results/activations/{model_slug}_train.pt  (for fitting projection)
        results/activations/{model_slug}_test.pt   (for evaluating at each round)
Output: results/figures/{model_slug}_inlp_curve.png
        results/probes/{model_slug}_inlp.json

inlp.json schema:
    {
        "model":         str,
        "layer":         int,     # layer used for INLP (best layer from eval)
        "round_accs":    [float], # test accuracy after each projection round
        "n_rounds":      int,
    }

Usage:
    python -m src.probes.inlp --model Qwen/Qwen2.5-7B-Instruct --layer 20
    python -m src.probes.inlp \
        --config config/experiment.yaml \
        --model Qwen/Qwen2.5-7B-Instruct \
        --layer 20 \
        --act-dir results/activations \
        --out-dir results/probes \
        --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression


def get_probe_direction(probe: LogisticRegression) -> np.ndarray:
    """Return the unit-norm weight vector of a binary logistic regression probe.

    This is the "deception direction" in activation space.

    Args:
        probe: Fitted sklearn LogisticRegression (binary).

    Returns:
        [hidden_dim] unit-norm float32 array.
    """
    raise NotImplementedError


def project_out_direction(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove a single direction from activation vectors via orthogonal projection.

    X_projected = X - (X @ d) * d^T   where d is the unit-norm direction.

    Args:
        X: [N, hidden_dim] activations.
        direction: [hidden_dim] unit-norm direction to remove.

    Returns:
        [N, hidden_dim] projected activations.
    """
    raise NotImplementedError


def run_inlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_rounds: int,
    floor_accuracy: float,
    C: float = 1.0,
    max_iter: int = 1000,
) -> list[float]:
    """Run INLP and return test accuracy after each projection round.

    Args:
        X_train: [N_train, hidden_dim].
        y_train: [N_train] binary labels.
        X_test:  [N_test, hidden_dim].
        y_test:  [N_test] binary labels.
        max_rounds: Maximum number of projection rounds.
        floor_accuracy: Stop early when test accuracy drops to this level.
        C: Logistic regression regularization.
        max_iter: Logistic regression solver iterations.

    Returns:
        List of test accuracies, one per round (length ≤ max_rounds).
        Index 0 = accuracy before any projection (original probe).
    """
    raise NotImplementedError


def plot_inlp_curve(round_accs: list[float], model_name: str, layer: int, out_path: str) -> None:
    """Plot INLP accuracy vs. projection round and save to file.

    Includes a horizontal dashed line at 0.5 (chance baseline).

    Args:
        round_accs: Output of run_inlp().
        model_name: Used in the plot title.
        layer: Layer index used, shown in the title.
        out_path: Destination .png path.
    """
    raise NotImplementedError


def main(
    config_path: str,
    model_name: str,
    layer: int,
    act_dir: str,
    out_dir: str,
    fig_dir: str,
) -> None:
    """Run INLP analysis on the specified layer → save curve and JSON."""
    raise NotImplementedError


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INLP analysis of deception directions.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to analyze")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--out-dir", default="results/probes")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()
    main(args.config, args.model, args.layer, args.act_dir, args.out_dir, args.fig_dir)
