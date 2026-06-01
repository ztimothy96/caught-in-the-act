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
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def load_activations(path: str) -> dict:
    """Load a .pt activation file produced by extract_activations.py.

    Args:
        path: Path to {model_slug}_train.pt.

    Returns:
        Dict with keys: activations (Tensor), labels (Tensor), ids, model,
        num_layers, hidden_dim.
    """
    raise NotImplementedError


def train_probe_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    max_iter: int,
    cv_folds: int | None,
) -> dict:
    """Fit a logistic regression probe on activations for a single layer.

    Args:
        X: [N, hidden_dim] float32 array of activations.
        y: [N] int array of labels (0=honest, 1=deceptive).
        C: Regularization strength (inverse of lambda).
        max_iter: Max solver iterations.
        cv_folds: If set, run stratified k-fold CV and record per-fold accuracy.

    Returns:
        Dict with keys: probe, train_acc, cv_accs.
    """
    raise NotImplementedError


def train_all_layers(
    activation_data: dict,
    config: dict,
) -> dict[int, dict]:
    """Train probes for all layers in the activation file.

    Args:
        activation_data: Output of load_activations().
        config: Full experiment config (uses config["probes"]).

    Returns:
        Dict mapping layer_idx → probe result dict.
    """
    raise NotImplementedError


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def main(config_path: str, model_name: str, act_dir: str, out_dir: str) -> None:
    """Run stage 5: train probes for all layers → save .pkl."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-layer deception probes.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--out-dir", default="results/probes")
    args = parser.parse_args()
    main(args.config, args.model, args.act_dir, args.out_dir)
