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

import plotly.graph_objects as go
import torch

from src.utils.data_utils import model_slug


def plot_layer_accuracy(
    layer_accs: list[float],
    model_name: str,
    out_path: str,
) -> None:
    """Plot probe accuracy vs. layer index and save to a PNG file.

    Includes:
      - Test accuracy curve (percentage y-axis)
      - Horizontal dashed line at 50 % (chance baseline)

    Args:
        layer_accs: Per-layer test accuracy values in [0, 1].
        model_name: Used in the plot title.
        out_path: Destination .png path.
    """
    layers = list(range(len(layer_accs)))
    pct = [acc * 100 for acc in layer_accs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=pct,
        mode="lines+markers",
        name="Test Accuracy",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
    ))
    fig.add_hline(
        y=50, line_dash="dash", line_color="red",
        annotation_text="Chance (50%)", annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"{model_name} — Layer-wise Probing Accuracy",
        xaxis_title="Layer Index",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(x=0.01, y=0.99),
        width=900, height=550,
        template="plotly_white",
    )
    fig.write_image(out_path)


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

    best_layer = int(max(range(len(layer_accs)), key=lambda i: layer_accs[i]))
    eval_result = {
        "model":      model_name,
        "best_layer": best_layer,
        "best_acc":   layer_accs[best_layer],
        "layer_accs": {str(i): acc for i, acc in enumerate(layer_accs)},
    }
    eval_path = Path(probe_dir) / f"{slug}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"Best layer: {best_layer} (acc={layer_accs[best_layer]:.3f}) → {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate per-layer deception probes.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--probe-dir", default="results/probes")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()
    main(args.model, args.act_dir, args.probe_dir, args.fig_dir)
