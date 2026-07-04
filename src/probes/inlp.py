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
import pickle

import plotly.graph_objects as go
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.utils.data_utils import load_config, model_slug

# Colour-blind-friendly qualitative palette (matches evaluate_probes.py)
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _param_count(model_name: str) -> float:
    """Extract parameter count in billions from a model name string."""
    import re
    m = re.search(r"(\d+\.?\d*)[Bb]", model_name)
    return float(m.group(1)) if m else 0.0


def _short_label(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_probe_direction(probe: LogisticRegression) -> np.ndarray:
    """Return the unit-norm "deception direction" in activation space.

    Args:
        probe: Fitted sklearn LogisticRegression (binary).

    Returns:
        [hidden_dim] unit-norm float32 array.
    """
    coef = probe.coef_.flatten()
    return coef / np.linalg.norm(coef)


def project_out_direction(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove a single direction from activation vectors via orthogonal projection.
    Args:
        X: [N, hidden_dim] activations.
        direction: [hidden_dim] unit-norm direction to remove.

    Returns:
        [N, hidden_dim] projected activations.
    """
    return X - (X @ direction)[:, np.newaxis] * direction


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
    round_accs = []
    for _ in tqdm(range(max_rounds)):
        probe = LogisticRegression(C=C, max_iter=max_iter)
        probe.fit(X_train, y_train)
        round_accs.append(probe.score(X_test, y_test))
        if round_accs[-1] < floor_accuracy:
            break
        direction = get_probe_direction(probe)
        X_train = project_out_direction(X_train, direction)
        X_test = project_out_direction(X_test, direction)
    return round_accs


def plot_inlp_curve(round_accs: list[float], model_name: str, layer: int,
                    out_path: str) -> None:
    """Plot INLP accuracy vs. projection round and save to file.

    Includes a horizontal dashed line at 0.5 (chance baseline).

    Args:
        round_accs: Output of run_inlp().
        model_name: Used in the plot title.
        layer: Layer index used, shown in the title.
        out_path: Destination .png path.
    """
    rounds = list(range(len(round_accs)))
    pct = [acc * 100 for acc in round_accs]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=rounds,
                   y=pct,
                   mode="lines+markers",
                   name=f"{model_name} (Layer {layer})",
                   line=dict(color="#1f77b4", width=2),
                   marker=dict(size=6)))
    fig.add_hline(y=50,
                  line_dash="dash",
                  line_color="red",
                  annotation_text="Chance (50%)",
                  annotation_position="bottom right")
    fig.update_layout(title=f"Layerwise Out-Projection Accuracy Comparison",
                      xaxis_title="INLP Rounds",
                      yaxis_title="Accuracy",
                      yaxis=dict(range=[0, 100], ticksuffix="%"),
                      legend=dict(x=0.01, y=0.99),
                      width=900,
                      height=550,
                      template="plotly_white")
    fig.write_image(out_path)


def plot_all_models_inlp(probe_dir: str, fig_dir: str) -> None:
    """Read all *_inlp.json files and plot every model's INLP curve on one figure.

    Models are sorted by parameter count (ascending) and assigned colours from
    the palette in that order.  Saves to {fig_dir}/all_models_inlp_curve.png.
    """
    inlp_files = sorted(Path(probe_dir).glob("*_inlp.json"))
    if not inlp_files:
        return

    results = []
    for path in inlp_files:
        with open(path) as f:
            results.append(json.load(f))

    results.sort(key=lambda d: _param_count(d["model"]))

    fig = go.Figure()
    for i, data in enumerate(results):
        pct = [acc * 100 for acc in data["round_accs"]]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(pct))),
                y=pct,
                mode="lines+markers",
                name=_short_label(data["model"]),
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                marker=dict(size=5),
            ))

    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        annotation_text="Chance (50%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Layerwise Out-Projection Accuracy Comparison",
        xaxis_title="INLP Rounds",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(x=0.7, y=0.99),
        width=1000,
        height=580,
        template="plotly_white",
    )
    out_path = Path(fig_dir) / "all_models_inlp_curve.png"
    fig.write_image(str(out_path))
    print(f"Combined INLP figure → {out_path}")


def main(
    config_path: str,
    model_name: str,
    act_dir: str,
    probe_dir: str,
    fig_dir: str,
    layer: int | None,
) -> None:
    """Run INLP analysis on the specified layer → save curve and JSON."""
    slug = model_slug(model_name)
    config = load_config(config_path)
    train_data = torch.load(f"{act_dir}/{slug}_train.pt")
    if layer is None:
        with open(f"{probe_dir}/{slug}_eval.json", "r") as f:
            eval_data = json.load(f)
        layer = eval_data["best_layer"]
    scalers = pickle.load(open(f"{probe_dir}/{slug}_scalers.pkl", "rb"))
    scaler = scalers[layer]
    X_train = scaler.transform(train_data["activations"][:, layer, :].numpy())
    y_train = train_data["labels"].numpy()
    test_data = torch.load(f"{act_dir}/{slug}_test.pt")
    X_test = scaler.transform(test_data["activations"][:, layer, :].numpy())
    y_test = test_data["labels"].numpy()

    # Reduce to a PCA subspace before INLP so that the probe is always
    # operating in an overdetermined regime (N > D)
    n_components = min(X_train.shape[0] - 1, 512)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    round_accs = run_inlp(X_train, y_train, X_test, y_test,
                          config["inlp"]["max_rounds"],
                          config["inlp"]["floor_accuracy"])
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plot_inlp_curve(round_accs, model_name, layer,
                    f"{fig_dir}/{slug}_inlp_curve.png")
    with open(Path(probe_dir) / f"{slug}_inlp.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "layer": layer,
                "round_accs": round_accs,
                "n_rounds": len(round_accs)
            }, f)
    plot_all_models_inlp(probe_dir, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="INLP analysis of deception directions.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to analyze (None = best layer from eval)")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--probe-dir", default="results/probes")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()
    main(args.config, args.model, args.act_dir, args.probe_dir, args.fig_dir,
         args.layer)
