"""Shared utilities for loading HuggingFace models and managing hooks."""

from __future__ import annotations
from typing import Callable

import torch
from transformer_lens.model_bridge import TransformerBridge


def load_model(model_name: str):
    """Load a HuggingFace causal LM in eval mode.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        TransformerBridge object.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    bridge = TransformerBridge.boot_transformers(model_name,
                                                 device=device,
                                                 dtype=dtype,
                                                 trust_remote_code=True)
    bridge.enable_compatibility_mode()
    bridge.tokenizer.padding_side = "left"
    bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


def _get_residual_hooks(
    bridge: TransformerBridge
) -> tuple[list[torch.Tensor], list[tuple[str, Callable]]]:

    n_layers = bridge.cfg.n_layers
    residuals = [None] * n_layers

    def make_hook(layer):

        def store_residual(activation, hook):
            residuals[layer] = activation[:, -1, :].detach().cpu()

        return store_residual

    hooks = [(f"blocks.{layer}.hook_resid_post", make_hook(layer))
             for layer in range(n_layers)]

    return residuals, hooks
