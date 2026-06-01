"""Shared utilities for loading HuggingFace models and managing hooks."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, dtype: torch.dtype = torch.bfloat16):
    """Load a HuggingFace causal LM in eval mode.

    Args:
        model_name: HuggingFace model ID or local path.
        dtype: Weight dtype; bfloat16 is default for memory efficiency.

    Returns:
        model in eval mode, gradients disabled.
    """
    raise NotImplementedError


def load_tokenizer(model_name: str):
    """Load tokenizer with left-padding (required for batch inference).

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        Tokenizer with padding_side="left" and pad_token set.
    """
    raise NotImplementedError


def get_num_layers(model) -> int:
    """Return the number of transformer hidden layers in the model.

    Works for Qwen2 and LLaMA-style architectures.
    """
    raise NotImplementedError


def attach_layer_hooks(model) -> tuple[dict[int, torch.Tensor], list]:
    """Register hooks that capture the output of each transformer layer.

    Hooks write to a shared dict keyed by layer index. Call .remove() on
    each handle in the returned list when extraction is complete.

    Args:
        model: HuggingFace causal LM.

    Returns:
        (store, handles)
        store:   dict that will contain {layer_idx: Tensor[batch, seq, hidden]}
                 after each forward pass.
        handles: list of hook handles to remove after use.
    """
    raise NotImplementedError
