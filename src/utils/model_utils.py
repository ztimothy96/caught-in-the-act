"""Shared utilities for loading HuggingFace models and managing hooks."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """Thin wrapper around a HuggingFace causal LM that mimics the parts of the
    TransformerBridge API used by this project (.tokenizer, .cfg, .run_with_hooks).

    Loading via HuggingFace directly keeps the model in bfloat16 with a single
    copy on the GPU, avoiding the ~2× memory overhead that TransformerBridge's
    enable_compatibility_mode() incurs when it converts weights to float32.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = SimpleNamespace(
            n_layers=model.config.num_hidden_layers,
            d_model=model.config.hidden_size,
        )
        self._device = next(model.parameters()).device

    def run_with_hooks(
        self,
        input_ids: torch.Tensor,
        fwd_hooks: list[tuple[str, Callable]] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> object:
        """Forward pass with optional residual-stream hooks.

        Hook names follow the pattern 'blocks.{layer}.hook_resid_post' (matching
        the TransformerLens convention used by _get_residual_hooks) so callers
        need no changes.

        Each hook callback receives (hidden_state, None) where hidden_state is the
        float32 CPU tensor at position [:, -1, :] for the batch — consistent with
        what the original TransformerBridge code produced.
        """
        handles: list = []
        if fwd_hooks:
            hook_map = dict(fwd_hooks)
            for layer_idx in range(self.cfg.n_layers):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name not in hook_map:
                    continue
                fn = hook_map[hook_name]
                layer = self.model.model.layers[layer_idx]

                def _make_hook(hook_fn: Callable):
                    def _hook(module, inp, output):
                        # Older transformers: decoder layer returns (hidden_state, ...)
                        # Newer transformers: returns the hidden-state tensor directly.
                        hidden = output[0] if isinstance(output, tuple) else output
                        # Mirror the TransformerLens callback signature: (activation, hook)
                        hook_fn(hidden, None)
                    return _hook

                handles.append(layer.register_forward_hook(_make_hook(fn)))

        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        try:
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        return output


def load_model(model_name: str) -> ModelWrapper:
    """Load a HuggingFace causal LM wrapped for activation extraction.

    Args:
        model_name: HuggingFace model ID or local path.

    Returns:
        ModelWrapper exposing .tokenizer, .cfg, and .run_with_hooks().
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return ModelWrapper(model, tokenizer)


def _get_residual_hooks(
    wrapper: ModelWrapper,
) -> tuple[list[torch.Tensor | None], list[tuple[str, Callable]]]:
    """Build per-layer hooks that capture the last-token residual-stream vector.

    Returns:
        residuals: list of length n_layers, filled in-place during the forward pass.
        hooks:     list of (hook_name, callback) pairs for run_with_hooks().
    """
    n_layers = wrapper.cfg.n_layers
    residuals: list[torch.Tensor | None] = [None] * n_layers

    def make_hook(layer: int) -> Callable:
        def store_residual(activation: torch.Tensor, hook) -> None:
            # Cast to float32 for downstream probe training
            residuals[layer] = activation[:, -1, :].detach().cpu().float()
        return store_residual

    hooks = [(f"blocks.{layer}.hook_resid_post", make_hook(layer))
             for layer in range(n_layers)]

    return residuals, hooks
