"""Weight pseudo quantization for LLaVA-like models (SpQR-style mixed precision)."""
from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .quant_funcs import (
    pseudo_quantize_tensor,
    pseudo_quantize_weight_per_column,
    pseudo_quantize_weight_spqr_style,
)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    """Return list of block modules (language part) for LLaVA / Qwen2-VL style."""
    cls_name = model.__class__.__name__
    if cls_name in ("LlavaLlamaForCausalLM", "LlavaQwenForCausalLM", "LlavaLlamaModel"):
        return model.model.layers
    if "Llama" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "Qwen2" in cls_name and hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if "InternVL" in cls_name and hasattr(model, "language_model"):
        return model.language_model.model.layers
    raise NotImplementedError(f"get_blocks not implemented for {cls_name}")


def _linear_layer_key(block_idx: int, linear_name: str) -> str:
    """Canonical key for a Linear layer, shared by hessian_collector and mixed_precision."""
    return f"layers.{block_idx}.{linear_name}"


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit: int = 4,
    q_group_size: int = -1,
    zero_point: bool = True,
    high_precision_columns: Set[Tuple[str, int]] | None = None,
    low_w_bit: int = 4,
):
    """
    In-place pseudo quantize Linear weights in model (language blocks only).

    If high_precision_columns is provided, uses SpQR-style mixed precision:
    group = one row x q_group_size columns; outlier columns are excluded when
    fitting scale/zero (replaced by row mean), then original float values are
    written back after quantization. The output weight is the merged result of
    "quantized part + outlier originals", so inference is a plain matmul.
    """
    layers = get_blocks(model)
    q_config = {"zero_point": zero_point, "q_group_size": q_group_size}
    use_mixed = high_precision_columns is not None
    if use_mixed and q_group_size <= 0:
        q_group_size = 128

    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            print(f"  [Block {i}] Quantizing {n} {tuple(m.weight.shape)}", flush=True)
            w = m.weight.data
            if use_mixed and high_precision_columns is not None:
                key = _linear_layer_key(i, n)
                m.weight.data = pseudo_quantize_weight_spqr_style(
                    w,
                    q_group_size=q_group_size,
                    high_precision_columns=high_precision_columns,
                    layer_key=key,
                    n_bits=low_w_bit,
                    zero_point=zero_point,
                )
            else:
                m.weight.data = pseudo_quantize_tensor(w, n_bits=w_bit, **q_config)
