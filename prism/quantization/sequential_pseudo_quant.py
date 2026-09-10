"""
Sequential PRISM path: per-block ASD select → pseudo-quant → short MSE on
kept columns → forward with quantized weights (error propagation).

Contrasts with the offline path (global FP Hessian → one-shot pseudo-quant).
"""
from __future__ import annotations

import gc
from typing import Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from prism.calibration.hessian_collector import capture_first_block_inputs
from prism.metrics.asd import compute_importance
from prism.quantization.mixed_precision import select_high_precision_columns_local
from prism.quantization.quant_funcs import pseudo_quantize_weight_prism
from prism.quantization.quantize import (
    _linear_layer_key,
    get_blocks,
    get_named_linears,
)


def _accumulate_diag_h(
    layer: nn.Module,
    layer_idx: int,
    all_inps: list[torch.Tensor],
    all_layer_kwargs: list[dict],
) -> dict[str, torch.Tensor]:
    """Collect E[x_c^2] for every Linear in ``layer`` under current activations."""
    sum_x2: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}

    def _make_hook(key: str):
        def hook(_module, args, _result):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach().float().view(-1, x.shape[-1])
            if x.numel() == 0:
                return
            x2 = x.pow(2).sum(dim=0)
            n = x.shape[0]
            if key not in sum_x2:
                sum_x2[key] = x2
                count[key] = n
            else:
                sum_x2[key] += x2
                count[key] += n

        return hook

    handles = []
    named = get_named_linears(layer)
    for name, linear in named.items():
        key = _linear_layer_key(layer_idx, name)
        handles.append(linear.register_forward_hook(_make_hook(key)))

    with torch.no_grad():
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in all_layer_kwargs[batch_idx].items()
            }
            kw["use_cache"] = False
            _ = layer(inp, **kw)[0]
            del inp, kw
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    return {k: sum_x2[k] / count[k] for k in sum_x2}


def _cache_fp_outputs(
    layer: nn.Module,
    all_inps: list[torch.Tensor],
    all_layer_kwargs: list[dict],
) -> list[torch.Tensor]:
    """FP block outputs on CPU (same order as ``all_inps``)."""
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in all_layer_kwargs[batch_idx].items()
            }
            kw["use_cache"] = False
            out = layer(inp, **kw)[0]
            outs.append(out.detach().float().cpu())
            del inp, out, kw
            torch.cuda.empty_cache()
    return outs


def _pseudo_quantize_block(
    layer: nn.Module,
    layer_idx: int,
    high_precision_columns: Set[Tuple[str, int]],
    q_group_size: int,
    low_w_bit: int,
    zero_point: bool = True,
) -> dict[str, torch.Tensor]:
    """In-place PRISM pseudo-quant for linears in one block. Returns HP masks."""
    hp_masks: dict[str, torch.Tensor] = {}
    named = get_named_linears(layer)
    for name, linear in named.items():
        key = _linear_layer_key(layer_idx, name)
        in_f = linear.weight.shape[1]
        mask = torch.zeros(in_f, dtype=torch.bool, device=linear.weight.device)
        for c in range(in_f):
            if (key, c) in high_precision_columns:
                mask[c] = True
        hp_masks[name] = mask
        linear.weight.data = pseudo_quantize_weight_prism(
            linear.weight.data,
            q_group_size=q_group_size,
            high_precision_columns=high_precision_columns,
            layer_key=key,
            n_bits=low_w_bit,
            zero_point=zero_point,
        )
    return hp_masks


def _mse_optimize_kept_columns(
    layer: nn.Module,
    all_inps: list[torch.Tensor],
    all_layer_kwargs: list[dict],
    fp_outs: list[torch.Tensor],
    hp_masks: dict[str, torch.Tensor],
    *,
    mse_epochs: int,
    mse_lr: float,
) -> float:
    """
    Short AdamW: only high-precision columns (and biases) receive gradients.
    Quantized columns stay frozen. Returns last-epoch mean MSE.
    """
    if mse_epochs <= 0:
        return float("nan")

    named = get_named_linears(layer)
    for linear in named.values():
        linear.weight.requires_grad_(True)
        if linear.bias is not None:
            linear.bias.requires_grad_(True)

    params = [p for p in layer.parameters() if p.requires_grad]
    if not params:
        return float("nan")

    optimizer = torch.optim.AdamW(params, lr=mse_lr)
    layer.train(False)  # keep dropout/etc off; grads still flow to weights

    last_mean = float("nan")
    for _epoch in range(mse_epochs):
        total = 0.0
        n_tok = 0
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            target = fp_outs[batch_idx].cuda()
            kw = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in all_layer_kwargs[batch_idx].items()
            }
            kw["use_cache"] = False

            out = layer(inp, **kw)[0]
            loss = F.mse_loss(out.float(), target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            for name, linear in named.items():
                mask = hp_masks[name].to(linear.weight.device)
                if linear.weight.grad is not None:
                    linear.weight.grad[:, ~mask] = 0

            optimizer.step()

            with torch.no_grad():
                total += loss.item() * out.shape[0] * out.shape[1]
                n_tok += out.shape[0] * out.shape[1]

            del inp, target, out, kw, loss
            torch.cuda.empty_cache()

        last_mean = total / max(n_tok, 1)

    for linear in named.values():
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    return last_mean


def _forward_block_to_next_inputs(
    layer: nn.Module,
    all_inps: list[torch.Tensor],
    all_layer_kwargs: list[dict],
) -> list[torch.Tensor]:
    new_inps: list[torch.Tensor] = []
    for batch_idx in range(len(all_inps)):
        inp = all_inps[batch_idx].cuda()
        kw = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in all_layer_kwargs[batch_idx].items()
        }
        kw["use_cache"] = False
        out = layer(inp, **kw)[0]
        new_inps.append(out.detach().cpu())
        del inp, out, kw
        torch.cuda.empty_cache()
    return new_inps


def sequential_pseudo_quantize_model(
    model_wrapper,
    forward_kwargs_list: list[dict],
    *,
    theta1: float = 0.8,
    theta2: float = 0.2,
    keep_ratio: float = 0.000833,
    w_bit: int = 4,
    q_group_size: int = 128,
    low_w_bit: int = 4,
    zero_point: bool = True,
    mse_epochs: int = 1,
    mse_lr: float = 1e-5,
) -> dict:
    """
    Route B: for each transformer block under current (possibly error-propagated)
    activations: ASD select → PRISM pseudo-quant → MSE on kept columns → forward.

    Returns a small stats dict (kept columns, last MSE per block).
    """
    del w_bit  # low_w_bit is the effective quant width for non-kept columns
    model = model_wrapper.model
    layers = get_blocks(model)
    if q_group_size <= 0:
        q_group_size = 128

    all_inps, all_layer_kwargs = capture_first_block_inputs(
        model_wrapper, forward_kwargs_list,
    )

    stats: dict = {"blocks": [], "total_kept": 0}
    desc = "[PRISM] Sequential MSE + error prop"

    for layer_idx in tqdm(range(len(layers)), desc=desc):
        layer = layers[layer_idx].cuda()
        named = get_named_linears(layer)

        diag_h = _accumulate_diag_h(layer, layer_idx, all_inps, all_layer_kwargs)
        fp_outs = _cache_fp_outputs(layer, all_inps, all_layer_kwargs)

        hp_cols: Set[Tuple[str, int]] = set()
        for name, linear in named.items():
            key = _linear_layer_key(layer_idx, name)
            if key not in diag_h:
                continue
            importance = compute_importance(linear.weight.data, diag_h[key])
            local = select_high_precision_columns_local(
                importance, keep_ratio, theta1=theta1, theta2=theta2,
            )
            for c in local:
                hp_cols.add((key, c))

        hp_masks = _pseudo_quantize_block(
            layer,
            layer_idx,
            hp_cols,
            q_group_size=q_group_size,
            low_w_bit=low_w_bit,
            zero_point=zero_point,
        )

        last_mse = _mse_optimize_kept_columns(
            layer,
            all_inps,
            all_layer_kwargs,
            fp_outs,
            hp_masks,
            mse_epochs=mse_epochs,
            mse_lr=mse_lr,
        )

        all_inps = _forward_block_to_next_inputs(
            layer, all_inps, all_layer_kwargs,
        )

        n_kept = len(hp_cols)
        stats["blocks"].append(
            {"layer": layer_idx, "kept": n_kept, "mse": last_mse},
        )
        stats["total_kept"] += n_kept
        print(
            f"  [Block {layer_idx}] kept={n_kept}, "
            f"mse_epochs={mse_epochs}, last_mse={last_mse:.6e}",
            flush=True,
        )

        layers[layer_idx] = layer.cpu()
        del layer, fp_outs, diag_h, hp_masks
        gc.collect()
        torch.cuda.empty_cache()

    return stats
