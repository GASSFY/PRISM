"""
Mixed precision: Hessian-based global ASD ranking + top-ratio column selection.

Flow:
  1. For each Linear layer, compute importance_c = ||W[:, c]||^2 * diag(H)_c
  2. Global normalize K = importance / global_max  →  [0, 1]
  3. Per-layer z-score → Psi, then global normalize Psi / global_max_psi → [0, 1]
  4. ASD = theta1 * K_normalized + theta2 * Psi_normalized
  5. Global sort, take top ratio as high_precision_columns

Budget: prefer ``target_bit`` (average bit-width). Kept columns count as
``high_bit`` (default 16, FP16), others as ``low_bit`` (e.g. 4):

    target_bit = (1 - r) * low_bit + r * high_bit
    r = (target_bit - low_bit) / (high_bit - low_bit)
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from prism.metrics.asd import compute_importance, compute_Psi

_EPS = 1e-8


def ratio_from_target_bit(
    target_bit: float,
    low_bit: float,
    high_bit: float = 16.0,
) -> float:
    """
    Convert average-bit budget to high-precision column fraction.

    target_bit = (1 - r) * low_bit + r * high_bit
    → r = (target_bit - low_bit) / (high_bit - low_bit)
    """
    if high_bit <= low_bit:
        raise ValueError(f"high_bit ({high_bit}) must be > low_bit ({low_bit})")
    if target_bit <= low_bit:
        return 0.0
    if target_bit >= high_bit:
        return 1.0
    return float(target_bit - low_bit) / float(high_bit - low_bit)


def compute_global_asd_list(
    model: nn.Module,
    hessian_diag: dict[str, torch.Tensor],
    theta1: float = 0.8,
    theta2: float = 0.2,
) -> list[tuple[str, int, float]]:
    """
    Compute per-channel ASD scores across all layers using Hessian-based importance.

    model: root model (with .model.layers).
    hessian_diag: {layer_key: diag_H tensor [C]} from collect_hessian_diag.
    theta1, theta2: weights for K (absolute) and Psi (relative).

    Returns:
        [(layer_key, channel_idx, asd_value), ...], unsorted.
    """
    from prism.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key

    # Step 1: compute raw importance for every (layer, channel)
    layer_importance: dict[str, torch.Tensor] = {}
    layers = get_blocks(model)
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            if key not in hessian_diag:
                continue
            importance = compute_importance(linear.weight.data, hessian_diag[key])
            layer_importance[key] = importance

    if not layer_importance:
        return []

    # Step 2: global normalize K = importance / global_max → [0, 1]
    global_max = max(imp.max().item() for imp in layer_importance.values())
    global_max = max(global_max, _EPS)

    # Step 3: per-layer z-score → Psi, collect all Psi values
    layer_psi: dict[str, torch.Tensor] = {}
    for key, importance in layer_importance.items():
        layer_psi[key] = compute_Psi(importance, method="zscore")

    # Global normalize Psi → [0, 1]
    global_max_psi = max(psi.max().item() for psi in layer_psi.values())
    global_max_psi = max(global_max_psi, _EPS)

    # Step 4: ASD = theta1 * K_norm + theta2 * Psi_norm
    result: list[tuple[str, int, float]] = []
    for key in layer_importance:
        K_norm = layer_importance[key] / global_max
        Psi_norm = layer_psi[key] / global_max_psi
        asd = theta1 * K_norm + theta2 * Psi_norm
        for c in range(asd.shape[0]):
            result.append((key, c, asd[c].item()))

    return result


def select_high_precision_columns(
    global_asd_list: list[tuple[str, int, float]],
    ratio: float,
) -> set[tuple[str, int]]:
    """
    Sort by ASD descending, take top ratio fraction as high-precision columns.

    ratio: global fraction, e.g. 0.1 means top 10% of all channels across all layers.
    Returns:
        set of (layer_key, channel_idx) that keep original float precision.
    """
    if ratio <= 0 or not global_asd_list:
        return set()
    if ratio >= 1.0:
        return {(k, c) for k, c, _ in global_asd_list}

    sorted_list = sorted(global_asd_list, key=lambda t: t[2], reverse=True)
    n_total = len(sorted_list)
    # Non-zero ratio keeps at least one column (avoids rounding to 0 when budget is tiny).
    n_high = max(1, int(round(n_total * ratio)))
    return {(t[0], t[1]) for t in sorted_list[:n_high]}


def select_high_precision_columns_local(
    importance: torch.Tensor,
    ratio: float,
    theta1: float = 0.8,
    theta2: float = 0.2,
) -> set[int]:
    """
    Within one Linear: ASD = θ1*K_norm + θ2*Ψ_norm, keep top-ratio columns.

    importance: [C] raw K scores for this layer.
    Returns set of channel indices (ints).
    """
    if ratio <= 0 or importance.numel() == 0:
        return set()
    importance = importance.float().view(-1)
    n = importance.numel()
    if ratio >= 1.0:
        return set(range(n))

    psi = compute_Psi(importance, method="zscore")
    k_norm = importance / importance.max().clamp(min=_EPS)
    psi_norm = psi / psi.max().clamp(min=_EPS)
    asd = theta1 * k_norm + theta2 * psi_norm
    n_high = max(1, int(round(n * ratio)))
    top = torch.topk(asd, k=min(n_high, n), largest=True)
    return {int(i) for i in top.indices.tolist()}


def resolve_high_precision_ratio(
    *,
    target_bit: float | None = None,
    low_bit: float = 4.0,
    high_bit: float = 16.0,
    legacy_ratio: float | None = None,
) -> float:
    """
    Prefer target_bit → ratio. If target_bit is unset, fall back to legacy_ratio.
    """
    if target_bit is not None:
        return ratio_from_target_bit(target_bit, low_bit=low_bit, high_bit=high_bit)
    if legacy_ratio is not None:
        return float(legacy_ratio)
    return ratio_from_target_bit(4.01, low_bit=low_bit, high_bit=high_bit)
