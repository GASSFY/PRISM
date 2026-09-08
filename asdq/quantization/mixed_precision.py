"""
Mixed precision: Hessian-based global ASD ranking + top-ratio column selection.

Flow:
  1. For each Linear layer, compute importance_c = ||W[:, c]||^2 * diag(H)_c
  2. Global normalize K = importance / global_max  →  [0, 1]
  3. Per-layer z-score → Psi, then global normalize Psi / global_max_psi → [0, 1]
  4. ASD = theta1 * K_normalized + theta2 * Psi_normalized
  5. Global sort, take top ratio as high_precision_columns
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from asdq.metrics.asd import compute_importance, compute_Psi

_EPS = 1e-8


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
    from asdq.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key

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
    n_high = max(1, int(round(n_total * ratio)))        # 确保如果你设置了ratio非零，那么至少会保留一个高精度列，防止ratio太小（0.01 * 5)
    return {(t[0], t[1]) for t in sorted_list[:n_high]}
