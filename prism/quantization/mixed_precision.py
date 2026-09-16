"""
Mixed precision: second-order global K ranking + top-ratio column selection.

Flow (mixed / default):
  1. importance_c = ||W[:, c]||^2 * E[x_c^2]
  2. Global normalize K = importance / global_max  →  [0, 1]
  3. Global sort, take top ratio as high_precision_columns

Flow (modality fusion):
  1. K^V_c = ||W[:, c]||^2 * E[x_c^2 | vision]
  2. K^T_c = ||W[:, c]||^2 * E[x_c^2 | text]
  3. Globally normalize each modality, then
       K = modality_theta * K^T_norm + (1 - modality_theta) * K^V_norm
     so modality_theta=1 is text-only, modality_theta=0 is vision-only.
  4. Global sort on fused K.

Flow (worst-modality greedy / strategy A):
  Same K^V / K^T, then greedily keep columns that always rescue the
  modality with larger remaining uncovered gain. No fusion weight.

Flow (shared→only quota / strategy B):
  Top-ρB per modality → Shared / OnlyV / OnlyT → fill Shared first,
  then split remaining budget evenly between OnlyV and OnlyT.

Budget: prefer ``target_bit`` (average bit-width). Kept columns count as
``high_bit`` (default 16, FP16), others as ``low_bit`` (e.g. 4):

    target_bit = (1 - r) * low_bit + r * high_bit
    r = (target_bit - low_bit) / (high_bit - low_bit)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from prism.metrics.asd import compute_importance

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


def _layer_importance_from_energy(
    model: nn.Module,
    energy_diag: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from prism.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key

    layer_importance: dict[str, torch.Tensor] = {}
    layers = get_blocks(model)
    for i in range(len(layers)):
        for name, linear in get_named_linears(layers[i]).items():
            key = _linear_layer_key(i, name)
            if key not in energy_diag:
                continue
            layer_importance[key] = compute_importance(
                linear.weight.data, energy_diag[key],
            )
    return layer_importance


def _scores_from_importance(
    layer_importance: dict[str, torch.Tensor],
) -> list[tuple[str, int, float]]:
    """Global-max normalize importance, emit (layer_key, channel, score)."""
    if not layer_importance:
        return []

    global_max = max(imp.max().item() for imp in layer_importance.values())
    global_max = max(global_max, _EPS)

    result: list[tuple[str, int, float]] = []
    for key, importance in layer_importance.items():
        K_norm = importance / global_max
        for c in range(K_norm.shape[0]):
            result.append((key, c, K_norm[c].item()))
    return result


def compute_global_asd_list(
    model: nn.Module,
    hessian_diag: dict[str, torch.Tensor] | None = None,
    *,
    hessian_vision: dict[str, torch.Tensor] | None = None,
    hessian_text: dict[str, torch.Tensor] | None = None,
    modality_theta: float | None = None,
) -> list[tuple[str, int, float]]:
    """
    Compute per-channel keep scores across all layers (name kept for compatibility).

    Mixed path: pass ``hessian_diag`` only (activation energy E[x_c^2]).

    Modality fusion path:
        pass ``hessian_vision``, ``hessian_text``, and ``modality_theta``.
        Score uses K = θ·K^T_norm + (1-θ)·K^V_norm.

    Returns:
        [(layer_key, channel_idx, score), ...], unsorted.
    """
    if modality_theta is not None:
        if hessian_vision is None or hessian_text is None:
            raise ValueError(
                "modality_theta requires hessian_vision and hessian_text"
            )
        theta = float(modality_theta)
        if not (0.0 <= theta <= 1.0):
            raise ValueError(f"modality_theta must be in [0, 1], got {theta}")

        imp_v = _layer_importance_from_energy(model, hessian_vision)
        imp_t = _layer_importance_from_energy(model, hessian_text)
        keys = sorted(set(imp_v) & set(imp_t))
        if not keys:
            return []

        max_v = max(imp_v[k].max().item() for k in keys)
        max_t = max(imp_t[k].max().item() for k in keys)
        max_v = max(max_v, _EPS)
        max_t = max(max_t, _EPS)

        fused: dict[str, torch.Tensor] = {}
        for key in keys:
            kv = imp_v[key] / max_v
            kt = imp_t[key] / max_t
            fused[key] = theta * kt + (1.0 - theta) * kv

        # Already ~[0, 1] per modality; emit fused scores directly.
        result: list[tuple[str, int, float]] = []
        for key, score in fused.items():
            for c in range(score.shape[0]):
                result.append((key, c, score[c].item()))
        return result

    if hessian_diag is None:
        raise ValueError("hessian_diag is required when modality_theta is None")

    layer_importance = _layer_importance_from_energy(model, hessian_diag)
    return _scores_from_importance(layer_importance)


def select_high_precision_columns(
    global_asd_list: list[tuple[str, int, float]],
    ratio: float,
) -> set[tuple[str, int]]:
    """
    Sort by score descending, take top ratio fraction as high-precision columns.

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
