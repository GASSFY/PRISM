"""Minimal quantization helpers (RTN-style) for ASDQ. No dependency on MBQ."""
from __future__ import annotations

from typing import Set, Tuple

import torch

_EPS = 1e-9


@torch.no_grad()
def pseudo_quantize_tensor(
    tensor: torch.Tensor,
    n_bits: int = 8,
    zero_point: bool = True,
    q_group_size: int = -1,
    per_tensor: bool = False,
) -> torch.Tensor:
    """Per-group or per-tensor symmetric/asymmetric pseudo quantization (no inplace)."""
    org_shape = tensor.shape
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)       # 准确来说，现在就是分成（有多少组，每组多少样本）
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    tensor = (
        (torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros)
        * scales
    )
    return tensor.reshape(org_shape)


@torch.no_grad()
def pseudo_quantize_weight_per_column(
    weight: torch.Tensor,
    n_bits_per_column: list[int] | torch.Tensor,
    zero_point: bool = True,
) -> torch.Tensor:
    """
    对权重矩阵按列使用不同比特伪量化（用于 ASD 混合精度：高 ASD 列高 bit，低 ASD 列低 bit）。

    weight: (out_features, in_features)，即每个列对应一个输入通道。
    n_bits_per_column: 长度为 in_features，每列使用的比特数。
    """
    out_f, in_f = weight.shape
    result = weight.clone()
    for j in range(in_f):
        bits = (
            int(n_bits_per_column[j].item())
            if isinstance(n_bits_per_column, torch.Tensor)
            else int(n_bits_per_column[j])
        )
        col = result[:, j].view(1, -1)  # (1, out) → 每列一个 scale/zero
        q_col = pseudo_quantize_tensor(
            col, n_bits=bits, zero_point=zero_point, q_group_size=col.shape[1]
        )
        result[:, j] = q_col.view(-1)
    return result


# ---------- SpQR-style: 一行×一坨列 分组，组内剔除「保存精度」后算 scale/zero，推理时合并 outlier ----------


def _get_scale_zero_per_row(
    tensor_2d: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 2D 张量 (out_f, group_size) 按行算 scale/zero，返回 scale (out_f, 1), zero (out_f, 1)。"""
    assert tensor_2d.dim() == 2
    x = tensor_2d.float()
    xmin = x.amin(dim=1, keepdim=True)
    xmax = x.amax(dim=1, keepdim=True)
    if not zero_point:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = torch.where(xmin < 0, -xmax, xmin)
    tmp = (xmin == xmax).squeeze(1)
    if tmp.any():
        xmin = xmin.clone()
        xmax = xmax.clone()
        xmin[tmp.unsqueeze(1)] = -1
        xmax[tmp.unsqueeze(1)] = 1
    max_int = (2**n_bits - 1) if zero_point else (2 ** (n_bits - 1) - 1)
    min_int = 0 if zero_point else -(2 ** (n_bits - 1))
    scale = (xmax - xmin).clamp(min=_EPS) / max_int
    zero = (
        (-torch.round(xmin / scale)).clamp(min_int, max_int)
        if zero_point
        else torch.full_like(scale, (max_int + 1) / 2)
    )
    return scale, zero


def _quantize_dequantize_with_scale_zero(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    n_bits: int,
    zero_point: bool = True,
) -> torch.Tensor:
    """用给定的 scale/zero 做量化再反量化，返回与 tensor 同形的张量。"""
    max_int = (2**n_bits - 1) if zero_point else (2 ** (n_bits - 1) - 1)
    min_int = 0 if zero_point else -(2 ** (n_bits - 1))
    scale = scale.expand_as(tensor).clamp(min=_EPS)
    zero = zero.expand_as(tensor)
    q = torch.clamp(torch.round(tensor / scale + zero), min_int, max_int)
    return scale * (q - zero)


def _fill_saved_with_mean(
    group: torch.Tensor,
    saved_mask: torch.Tensor,
) -> torch.Tensor:
    """
    组内把「保存精度」位置用该行非保存位置的均值填上（SpQR 方式 1）。
    group (out_f, g), saved_mask (out_f, g) bool，True = 保存精度。

    saved_mask 是按列定义的（同一列所有行相同），因此可以向量化。
    """
    col_mask = saved_mask[0]  # (g,) — 列级 mask，所有行一致
    not_saved = ~col_mask
    if not_saved.all():
        return group.clone()
    if not not_saved.any():
        return group.clone()
    row_mean = group[:, not_saved].mean(dim=1, keepdim=True)  # (out_f, 1)
    group_filled = group.clone()
    group_filled[:, col_mask] = row_mean.expand(-1, int(col_mask.sum()))
    return group_filled


@torch.no_grad()
def pseudo_quantize_weight_spqr_style(
    weight: torch.Tensor,
    q_group_size: int,
    high_precision_columns: Set[Tuple[str, int]],
    layer_key: str,
    n_bits: int = 4,
    zero_point: bool = True,
) -> torch.Tensor:
    """
    SpQR 风格混合精度：分组 = 一行×一坨列；组内若有保存精度列则剔除后算 scale/zero（用非保存位置均值填充再 fit），
    量化后用原权重重写保存位置；返回合并后的权重（推理时等价于 量化部分 + outlier 加回）。

    weight: (out_features, in_features)
    high_precision_columns: (layer_key, col_idx) 的集合，该列整列视为保存精度（outlier）。
    """
    out_f, in_f = weight.shape
    result = weight.clone()
    w = weight.float()

    hp_col_mask = torch.zeros(in_f, dtype=torch.bool, device=weight.device)
    for col_idx in range(in_f):
        if (layer_key, col_idx) in high_precision_columns:
            hp_col_mask[col_idx] = True

    for j in range(0, in_f, q_group_size):
        g = min(q_group_size, in_f - j)
        group = w[:, j : j + g]
        saved_mask = hp_col_mask[j : j + g].unsqueeze(0).expand(out_f, -1)

        group_filled = _fill_saved_with_mean(group, saved_mask)
        scale, zero = _get_scale_zero_per_row(group_filled, n_bits, zero_point)
        group_q = _quantize_dequantize_with_scale_zero(
            group, scale, zero, n_bits, zero_point
        )
        saved_f = saved_mask.float()
        group_merged = group_q * (1 - saved_f) + group * saved_f
        result[:, j : j + g] = group_merged

    return result
