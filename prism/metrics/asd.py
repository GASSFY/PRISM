"""
ASD (Activation-aware Significance for Data-free quantization) metrics.

Hessian-based importance: importance_c = ||W[:, c]||^2 * diag(H)_c
where diag(H)_c = E[x_c^2] is the Hessian diagonal collected via streaming.

K (absolute significance) = importance_c, globally normalized to [0, 1].
Psi (relative significance) = z-score of importance within a layer.
ASD_c = theta1 * K_normalized + theta2 * Psi_normalized.
"""
from __future__ import annotations

from typing import Any

import torch

_EPS = 1e-8

PSI_METHODS = ("mean", "max", "zscore")


@torch.no_grad()
def compute_Psi(s: torch.Tensor, method: str = "zscore") -> torch.Tensor:
    """
    Per-channel relative significance Psi_c (how much this channel stands out
    compared to its layer-mates).

    s: [C] channel-wise significance (e.g. importance from compute_importance).
    method: "mean" (Psi1), "max" (Psi2), "zscore" (Psi3).
    Returns: [C] tensor.
    """
    s = s.float()
    if s.dim() != 1:
        s = s.view(-1)

    if method == "mean":
        s_mean = s.mean().clamp(min=_EPS)
        Psi = s / s_mean
    elif method == "max":
        s_max = s.amax().clamp(min=_EPS)
        Psi = s / s_max
    elif method == "zscore":
        s_mean = s.mean()
        s_std = s.std().clamp(min=_EPS)
        Psi = ((s - s_mean) / s_std).clamp(min=0.0)
    else:
        raise ValueError(f"Unknown Psi method: {method}. Use one of {PSI_METHODS}.")

    return Psi


@torch.no_grad()
def compute_importance(weight: torch.Tensor, diag_H: torch.Tensor) -> torch.Tensor:
    """
    Per-channel importance: importance_c = ||W[:, c]||^2 * diag(H)_c.

    Measures the contribution of input channel c to the layer's output variance.
    Naturally cross-layer comparable (same physical unit: output variance).

    weight: (out_features, in_features).
    diag_H: (in_features,) — E[x_c^2] from streaming Hessian diagonal.
    Returns: (in_features,) importance per input channel.
    """
    w_col_norm_sq = weight.float().pow(2).sum(dim=0)  # ||W[:, c]||^2
    return w_col_norm_sq * diag_H.float()


@torch.no_grad()
def compute_ASD(
    weight: torch.Tensor,
    diag_H: torch.Tensor,
    theta1: float = 0.8,
    theta2: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel ASD computation for a single layer.

    Returns (importance, Psi) — both [C] tensors.
    The caller (mixed_precision.compute_global_asd_list) handles global
    normalization of K and Psi before combining into the final ASD score.
    """
    importance = compute_importance(weight, diag_H)
    Psi = compute_Psi(importance, method="zscore")
    return importance, Psi


def asd_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract ASD parameters from a config dict."""
    return {
        "theta1": config.get("asd_theta1", 0.8),
        "theta2": config.get("asd_theta2", 0.2),
    }
