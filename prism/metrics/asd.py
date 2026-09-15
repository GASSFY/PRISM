"""
Column importance for mixed-precision keep-set selection.

Second-order proxy (OWQ-style, not a full GPTQ Hessian):
  importance_c = ||W[:, c]||^2 * E[x_c^2]
where E[x_c^2] is streaming activation energy (optionally modality-conditioned).

Ranking score is globally normalized K (importance / global_max).
Optional cross-modal fusion (in mixed_precision):
  K = modality_theta * K^T_norm + (1 - modality_theta) * K^V_norm

Ψ / asd_theta1·K+asd_theta2·Ψ was removed after psi_ablation (no gain).
"""
from __future__ import annotations

from typing import Any

import torch

_EPS = 1e-8


@torch.no_grad()
def compute_importance(weight: torch.Tensor, diag_H: torch.Tensor) -> torch.Tensor:
    """
    Per-channel importance: importance_c = ||W[:, c]||^2 * E[x_c^2].

    weight: (out_features, in_features).
    diag_H: (in_features,) — E[x_c^2] activation energy (mixed or modality-split).
    Returns: (in_features,) importance per input channel.
    """
    w_col_norm_sq = weight.float().pow(2).sum(dim=0)
    return w_col_norm_sq * diag_H.float()


def importance_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract optional modality fusion params from a config dict."""
    out: dict[str, Any] = {}
    if config.get("modality_theta", None) is not None:
        out["modality_theta"] = float(config["modality_theta"])
    return out
