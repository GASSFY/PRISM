"""Save/load PRISM pseudo-quant checkpoints (float weights after fake quant)."""
from __future__ import annotations

import os
from typing import Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    *,
    quant_payload: Optional[dict] = None,
    w_group: int = 128,
) -> None:
    """Save model state_dict. ``quant_payload`` / ``w_group`` kept for API compat but ignored (Phase-1 pseudo only)."""
    del quant_payload, w_group  # Phase-1: no real-int4 payload
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(state, dict):
        model.load_state_dict(state, strict=False)
        print(f"[PRISM] Loaded checkpoint from {path}")
        return

    if "quant_payload" in state or state.get("format") == "asdq_int4_v2":
        raise ValueError(
            f"Checkpoint {path} looks like a real-int4 / v2 deploy format. "
            "Phase-1 PRISM only supports pseudo-quant float state_dict. "
            "Re-run main_quant.py with pseudo_quant."
        )

    if "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
        print(f"[PRISM] Loaded pseudo-quant checkpoint from {path}")
        return

    model.load_state_dict(state, strict=False)
    print(f"[PRISM] Loaded checkpoint from {path}")
