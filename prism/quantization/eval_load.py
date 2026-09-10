"""Evaluation-time model loading: FP16 baseline or pseudo-quant float weights."""
from __future__ import annotations

import os
from typing import Any, Optional, Type

from .checkpoint import load_checkpoint


def resolve_eval_load_mode(
    scale_path: Optional[str],
    *,
    real_quant: bool = False,
    pseudo_quant: bool = False,
) -> str:
    """Return ``fp16`` or ``pseudo_quant``. Real-int4 deploy path is not supported in Phase-1."""
    del real_quant  # Phase-1: ignored; kept so old yaml keys do not crash callers
    if not scale_path or not os.path.exists(scale_path):
        return "fp16"
    if pseudo_quant or scale_path:
        return "pseudo_quant"
    return "pseudo_quant"


def load_model_for_eval(ModelClass: Type[Any], args: Any) -> Any:
    """Load lmms-eval model wrapper: FP16 or overwrite with pseudo-quant state_dict."""
    scale_path = getattr(args, "scale_path", None)
    mode = resolve_eval_load_mode(
        scale_path,
        real_quant=getattr(args, "real_quant", False),
        pseudo_quant=getattr(args, "pseudo_quant", True),
    )
    print(f"[PRISM] Eval load mode: {mode}")

    model_args = getattr(args, "model_args", "") or ""
    batch_size = getattr(args, "batch_size", "1")
    device = getattr(args, "device", None)

    lm = ModelClass.create_from_arg_string(
        model_args,
        {"batch_size": batch_size, "device": device},
    )
    if mode == "pseudo_quant":
        load_checkpoint(lm._model, scale_path)
    return lm
