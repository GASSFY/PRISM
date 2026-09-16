#!/usr/bin/env python3
"""Build one offline pseudo-quant checkpoint for a given alpha (=theta2)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import torch
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lmms_eval.models import get_model

from prism.calibration.coco_vl import get_multimodal_calib_dataset
from prism.calibration.hessian_collector import collect_hessian_diag
from prism.models import get_process_model
from prism.quantization.checkpoint import save_checkpoint
from prism.quantization.mixed_precision import (
    compute_global_asd_list,
    select_high_precision_columns,
)
from prism.quantization.quantize import pseudo_quantize_model_weight


@torch.no_grad()
def compute_ce(process_model, forward_kwargs_list, microbatch: int = 1) -> float:
    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    total_loss = 0.0
    total_tok = 0
    for kw in forward_kwargs_list:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
        bsz = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bsz = v.shape[0]
                break
        for start in range(0, bsz, microbatch):
            end = min(start + microbatch, bsz)
            micro = {
                k: (v[start:end] if isinstance(v, torch.Tensor) and v.shape[0] == bsz else v)
                for k, v in batch.items()
            }
            labels = micro.get("labels")
            n = int((labels != -100).sum().item()) if labels is not None else 1
            out = process_model.forward(**micro)
            loss = out.loss if hasattr(out, "loss") else out[0]
            total_loss += float(loss.item()) * n
            total_tok += max(n, 1)
            del out, micro
            torch.cuda.empty_cache()
        del batch
        torch.cuda.empty_cache()
    return total_loss / max(total_tok, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="theta2; theta1=1-alpha")
    parser.add_argument("--scale_path", required=True)
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("--cols_json", default="", help="optional path to dump kept column keys")
    parser.add_argument("--hessian_cache", default="", help="optional .pt path to reuse hessian")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    alpha = round(float(args.alpha), 2)
    theta1, theta2 = 1.0 - alpha, alpha
    ratio = float(cfg.get("asd_high_precision_ratio", 0.01))
    w_bit = int(cfg.get("w_bit", 4))
    w_group = int(cfg.get("w_group", 128))
    low_w_bit = int(cfg.get("asd_low_w_bit", 4))
    n_samples = int(cfg.get("n_samples", 128))

    print(f"[psi] load model {cfg['model']} alpha={alpha}", flush=True)
    ModelClass = get_model(cfg["model"])
    lm = ModelClass.create_from_arg_string(
        cfg.get("model_args") or "",
        {"batch_size": str(cfg.get("batch_size", "1")), "device": cfg.get("device")},
    )
    process_model = get_process_model(cfg["model"])(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )

    print("[psi] load calib", flush=True)
    all_kw, _ = get_multimodal_calib_dataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        model=process_model,
        n_samples=n_samples,
        few_shot_format=bool(cfg.get("few_shot_format", False)),
        interleave_format=bool(cfg.get("interleave_format", False)),
        text_data_path=cfg.get("text_data_path") or None,
    )

    fp_state = {k: v.detach().cpu().clone() for k, v in process_model.model.state_dict().items()}

    if args.hessian_cache and os.path.exists(args.hessian_cache):
        print(f"[psi] load hessian cache {args.hessian_cache}", flush=True)
        hessian = torch.load(args.hessian_cache, map_location="cpu", weights_only=True)
    else:
        t0 = time.perf_counter()
        hessian = collect_hessian_diag(process_model, all_kw)
        print(f"[psi] hessian done {time.perf_counter()-t0:.1f}s layers={len(hessian)}", flush=True)
        if args.hessian_cache:
            os.makedirs(os.path.dirname(args.hessian_cache) or ".", exist_ok=True)
            torch.save(hessian, args.hessian_cache)
            print(f"[psi] hessian saved {args.hessian_cache}", flush=True)

    process_model.model.load_state_dict(fp_state, strict=False)
    ce_fp = compute_ce(process_model, all_kw)
    print(f"[psi] CE_fp={ce_fp:.6f}", flush=True)

    process_model.model.load_state_dict(fp_state, strict=False)
    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    else:
        process_model.model.cpu()
    torch.cuda.empty_cache()

    global_asd = compute_global_asd_list(
        process_model.model, hessian, theta1=theta1, theta2=theta2,
    )
    hp_cols = select_high_precision_columns(global_asd, ratio)
    # serializable kept set
    kept_list = sorted([[k, int(c)] for k, c in hp_cols])
    print(f"[psi] kept={len(hp_cols)} alpha={alpha}", flush=True)

    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    else:
        process_model.model.cuda()
    t0 = time.perf_counter()
    pseudo_quantize_model_weight(
        process_model.model,
        w_bit=w_bit,
        q_group_size=w_group,
        zero_point=True,
        high_precision_columns=hp_cols,
        low_w_bit=low_w_bit,
        progress_label=f"psi_a{alpha:.2f}",
    )
    print(f"[psi] pseudo-quant done {time.perf_counter()-t0:.1f}s", flush=True)

    ce_q = compute_ce(process_model, all_kw)
    delta = ce_q - ce_fp
    print(f"[psi] CE_q={ce_q:.6f} DeltaCE={delta:.6f}", flush=True)

    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    os.makedirs(os.path.dirname(args.scale_path) or ".", exist_ok=True)
    save_checkpoint(process_model.model, args.scale_path)
    print(f"[psi] saved {args.scale_path}", flush=True)

    metrics: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": cfg["model"],
        "model_args": cfg.get("model_args"),
        "alpha": alpha,
        "theta1": theta1,
        "theta2": theta2,
        "ratio": ratio,
        "ce_fp": ce_fp,
        "ce_q": ce_q,
        "delta_ce": delta,
        "n_kept": len(hp_cols),
        "scale_path": args.scale_path,
    }
    os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if args.cols_json:
        with open(args.cols_json, "w", encoding="utf-8") as f:
            json.dump(kept_list, f)
    print("[psi] metrics written", args.metrics_json, flush=True)


if __name__ == "__main__":
    main()
