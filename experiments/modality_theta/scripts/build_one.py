#!/usr/bin/env python3
"""Build one offline pseudo-quant checkpoint for a fixed modality_theta.

K = θ · normalize(K^T) + (1-θ) · normalize(K^V)
θ=0 vision-only, θ=1 text-only, θ=0.5 equal fusion.
"""
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
    parser.add_argument(
        "--modality_theta",
        type=float,
        required=True,
        help="θ on TEXT: K=θ·norm(K^T)+(1-θ)·norm(K^V)",
    )
    parser.add_argument("--scale_path", required=True)
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("--cols_json", default="")
    parser.add_argument("--energy_cache", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    theta = round(float(args.modality_theta), 2)
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"modality_theta must be in [0,1], got {theta}")

    ratio = float(cfg.get("asd_high_precision_ratio", 0.01))
    w_bit = int(cfg.get("w_bit", 4))
    w_group = int(cfg.get("w_group", 128))
    low_w_bit = int(cfg.get("asd_low_w_bit", 4))
    n_samples = int(cfg.get("n_samples", 128))

    print(
        f"[modality] load model {cfg['model']} θ={theta:.2f} "
        f"(K=θ·norm(K^T)+(1-θ)·norm(K^V))",
        flush=True,
    )
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

    print("[modality] load calib", flush=True)
    all_kw, all_meta = get_multimodal_calib_dataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        model=process_model,
        n_samples=n_samples,
        few_shot_format=bool(cfg.get("few_shot_format", False)),
        interleave_format=bool(cfg.get("interleave_format", False)),
        text_data_path=cfg.get("text_data_path") or None,
    )

    fp_state = {k: v.detach().cpu().clone() for k, v in process_model.model.state_dict().items()}

    if args.energy_cache and os.path.exists(args.energy_cache):
        print(f"[modality] load energy cache {args.energy_cache}", flush=True)
        energy = torch.load(args.energy_cache, map_location="cpu", weights_only=False)
    else:
        t0 = time.perf_counter()
        energy = collect_hessian_diag(
            process_model,
            all_kw,
            metadata_list=all_meta,
            split_modality=True,
        )
        print(
            f"[modality] energy done {time.perf_counter()-t0:.1f}s "
            f"mixed={len(energy['mixed'])} "
            f"vision={len(energy['vision'])} text={len(energy['text'])}",
            flush=True,
        )
        if args.energy_cache:
            os.makedirs(os.path.dirname(args.energy_cache) or ".", exist_ok=True)
            torch.save(energy, args.energy_cache)
            print(f"[modality] energy saved {args.energy_cache}", flush=True)

    process_model.model.load_state_dict(fp_state, strict=False)
    ce_fp = compute_ce(process_model, all_kw)
    print(f"[modality] CE_fp={ce_fp:.6f}", flush=True)

    process_model.model.load_state_dict(fp_state, strict=False)
    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    else:
        process_model.model.cpu()
    torch.cuda.empty_cache()

    global_asd = compute_global_asd_list(
        process_model.model,
        hessian_diag=None,
        hessian_vision=energy["vision"],
        hessian_text=energy["text"],
        modality_theta=theta,
    )
    hp_cols = select_high_precision_columns(global_asd, ratio)
    kept_list = sorted([[k, int(c)] for k, c in hp_cols])
    print(f"[modality] kept={len(hp_cols)} θ={theta:.2f}", flush=True)

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
        progress_label=f"modality_t{theta:.2f}",
    )
    print(f"[modality] pseudo-quant done {time.perf_counter()-t0:.1f}s", flush=True)

    ce_q = compute_ce(process_model, all_kw)
    delta = ce_q - ce_fp
    print(f"[modality] CE_q={ce_q:.6f} DeltaCE={delta:.6f}", flush=True)

    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    os.makedirs(os.path.dirname(args.scale_path) or ".", exist_ok=True)
    save_checkpoint(process_model.model, args.scale_path)
    print(f"[modality] saved {args.scale_path}", flush=True)

    metrics: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": cfg["model"],
        "model_args": cfg.get("model_args"),
        "modality_theta": theta,
        "formula": "K = θ·norm(K^T) + (1-θ)·norm(K^V)",
        "normalization": "per-modality global max on importance before fusion",
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
    print("[modality] metrics written", args.metrics_json, flush=True)


if __name__ == "__main__":
    main()
