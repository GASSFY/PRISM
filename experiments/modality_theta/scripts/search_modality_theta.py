#!/usr/bin/env python3
"""
Grid-search modality fusion weight θ for column selection:

    K = θ · K^T_norm + (1 - θ) · K^V_norm

Protocol mirrors holdout_vs_full α search:
  5 coarse points {0, 0.25, 0.5, 0.75, 1.0} + 3 neighborhood refinements (0.1 grid)
  = 8 DeltaCE evaluations on the calibration set.

θ=1 → text-only, θ=0 → vision-only.
Score is fused K only (Ψ removed from the main path).
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
    resolve_high_precision_ratio,
    select_high_precision_columns,
)
from prism.quantization.quantize import pseudo_quantize_model_weight


def _round_theta(a: float) -> float:
    return round(float(a), 2)


def _coarse_thetas() -> list[float]:
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def _refine_thetas(best: float, evaluated: set[float], n: int = 3) -> list[float]:
    grid = [_round_theta(i / 10) for i in range(11)]
    evaluated_r = {_round_theta(x) for x in evaluated}
    remaining = [a for a in grid if a not in evaluated_r]
    remaining.sort(key=lambda a: (abs(a - best), a))
    return remaining[:n]


@torch.no_grad()
def compute_ce(
    process_model,
    forward_kwargs_list: list[dict],
    microbatch: int = 1,
    *,
    progress_label: str = "CE",
) -> float:
    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    total_batches = len(forward_kwargs_list)
    report_every = max(1, total_batches // 4)
    started_at = time.perf_counter()
    print(
        f"[{progress_label}] Forward CE started: "
        f"{total_batches} batches, microbatch={microbatch}",
        flush=True,
    )
    total_loss = 0.0
    total_tok = 0
    for batch_idx, kw in enumerate(forward_kwargs_list, start=1):
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in kw.items()
        }
        bsz = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bsz = v.shape[0]
                break
        for start in range(0, bsz, microbatch):
            end = min(start + microbatch, bsz)
            micro = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == bsz:
                    micro[k] = v[start:end]
                else:
                    micro[k] = v
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
        if batch_idx % report_every == 0 or batch_idx == total_batches:
            print(
                f"[{progress_label}] Forward CE {batch_idx}/{total_batches}: "
                f"running_ce={total_loss / max(total_tok, 1):.6f}",
                flush=True,
            )
    ce = total_loss / max(total_tok, 1)
    print(
        f"[{progress_label}] Forward CE done: ce={ce:.6f}, "
        f"time={time.perf_counter() - started_at:.1f}s",
        flush=True,
    )
    return ce


def _cache_fp_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _restore_fp(model: torch.nn.Module, fp_state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(fp_state, strict=False)
    torch.cuda.empty_cache()


def _resolve_ratio(cfg: dict) -> float:
    return resolve_high_precision_ratio(
        target_bit=cfg.get("target_bit", None),
        low_bit=float(cfg.get("asd_low_w_bit", cfg.get("w_bit", 4))),
        high_bit=float(cfg.get("asd_high_bit", 16.0)),
        legacy_ratio=cfg.get("asd_high_precision_ratio", None),
    )


def _apply_modality_pseudo_quant(
    process_model,
    energy: dict[str, dict[str, torch.Tensor]],
    modality_theta: float,
    *,
    ratio: float,
    w_bit: int,
    w_group: int,
    low_w_bit: int,
    progress_label: str = "PRISM",
) -> int:
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
        modality_theta=float(modality_theta),
    )
    hp_cols = select_high_precision_columns(global_asd, ratio)
    print(
        f"[{progress_label}] Modality ASD: θ={modality_theta:.2f}, "
        f"kept={len(hp_cols)}, ratio={ratio:.4f}",
        flush=True,
    )

    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    else:
        process_model.model.cuda()
    pseudo_quantize_model_weight(
        process_model.model,
        w_bit=w_bit,
        q_group_size=w_group,
        zero_point=True,
        high_precision_columns=hp_cols,
        low_w_bit=low_w_bit,
        progress_label=progress_label,
    )
    torch.cuda.empty_cache()
    return len(hp_cols)


def search_modality_theta(
    *,
    process_model,
    fp_state: dict[str, torch.Tensor],
    energy: dict[str, dict[str, torch.Tensor]],
    ce_kwargs: list[dict],
    ratio: float,
    w_bit: int,
    w_group: int,
    low_w_bit: int,
) -> dict[str, Any]:
    t_fp0 = time.perf_counter()
    _restore_fp(process_model.model, fp_state)
    ce_fp = compute_ce(process_model, ce_kwargs, progress_label="FP baseline")
    seconds_fp_ce = time.perf_counter() - t_fp0
    print(f"[modality] CE_fp={ce_fp:.6f} ({seconds_fp_ce:.1f}s)", flush=True)

    trials: list[dict] = []
    evaluated: set[float] = set()

    def eval_one(theta: float, stage: str) -> dict:
        theta = _round_theta(theta)
        trial_idx = len(trials) + 1
        label = f"modality trial {trial_idx}/8"
        print(
            f"\n[{label}] Start ({stage}): θ={theta:.2f} "
            f"(K=θ·K^T+(1-θ)·K^V)",
            flush=True,
        )
        _restore_fp(process_model.model, fp_state)
        t0 = time.perf_counter()
        n_kept = _apply_modality_pseudo_quant(
            process_model,
            energy,
            theta,
            ratio=ratio,
            w_bit=w_bit,
            w_group=w_group,
            low_w_bit=low_w_bit,
            progress_label=label,
        )
        ce_q = compute_ce(process_model, ce_kwargs, progress_label=f"{label} quantized")
        dce = ce_q - ce_fp
        elapsed = time.perf_counter() - t0
        row = {
            "modality_theta": theta,
            "ce_q": ce_q,
            "ce_fp": ce_fp,
            "delta_ce": dce,
            "n_kept": n_kept,
            "seconds": elapsed,
        }
        trials.append(row)
        evaluated.add(theta)
        print(
            f"[{label}] Done: θ={theta:.2f}, CE_q={ce_q:.6f}, "
            f"DeltaCE={dce:.6f}, kept={n_kept}, time={elapsed:.1f}s",
            flush=True,
        )
        return row

    for t in _coarse_thetas():
        eval_one(t, "coarse")

    best_so_far = min(trials, key=lambda r: r["delta_ce"])
    print(
        f"[modality] Coarse best: θ={best_so_far['modality_theta']:.2f}, "
        f"DeltaCE={best_so_far['delta_ce']:.6f}",
        flush=True,
    )
    for t in _refine_thetas(best_so_far["modality_theta"], evaluated, n=3):
        eval_one(t, "refine")

    best = min(trials, key=lambda r: r["delta_ce"])
    t_final0 = time.perf_counter()
    _restore_fp(process_model.model, fp_state)
    n_kept_best = _apply_modality_pseudo_quant(
        process_model,
        energy,
        best["modality_theta"],
        ratio=ratio,
        w_bit=w_bit,
        w_group=w_group,
        low_w_bit=low_w_bit,
        progress_label="modality best checkpoint",
    )
    seconds_final_ckpt = time.perf_counter() - t_final0
    best = {**best, "n_kept": n_kept_best}
    print(
        f"[modality] BEST θ={best['modality_theta']:.2f} "
        f"DeltaCE={best['delta_ce']:.6f} final_ckpt={seconds_final_ckpt:.1f}s",
        flush=True,
    )
    coarse_best = min(trials[:5], key=lambda r: r["delta_ce"]) if len(trials) >= 5 else best
    return {
        "ce_fp": ce_fp,
        "trials": trials,
        "best": best,
        "coarse_best": {
            "modality_theta": coarse_best["modality_theta"],
            "delta_ce": coarse_best["delta_ce"],
        },
        "timing": {
            "seconds_fp_ce": float(seconds_fp_ce),
            "seconds_search_trials": float(sum(t["seconds"] for t in trials)),
            "seconds_final_ckpt": float(seconds_final_ckpt),
            "n_trials": len(trials),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            _REPO, "experiments/modality_theta/configs/llava_ov.yaml"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_REPO, "experiments/modality_theta"),
    )
    parser.add_argument(
        "--energy_cache",
        default=None,
        help="Optional path to save/load modality energy dict (.pt).",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Skip search; build one checkpoint at this modality_theta.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "scale_cache"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)

    ratio = _resolve_ratio(cfg)
    w_bit = int(cfg.get("w_bit", 4))
    w_group = int(cfg.get("w_group", 128))
    low_w_bit = int(cfg.get("asd_low_w_bit", 4))
    energy_cache = args.energy_cache or os.path.join(
        out_dir, "scale_cache", "modality_energy.pt"
    )

    timing: dict[str, Any] = {}
    t_load0 = time.perf_counter()
    print(f"[modality] Loading model… ({datetime.now()})", flush=True)
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
    timing["seconds_load_model"] = time.perf_counter() - t_load0

    print("[modality] Loading COCO calib…", flush=True)
    t_data0 = time.perf_counter()
    all_kw, all_meta = get_multimodal_calib_dataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        model=process_model,
        n_samples=int(cfg.get("n_samples", 128)),
        few_shot_format=bool(cfg.get("few_shot_format", False)),
        interleave_format=bool(cfg.get("interleave_format", False)),
        text_data_path=cfg.get("text_data_path") or None,
    )
    timing["seconds_load_calib"] = time.perf_counter() - t_data0
    print(f"[modality] batches={len(all_kw)}", flush=True)

    fp_state = _cache_fp_state(process_model.model)
    print(f"[modality] FP state cached ({len(fp_state)} tensors)", flush=True)

    if os.path.exists(energy_cache):
        print(f"[modality] Load energy cache {energy_cache}", flush=True)
        t_e0 = time.perf_counter()
        energy = torch.load(energy_cache, map_location="cpu")
        timing["seconds_energy"] = time.perf_counter() - t_e0
        timing["energy_from_cache"] = True
    else:
        print("[modality] Collecting modality energy…", flush=True)
        t_e0 = time.perf_counter()
        energy = collect_hessian_diag(
            process_model,
            all_kw,
            metadata_list=all_meta,
            split_modality=True,
        )
        timing["seconds_energy"] = time.perf_counter() - t_e0
        timing["energy_from_cache"] = False
        torch.save(energy, energy_cache)
        print(f"[modality] Saved energy cache {energy_cache}", flush=True)

    ckpt_path = os.path.join(out_dir, "scale_cache", "prism_modality_best.pt")

    if args.theta is not None:
        theta = _round_theta(args.theta)
        print(f"[modality] Save-only θ={theta:.2f}", flush=True)
        _restore_fp(process_model.model, fp_state)
        n_kept = _apply_modality_pseudo_quant(
            process_model,
            energy,
            theta,
            ratio=ratio,
            w_bit=w_bit,
            w_group=w_group,
            low_w_bit=low_w_bit,
        )
        if hasattr(process_model, "to_cpu"):
            process_model.to_cpu()
        save_checkpoint(process_model.model, ckpt_path)
        summary = {
            "modality_theta": theta,
            "n_kept": n_kept,
            "ratio": ratio,
            "checkpoint": ckpt_path,
            "timing": timing,
        }
        with open(
            os.path.join(out_dir, "logs", f"save_theta_{theta:.2f}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)
        print(f"[modality] Saved {ckpt_path} kept={n_kept}", flush=True)
        return

    t_search0 = time.perf_counter()
    result = search_modality_theta(
        process_model=process_model,
        fp_state=fp_state,
        energy=energy,
        ce_kwargs=all_kw,
        ratio=ratio,
        w_bit=w_bit,
        w_group=w_group,
        low_w_bit=low_w_bit,
    )
    timing["seconds_search_wall"] = time.perf_counter() - t_search0
    if "timing" in result:
        timing.update(result["timing"])

    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    t_save0 = time.perf_counter()
    save_checkpoint(process_model.model, ckpt_path)
    timing["seconds_save_ckpt"] = time.perf_counter() - t_save0

    result["ratio"] = ratio
    result["checkpoint"] = ckpt_path
    result["energy_cache"] = energy_cache
    result["timing"] = timing
    result["model"] = cfg["model"]
    result["model_args"] = cfg.get("model_args")
    with open(os.path.join(out_dir, "logs", "search_modality_theta.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "metrics", "search_modality_theta.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[modality] Saved checkpoint {ckpt_path}", flush=True)
    print(f"[modality] timing={json.dumps(timing)}", flush=True)
    print(f"[modality] Done @ {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
