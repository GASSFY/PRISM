#!/usr/bin/env python3
"""
Compare protocol A (no hold-out) vs B (96/32 hold-out) for 1D ASD alpha search.

alpha = theta2, theta1 = 1 - alpha.
Each protocol: 5 coarse alphas + 3 neighborhood refinements (8 DeltaCE evals).
Then save best pseudo-quant checkpoints for downstream RWQA.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import torch
import yaml

# Repo root on sys.path (experiments/<name>/scripts/ -> ../../..)
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


def _round_alpha(a: float) -> float:
    return round(float(a), 2)


def _coarse_alphas() -> list[float]:
    return [0.0, 0.25, 0.5, 0.75, 1.0]


def _refine_alphas(best: float, evaluated: set[float], n: int = 3) -> list[float]:
    grid = [_round_alpha(i / 10) for i in range(11)]
    evaluated_r = {_round_alpha(x) for x in evaluated}
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
    """Token-weighted mean CE; microbatch=1 avoids OOM on long multimodal batches."""
    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    total_batches = len(forward_kwargs_list)
    report_every = max(1, total_batches // 4)
    started_at = time.perf_counter()
    print(
        f"[{progress_label}] Forward CE started: "
        f"{total_batches} calibration batches, microbatch={microbatch}",
        flush=True,
    )
    total_loss = 0.0
    total_tok = 0
    for batch_idx, kw in enumerate(forward_kwargs_list, start=1):
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in kw.items()
        }
        # Infer batch size from any 2D+ tensor
        bsz = None
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bsz = v.shape[0]
                break
        if bsz is None:
            bsz = 1
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
            running_ce = total_loss / max(total_tok, 1)
            print(
                f"[{progress_label}] Forward CE {batch_idx}/{total_batches}: "
                f"running_ce={running_ce:.6f}, tokens={total_tok}",
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


def _apply_alpha_pseudo_quant(
    process_model,
    hessian_diag: dict,
    alpha: float,
    *,
    ratio: float,
    w_bit: int,
    w_group: int,
    low_w_bit: int,
    progress_label: str = "PRISM",
) -> int:
    theta1 = 1.0 - alpha
    theta2 = alpha
    # ASD on CPU (hessian stays on CPU); then quantize on GPU
    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    else:
        process_model.model.cpu()
    torch.cuda.empty_cache()
    global_asd = compute_global_asd_list(
        process_model.model, hessian_diag, theta1=theta1, theta2=theta2,
    )
    hp_cols = select_high_precision_columns(global_asd, ratio)
    print(
        f"[{progress_label}] ASD selection done: alpha={alpha:.2f}, "
        f"kept_columns={len(hp_cols)}, ratio={ratio:.4f}",
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


def search_alpha(
    *,
    protocol: str,
    process_model,
    fp_state: dict[str, torch.Tensor],
    hessian_diag: dict,
    ce_kwargs: list[dict],
    ratio: float,
    w_bit: int,
    w_group: int,
    low_w_bit: int,
) -> dict[str, Any]:
    """8-eval coarse-to-fine search; leaves model at best alpha pseudo-quant."""
    _restore_fp(process_model.model, fp_state)
    ce_fp = compute_ce(
        process_model,
        ce_kwargs,
        progress_label=f"{protocol} FP baseline",
    )
    print(f"[{protocol}] CE_fp={ce_fp:.6f}", flush=True)

    trials: list[dict] = []
    evaluated: set[float] = set()

    def eval_one(alpha: float, stage: str) -> dict:
        alpha = _round_alpha(alpha)
        trial_idx = len(trials) + 1
        trial_label = f"{protocol} trial {trial_idx}/8"
        print(
            f"\n[{trial_label}] Start ({stage}): alpha={alpha:.2f}, "
            f"theta1={1.0 - alpha:.2f}, theta2={alpha:.2f}",
            flush=True,
        )
        _restore_fp(process_model.model, fp_state)
        t0 = time.perf_counter()
        n_kept = _apply_alpha_pseudo_quant(
            process_model,
            hessian_diag,
            alpha,
            ratio=ratio,
            w_bit=w_bit,
            w_group=w_group,
            low_w_bit=low_w_bit,
            progress_label=trial_label,
        )
        ce_q = compute_ce(
            process_model,
            ce_kwargs,
            progress_label=f"{trial_label} quantized",
        )
        dce = ce_q - ce_fp
        elapsed = time.perf_counter() - t0
        row = {
            "alpha": alpha,
            "theta1": 1.0 - alpha,
            "theta2": alpha,
            "ce_q": ce_q,
            "ce_fp": ce_fp,
            "delta_ce": dce,
            "n_kept": n_kept,
            "seconds": elapsed,
        }
        trials.append(row)
        evaluated.add(alpha)
        print(
            f"[{trial_label}] Done: alpha={alpha:.2f}, CE_q={ce_q:.6f}, "
            f"DeltaCE={dce:.6f}, kept={n_kept}, time={elapsed:.1f}s",
            flush=True,
        )
        return row

    for a in _coarse_alphas():
        eval_one(a, "coarse")

    best_so_far = min(trials, key=lambda r: r["delta_ce"])
    print(
        f"[{protocol}] Coarse search best: alpha={best_so_far['alpha']:.2f}, "
        f"DeltaCE={best_so_far['delta_ce']:.6f}",
        flush=True,
    )
    for a in _refine_alphas(best_so_far["alpha"], evaluated, n=3):
        eval_one(a, "refine")

    best = min(trials, key=lambda r: r["delta_ce"])
    # Restore best checkpoint in memory
    _restore_fp(process_model.model, fp_state)
    _apply_alpha_pseudo_quant(
        process_model,
        hessian_diag,
        best["alpha"],
        ratio=ratio,
        w_bit=w_bit,
        w_group=w_group,
        low_w_bit=low_w_bit,
        progress_label=f"{protocol} best checkpoint",
    )
    print(
        f"[{protocol}] BEST alpha={best['alpha']:.2f} DeltaCE={best['delta_ce']:.6f}",
        flush=True,
    )
    return {"protocol": protocol, "ce_fp": ce_fp, "trials": trials, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(
            _REPO, "experiments/offline_vs_sequential/configs/offline.yaml"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(_REPO, "experiments/holdout_vs_full"),
    )
    parser.add_argument("--n_train", type=int, default=96)
    parser.add_argument("--n_holdout", type=int, default=32)
    parser.add_argument(
        "--save_only",
        choices=["A", "B", "both"],
        default=None,
        help="Skip DeltaCE search; recompute Hessian and save checkpoint(s) at given alpha(s).",
    )
    parser.add_argument("--alpha_a", type=float, default=None)
    parser.add_argument("--alpha_b", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "scale_cache"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    ratio = float(cfg.get("asd_high_precision_ratio", 0.01))
    w_bit = int(cfg.get("w_bit", 4))
    w_group = int(cfg.get("w_group", 128))
    low_w_bit = int(cfg.get("asd_low_w_bit", 4))
    n_samples = int(cfg.get("n_samples", 128))
    assert args.n_train + args.n_holdout == n_samples

    print(f"[compare] Loading model… ({datetime.now()})", flush=True)
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

    print("[compare] Loading COCO calib…", flush=True)
    all_kw, _ = get_multimodal_calib_dataset(
        data_path=cfg["data_path"],
        image_folder=cfg["image_folder"],
        model=process_model,
        n_samples=n_samples,
        few_shot_format=bool(cfg.get("few_shot_format", False)),
        interleave_format=bool(cfg.get("interleave_format", False)),
        text_data_path=cfg.get("text_data_path") or None,
    )
    # batch_size=8 → 16 batches for 128 samples; 96/32 → 12 + 4 batches
    calib_bs = 8
    n_train_batches = args.n_train // calib_bs
    n_hold_batches = args.n_holdout // calib_bs
    assert len(all_kw) == n_train_batches + n_hold_batches, (
        f"Expected {n_train_batches + n_hold_batches} batches, got {len(all_kw)}"
    )
    train_kw = all_kw[:n_train_batches]
    hold_kw = all_kw[n_train_batches:]
    print(
        f"[compare] batches: all={len(all_kw)}, train={len(train_kw)}, holdout={len(hold_kw)}",
        flush=True,
    )

    fp_state = _cache_fp_state(process_model.model)
    print(f"[compare] FP state cached ({len(fp_state)} tensors)", flush=True)

    path_a = os.path.join(out_dir, "scale_cache", "prism_alphaA.pt")
    path_b = os.path.join(out_dir, "scale_cache", "prism_alphaB.pt")
    result_a: dict[str, Any] | None = None
    result_b: dict[str, Any] | None = None
    do_a = args.save_only in (None, "A", "both")
    do_b = args.save_only in (None, "B", "both")

    def _save_at_alpha(protocol: str, hess, alpha: float, dest: str) -> None:
        _restore_fp(process_model.model, fp_state)
        n_kept = _apply_alpha_pseudo_quant(
            process_model,
            hess,
            alpha,
            ratio=ratio,
            w_bit=w_bit,
            w_group=w_group,
            low_w_bit=low_w_bit,
        )
        if hasattr(process_model, "to_cpu"):
            process_model.to_cpu()
        save_checkpoint(process_model.model, dest)
        print(
            f"[{protocol}] Saved {dest} alpha={alpha:.2f} kept={n_kept}",
            flush=True,
        )

    if args.save_only is not None:
        # Resume path: skip DeltaCE search; rebuild checkpoint(s) only.
        if do_a:
            alpha_a = args.alpha_a
            if alpha_a is None:
                with open(os.path.join(out_dir, "logs", "search_A.json"), encoding="utf-8") as f:
                    alpha_a = json.load(f)["best"]["alpha"]
            print(f"\n======== SAVE-ONLY A alpha={alpha_a} ========", flush=True)
            _restore_fp(process_model.model, fp_state)
            t_h = time.perf_counter()
            hess_a = collect_hessian_diag(process_model, all_kw)
            print(f"[A] Hessian done in {time.perf_counter() - t_h:.1f}s", flush=True)
            _save_at_alpha("A", hess_a, float(alpha_a), path_a)
            del hess_a
        if do_b:
            alpha_b = args.alpha_b
            if alpha_b is None:
                with open(os.path.join(out_dir, "logs", "search_B.json"), encoding="utf-8") as f:
                    alpha_b = json.load(f)["best"]["alpha"]
            print(f"\n======== SAVE-ONLY B alpha={alpha_b} ========", flush=True)
            _restore_fp(process_model.model, fp_state)
            t_h = time.perf_counter()
            hess_b = collect_hessian_diag(process_model, train_kw)
            print(f"[B] Hessian done in {time.perf_counter() - t_h:.1f}s", flush=True)
            _save_at_alpha("B", hess_b, float(alpha_b), path_b)
            del hess_b
        print(f"[compare] Done save_only @ {datetime.now()}", flush=True)
        return

    # ----- Protocol A: no cut -----
    print("\n======== PROTOCOL A (no hold-out) ========", flush=True)
    _restore_fp(process_model.model, fp_state)
    t_h = time.perf_counter()
    hess_a = collect_hessian_diag(process_model, all_kw)
    print(f"[A] Hessian done in {time.perf_counter() - t_h:.1f}s, layers={len(hess_a)}", flush=True)
    result_a = search_alpha(
        protocol="A",
        process_model=process_model,
        fp_state=fp_state,
        hessian_diag=hess_a,
        ce_kwargs=all_kw,
        ratio=ratio,
        w_bit=w_bit,
        w_group=w_group,
        low_w_bit=low_w_bit,
    )
    with open(os.path.join(out_dir, "logs", "search_A.json"), "w", encoding="utf-8") as f:
        json.dump(result_a, f, indent=2)
    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    save_checkpoint(process_model.model, path_a)
    print(f"[A] Saved {path_a}", flush=True)
    del hess_a

    # ----- Protocol B: 96/32 -----
    print("\n======== PROTOCOL B (96/32 hold-out) ========", flush=True)
    _restore_fp(process_model.model, fp_state)
    t_h = time.perf_counter()
    hess_b = collect_hessian_diag(process_model, train_kw)
    print(f"[B] Hessian done in {time.perf_counter() - t_h:.1f}s, layers={len(hess_b)}", flush=True)
    result_b = search_alpha(
        protocol="B",
        process_model=process_model,
        fp_state=fp_state,
        hessian_diag=hess_b,
        ce_kwargs=hold_kw,
        ratio=ratio,
        w_bit=w_bit,
        w_group=w_group,
        low_w_bit=low_w_bit,
    )
    with open(os.path.join(out_dir, "logs", "search_B.json"), "w", encoding="utf-8") as f:
        json.dump(result_b, f, indent=2)
    if hasattr(process_model, "to_cpu"):
        process_model.to_cpu()
    save_checkpoint(process_model.model, path_b)
    print(f"[B] Saved {path_b}", flush=True)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ratio": ratio,
        "n_train": args.n_train,
        "n_holdout": args.n_holdout,
        "A_best": result_a["best"],
        "B_best": result_b["best"],
        "checkpoint_A": path_a,
        "checkpoint_B": path_b,
    }
    with open(os.path.join(out_dir, "logs", "search_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[compare] Summary:", json.dumps(summary, indent=2), flush=True)
    print(f"[compare] Done search @ {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
