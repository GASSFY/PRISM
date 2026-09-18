"""
PRISM Phase-1: K-based column selection + mixed-precision *pseudo* quantization.

Offline only: FP calibrate once → global K ranking → one-shot pseudo-quant.
Optional cross-modal fusion:
  linear:    K = θ·norm(K^T) + (1-θ)·norm(K^V)
  geometric: log K = θ·log K^T + (1-θ)·log K^V
"""
import argparse
import os
import time
import warnings
from typing import Union

import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval.models import get_model

from prism.models import get_process_model
from prism.calibration.coco_vl import get_multimodal_calib_dataset
from prism.calibration.hessian_collector import collect_hessian_diag
from prism.quantization.quantize import pseudo_quantize_model_weight
from prism.quantization.checkpoint import load_checkpoint, save_checkpoint
from prism.quantization.mixed_precision import (
    compute_global_asd_list,
    resolve_high_precision_ratio,
    select_high_precision_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="PRISM Phase-1: offline pseudo-quant with global K column keep.",
    )
    parser.add_argument("--config", default="", help="Path to yaml config (overrides CLI)")
    parser.add_argument("--model", default="llava_onevision")
    parser.add_argument("--model_args", default="")
    parser.add_argument("--batch_size", "-b", type=str, default="1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--calib_data", default="coco", choices=["coco"])
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)
    # quantization (pseudo only)
    parser.add_argument("--run_process", action="store_true", help="Run quant and save")
    parser.add_argument("--scale_path", default=None, type=str, help="Path to save/load pseudo-quant state_dict")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--w_group", type=int, default=128)
    parser.add_argument("--pseudo_quant", action="store_true", default=True)
    # mixed precision keep-set
    parser.add_argument("--asd_mixed_precision", action="store_true", default=True)
    parser.add_argument(
        "--target_bit",
        type=float,
        default=4.01,
        help="Average bit-width budget; r=(target_bit-low)/(high-low), high=16.",
    )
    parser.add_argument(
        "--asd_high_bit",
        type=float,
        default=16.0,
        help="Bit-width counted for kept (FP) columns when converting target_bit→ratio.",
    )
    parser.add_argument(
        "--asd_high_precision_ratio",
        type=float,
        default=None,
        help="Legacy override: direct keep-column fraction. Ignored if target_bit is set.",
    )
    parser.add_argument("--asd_low_w_bit", type=int, default=4)
    parser.add_argument(
        "--split_modality",
        action="store_true",
        help="Collect separate vision/text activation energies (needs vision_mask).",
    )
    parser.add_argument(
        "--modality_theta",
        type=float,
        default=None,
        help="Fusion weight on TEXT. Requires --split_modality.",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="linear",
        choices=["linear", "geometric"],
        help="Cross-modal fusion: linear (max-norm soft-OR) or geometric (log soft-AND).",
    )
    args = parser.parse_args()
    return args


def _apply_config(args: argparse.Namespace, config: dict) -> None:
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)


def _resolve_keep_ratio(args: argparse.Namespace) -> float:
    low_bit = float(getattr(args, "asd_low_w_bit", args.w_bit))
    high_bit = float(getattr(args, "asd_high_bit", 16.0))
    target_bit = getattr(args, "target_bit", None)
    legacy_ratio = getattr(args, "asd_high_precision_ratio", None)
    if target_bit is not None:
        return resolve_high_precision_ratio(
            target_bit=float(target_bit),
            low_bit=low_bit,
            high_bit=high_bit,
        )
    return resolve_high_precision_ratio(
        target_bit=None,
        low_bit=low_bit,
        high_bit=high_bit,
        legacy_ratio=legacy_ratio,
    )


def cli_main(args: Union[argparse.Namespace, None] = None) -> None:
    if args is None:
        args = parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = [cfg] if not isinstance(cfg, list) else cfg
        for c in cfg:
            args_copy = argparse.Namespace(**vars(args))
            _apply_config(args_copy, c)
            _run_single(args_copy)
    else:
        _run_single(args)


def _run_offline(
    args: argparse.Namespace,
    process_model,
    forward_kwargs_list,
    metadata_list=None,
) -> None:
    """FP energy → global K ranking → one-shot pseudo-quant."""
    high_precision_columns = None
    ratio = 0.0
    if getattr(args, "asd_mixed_precision", True) and forward_kwargs_list is not None:
        split_modality = bool(getattr(args, "split_modality", False))
        modality_theta = getattr(args, "modality_theta", None)
        fusion_mode = str(getattr(args, "fusion_mode", "linear") or "linear")
        if modality_theta is not None and not split_modality:
            raise ValueError("modality_theta requires --split_modality")

        energy = collect_hessian_diag(
            process_model,
            forward_kwargs_list,
            metadata_list=metadata_list if split_modality else None,
            split_modality=split_modality,
        )

        if split_modality:
            assert isinstance(energy, dict) and "vision" in energy and "text" in energy
            print(
                f"[PRISM] Modality energy collected: "
                f"mixed={len(energy['mixed'])}, "
                f"vision={len(energy['vision'])}, "
                f"text={len(energy['text'])} layers."
            )
            if modality_theta is None:
                raise ValueError(
                    "split_modality offline path requires modality_theta "
                    "(use 0.5 for equal fusion, or run the θ search script)."
                )
            global_asd_list = compute_global_asd_list(
                process_model.model,
                hessian_diag=None,
                hessian_vision=energy["vision"],
                hessian_text=energy["text"],
                modality_theta=float(modality_theta),
                fusion_mode=fusion_mode,  # type: ignore[arg-type]
            )
            if fusion_mode == "geometric":
                formula = "logK = θ·logK^T + (1-θ)·logK^V"
            else:
                formula = "K = θ·norm(K^T) + (1-θ)·norm(K^V)"
            print(
                f"[PRISM] Modality fusion ({fusion_mode}): "
                f"θ={float(modality_theta):.3f} ({formula})"
            )
        else:
            assert isinstance(energy, dict)
            print(f"[PRISM] Activation energy collected for {len(energy)} layers.")
            global_asd_list = compute_global_asd_list(
                process_model.model,
                energy,
            )

        ratio = _resolve_keep_ratio(args)
        high_precision_columns = select_high_precision_columns(global_asd_list, ratio)
        print(
            f"[PRISM] Offline mixed precision: target_bit={getattr(args, 'target_bit', None)}, "
            f"ratio={ratio:.6f}, {len(high_precision_columns)} high-precision columns."
        )

    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    elif hasattr(process_model.model, "cuda"):
        process_model.model.cuda()

    if high_precision_columns is not None:
        pseudo_quantize_model_weight(
            process_model.model,
            w_bit=args.w_bit,
            q_group_size=args.w_group,
            zero_point=True,
            high_precision_columns=high_precision_columns,
            low_w_bit=getattr(args, "asd_low_w_bit", 4),
        )
        print("[PRISM] Offline pseudo quantization applied (K mixed precision).")
    else:
        pseudo_quantize_model_weight(
            process_model.model,
            w_bit=args.w_bit,
            q_group_size=args.w_group,
            zero_point=True,
        )
        print(f"[PRISM] Offline pseudo quantization applied (uniform w_bit={args.w_bit}).")


def _run_single(args: argparse.Namespace) -> None:
    if args.model_args is None:
        args.model_args = ""

    ModelClass = get_model(args.model)
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {"batch_size": args.batch_size, "device": args.device},
    )
    process_model = get_process_model(args.model)(
        lm._model,
        lm._tokenizer,
        lm.processor if hasattr(lm, "processor") else None,
    )

    if not args.run_process:
        if args.scale_path and os.path.exists(args.scale_path):
            load_checkpoint(lm._model, args.scale_path)
        return

    forward_kwargs_list, metadata_list = None, None
    if args.calib_data == "coco" and args.data_path and args.image_folder:
        forward_kwargs_list, metadata_list = get_multimodal_calib_dataset(
            data_path=args.data_path,
            image_folder=args.image_folder,
            model=process_model,
            n_samples=args.n_samples,
            few_shot_format=args.few_shot_format,
            interleave_format=args.interleave_format,
            text_data_path=args.text_data_path or None,
        )
        print(f"[PRISM] Calibration data loaded ({len(forward_kwargs_list)} mini-batches).")

    if not args.pseudo_quant:
        raise ValueError("Phase-1 PRISM requires pseudo_quant=True (real-int4 deploy path removed).")

    t0 = time.perf_counter()
    _run_offline(args, process_model, forward_kwargs_list, metadata_list)
    elapsed = time.perf_counter() - t0
    print(f"[PRISM] Quant wall time: {elapsed:.1f}s (mode=offline)")

    if args.scale_path:
        save_checkpoint(lm._model, args.scale_path)
        print(f"[PRISM] Saved pseudo-quant state to {args.scale_path}")


if __name__ == "__main__":
    cli_main()
