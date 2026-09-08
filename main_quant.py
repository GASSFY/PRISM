"""
PRISM Phase-1: calibrate, ASD column selection, SpQR-style *pseudo* quantization,
save float weights (fake-quantized). No real-int4 / CUDA deploy path.
"""
import argparse
import os
import warnings
from typing import Union

import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from lmms_eval.models import get_model

from asdq.models import get_process_model
from asdq.calibration.coco_vl import get_multimodal_calib_dataset
from asdq.calibration.hessian_collector import collect_hessian_diag
from asdq.metrics import asd_kwargs_from_config
from asdq.quantization.quantize import pseudo_quantize_model_weight
from asdq.quantization.checkpoint import load_checkpoint, save_checkpoint
from asdq.quantization.mixed_precision import (
    compute_global_asd_list,
    select_high_precision_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="PRISM Phase-1: ASD + SpQR-style pseudo-quant (accuracy study).",
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
    # ASD mixed precision
    parser.add_argument("--asd_mixed_precision", action="store_true", default=True)
    parser.add_argument("--asd_theta1", type=float, default=0.8)
    parser.add_argument("--asd_theta2", type=float, default=0.2)
    parser.add_argument("--asd_high_precision_ratio", type=float, default=0.1)
    parser.add_argument("--asd_low_w_bit", type=int, default=4)
    args = parser.parse_args()
    return args


def _apply_config(args: argparse.Namespace, config: dict) -> None:
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)


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

    # Load calibration data (returned as lists of mini-batches)
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

    # ASD: Hessian diag → importance → global ranking → high-precision columns
    high_precision_columns = None
    if getattr(args, "asd_mixed_precision", True) and forward_kwargs_list is not None:
        hessian_diag = collect_hessian_diag(
            process_model,
            forward_kwargs_list,
        )
        print(f"[PRISM] Hessian diag collected for {len(hessian_diag)} layers.")

        asd_kw = asd_kwargs_from_config(vars(args))
        global_asd_list = compute_global_asd_list(
            process_model.model, hessian_diag, **asd_kw,
        )
        ratio = getattr(args, "asd_high_precision_ratio", 0.1)
        high_precision_columns = select_high_precision_columns(global_asd_list, ratio)
        print(f"[PRISM] Mixed precision: {len(high_precision_columns)} high-precision columns (ratio={ratio}).")

    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    elif hasattr(process_model.model, "cuda"):
        process_model.model.cuda()

    if not args.pseudo_quant:
        raise ValueError("Phase-1 PRISM requires pseudo_quant=True (real-int4 deploy path removed).")

    if high_precision_columns is not None:
        pseudo_quantize_model_weight(
            process_model.model,
            w_bit=args.w_bit,
            q_group_size=args.w_group,
            zero_point=True,
            high_precision_columns=high_precision_columns,
            low_w_bit=getattr(args, "asd_low_w_bit", 4),
        )
        print("[PRISM] Pseudo quantization applied (ASD mixed precision).")
    else:
        pseudo_quantize_model_weight(
            process_model.model,
            w_bit=args.w_bit,
            q_group_size=args.w_group,
            zero_point=True,
        )
        print(f"[PRISM] Pseudo quantization applied (uniform w_bit={args.w_bit}).")

    if args.scale_path:
        save_checkpoint(lm._model, args.scale_path)
        print(f"[PRISM] Saved pseudo-quant state to {args.scale_path}")


if __name__ == "__main__":
    cli_main()
