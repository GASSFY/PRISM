"""Eight-cell Vision / Projector / LLM pseudo-quantization probe.

Does not change PRISM's column-selection path. Each cell restores full
precision, then applies the same RTN recipe to the selected components,
re-encodes the image (vision and projector run inside generate_input),
and compares hooked activations and next-token distributions to the FP run.

Run from the PRISM repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from interaction import CELL_NAMES, interaction_row  # noqa: E402
from lmms_eval.models import get_model  # noqa: E402
from prism.calibration.coco_vl import load_image  # noqa: E402
from prism.models import get_process_model  # noqa: E402
from prism.quantization.quant_funcs import pseudo_quantize_tensor  # noqa: E402
from prism.quantization.quantize import get_blocks, get_named_linears  # noqa: E402

SITES = ("vision_out", "proj_out", "llm_early", "llm_mid", "llm_late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component-interaction probe (8 cells).")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("data_path") or not cfg.get("image_folder"):
        raise SystemExit(
            "Fill data_path and image_folder in the config before running. "
            "Both are empty."
        )
    if not cfg.get("model_args"):
        raise SystemExit("Fill model_args (pretrained=...) in the config.")
    return cfg


def _as_tensor(out: Any) -> torch.Tensor | None:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
        return None
    hidden = getattr(out, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    return None


def _clone_batch(batch: dict) -> dict:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            cloned[key] = [
                item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            cloned[key] = value
    return cloned


def load_collated_batches(process_model, cfg: dict) -> list[dict]:
    """COCO batches stopped before generate_input, so each cell re-encodes."""
    data_path = cfg["data_path"]
    image_folder = cfg["image_folder"]
    n_samples = int(cfg.get("n_samples", 32))
    batch_size = int(cfg.get("calib_batch_size", 1))

    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                dataset.append(json.loads(line))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        raise ValueError(f"Unsupported calibration file: {data_path}")

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(dataset)

    data_list = []
    taken = 0
    for item in dataset:
        image_field = item.get("image")
        if not image_field:
            continue
        paths = image_field if isinstance(image_field, list) else [image_field]
        images = [load_image(os.path.join(image_folder, p)) for p in paths]
        data_list.append(process_model.preprocess_data(images, item))
        taken += 1
        if taken >= n_samples:
            break
    if not data_list:
        raise RuntimeError("No image samples were loaded from the calibration file.")

    batches = []
    for start in range(0, len(data_list), batch_size):
        chunk = data_list[start : start + batch_size]
        batches.append(_clone_batch(process_model.data_collator(chunk)))
    print(f"[interaction] loaded {taken} samples in {len(batches)} batches", flush=True)
    return batches


class WeightBook:
    """CPU copies of every Linear weight we are allowed to quantize."""

    def __init__(self, groups: dict[str, list[nn.Linear]]):
        self.groups = groups
        self.fp: dict[int, torch.Tensor] = {}
        self.meta: dict[str, dict[str, int]] = {}
        for name, modules in groups.items():
            n_param = 0
            n_linear = 0
            for module in modules:
                for linear in get_named_linears(module).values():
                    self.fp[id(linear.weight)] = linear.weight.detach().float().cpu().clone()
                    n_param += linear.weight.numel()
                    if linear.bias is not None:
                        n_param += linear.bias.numel()
                    n_linear += 1
            self.meta[name] = {"n_linear": n_linear, "n_param": n_param}

    def restore(self) -> None:
        for modules in self.groups.values():
            for module in modules:
                for linear in get_named_linears(module).values():
                    saved = self.fp[id(linear.weight)]
                    linear.weight.data.copy_(saved.to(device=linear.weight.device, dtype=linear.weight.dtype))

    def quantize(self, letters: str, w_bit: int, w_group: int, zero_point: bool) -> dict[str, Any]:
        """RTN the selected components. Returns how many linears used a fallback group."""
        fallback = 0
        applied = 0
        seen: set[int] = set()
        num = {letter: 0.0 for letter in letters}
        den = {letter: 0.0 for letter in letters}
        for letter in letters:
            for module in self.groups[letter]:
                for linear in get_named_linears(module).values():
                    key = id(linear.weight)
                    if key in seen:
                        continue
                    seen.add(key)
                    applied += 1
                    fallback += _quantize_linear(linear, w_bit, w_group, zero_point)
                    orig = self.fp[key]
                    delta = linear.weight.detach().float().cpu() - orig
                    num[letter] += float(delta.pow(2).sum().item())
                    den[letter] += float(orig.pow(2).sum().item())
        weight_rel_l2 = {
            letter: (num[letter] / den[letter]) ** 0.5 if den[letter] else None
            for letter in letters
        }
        return {
            "linears": applied,
            "group_fallbacks": fallback,
            "weight_rel_l2": weight_rel_l2,
        }


def _quantize_linear(linear: nn.Linear, w_bit: int, w_group: int, zero_point: bool) -> int:
    weight = linear.weight.data
    group = w_group
    fallback = 0
    if group > 0 and weight.shape[-1] % group != 0:
        group = -1
        fallback = 1
    q = pseudo_quantize_tensor(
        weight.float(),
        n_bits=w_bit,
        zero_point=zero_point,
        q_group_size=group,
    )
    linear.weight.data.copy_(q.to(dtype=weight.dtype))
    return fallback


def _language_blocks(model) -> list[nn.Module]:
    try:
        return list(get_blocks(model))
    except NotImplementedError:
        pass
    for path in (
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    ):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return list(cur)
    raise NotImplementedError(f"get_blocks not implemented for {type(model).__name__}")


def _layout(process_model) -> tuple[dict[str, list[nn.Module]], nn.Module, nn.Module, list[nn.Module], str]:
    """Map Vision / interface / LLM onto the modules that actually run.

    Qwen2.5-VL has no mm_projector. The patch merger inside the vision tower
    is the interface, so it is P and the vision blocks before it are V.
    Hooking the whole visual module would put the merger on both sides.
    """
    kind = type(process_model).__name__
    llm_blocks = _language_blocks(process_model.model)
    if not llm_blocks:
        raise RuntimeError("Language backbone has no blocks.")
    if kind == "Qwen2_5_VL":
        visual = process_model.model.visual
        groups = {"V": list(visual.blocks), "P": [visual.merger], "L": llm_blocks}
        return (
            groups,
            visual.blocks[-1],
            visual.merger,
            llm_blocks,
            "V=visual.blocks (before the merger), P=visual.merger, L=language layers",
        )
    proj = process_model.fetch_proj()
    if proj is None:
        raise RuntimeError(f"{kind} has no projector module to use as P.")
    groups = {"V": [process_model.fetch_vit()], "P": [proj], "L": llm_blocks}
    if kind == "InternVL2":
        note = "V=vision_model, P=mlp1, L=language layers. pixel_shuffle between V and P has no weights."
    else:
        note = "V=vision tower, P=mm_projector, L=language layers"
    return groups, process_model.fetch_vit(), proj, llm_blocks, note


class Probe:
    def __init__(self, vision_module: nn.Module, proj_module: nn.Module, llm_blocks: list[nn.Module]):
        mid = llm_blocks[len(llm_blocks) // 2]
        self.sites: list[tuple[str, nn.Module]] = [
            ("vision_out", vision_module),
            ("proj_out", proj_module),
            ("llm_early", llm_blocks[0]),
            ("llm_mid", mid),
            ("llm_late", llm_blocks[-1]),
        ]
        self.captured: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def clear(self) -> None:
        self.captured = {}

    def install(self) -> None:
        self.remove()
        for name, module in self.sites:
            self._handles.append(module.register_forward_hook(self._make(name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _make(self, name: str):
        def hook(_module, _args, output):
            tensor = _as_tensor(output)
            if tensor is None:
                return
            # float16 only for the stored reference. W4 error is far above this rounding.
            self.captured[name] = tensor.detach().float().cpu().half()

        return hook


def _logits_of(outputs) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(outputs, (tuple, list)) and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    raise RuntimeError(f"Cannot find logits on output type {type(outputs)}")


def _masked_kl(logits_fp: torch.Tensor, logits_q: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """Mean KL(softmax FP || softmax Q) on supervised tokens. Returns (sum, count)."""
    lf = logits_fp[:, :-1, :].float()
    lq = logits_q[:, :-1, :].float()
    lab = labels[:, 1:]
    mask = lab.ne(-100)
    if int(mask.sum()) == 0:
        return 0.0, 0
    log_p = torch.log_softmax(lf, dim=-1)
    log_q = torch.log_softmax(lq, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    selected = kl[mask]
    return float(selected.sum().item()), int(selected.numel())


class RunningMSE:
    def __init__(self) -> None:
        self.sse = 0.0
        self.numel = 0
        self.n_mismatch = 0

    def add(self, fp: torch.Tensor, q: torch.Tensor) -> None:
        if fp.shape != q.shape:
            self.n_mismatch += 1
            return
        diff = (q.float() - fp.float()).pow(2)
        self.sse += float(diff.sum().item())
        self.numel += int(diff.numel())

    def value(self) -> float | None:
        if self.numel == 0:
            return None
        return self.sse / self.numel


def run_cell_against_fp(
    process_model,
    batches,
    probe: Probe,
    fp_cache: list[dict] | None,
    save_error: bool = False,
    baselines: list[dict[str, list[torch.Tensor]]] | None = None,
):
    """If fp_cache is None, this is the FP pass and the cache is returned.

    Otherwise each batch is compared to fp_cache[i]. When save_error is set,
    the per-sample activation error (q - fp) is kept so a later joint cell can
    measure ||e_AB - e_A - e_B||. A near-zero interaction with a large residual
    would mean the scalar losses happened to add while the errors did not.
    """
    mse = {site: RunningMSE() for site in SITES}
    cross = {site: RunningMSE() for site in SITES}
    kl_sum = 0.0
    kl_count = 0
    missing = {site: 0 for site in SITES}
    built: list[dict] = []
    errors: dict[str, list[torch.Tensor]] = {site: [] for site in SITES}

    for index, batch in enumerate(batches):
        probe.clear()
        probe.install()
        fresh = _clone_batch(batch)
        forward_kwargs, _meta = process_model.generate_input(fresh)
        # KL is computed from logits below. The KV cache is not used, and
        # keeping it on GPU overflows a 32GB card once more than one sample
        # is in flight.
        forward_kwargs["use_cache"] = False
        outputs = process_model.forward(**forward_kwargs)
        logits = _logits_of(outputs).detach().float().cpu()
        labels = forward_kwargs["labels"].detach().cpu()
        probe.remove()
        captured = dict(probe.captured)
        for site in SITES:
            if site not in captured:
                missing[site] += 1
        del outputs, forward_kwargs, fresh
        torch.cuda.empty_cache()

        if fp_cache is None:
            built.append({"activations": captured, "logits": logits, "labels": labels})
            continue

        ref = fp_cache[index]
        for site in SITES:
            if site not in captured or site not in ref["activations"]:
                if save_error:
                    errors[site].append(torch.zeros(0))
                continue
            fp_act = ref["activations"][site]
            mse[site].add(fp_act, captured[site])
            if fp_act.shape != captured[site].shape:
                if save_error:
                    errors[site].append(torch.zeros(0))
                continue
            err = captured[site].float() - fp_act.float()
            if save_error:
                errors[site].append(err.half())
            if baselines:
                acc = None
                aligned = True
                for prev in baselines:
                    piece = prev[site][index]
                    if piece.numel() == 0 or piece.shape != err.shape:
                        aligned = False
                        break
                    piece_f = piece.float()
                    acc = piece_f if acc is None else acc + piece_f
                if aligned and acc is not None:
                    cross[site].add(acc, err)
        part_sum, part_count = _masked_kl(ref["logits"], logits, ref["labels"])
        kl_sum += part_sum
        kl_count += part_count
        del logits

    if fp_cache is None:
        return built, None

    cell = {}
    for site, stat in mse.items():
        cell[site] = {
            "mse": stat.value(),
            "numel": stat.numel,
            "n_shape_mismatch": stat.n_mismatch,
            "n_missing_hook": missing[site],
        }
        if baselines:
            cell[site]["cross_residual_mse"] = cross[site].value()
    cell["logit_kl"] = {
        "mean": (kl_sum / kl_count) if kl_count else None,
        "n_tokens": kl_count,
    }
    return cell, (errors if save_error else None)


def _signal_rms(fp_cache: list[dict]) -> dict[str, float | None]:
    """RMS of the full-precision activation at each probe. MSE / RMS^2 is a relative error."""
    out: dict[str, float | None] = {}
    for site in SITES:
        sse = 0.0
        numel = 0
        for item in fp_cache:
            tensor = item["activations"].get(site)
            if tensor is None:
                continue
            sse += float(tensor.float().pow(2).sum().item())
            numel += int(tensor.numel())
        out[site] = (sse / numel) ** 0.5 if numel else None
    return out


def _finite(value: float | None) -> float:
    if value is None:
        raise RuntimeError("A required metric is missing; a hook did not fire.")
    return float(value)


def build_interactions(cells: dict[str, dict]) -> dict[str, dict[str, float]]:
    out = {}
    metric_names = list(SITES) + ["logit_kl"]
    for metric in metric_names:
        flat = {}
        for name in CELL_NAMES:
            payload = cells[name][metric]
            flat[name] = _finite(payload["mse"] if metric != "logit_kl" else payload["mean"])
        out[metric] = interaction_row(flat)
    return out


def _mse(cells: dict[str, dict], cell: str, site: str) -> float | None:
    return cells[cell][site]["mse"]


def sanity_flags(cells: dict[str, dict]) -> list[str]:
    """Isolation checks, scaled by the vision-only error so float16 storage does not dominate."""
    flags = []
    v_vision = _mse(cells, "V", "vision_out")
    if v_vision is None or v_vision <= 1e-6:
        flags.append(
            "Vision-only quantization left vision_out unchanged. "
            "RTN did not reach the vision tower, or the hook is downstream of a cache."
        )
        return flags

    fp_vision = _mse(cells, "", "vision_out")
    if fp_vision is None or fp_vision > 1e-3 * v_vision:
        flags.append(
            f"FP rerun vision_out mse={fp_vision} is not negligible next to "
            f"vision-only mse={v_vision}. The reference forward is not stable."
        )

    def _isolated(cell: str, site: str, what: str) -> None:
        value = _mse(cells, cell, site)
        if value is None:
            flags.append(f"{what}: metric missing.")
            return
        if value > 1e-3 * v_vision:
            flags.append(
                f"{what} (mse={value}, vision-only mse={v_vision}). "
                "The component split is leaking."
            )

    _isolated("P", "vision_out", "Projector-only quantization moved vision_out")
    _isolated("L", "vision_out", "LLM-only quantization moved vision_out")
    _isolated("L", "proj_out", "LLM-only quantization moved proj_out")
    return flags


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

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
    if hasattr(process_model, "to_cuda"):
        process_model.to_cuda()
    else:
        process_model.model.cuda()

    batches = load_collated_batches(process_model, cfg)
    groups, vision_hook, proj_hook, llm_blocks, component_map = _layout(process_model)
    print(f"[interaction] components: {component_map}", flush=True)
    book = WeightBook(groups)
    probe = Probe(vision_hook, proj_hook, llm_blocks)

    w_bit = int(cfg.get("w_bit", 4))
    w_group = int(cfg.get("w_group", 128))
    zero_point = bool(cfg.get("zero_point", True))

    print("[interaction] FP reference pass", flush=True)
    t0 = time.perf_counter()
    book.restore()
    fp_cache, _ = run_cell_against_fp(process_model, batches, probe, None)
    print(f"[interaction] FP reference done in {time.perf_counter() - t0:.1f}s", flush=True)
    signal_rms = _signal_rms(fp_cache)

    cells: dict[str, dict] = {}
    quant_stats: dict[str, dict] = {}
    saved_errors: dict[str, dict[str, list[torch.Tensor]]] = {}
    for name in CELL_NAMES:
        print(f"[interaction] cell {name or 'fp'}", flush=True)
        book.restore()
        quant_stats[name] = book.quantize(name, w_bit, w_group, zero_point)
        baselines = None
        if len(name) == 2:
            baselines = [saved_errors[ch] for ch in name]
        cell_t0 = time.perf_counter()
        cells[name], errors = run_cell_against_fp(
            process_model,
            batches,
            probe,
            fp_cache,
            save_error=(name in ("V", "P", "L")),
            baselines=baselines,
        )
        if errors is not None:
            saved_errors[name] = errors
        resid = cells[name]["proj_out"].get("cross_residual_mse")
        print(
            f"[interaction] cell {name or 'fp'} done in {time.perf_counter() - cell_t0:.1f}s "
            f"vision_mse={cells[name]['vision_out']['mse']} "
            f"logit_kl={cells[name]['logit_kl']['mean']} "
            f"proj_cross={resid}",
            flush=True,
        )
    book.restore()
    probe.remove()
    del saved_errors, fp_cache

    flags = sanity_flags(cells)
    interactions = None
    interaction_error = None
    if not flags:
        try:
            interactions = build_interactions(cells)
        except RuntimeError as exc:
            interaction_error = str(exc)
            flags.append(str(exc))
    else:
        interaction_error = "Skipped interaction numbers because sanity checks failed."

    payload = {
        "experiment": "01_component_interaction",
        "config": {k: cfg[k] for k in cfg if k != "model_args"},
        "model_args": cfg.get("model_args"),
        "quantizer": {
            "method": "RTN",
            "weight_only": True,
            "w_bit": w_bit,
            "w_group": w_group,
            "zero_point": zero_point,
            "excluded": [
                "patch embedding conv",
                "lm_head",
                "token embeddings",
                "normalization",
            ],
        },
        "component_map": component_map,
        "param_counts": book.meta,
        "signal_rms": signal_rms,
        "quant_stats": quant_stats,
        "cells": cells,
        "interactions": interactions,
        "interaction_error": interaction_error,
        "sanity_flags": flags,
        "sign": "Positive interaction means the joint loss exceeds the sum of the single-component losses.",
    }

    out_path = cfg.get("output_json") or "new_method/results/01_component_interaction/w4.json"
    if not os.path.isabs(out_path):
        out_path = os.path.join(_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[interaction] wrote {out_path}", flush=True)
    if flags:
        print("[interaction] SANITY FLAGS:", flush=True)
        for flag in flags:
            print(" -", flag, flush=True)


if __name__ == "__main__":
    main()
