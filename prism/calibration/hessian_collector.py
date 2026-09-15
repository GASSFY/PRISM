"""
Streaming second-order energy collection for ASD importance ranking.

Uses MBQ-style Catcher + layer-by-layer forward to avoid CUDA OOM:
  1. Catcher intercepts the first transformer block's input (hidden_states + kwargs).
  2. Each block is moved to GPU one at a time; hooks accumulate E[x_c^2] on CPU.
  3. After processing, the block returns to CPU before the next one loads.

Optional modality split (vision vs text tokens) uses metadata ``vision_mask``
together with ``attention_mask`` so padding is excluded from both buckets.
"""
from __future__ import annotations

import gc
import time
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from prism.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key

_EPS = 1e-8


def move_embed(model: nn.Module, device: str) -> None:
    """Move embedding (pre-block) layers between CPU and CUDA.

    Mirrors MBQ ``pre_quant.move_embed`` to keep only the necessary
    sub-modules on the target device during the Catcher phase.
    """
    cls_name = model.__class__.__name__
    if cls_name in ("LlavaQwenForCausalLM", "LlavaLlamaForCausalLM", "LlavaLlamaModel"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "Qwen2" in cls_name and hasattr(model, "model"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "Llama" in cls_name and hasattr(model, "model"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif "InternVL" in cls_name and hasattr(model, "language_model"):
        lm = model.language_model
        inner = getattr(lm, "model", lm)
        if hasattr(inner, "tok_embeddings"):
            inner.tok_embeddings = inner.tok_embeddings.to(device)
        elif hasattr(inner, "embed_tokens"):
            inner.embed_tokens = inner.embed_tokens.to(device)
    else:
        pass


def _to_bs_bool_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert attention / vision masks to shape (B, S) bool."""
    if mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    m = mask.detach().to(device)
    if m.dim() == 1:
        if m.numel() == seq_len:
            m = m.unsqueeze(0).expand(batch_size, -1)
        elif m.numel() == batch_size * seq_len:
            m = m.view(batch_size, seq_len)
        else:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    elif m.dim() == 2:
        if m.shape[0] != batch_size or m.shape[1] < seq_len:
            if m.numel() == batch_size * seq_len:
                m = m.view(batch_size, seq_len)
            else:
                return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        m = m[:, :seq_len]
    elif m.dim() == 3:
        # (B, 1, S) or (B, S, S)
        if m.shape[-1] >= seq_len and m.shape[1] == 1:
            m = m[:, 0, :seq_len]
        elif m.shape[1] >= seq_len and m.shape[2] >= seq_len:
            m = m[:, :seq_len, :seq_len].diagonal(dim1=1, dim2=2)
        else:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    elif m.dim() == 4:
        # (B, heads, Q, K) additive / bool mask — use last query over keys
        key = m[:, 0, -1, :seq_len]
        if key.dtype == torch.bool:
            m = key
        else:
            m = key > -1e3
    else:
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    if m.dtype != torch.bool:
        m = m != 0
    if m.shape != (batch_size, seq_len):
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return m


def capture_first_block_inputs(
    model_wrapper,
    forward_kwargs_list: list[dict],
) -> tuple[list[torch.Tensor], list[dict]]:
    """
    Run Catcher on block 0 to cache per-mini-batch hidden states + kwargs (on CPU).
    Leaves the model on CPU with embed offloaded.
    """
    model = model_wrapper.model
    layers = get_blocks(model)

    all_inps: list[torch.Tensor] = []
    all_layer_kwargs: list[dict] = []

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            all_inps.append(inp.detach().cpu())
            # Drop mutable KV cache; clone tensors so later block forwards
            # cannot poison shared Catcher state (mask length S vs 2S).
            saved_kw = {}
            for k, v in kwargs.items():
                if k in ("past_key_value", "past_key_values"):
                    continue
                if isinstance(v, torch.Tensor):
                    saved_kw[k] = v.detach().cpu().clone()
                elif isinstance(v, tuple) and v and all(isinstance(x, torch.Tensor) for x in v):
                    saved_kw[k] = tuple(x.detach().cpu().clone() for x in v)
                else:
                    saved_kw[k] = v
            saved_kw["use_cache"] = False
            saved_kw["past_key_value"] = None
            all_layer_kwargs.append(saved_kw)
            raise ValueError

    layers[0] = Catcher(layers[0])

    total_batches = len(forward_kwargs_list)
    report_every = max(1, total_batches // 4)
    capture_t0 = time.perf_counter()
    print(
        f"[PRISM] Calibration forward capture started: {total_batches} batches",
        flush=True,
    )
    model_wrapper.to_cuda()
    for batch_idx, kwargs in enumerate(forward_kwargs_list, start=1):
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        try:
            model_wrapper.forward(**batch)
        except ValueError:
            pass
        del batch
        torch.cuda.empty_cache()
        if batch_idx % report_every == 0 or batch_idx == total_batches:
            print(
                f"[PRISM] Calibration forward capture "
                f"{batch_idx}/{total_batches}",
                flush=True,
            )

    model_wrapper.to_cpu()
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[PRISM] Calibration forward capture done: "
        f"{len(all_inps)} cached batches, "
        f"time={time.perf_counter() - capture_t0:.1f}s",
        flush=True,
    )
    return all_inps, all_layer_kwargs


def _prepare_layer_kwargs(raw: dict) -> dict:
    kw: dict[str, Any] = {}
    for k, v in raw.items():
        if k in ("past_key_value", "past_key_values"):
            continue
        if isinstance(v, torch.Tensor):
            kw[k] = v.detach().to("cuda", copy=True)
        elif isinstance(v, tuple) and v and all(isinstance(x, torch.Tensor) for x in v):
            kw[k] = tuple(x.detach().to("cuda", copy=True) for x in v)
        else:
            kw[k] = v
    kw["use_cache"] = False
    kw["past_key_value"] = None
    return kw


def _accumulate(
    store_sum: dict[str, torch.Tensor],
    store_count: dict[str, int],
    key: str,
    x2: torch.Tensor,
    n: int,
) -> None:
    if n <= 0:
        return
    x2 = x2.detach().float().cpu()
    if key not in store_sum:
        store_sum[key] = x2
        store_count[key] = n
    else:
        store_sum[key] += x2
        store_count[key] += n


def _finalize(store_sum: dict[str, torch.Tensor], store_count: dict[str, int]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, s in store_sum.items():
        n = max(int(store_count.get(key, 0)), 1)
        out[key] = s / n
    return out


@torch.no_grad()
def collect_hessian_diag(
    model_wrapper,
    forward_kwargs_list: list[dict],
    metadata_list: list[dict] | None = None,
    split_modality: bool = False,
) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
    """Collect per-column second-order energy E[x_c^2] for every Linear layer.

    This is an OWQ-style diagonal activation energy, not a full GPTQ Hessian.

    Args:
        model_wrapper: process_model with ``.model``, ``.forward()``,
            ``.to_cuda()`` and ``.to_cpu()``.
        forward_kwargs_list: mini-batch dicts for ``model_wrapper.forward()``.
        metadata_list: optional per-batch metadata; must contain ``vision_mask``
            when ``split_modality=True``.
        split_modality: if True, also accumulate vision/text energies using
            ``vision_mask`` and ``attention_mask``.

    Returns:
        If ``split_modality`` is False:
            ``{layer_key: E[x_c^2]}``
        If True:
            ``{"mixed": {...}, "vision": {...}, "text": {...}}``
    """
    if split_modality:
        if metadata_list is None or len(metadata_list) != len(forward_kwargs_list):
            raise ValueError(
                "split_modality=True requires metadata_list aligned with "
                "forward_kwargs_list (need vision_mask)."
            )
        for i, meta in enumerate(metadata_list):
            if "vision_mask" not in meta:
                raise ValueError(f"metadata_list[{i}] missing vision_mask")

    model = model_wrapper.model
    layers = get_blocks(model)

    all_inps, all_layer_kwargs = capture_first_block_inputs(
        model_wrapper, forward_kwargs_list,
    )

    sum_mixed: dict[str, torch.Tensor] = {}
    count_mixed: dict[str, int] = {}
    sum_vision: dict[str, torch.Tensor] = {}
    count_vision: dict[str, int] = {}
    sum_text: dict[str, torch.Tensor] = {}
    count_text: dict[str, int] = {}

    # Filled per mini-batch before layer forward; read by hooks.
    batch_ctx: dict[str, torch.Tensor | None] = {
        "attn_mask": None,
        "vision_mask": None,
    }

    def _make_hook(key: str):
        def hook(_module, args, _result):
            x = args[0]
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return
            if x.dim() == 2:
                # (tokens, C) — no reliable modality split
                x2 = x.detach().float().pow(2).sum(dim=0)
                _accumulate(sum_mixed, count_mixed, key, x2, x.shape[0])
                return

            if x.dim() != 3:
                x_flat = x.detach().float().view(-1, x.shape[-1])
                x2 = x_flat.pow(2).sum(dim=0)
                _accumulate(sum_mixed, count_mixed, key, x2, x_flat.shape[0])
                return

            bsz, seq_len, _ = x.shape
            x_f = x.detach().float()
            device = x_f.device
            attn = _to_bs_bool_mask(batch_ctx["attn_mask"], bsz, seq_len, device)
            valid = attn.reshape(-1)
            x_flat = x_f.reshape(-1, x_f.shape[-1])
            if valid.any():
                x2 = (x_flat[valid].pow(2)).sum(dim=0)
                _accumulate(sum_mixed, count_mixed, key, x2, int(valid.sum().item()))

            if not split_modality:
                return

            vision = _to_bs_bool_mask(batch_ctx["vision_mask"], bsz, seq_len, device)
            vision_flat = vision.reshape(-1)
            v_sel = valid & vision_flat
            t_sel = valid & (~vision_flat)
            if v_sel.any():
                x2_v = x_flat[v_sel].pow(2).sum(dim=0)
                _accumulate(sum_vision, count_vision, key, x2_v, int(v_sel.sum().item()))
            if t_sel.any():
                x2_t = x_flat[t_sel].pow(2).sum(dim=0)
                _accumulate(sum_text, count_text, key, x2_t, int(t_sel.sum().item()))

        return hook

    desc = "[PRISM] Collecting modality energy..." if split_modality else "[PRISM] Collecting Hessian diag..."
    for layer_idx in tqdm(range(len(layers)), desc=desc):
        layer = layers[layer_idx].cuda()

        handles = []
        for name, linear in get_named_linears(layer).items():
            key = _linear_layer_key(layer_idx, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

        new_inps = []
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = _prepare_layer_kwargs(all_layer_kwargs[batch_idx])
            batch_ctx["attn_mask"] = kw.get("attention_mask")
            if split_modality:
                vm = metadata_list[batch_idx]["vision_mask"]
                batch_ctx["vision_mask"] = (
                    vm.cuda() if isinstance(vm, torch.Tensor) else vm
                )
            else:
                batch_ctx["vision_mask"] = None

            out = layer(inp, **kw)[0]
            new_inps.append(out.detach().cpu())
            del inp, out, kw
            batch_ctx["attn_mask"] = None
            batch_ctx["vision_mask"] = None
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        all_inps = new_inps
        layers[layer_idx] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    mixed = _finalize(sum_mixed, count_mixed)
    if not split_modality:
        return mixed

    vision = _finalize(sum_vision, count_vision)
    text = _finalize(sum_text, count_text)
    # Layers that never saw a modality fall back to mixed energy.
    for key in mixed:
        if key not in vision:
            vision[key] = mixed[key].clone()
        if key not in text:
            text[key] = mixed[key].clone()

    n_v = sum(count_vision.values())
    n_t = sum(count_text.values())
    print(
        f"[PRISM] Modality energy tokens: vision={n_v}, text={n_t}, "
        f"layers_mixed={len(mixed)}",
        flush=True,
    )
    return {"mixed": mixed, "vision": vision, "text": text}
