"""
Streaming Hessian diagonal collection for ASD importance ranking.

Uses MBQ-style Catcher + layer-by-layer forward to avoid CUDA OOM:
  1. Catcher intercepts the first transformer block's input (hidden_states + kwargs).
  2. Each block is moved to GPU one at a time; hooks accumulate diag(H) on CPU.
  3. After processing, the block returns to CPU before the next one loads.

Memory cost: O(mini_batches * seq * hidden) for cached hidden states on CPU,
plus one transformer block on GPU at a time.
"""
from __future__ import annotations

import gc
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from prism.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key


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


@torch.no_grad()
def collect_hessian_diag(
    model_wrapper,
    forward_kwargs_list: list[dict],
) -> dict[str, torch.Tensor]:
    """Collect Hessian diagonal for every Linear layer via layer-by-layer forward.

    Follows the MBQ ``run_mbq`` pattern (Catcher -> layer-by-layer -> CPU offload)
    to keep GPU memory bounded to a single transformer block at a time.

    Args:
        model_wrapper: process_model object with ``.model``, ``.forward()``,
            ``.to_cuda()`` and ``.to_cpu()`` methods (same role as MBQ's
            ``model`` argument in ``run_mbq``).
        forward_kwargs_list: list of mini-batch dicts, each containing keys
            accepted by ``model_wrapper.forward()`` (e.g. inputs_embeds,
            labels, attention_mask).  Tensors should reside on CPU.

    Returns:
        ``{layer_key: diag_H}`` where ``diag_H`` is shape ``[C] = E[x_c^2]``.
    """
    model = model_wrapper.model
    layers = get_blocks(model)

    all_inps, all_layer_kwargs = capture_first_block_inputs(
        model_wrapper, forward_kwargs_list,
    )

    sum_x2: dict[str, torch.Tensor] = {}
    count: dict[str, int] = {}

    def _make_hook(key: str):
        def hook(_module, args, _result):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return
            x = x.detach().float().view(-1, x.shape[-1])  # (tokens, C)
            if x.numel() == 0:
                return
            x2 = x.pow(2).sum(dim=0).cpu()  # (C,) -> CPU immediately
            n = x.shape[0]
            if key not in sum_x2:
                sum_x2[key] = x2
                count[key] = n
            else:
                sum_x2[key] += x2
                count[key] += n
        return hook

    for layer_idx in tqdm(range(len(layers)), desc="[PRISM] Collecting Hessian diag..."):
        layer = layers[layer_idx].cuda()

        handles = []
        for name, linear in get_named_linears(layer).items():
            key = _linear_layer_key(layer_idx, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

        new_inps = []
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = {}
            for k, v in all_layer_kwargs[batch_idx].items():
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
            out = layer(inp, **kw)[0]
            new_inps.append(out.detach().cpu())
            del inp, out, kw
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        all_inps = new_inps

        layers[layer_idx] = layer.cpu()
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    return {key: sum_x2[key] / count[key] for key in sum_x2}
