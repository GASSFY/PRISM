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

import torch
import torch.nn as nn
from tqdm import tqdm

from asdq.quantization.quantize import get_blocks, get_named_linears, _linear_layer_key


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

    # ------------------------------------------------------------------ #
    # Phase 1: Catcher – capture first-block inputs per mini-batch       #
    # (cf. MBQ pre_quant.py:229-259)                                     #
    # ------------------------------------------------------------------ #
    all_inps: list[torch.Tensor] = []
    all_layer_kwargs: list[dict] = []

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            all_inps.append(inp.detach().cpu())
            saved_kw = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            all_layer_kwargs.append(saved_kw)
            raise ValueError

    layers[0] = Catcher(layers[0])

    model_wrapper.to_cuda()
    for kwargs in forward_kwargs_list:
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

    model_wrapper.to_cpu()
    layers[0] = layers[0].module  # restore original block
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Phase 2: layer-by-layer forward with hooks                         #
    # (cf. MBQ pre_quant.py:317-493)                                     #
    # ------------------------------------------------------------------ #
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

    for layer_idx in tqdm(range(len(layers)), desc="[ASDQ] Collecting Hessian diag..."):
        layer = layers[layer_idx].cuda()

        handles = []
        for name, linear in get_named_linears(layer).items():
            key = _linear_layer_key(layer_idx, name)
            handles.append(linear.register_forward_hook(_make_hook(key)))

        new_inps = []
        for batch_idx in range(len(all_inps)):
            inp = all_inps[batch_idx].cuda()
            kw = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in all_layer_kwargs[batch_idx].items()
            }
            kw["use_cache"] = False
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
