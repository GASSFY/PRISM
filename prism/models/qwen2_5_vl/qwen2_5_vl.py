"""PRISM process-model adapter for Qwen2.5-VL."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from prism.models.base import BaseModel
from prism.utils.registry import MODEL_REGISTRY

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover
    process_vision_info = None


def _sharegpt_turns(conversations: list[dict]) -> tuple[str, str]:
    human, gpt = "", ""
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user") and not human:
            human = value
        elif role in ("gpt", "assistant") and not gpt:
            gpt = value
    return human, gpt


@MODEL_REGISTRY.register("qwen2_5_vl")
class Qwen2_5_VL(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        if self.processor is None:
            raise ValueError("Qwen2.5-VL adapter requires a processor")
        if process_vision_info is None:
            raise ImportError("qwen-vl-utils is required for Qwen2.5-VL calibration")
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.image_token_id = int(getattr(self.model.config, "image_token_id", 151655))
        self.ignore_index = -100
        # Keep vision resolution bounded for calib speed / memory.
        self.max_pixels = 512 * 512

    def fetch_vit(self):
        return self.model.visual

    def fetch_llm(self):
        return self.model.model

    def fetch_proj(self):
        return None

    def vision_preprocess(self, image):
        return image

    def language_preprocess(self, text):
        return self.tokenizer(text)

    def to_cuda(self):
        self.model = self.model.cuda()

    def to_cpu(self):
        self.model = self.model.cpu()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Language-side forward used by Hessian Catcher (vision already merged)."""
        del input_ids, kwargs  # calibration path always uses inputs_embeds
        return_dict = return_dict if return_dict is not None else True
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False if use_cache is None else use_cache,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.model.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss = CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def preprocess_data(self, images, data_item):
        conversations = deepcopy(data_item["conversations"])
        human, gpt = _sharegpt_turns(conversations)
        human_text = human.replace("<image>", "").strip()
        if not human_text:
            human_text = "Describe the image."

        pil_images: list[Image.Image] = []
        if images is not None:
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                else:
                    pil_images.append(Image.open(img).convert("RGB"))
        if not pil_images:
            pil_images = [Image.new("RGB", (224, 224), (255, 255, 255))]

        content: list[dict[str, Any]] = []
        for img in pil_images:
            content.append(
                {
                    "type": "image",
                    "image": img,
                    "max_pixels": self.max_pixels,
                }
            )
        content.append({"type": "text", "text": human_text})
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        prompt_inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        answer = gpt.strip() if gpt else ""
        # Match chat template assistant termination.
        answer_ids = self.tokenizer(
            answer + "<|im_end|>\n",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1).squeeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, self.ignore_index)
        labels[prompt_inputs["input_ids"].shape[1] :] = answer_ids.squeeze(0)

        igt = prompt_inputs["image_grid_thw"]
        # Per-sample tensor shaped [n_img, 3]; collapse leading singleton batch dim from processor.
        if igt.dim() == 2 and igt.shape[0] == 1:
            igt = igt  # already [1, 3] for one image — keep as [n_img, 3]
        elif igt.dim() == 3 and igt.shape[0] == 1:
            igt = igt.squeeze(0)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": prompt_inputs["pixel_values"],
            "image_grid_thw": igt,
        }
        if "id" in data_item:
            data_dict["sample_id"] = data_item["id"]
        return data_dict

    def data_collator(self, instances):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        max_len = max(feat["input_ids"].shape[0] for feat in instances)

        input_ids, labels, attention_mask = [], [], []
        for feat in instances:
            cur_len = feat["input_ids"].shape[0]
            pad_len = max_len - cur_len
            input_ids.append(
                torch.cat(
                    [
                        feat["input_ids"],
                        torch.full((pad_len,), pad_id, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        feat["labels"],
                        torch.full((pad_len,), self.ignore_index, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        feat["attention_mask"],
                        torch.zeros(pad_len, dtype=feat["attention_mask"].dtype),
                    ]
                )
            )

        batch = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.cat([f["pixel_values"] for f in instances], dim=0),
            "image_grid_thw": torch.cat(
                [f["image_grid_thw"].reshape(-1, 3) for f in instances], dim=0
            ),
        }
        if "sample_id" in instances[0]:
            batch["sample_id"] = [f["sample_id"] for f in instances]
        return batch

    @torch.no_grad()
    def generate_input(self, data_samples):
        device = next(self.model.parameters()).device
        dtype = self.model.dtype

        input_ids = data_samples["input_ids"].to(device)
        attention_mask = data_samples["attention_mask"].to(device)
        labels = data_samples["labels"].to(device)
        pixel_values = data_samples["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = data_samples["image_grid_thw"].to(device)

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}"
            )
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        self.model.rope_deltas = rope_deltas

        vision_mask = input_ids == self.image_token_id
        answer_mask = labels != self.ignore_index
        forward_kwargs = {
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        metadata = {
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }
        return forward_kwargs, metadata

    @torch.no_grad()
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        raise NotImplementedError("few_shot_format is not supported for Qwen2.5-VL adapter yet")

    @torch.no_grad()
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):
        raise NotImplementedError("interleave_format is not supported for Qwen2.5-VL adapter yet")
