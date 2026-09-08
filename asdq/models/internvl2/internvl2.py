import math
from copy import deepcopy
from typing import Optional

import torch
from PIL import Image
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constants import IMG_CONTEXT_TOKEN
from .dataset import build_transform, dynamic_preprocess, preprocess, preprocess_internlm, preprocess_mpt, preprocess_phi3
from asdq.models.base import BaseModel
from asdq.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("internvl2")
class InternVL2(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.template_name = "internlm2-chat"
        self.num_image_token = self.model.num_image_token
        self.image_size = 448
        self.pad2square = False
        self.dynamic_image_size = True
        self.use_thumbnail = True
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 1
        self.normalize_type = "imagenet"
        self.group_by_length = True
        self.device_map = None

    def fetch_vit(self):
        return self.model.vision_model

    def fetch_llm(self):
        return self.model.language_model

    def fetch_proj(self):
        return self.model.mlp1

    def vision_preprocess(self, image):
        transform = self.get_transform()
        if len(image) == 1:
            img = image[0]
            if self.dynamic_image_size:
                images = dynamic_preprocess(
                    img,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                num_tiles = [len(images)]
            else:
                images = [img]
                num_tiles = [1]
            pixel_values = torch.stack([transform(i) for i in images])
            num_patches = pixel_values.size(0)
            if not self.dynamic_image_size:
                assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."
        else:
            images, num_tiles = [], []
            num_image = len(image)
            for i in range(num_image):
                img = image[i]
                if self.dynamic_image_size:
                    img = dynamic_preprocess(
                        img,
                        min_num=self.min_dynamic_patch,
                        max_num=self.max_dynamic_patch // num_image,
                        image_size=self.image_size,
                        use_thumbnail=self.use_thumbnail,
                    )
                    images += img
                    num_tiles.append(len(img))
                else:
                    images.append(img)
                    num_tiles.append(1)
            pixel_values = torch.stack([transform(i) for i in images])
            num_patches = pixel_values.size(0)
        return pixel_values, num_patches, num_tiles

    def language_preprocess(self, text):
        return self.tokenizer(text)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        lm = self.fetch_llm()
        if input_ids is not None:
            outputs = lm(
                input_ids=input_ids.to(next(lm.parameters()).device),
                attention_mask=attention_mask.to(next(lm.parameters()).device) if attention_mask is not None else None,
                use_cache=use_cache,
            )
        else:
            outputs = lm(
                inputs_embeds=inputs_embeds.to(next(lm.parameters()).device),
                attention_mask=attention_mask.to(next(lm.parameters()).device) if attention_mask is not None else None,
                use_cache=use_cache,
            )

        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, lm.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss = CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def split_model(self, num_layers, vit_alpha=0.5):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    def to_cuda(self):
        if self.num_params > 20 * 10 ** 9:
            existing_map = getattr(self.model, "hf_device_map", None)
            if isinstance(existing_map, dict) and len(existing_map) > 0:
                self.device_map = existing_map
            else:
                num_layers = self.model.language_model.config.num_hidden_layers
                self.device_map = self.split_model(num_layers)
            self.model = dispatch_model(self.model, device_map=self.device_map)
        else:
            self.device_map = None
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9:
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()

    def get_preprocess_function(self):
        if self.template_name == "Hermes-2":
            return preprocess_mpt
        if self.template_name == "internlm2-chat":
            return preprocess_internlm
        if self.template_name == "phi3-chat":
            return preprocess_phi3
        return preprocess

    def get_transform(self):
        return build_transform(
            is_train=False,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )

    def preprocess_data(self, images, data_item):
        if images is not None:
            pixel_values, num_patches, num_tiles = self.vision_preprocess(images)
            preprocess_function = self.get_preprocess_function()
            if len(images) == 1:
                if "<image>" not in data_item["conversations"][0]["value"]:
                    data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]
                ret = preprocess_function(
                    self.template_name,
                    [deepcopy(data_item["conversations"])],
                    self.tokenizer,
                    [self.num_image_token * num_patches],
                    group_by_length=self.group_by_length,
                    ds_name="sharegpt4v",
                )
            else:
                num_image = len(data_item["image"])
                num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
                ret = preprocess_function(
                    self.template_name,
                    [deepcopy(data_item["conversations"])],
                    self.tokenizer,
                    num_image_tokens,
                    group_by_length=self.group_by_length,
                    ds_name="sharegpt4v",
                    num_image=num_image,
                )
            data_dict = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            )
        else:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
            pixel_values, num_patches, _ = self.vision_preprocess([image])
            preprocess_function = self.get_preprocess_function()
            ret = preprocess_function(
                self.template_name,
                [deepcopy(data_item["conversations"])],
                self.tokenizer,
                [self.num_image_token * num_patches],
                text_only=True,
                group_by_length=self.group_by_length,
                ds_name="sharegpt4v",
            )
            data_dict = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                attention_mask=ret["attention_mask"][0],
                pixel_values=pixel_values,
                image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            )
        if "id" in data_item:
            data_dict["sample_id"] = data_item["id"]
        return data_dict

    @torch.no_grad()
    def generate_input(self, data_samples):
        viz_dev = next(self.model.vision_model.parameters()).device
        input_ids = data_samples["input_ids"].to(viz_dev)
        attention_mask = data_samples["attention_mask"].to(viz_dev)
        labels = data_samples["labels"].to(viz_dev)
        pixel_values = data_samples["pixel_values"].to(dtype=self.model.dtype, device=viz_dev)
        image_flags = data_samples["image_flags"].to(viz_dev).squeeze(-1)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        vit_embeds = self.model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        bsz, seq_len, dim = input_embeds.shape
        flat_embeds = input_embeds.reshape(bsz * seq_len, dim)
        flat_ids = input_ids.reshape(bsz * seq_len)
        selected = flat_ids == self.model.img_context_token_id
        try:
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, dim)
        except Exception:
            vit_flat = vit_embeds.reshape(-1, dim)
            n_token = selected.sum()
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_flat[:n_token]
        input_embeds = flat_embeds.reshape(bsz, seq_len, dim)

        vision_mask = selected.reshape(bsz, seq_len)
        answer_mask = labels != -100
        forward_kwargs = {
            "inputs_embeds": input_embeds,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        metadata = {
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }
        return forward_kwargs, metadata

    def data_collator(self, instances):
        pad_id = 0
        ignore_index = -100
        first = instances[0]
        batch = {}
        max_item_length = max(feat["input_ids"].shape[0] for feat in instances)
        for feat in instances:
            temp_input_ids = torch.full((max_item_length,), pad_id, dtype=torch.long)
            temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = torch.full((max_item_length,), ignore_index, dtype=torch.long)
            temp_labels[: feat["labels"].shape[0]] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        for k, v in first.items():
            if k not in ("label", "label_ids", "pixel_values", "image_flags", "sample_id") and v is not None and not isinstance(v, str):
                batch[k] = torch.stack([f[k] for f in instances])
            elif k in ("pixel_values", "image_flags"):
                batch[k] = torch.concat([f[k] for f in instances])
            elif k == "sample_id":
                batch[k] = [f[k] for f in instances]
        return batch

    @torch.no_grad()
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_flags = data_samples["image_flags"]
        sample_id = data_samples["sample_id"]

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_ids = [torch.cat(input_ids[i : i + interleave_freq], dim=0) for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq)]
        new_labels = [torch.cat(labels[i : i + interleave_freq], dim=0) for i in range(0, len(labels) - interleave_freq + 1, interleave_freq)]

        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                new_labels_padded[i, -cur_len:] = cur_new_labels
                new_attention_mask[i, -cur_len:] = True
            else:
                new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_attention_mask[i, :cur_len] = True
        new_sample_id = [sample_id[i : i + interleave_freq] for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq)]
        return {
            "input_ids": new_input_ids_padded,
            "labels": new_labels_padded,
            "attention_mask": new_attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "sample_id": new_sample_id,
        }

    @torch.no_grad()
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_flags = data_samples["image_flags"]
        sample_id = data_samples["sample_id"]

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_ids, new_labels = [], []
        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = [input_ids[i]]
            cur_labels = [labels[i]]
            for j in range(interleave_freq - 1):
                cur_input_ids.append(pure_text[i + j])
                cur_input_ids.append(input_ids[i + 1 + j])
                cur_labels.append(pure_text[i + j])
                cur_labels.append(labels[i + 1 + j])
            new_input_ids.append(torch.cat(cur_input_ids, dim=0))
            new_labels.append(torch.cat(cur_labels, dim=0))

        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                new_labels_padded[i, -cur_len:] = cur_new_labels
                new_attention_mask[i, -cur_len:] = True
            else:
                new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_attention_mask[i, :cur_len] = True
        new_sample_id = [sample_id[i : i + interleave_freq] for i in range(0, len(sample_id) - interleave_freq + 1, interleave_freq)]
        return {
            "input_ids": new_input_ids_padded,
            "labels": new_labels_padded,
            "attention_mask": new_attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "sample_id": new_sample_id,
        }

