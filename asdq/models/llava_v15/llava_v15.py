import copy
import re
from typing import Dict, Optional, Sequence

import torch
import transformers
from PIL import Image
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules

from .dataset import (
    _add_speaker_and_signal,
    _mask_targets,
    _tokenize_fn,
    preprocess_gemma,
    preprocess_llama_2,
    preprocess_llama3,
    preprocess_mpt,
    preprocess_plain,
    preprocess_qwen,
    preprocess_v1,
)
from asdq.models.base import BaseModel
from asdq.utils.registry import MODEL_REGISTRY

try:
    from llava import conversation as conversation_lib
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.mm_utils import process_anyres_image, process_highres_image, tokenizer_image_token
except ImportError as e:
    print(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


@MODEL_REGISTRY.register("llava")
class LLaVA_v15(BaseModel):
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.device_map = getattr(model, "hf_device_map", {})
        self.vision_tower = model.get_vision_tower()
        self.image_aspect_ratio = "pad"
        self.image_processor = self.vision_tower.image_processor
        self.is_multimodal = True
        self.mm_use_im_start_end = model.config.mm_use_im_start_end
        self.image_grid_pinpoints = getattr(model.config, "image_grid_pinpoints", None)

    def fetch_vit(self):
        return self.model.model.vision_tower

    def fetch_llm(self):
        return self.model.model

    def fetch_proj(self):
        return self.model.model.mm_projector

    def vision_preprocess(self, image: torch.FloatTensor):
        image_size = image.size
        image_aspect_ratio = self.image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.image_processor, self.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.image_processor, self.image_grid_pinpoints)
        elif image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                if width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        outputs = self.model(
            input_ids=input_ids.to(next(self.model.parameters()).device) if input_ids is not None else None,
            inputs_embeds=inputs_embeds.to(next(self.model.parameters()).device) if inputs_embeds is not None else None,
            attention_mask=attention_mask.to(next(self.model.parameters()).device) if attention_mask is not None else None,
            labels=labels.to(next(self.model.parameters()).device),
            use_cache=use_cache,
            return_dict=return_dict,
        )
        return outputs

    def to_cuda(self):
        if self.num_params > 20 * 10 ** 9:
            self.model = dispatch_model(self.model, device_map=self.device_map)
        else:
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9:
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()

    def preprocess_multimodal(self, sources: Sequence[str]) -> Dict:
        if not self.is_multimodal:
            return sources
        for source in sources:
            for sentence in source:
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
                replace_token = DEFAULT_IMAGE_TOKEN
                if self.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
                sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")
        return sources

    def preprocess(self, sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
        if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            return preprocess_plain(sources, tokenizer)
        if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
            return preprocess_llama_2(sources, tokenizer, has_image=has_image)
        if conversation_lib.default_conversation.version.startswith("v1"):
            return preprocess_v1(sources, tokenizer, has_image=has_image)
        if conversation_lib.default_conversation.version == "mpt":
            return preprocess_mpt(sources, tokenizer, has_image=has_image)
        if conversation_lib.default_conversation.version == "qwen":
            return preprocess_qwen(sources, tokenizer, has_image=has_image)
        if conversation_lib.default_conversation.version == "gemma":
            return preprocess_gemma(sources, tokenizer, has_image=has_image)
        if conversation_lib.default_conversation.version == "llama_v3":
            return preprocess_llama3(sources, tokenizer, has_image=has_image)

        conversations = []
        for source in sources:
            header = f"{conversation_lib.default_conversation.system}\n\n"
            conversations.append(_add_speaker_and_signal(header, source))

        def get_tokenize_len(prompts):
            return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

        if has_image:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
        else:
            input_ids = _tokenize_fn(conversations, tokenizer)["input_ids"]

        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            if has_image:
                tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
            else:
                tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
            _mask_targets(target, tokenized_lens, [sentence["from"] for sentence in source])
        return dict(input_ids=input_ids, labels=targets)

    def preprocess_data(self, images, data_item):
        sources = [data_item]
        if images is not None:
            if len(images) == 1:
                image = [self.vision_preprocess(images[0])]
            else:
                image = [self.vision_preprocess(f) for f in images]
                image = [[im[0], im[1], "image"] for im in image]
            sources = self.preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = self.preprocess(sources, self.tokenizer, has_image="image" in data_item)
        prompt = data_dict.get("prompt", None)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if "image" in data_item:
            data_dict["image"] = image
        elif self.is_multimodal:
            crop_size = self.image_processor.crop_size
            data_dict["image"] = [(torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text")]
        if prompt is not None:
            data_dict["prompt"] = prompt
        if "id" in data_item:
            data_dict["id"] = data_item["id"]
        return data_dict

    @torch.no_grad()
    def generate_input(self, data_samples):
        data_samples["input_ids"] = data_samples["input_ids"].cuda()
        data_samples["attention_mask"] = data_samples["attention_mask"].cuda()
        data_samples["labels"] = data_samples["labels"].cuda()
        for i, _ in enumerate(data_samples["images"]):
            data_samples["images"][i] = data_samples["images"][i].to(self.model.dtype)

        (_, _, attention_mask, _, input_embeds, labels) = self.model.prepare_inputs_labels_for_multimodal(
            data_samples["input_ids"],
            None,
            data_samples["attention_mask"],
            None,
            data_samples["labels"],
            data_samples["images"],
            data_samples["modalities"],
            data_samples["image_sizes"],
        )

        vision_sel = []
        for batch_idx, pre_input_ids in enumerate(data_samples["input_ids"]):
            num_images = (pre_input_ids == IMAGE_TOKEN_INDEX).sum()
            pre_labels = data_samples["labels"][batch_idx]
            pre_attn_mask = data_samples["attention_mask"][batch_idx]
            pre_len = pre_labels[pre_attn_mask].shape[0]
            post_labels = labels[batch_idx]
            post_attn_mask = attention_mask[batch_idx]
            post_len = post_labels[post_attn_mask].shape[0]

            image_emb_len = int((post_len - pre_len + num_images) / num_images)
            image_emb_start = torch.where(pre_input_ids == IMAGE_TOKEN_INDEX)[0]
            for idx, _ in enumerate(image_emb_start):
                image_emb_start[idx] = image_emb_start[idx] + (image_emb_len - 1) * idx
            image_emb_end = image_emb_start + image_emb_len

            cur_vision_sel = torch.zeros(post_attn_mask.shape[0], dtype=torch.bool)
            for jdx in range(num_images):
                cur_vision_sel[image_emb_start[jdx] : image_emb_end[jdx]] = True
            vision_sel.append(cur_vision_sel)

        vision_mask = torch.stack(vision_sel)
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

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def data_collator(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            batch["images"] = [im[0] for im_list in images for im in im_list]
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        if "id" in instances[0]:
            batch["sample_id"] = [instance["id"] for instance in instances]
        return batch

    @torch.no_grad()
    def few_shot_data_samples(self, data_samples, pad_side="right", interleave_freq=2):
        from asdq.models.llava_onevision.llava_onevision import LLaVA_onevision

        return LLaVA_onevision.few_shot_data_samples(self, data_samples, pad_side, interleave_freq)

    @torch.no_grad()
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="right", interleave_freq=2):
        from asdq.models.llava_onevision.llava_onevision import LLaVA_onevision

        return LLaVA_onevision.interleave_data_samples(self, data_samples, pure_text, pad_side, interleave_freq)

