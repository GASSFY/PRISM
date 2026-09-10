import os
import json
import torch
import numpy as np

from PIL import Image
from datasets import load_dataset


def load_image(image_path):
    # Load the image using use PIL, we don't support tcs_loader
    return Image.open(image_path).convert('RGB')


def get_multimodal_calib_dataset(
    data_path,
    image_folder,
    model,
    n_samples=128,
    few_shot_format=False,
    interleave_format=False,
    text_data_path=None,
    shuffle=True,
    calib_batch_size=8,
):
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r") as json_file:
            for line in json_file:
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r") as json_file:
            dataset = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    if shuffle:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(dataset)

    data_list = []
    for i in range(n_samples):
        i = i % len(dataset)
        data_item = dataset[i]
        if 'image' in data_item and len(data_item['image']) != 0:
            if type(data_item['image']) == list:
                images = []
                for image_path in data_item['image']:
                    full_image_path = os.path.join(image_folder, image_path)
                    image = load_image(full_image_path)
                    images.append(image)
            else:
                images = []
                image_path = data_item['image']
                full_image_path = os.path.join(image_folder, image_path)
                image = load_image(full_image_path)
                images.append(image)
        else:
            images = None

        data_dict = model.preprocess_data(images, data_item)
        data_list.append(data_dict)

    if few_shot_format and interleave_format:
        raise ValueError('You cannot specify both few_shot_format and interleave_format at the same time!')

    all_forward_kwargs = []
    all_metadata = []

    for start in range(0, len(data_list), calib_batch_size):
        batch = data_list[start : start + calib_batch_size]
        examples = model.data_collator(batch)

        if few_shot_format:
            examples = model.few_shot_data_samples(examples)
        if interleave_format:
            if not text_data_path:
                _dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            else:
                _dataset = load_dataset(data_path, split="validation")
            if shuffle:
                _dataset = _dataset.shuffle(seed=42)
            samples = []
            n_run = 0
            for data in _dataset:
                line = data["text"].strip()
                line_encoded = model.tokenizer.encode(line)
                if len(line_encoded) > 512:
                    sample = torch.tensor(line_encoded[:512])
                    samples.append(sample)
                    n_run += 1
                if n_run == 128:
                    break
            examples = model.interleave_data_samples(examples, pure_text=samples)

        fwd_kw, meta = model.generate_input(examples)

        for k, v in fwd_kw.items():
            if isinstance(v, torch.Tensor):
                fwd_kw[k] = v.cpu()
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                meta[k] = v.cpu()
        torch.cuda.empty_cache()

        all_forward_kwargs.append(fwd_kw)
        all_metadata.append(meta)

    return all_forward_kwargs, all_metadata
