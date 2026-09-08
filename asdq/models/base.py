import abc
from typing import Dict, Tuple


class BaseModel:
    @abc.abstractmethod
    def fetch_vit(self):
        pass

    @abc.abstractmethod
    def fetch_llm(self):
        pass

    @abc.abstractmethod
    def fetch_proj(self):
        pass

    @abc.abstractmethod
    def vision_preprocess(self, image):
        pass

    @abc.abstractmethod
    def language_preprocess(self, text):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def generate_input(self, data_samples) -> Tuple[Dict, Dict]:
        """Prepare model inputs from collated data samples.

        Returns:
            forward_kwargs: dict with keys accepted by ``forward()``
                (e.g. inputs_embeds, labels, attention_mask).
            metadata: dict with auxiliary tensors for analysis only
                (e.g. vision_mask, caption_mask). These must NOT be
                passed to ``forward()``.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
