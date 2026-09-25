from prism.models.llava_onevision import LLaVA_onevision  # noqa: F401 - registers "llava_onevision"
from prism.models.llava_v15 import LLaVA_v15  # noqa: F401 - registers "llava"
from prism.models.internvl2 import InternVL2  # noqa: F401 - registers "internvl2"
from prism.models.qwen2_5_vl import Qwen2_5_VL  # noqa: F401 - registers "qwen2_5_vl"
from prism.utils.registry import MODEL_REGISTRY


def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]
