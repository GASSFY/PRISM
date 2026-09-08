from asdq.models.llava_onevision import LLaVA_onevision  # noqa: F401 - registers "llava_onevision"
from asdq.models.llava_v15 import LLaVA_v15  # noqa: F401 - registers "llava"
from asdq.models.internvl2 import InternVL2  # noqa: F401 - registers "internvl2"
from asdq.utils.registry import MODEL_REGISTRY


def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]
