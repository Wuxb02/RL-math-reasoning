from .data import GSM8KDataset, get_gsm8k_dataset
from .models import ModelLoader
from .methods import BaseMethod, CoTMethod, RLOOMethod
from .rewards import correctness_reward_func, int_reward_func
from .evaluation import ExperimentEvaluator


def __getattr__(name):
    if name == "GRPOMethod":
        from .methods import GRPOMethod

        return GRPOMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GSM8KDataset",
    "get_gsm8k_dataset",
    "ModelLoader",
    "BaseMethod",
    "CoTMethod",
    "RLOOMethod",
    "GRPOMethod",
    "correctness_reward_func",
    "int_reward_func",
    "extract_xml_answer",
    "ExperimentEvaluator",
]
