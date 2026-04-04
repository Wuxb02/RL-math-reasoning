from .base import BaseMethod
from .cot import CoTMethod
from .rloo import RLOOMethod


def __getattr__(name):
    if name == "GRPOMethod":
        from .grpo import GRPOMethod

        return GRPOMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseMethod", "CoTMethod", "RLOOMethod", "GRPOMethod"]
