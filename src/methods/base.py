from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class BaseMethod(ABC):
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.method_config = self.config["method"]

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        pass

    def get_method_name(self) -> str:
        return self.method_config["name"]

    def get_method_type(self) -> str:
        return self.method_config["type"]
