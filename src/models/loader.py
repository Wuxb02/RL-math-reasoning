import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import yaml


class ModelLoader:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model_config = self.config["model"]
        self.device = self._detect_device()

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model_and_tokenizer(
        self, dtype: Optional[str] = None, device_map: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_name = self.model_config["huggingface_id"]
        local_dir = self.model_config.get("local_dir")

        if local_dir and Path(local_dir).exists():
            model_name = local_dir

        if dtype is None:
            dtype = self.model_config.get("dtype", "bfloat16")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            device_map=device_map,
        )

        if device_map is None:
            model = model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def get_generation_config(self) -> dict:
        return self.model_config.get(
            "generation",
            {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
        )

    def get_model_name(self) -> str:
        return self.model_config["name"]

    def get_recommended_batch_size(self, gpu_memory: str = "16GB") -> int:
        batch_sizes = self.model_config.get("batch_sizes", {})
        return batch_sizes.get(gpu_memory, 4)
