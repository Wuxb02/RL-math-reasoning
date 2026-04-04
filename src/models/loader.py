"""
模型加载器模块。

负责从配置文件加载 HuggingFace 因果语言模型（CausalLM）和对应的分词器。
支持自动设备检测（CUDA → MPS → CPU）、数据类型选择和本地模型优先加载。

支持的模型系列:
    - Qwen3-0.6B
    - Qwen3-1.7B
    - Qwen3-4B

配置示例（configs/models/qwen3-0.6b.yaml）:
    model:
      name: "Qwen3-0.6B"
      huggingface_id: "Qwen/Qwen3-0.6B"
      local_dir: "./resources/Qwen3-0.6B"
      dtype: "bfloat16"
"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import yaml


class ModelLoader:
    """
    模型和分词器的统一加载器。

    从 YAML 配置文件读取模型信息，自动处理:
    1. 设备检测（优先使用 GPU）
    2. 数据类型选择（bfloat16/float16/float32）
    3. 本地模型优先（如果已下载则从本地加载）
    4. 分词器配置（pad_token 和 padding_side）
    """

    def __init__(self, config_path: str):
        """
        初始化模型加载器。

        Args:
            config_path: 模型配置文件路径（如 configs/models/qwen3-0.6b.yaml）
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config["model"]
        self.device = self._detect_device()

    def _load_config(self, config_path: str) -> dict:
        """读取并解析 YAML 配置文件。"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _detect_device(self) -> str:
        """
        自动检测可用的计算设备。

        检测顺序:
        1. CUDA（NVIDIA GPU）— 训练首选
        2. MPS（Apple Silicon GPU）— Mac 设备
        3. CPU — 兜底方案

        Returns:
            str: 设备名称 ("cuda", "mps", 或 "cpu")
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model_and_tokenizer(
        self, dtype: Optional[str] = None, device_map: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载预训练模型和分词器。

        加载策略:
        1. 优先从本地目录加载（如果已下载）
        2. 否则从 HuggingFace Hub 下载
        3. 自动设置数据类型和设备

        Args:
            dtype: 数据类型，可选。默认使用配置文件中的设置。
                可选值: "bfloat16", "float16", "float32"
            device_map: 设备映射策略，可选。默认不使用 device_map，
                手动将模型移动到检测到的设备。

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: 加载后的模型和分词器

        注意事项:
            - pad_token 设置为 eos_token，因为部分模型没有独立的 pad_token
            - padding_side 设置为 "left"，这是生成任务的推荐设置
              （确保右对齐，生成的 token 在序列末尾）
        """
        model_name = self.model_config["huggingface_id"]
        local_dir = self.model_config.get("local_dir")

        # 如果本地目录存在，优先从本地加载（避免重复下载）
        if local_dir and Path(local_dir).exists():
            model_name = local_dir

        if dtype is None:
            dtype = self.model_config.get("dtype", "bfloat16")

        # 数据类型映射
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        # 加载因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            device_map=device_map,
        )

        # 如果没有使用 device_map，手动将模型移动到检测到的设备
        if device_map is None:
            model = model.to(self.device)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 设置 pad_token（部分模型没有独立的 pad_token，使用 eos_token 代替）
        tokenizer.pad_token = tokenizer.eos_token
        # 设置左侧填充（生成任务推荐设置，确保生成内容在序列右侧）
        tokenizer.padding_side = "left"

        return model, tokenizer

    def get_generation_config(self) -> dict:
        """
        获取生成配置。

        Returns:
            dict: 生成参数字典，包含 max_new_tokens、temperature 等
        """
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
        """
        获取模型名称（用于日志和输出目录命名）。

        Returns:
            str: 模型名称（如 "Qwen3-0.6B"）
        """
        return self.model_config["name"]

    def get_recommended_batch_size(self, gpu_memory: str = "16GB") -> int:
        """
        根据 GPU 显存获取推荐的批次大小。

        Args:
            gpu_memory: GPU 显存规格（如 "16GB", "24GB", "40GB"）

        Returns:
            int: 推荐的批次大小，默认返回 4
        """
        batch_sizes = self.model_config.get("batch_sizes", {})
        return batch_sizes.get(gpu_memory, 4)
