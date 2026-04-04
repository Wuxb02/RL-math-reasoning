"""方法抽象基类定义。

该模块统一三类方法（CoT、RLOO、GRPO）的最小接口，便于训练脚本与
评估脚本通过多态方式调用，不需要关心具体算法实现细节。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class BaseMethod(ABC):
    """所有方法实现的抽象基类。

    设计决策：
    - 配置统一由 YAML 驱动，避免在代码中硬编码超参数。
    - `run` 与 `evaluate` 分离，便于对训练型方法与纯推理方法复用同一接口。
    """

    def __init__(self, config_path: str):
        """初始化方法配置。

        参数:
            config_path: 方法配置文件路径（如 `configs/methods/grpo.yaml`）。
        """
        self.config = self._load_config(config_path)
        self.method_config = self.config["method"]

    def _load_config(self, config_path: str) -> dict:
        """读取并解析 YAML 配置。

        参数:
            config_path: 配置文件路径。

        返回:
            dict: 解析后的配置字典。
        """
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        """执行方法主流程（训练或推理）。"""
        pass

    @abstractmethod
    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        """在测试集上评估方法效果并返回指标。"""
        pass

    def get_method_name(self) -> str:
        """返回方法名称（来自配置中的 `method.name`）。"""
        return self.method_config["name"]

    def get_method_type(self) -> str:
        """返回方法类型（如 `inference` 或 `rl_training`）。"""
        return self.method_config["type"]
