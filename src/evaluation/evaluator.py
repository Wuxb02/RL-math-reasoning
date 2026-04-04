"""
实验评估编排模块。

负责运行单个实验（模型 + 方法的组合）以及批量对比实验。
自动化整个评估流程:
    1. 加载模型和分词器
    2. 根据方法名选择对应的实现（CoT/RLOO/GRPO）
    3. 执行训练（如需要）和评估
    4. 收集指标并记录

与 scripts/ 下的脚本不同，本模块提供编程式 API，
适合在 Jupyter Notebook 或其他自动化流程中使用。
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from tqdm import tqdm

from ..data.gsm8k import get_gsm8k_dataset
from ..models.loader import ModelLoader
from ..methods.cot import CoTMethod
from ..methods.rloo import RLOOMethod
from .metrics import ExperimentEvaluator


class ExperimentEvaluator:
    """
    实验评估编排器。

    封装完整的实验运行流程：加载模型 → 选择方法 → 训练/评估 → 收集指标。
    内部使用 metrics.ExperimentEvaluator 来聚合多个实验的结果。

    使用方式:
        >>> evaluator = ExperimentEvaluator()
        >>> result = evaluator.run_single_experiment(
        ...     "configs/models/qwen3-0.6b.yaml",
        ...     "configs/methods/rloo.yaml"
        ... )
        >>> evaluator.run_comparison(model_configs, method_configs)
    """

    def __init__(self, results_dir: str = "outputs/results"):
        """
        初始化评估器。

        Args:
            results_dir: 结果输出目录，默认 "outputs/results"
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ExperimentEvaluator(results_dir)

    def run_single_experiment(
        self, model_config: str, method_config: str, output_dir: str = None
    ) -> Dict[str, Any]:
        """
        运行单个实验（一个模型 + 一个方法的组合）。

        实验流程:
        1. 加载模型和分词器
        2. 根据方法名选择对应的实现
        3. 对于训练型方法（RLOO/GRPO）: 先训练再评估
        4. 对于推理型方法（CoT）: 直接评估
        5. 收集指标并记录

        Args:
            model_config: 模型配置文件路径
            method_config: 方法配置文件路径
            output_dir: 输出目录（可选，默认根据模型名和方法名自动生成）

        Returns:
            Dict[str, Any]: 实验结果，包含准确率和格式合规率等指标
        """
        # 加载模型
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()

        model_name = model_loader.get_model_name()

        # 读取方法配置
        with open(method_config, "r", encoding="utf-8") as f:
            method_cfg = yaml.safe_load(f)
        method_name = method_cfg["method"]["name"]

        if output_dir is None:
            output_dir = f"outputs/{model_name}-{method_name}"

        test_dataset = get_gsm8k_dataset(split="test")

        # 根据方法名选择对应的实现
        if method_name == "CoT":
            # CoT 是纯推理方法，无需训练
            method = CoTMethod(method_config)
            results = method.run(model, tokenizer, test_dataset, output_dir)
        elif method_name == "RLOO":
            # RLOO 需要先训练再评估
            train_dataset = get_gsm8k_dataset(split="train")
            method = RLOOMethod(method_config)
            results = method.run(model, tokenizer, train_dataset, output_dir)
            results = method.evaluate(model, tokenizer, test_dataset)
        elif method_name == "GRPO":
            # GRPO 需要先训练再评估
            from ..methods.grpo import GRPOMethod

            train_dataset = get_gsm8k_dataset(split="train")
            method = GRPOMethod(method_config)
            results = method.run(model, tokenizer, train_dataset, output_dir)
            results = method.evaluate(model, tokenizer, test_dataset)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # 提取关键指标
        metrics = {
            "accuracy": results.get("accuracy", 0),
            "format_compliance": results.get("format_compliance", 0),
            "correct": results.get("correct", 0),
            "total": results.get("total", 0),
        }

        # 记录到聚合器
        self.evaluator.add_result(model_name, method_name, metrics)

        return results

    def run_comparison(
        self,
        model_configs: list,
        method_configs: list,
        output_base_dir: str = "outputs",
    ):
        """
        运行批量对比实验（所有模型 × 方法的组合）。

        对每一对 (model_config, method_config) 调用 run_single_experiment，
        最后保存所有结果并打印汇总表格。

        Args:
            model_configs: 模型配置文件路径列表
            method_configs: 方法配置文件路径列表
            output_base_dir: 输出基础目录
        """
        for model_config in model_configs:
            for method_config in method_configs:
                print(f"\n{'=' * 60}")
                print(
                    f"Running: {Path(model_config).stem} + {Path(method_config).stem}"
                )
                print(f"{'=' * 60}\n")

                try:
                    self.run_single_experiment(
                        model_config,
                        method_config,
                        output_dir=f"{output_base_dir}/{Path(model_config).stem}-{Path(method_config).stem}",
                    )
                except Exception as e:
                    print(f"Error running experiment: {e}")
                    continue

        # 保存结果并打印汇总
        self.evaluator.save_results()
        self.evaluator.print_summary()
