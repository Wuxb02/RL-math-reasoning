"""
实验指标收集与报告模块。

提供两个核心功能:
    1. compute_metrics(): 计算预测结果的准确率和格式合规率
    2. ExperimentEvaluator: 聚合多个实验结果，生成对比表格和汇总报告

结果存储格式:
    {
        "Qwen3-0.6B": {
            "CoT": {"accuracy": 0.45, "format_compliance": 0.78, ...},
            "RLOO": {"accuracy": 0.58, "format_compliance": 0.89, ...},
            "GRPO": {"accuracy": 0.62, "format_compliance": 0.92, ...}
        },
        "Qwen3-1.7B": {...},
        "Qwen3-4B": {...}
    }
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def compute_metrics(predictions: List[Dict], expected: List[str]) -> Dict[str, float]:
    """
    计算预测结果的评估指标。

    比较模型预测与标准答案，计算准确率和格式合规率。

    Args:
        predictions: 预测结果列表，每个元素为字典，包含:
            - extracted_answer: 从模型输出中提取的答案
            - has_format: 是否包含正确的 XML 格式标签
        expected: 标准答案列表

    Returns:
        Dict[str, float]: 评估指标
            - accuracy: 答案正确率
            - format_compliance: 格式合规率
            - correct_count: 正确回答数量
            - total_count: 总样本数量
    """
    correct = 0
    format_correct = 0
    total = len(predictions)

    for pred, exp in zip(predictions, expected):
        if pred["extracted_answer"] == exp:
            correct += 1
        if pred["has_format"]:
            format_correct += 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "format_compliance": format_correct / total if total > 0 else 0,
        "correct_count": correct,
        "total_count": total,
    }


class ExperimentEvaluator:
    """
    实验结果聚合器与报告生成器。

    用于收集和管理多个实验（模型 × 方法组合）的评估结果，
    支持生成对比表格、保存/加载 JSON 结果、打印 ASCII 汇总表格。

    使用方式:
        >>> evaluator = ExperimentEvaluator()
        >>> evaluator.add_result("Qwen3-0.6B", "RLOO", {"accuracy": 0.58, ...})
        >>> evaluator.add_result("Qwen3-0.6B", "GRPO", {"accuracy": 0.62, ...})
        >>> evaluator.print_summary()  # 打印对比表格
        >>> evaluator.save_results()   # 保存到 JSON 文件
    """

    def __init__(self, output_dir: str = "outputs/results"):
        """
        初始化评估结果聚合器。

        Args:
            output_dir: 结果输出目录，默认 "outputs/results"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(self, model_name: str, method_name: str, metrics: Dict[str, float]):
        """
        添加单个实验的评估结果。

        结果按模型名和方法名两级索引存储。

        Args:
            model_name: 模型名称（如 "Qwen3-0.6B"）
            method_name: 方法名称（如 "RLOO"）
            metrics: 评估指标字典，通常包含 accuracy、format_compliance 等
        """
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][method_name] = metrics

    def get_comparison_table(self) -> Dict[str, Any]:
        """
        生成对比表格数据结构。

        将所有实验结果组织为二维表格格式，便于打印或导出。
        缺失的实验组合用 None 填充。

        Returns:
            Dict[str, Any]: 表格数据，包含:
                - models: 模型名称列表
                - methods: 方法名称列表（已排序）
                - data: 二维数据 {model: {method: metrics_or_None}}
        """
        table = {"models": [], "methods": [], "data": {}}

        # 收集所有方法名
        all_methods = set()
        for model_name, methods in self.results.items():
            table["models"].append(model_name)
            for method_name in methods:
                all_methods.add(method_name)

        table["methods"] = sorted(list(all_methods))

        # 填充数据（缺失的组合用 None 表示）
        for model_name in table["models"]:
            table["data"][model_name] = {}
            for method_name in table["methods"]:
                if method_name in self.results.get(model_name, {}):
                    table["data"][model_name][method_name] = self.results[model_name][
                        method_name
                    ]
                else:
                    table["data"][model_name][method_name] = None

        return table

    def save_results(self, filename: str = None):
        """
        将所有实验结果保存为 JSON 文件。

        Args:
            filename: 文件名（可选）。如果不指定，使用时间戳自动生成。

        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        return str(filepath)

    def load_results(self, filepath: str):
        """
        从 JSON 文件加载实验结果。

        Args:
            filepath: JSON 文件路径
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.results = json.load(f)

    def print_summary(self):
        """
        在终端打印 ASCII 格式的对比表格。

        表格格式:
            Model               | CoT                      | GRPO                     | RLOO
            --------------------|---------------------------------------------------------------------------
            Qwen3-0.6B          | Acc:45.2% Fmt:78.3%      | Acc:62.1% Fmt:92.5%      | Acc:58.7% Fmt:89.2%
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)

        table = self.get_comparison_table()

        # 打印表头
        header = f"{'Model':<20}"
        for method in table["methods"]:
            header += f"| {method:<25}"
        print(header)
        print("-" * len(header))

        # 打印每一行数据
        for model_name in table["models"]:
            row = f"{model_name:<20}"
            for method_name in table["methods"]:
                metrics = table["data"][model_name].get(method_name)
                if metrics:
                    acc = metrics.get("accuracy", 0)
                    fmt = metrics.get("format_compliance", 0)
                    row += f"| Acc:{acc:.1%} Fmt:{fmt:.1%}   "
                else:
                    row += f"| {'N/A':<25}"
            print(row)

        print("=" * 80)
