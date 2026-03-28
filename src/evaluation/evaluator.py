import torch
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from tqdm import tqdm

from ..data.gsm8k import get_gsm8k_dataset
from ..models.loader import ModelLoader
from ..methods.cot import CoTMethod
from ..methods.ppo import PPOMethod
from ..methods.grpo import GRPOMethod
from .metrics import ExperimentEvaluator


class ExperimentEvaluator:
    def __init__(self, results_dir: str = "outputs/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ExperimentEvaluator(results_dir)

    def run_single_experiment(
        self, model_config: str, method_config: str, output_dir: str = None
    ) -> Dict[str, Any]:
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()

        model_name = model_loader.get_model_name()

        with open(method_config, "r") as f:
            method_cfg = yaml.safe_load(f)
        method_name = method_cfg["method"]["name"]

        if output_dir is None:
            output_dir = f"outputs/{model_name}-{method_name}"

        test_dataset = get_gsm8k_dataset(split="test")

        if method_name == "CoT":
            method = CoTMethod(method_config)
            results = method.run(model, tokenizer, test_dataset, output_dir)
        elif method_name == "PPO":
            train_dataset = get_gsm8k_dataset(split="train")
            method = PPOMethod(method_config)
            results = method.run(model, tokenizer, train_dataset, output_dir)
            results = method.evaluate(model, tokenizer, test_dataset)
        elif method_name == "GRPO":
            train_dataset = get_gsm8k_dataset(split="train")
            method = GRPOMethod(method_config)
            results = method.run(model, tokenizer, train_dataset, output_dir)
            results = method.evaluate(model, tokenizer, test_dataset)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        metrics = {
            "accuracy": results.get("accuracy", 0),
            "format_compliance": results.get("format_compliance", 0),
            "correct": results.get("correct", 0),
            "total": results.get("total", 0),
        }

        self.evaluator.add_result(model_name, method_name, metrics)

        return results

    def run_comparison(
        self,
        model_configs: list,
        method_configs: list,
        output_base_dir: str = "outputs",
    ):
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

        self.evaluator.save_results()
        self.evaluator.print_summary()
