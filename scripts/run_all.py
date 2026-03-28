import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import itertools

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env

load_env()

import wandb
from src.data.gsm8k import get_gsm8k_dataset
from src.models.loader import ModelLoader
from src.methods.cot import CoTMethod
from src.methods.ppo import PPOMethod
from src.methods.grpo import GRPOMethod


DEFAULT_MODELS = [
    "configs/models/qwen3-0.6b.yaml",
    "configs/models/qwen3-1.7b.yaml",
    "configs/models/qwen3-4b.yaml",
]

DEFAULT_METHODS = [
    "configs/methods/cot.yaml",
    "configs/methods/ppo.yaml",
    "configs/methods/grpo.yaml",
]


def run_experiment(model_config: str, method_config: str, output_base: str) -> dict:
    print(f"\n{'=' * 80}")
    print(
        f"Starting experiment: {Path(model_config).stem} + {Path(method_config).stem}"
    )
    print(f"{'=' * 80}\n")

    try:
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()

        model_name = model_loader.get_model_name()

        import yaml

        with open(method_config, "r") as f:
            method_cfg = yaml.safe_load(f)
        method_name = method_cfg["method"]["name"]

        output_dir = f"{output_base}/{model_name}-{method_name}"

        if method_name == "CoT":
            dataset = get_gsm8k_dataset(split="test")
            method = CoTMethod(method_config)
            results = method.run(model, tokenizer, dataset, output_dir)
        elif method_name == "PPO":
            dataset = get_gsm8k_dataset(split="train")
            method = PPOMethod(method_config)
            results = method.run(model, tokenizer, dataset, output_dir)
            test_dataset = get_gsm8k_dataset(split="test")
            results = method.evaluate(model, tokenizer, test_dataset)
        elif method_name == "GRPO":
            dataset = get_gsm8k_dataset(split="train")
            method = GRPOMethod(method_config)
            results = method.run(model, tokenizer, dataset, output_dir)
            test_dataset = get_gsm8k_dataset(split="test")
            results = method.evaluate(model, tokenizer, test_dataset)
        else:
            return {"error": f"Unknown method: {method_name}"}

        return {
            "model": model_name,
            "method": method_name,
            "accuracy": results.get("accuracy", 0),
            "format_compliance": results.get("format_compliance", 0),
            "correct": results.get("correct", 0),
            "total": results.get("total", 0),
            "output_dir": output_dir,
            "status": "completed",
        }

    except Exception as e:
        print(f"Error in experiment: {e}")
        return {
            "model": Path(model_config).stem,
            "method": Path(method_config).stem,
            "error": str(e),
            "status": "failed",
        }


def generate_comparison_report(results: list) -> str:
    completed = [r for r in results if r.get("status") == "completed"]

    if not completed:
        return "No completed experiments to compare."

    by_model = {}
    by_method = {}

    for r in completed:
        model = r["model"]
        method = r["method"]

        if model not in by_model:
            by_model[model] = {}
        by_model[model][method] = r

        if method not in by_method:
            by_method[method] = {}
        by_method[method][model] = r

    report = []
    report.append("\n" + "=" * 80)
    report.append("EXPERIMENT COMPARISON REPORT")
    report.append("=" * 80)

    report.append("\n## 1. 方法对比 (CoT vs PPO vs GRPO)")
    report.append("-" * 60)
    report.append(
        f"{'方法':<15} {'平均准确率':<15} {'平均格式合规':<15} {'最佳模型':<20}"
    )
    report.append("-" * 60)

    for method, models in sorted(by_method.items()):
        accs = [r["accuracy"] for r in models.values()]
        fmts = [r["format_compliance"] for r in models.values()]
        avg_acc = sum(accs) / len(accs) if accs else 0
        avg_fmt = sum(fmts) / len(fmts) if fmts else 0
        best_model = max(models.items(), key=lambda x: x[1]["accuracy"])[0]
        report.append(
            f"{method:<15} {avg_acc:<15.2%} {avg_fmt:<15.2%} {best_model:<20}"
        )

    report.append("\n## 2. 模型规模对比 (0.6B vs 1.7B vs 4B)")
    report.append("-" * 60)
    report.append(
        f"{'模型':<20} {'平均准确率':<15} {'最佳方法':<15} {'最高准确率':<15}"
    )
    report.append("-" * 60)

    for model, methods in sorted(by_model.items()):
        accs = [r["accuracy"] for r in methods.values()]
        avg_acc = sum(accs) / len(accs) if accs else 0
        best_method = max(methods.items(), key=lambda x: x[1]["accuracy"])[0]
        best_acc = max(accs) if accs else 0
        report.append(
            f"{model:<20} {avg_acc:<15.2%} {best_method:<15} {best_acc:<15.2%}"
        )

    report.append("\n## 3. 完整结果矩阵")
    report.append("-" * 80)

    methods = sorted(by_method.keys())
    header = f"{'模型':<20}"
    for method in methods:
        header += f"| {method:<18}"
    report.append(header)
    report.append("-" * len(header))

    for model in sorted(by_model.keys()):
        row = f"{model:<20}"
        for method in methods:
            if method in by_model[model]:
                acc = by_model[model][method]["accuracy"]
                fmt = by_model[model][method]["format_compliance"]
                row += f"| {acc:.1%} ({fmt:.0%})   "
            else:
                row += f"| {'N/A':<18}"
        report.append(row)

    report.append("\n## 4. 关键发现")
    report.append("-" * 60)

    best_overall = max(completed, key=lambda x: x["accuracy"])
    report.append(
        f"  最佳组合: {best_overall['model']} + {best_overall['method']} ({best_overall['accuracy']:.2%})"
    )

    worst_overall = min(completed, key=lambda x: x["accuracy"])
    report.append(
        f"  最差组合: {worst_overall['model']} + {worst_overall['method']} ({worst_overall['accuracy']:.2%})"
    )

    report.append("\n  方法效果排名 (按平均准确率):")
    method_avgs = []
    for method, models in by_method.items():
        avg = sum(r["accuracy"] for r in models.values()) / len(models)
        method_avgs.append((method, avg))
    method_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (method, avg) in enumerate(method_avgs, 1):
        report.append(f"    {i}. {method}: {avg:.2%}")

    report.append("\n  模型规模效果排名 (按平均准确率):")
    model_avgs = []
    for model, methods in by_model.items():
        avg = sum(r["accuracy"] for r in methods.values()) / len(methods)
        model_avgs.append((model, avg))
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (model, avg) in enumerate(model_avgs, 1):
        report.append(f"    {i}. {model}: {avg:.2%}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, help="Model config files"
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS, help="Method config files"
    )
    parser.add_argument(
        "--output", type=str, default="outputs", help="Output base directory"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip if output directory exists"
    )

    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="grpo-math-comparison",
            name=f"batch-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    all_results = []

    experiments = list(itertools.product(args.models, args.methods))

    print(f"\nRunning {len(experiments)} experiments:")
    for model_config, method_config in experiments:
        print(f"  - {Path(model_config).stem} + {Path(method_config).stem}")

    for i, (model_config, method_config) in enumerate(experiments, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# Experiment {i}/{len(experiments)}")
        print(f"{'#' * 80}")

        result = run_experiment(model_config, method_config, args.output)
        all_results.append(result)

        results_file = Path(args.output) / "all_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    comparison_report = generate_comparison_report(all_results)
    print(comparison_report)

    report_file = Path(args.output) / "comparison_report.txt"
    with open(report_file, "w") as f:
        f.write(comparison_report)
    print(f"\nComparison report saved to: {report_file}")

    results_file = Path(args.output) / "all_results.json"
    print(f"All results saved to: {results_file}")


if __name__ == "__main__":
    main()
