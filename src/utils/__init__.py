import json
import os
from pathlib import Path
from typing import Dict, List, Any
import math

from dotenv import load_dotenv


def load_env():
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint


load_env()


def generate_ascii_bar(value: float, max_width: int = 30) -> str:
    filled = int(value * max_width)
    return "█" * filled + "░" * (max_width - filled)


def generate_comparison_table(results: Dict[str, Dict[str, Dict]]) -> str:
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("EXPERIMENT RESULTS COMPARISON")
    lines.append("=" * 100)

    all_methods = set()
    for model_results in results.values():
        all_methods.update(model_results.keys())
    methods = sorted(list(all_methods))
    models = sorted(results.keys())

    header = f"{'Model':<20}"
    for method in methods:
        header += f"| {method:<30}"
    lines.append(header)
    lines.append("-" * len(header))

    for model in models:
        row = f"{model:<20}"
        for method in methods:
            if method in results.get(model, {}):
                metrics = results[model][method]
                acc = metrics.get("accuracy", 0)
                fmt = metrics.get("format_compliance", 0)
                row += f"| Acc:{acc:.1%} Fmt:{fmt:.1%}        "
            else:
                row += f"| {'N/A':<30}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def generate_accuracy_chart(results: Dict[str, Dict[str, Dict]]) -> str:
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("ACCURACY COMPARISON CHART")
    lines.append("=" * 60)

    all_methods = set()
    for model_results in results.values():
        all_methods.update(model_results.keys())
    methods = sorted(list(all_methods))
    models = sorted(results.keys())

    for model in models:
        lines.append(f"\n{model}:")
        for method in methods:
            if method in results.get(model, {}):
                acc = results[model][method].get("accuracy", 0)
                bar = generate_ascii_bar(acc)
                lines.append(f"  {method:<10} {bar} {acc:.1%}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def generate_analysis_report(results: Dict[str, Dict[str, Dict]]) -> str:
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("ANALYSIS REPORT")
    lines.append("=" * 80)

    lines.append("\n## Method Comparison (CoT vs PPO vs GRPO)")
    lines.append("-" * 40)

    method_accs = {}
    for model_results in results.values():
        for method, metrics in model_results.items():
            if method not in method_accs:
                method_accs[method] = []
            method_accs[method].append(metrics.get("accuracy", 0))

    for method, accs in method_accs.items():
        avg_acc = sum(accs) / len(accs) if accs else 0
        lines.append(f"  {method}: Average Accuracy = {avg_acc:.1%}")

    lines.append("\n## Model Size Comparison (0.6B vs 1.7B vs 4B)")
    lines.append("-" * 40)

    for model in sorted(results.keys()):
        model_results = results[model]
        if model_results:
            avg_acc = sum(m.get("accuracy", 0) for m in model_results.values()) / len(
                model_results
            )
            lines.append(f"  {model}: Average Accuracy = {avg_acc:.1%}")

    lines.append("\n## Key Findings")
    lines.append("-" * 40)

    best_model = None
    best_method = None
    best_acc = 0

    for model, model_results in results.items():
        for method, metrics in model_results.items():
            acc = metrics.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_method = method

    if best_model:
        lines.append(
            f"  Best Performance: {best_model} + {best_method} ({best_acc:.1%})"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def visualize_results(results_file: str, output_file: str = None):
    with open(results_file, "r") as f:
        results = json.load(f)

    report = []
    report.append(generate_comparison_table(results))
    report.append(generate_accuracy_chart(results))
    report.append(generate_analysis_report(results))

    full_report = "\n".join(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(full_report)
        print(f"Report saved to {output_file}")

    print(full_report)
    return full_report
