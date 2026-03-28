import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def compute_metrics(predictions: List[Dict], expected: List[str]) -> Dict[str, float]:
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
    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(self, model_name: str, method_name: str, metrics: Dict[str, float]):
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][method_name] = metrics

    def get_comparison_table(self) -> Dict[str, Any]:
        table = {"models": [], "methods": [], "data": {}}

        all_methods = set()
        for model_name, methods in self.results.items():
            table["models"].append(model_name)
            for method_name in methods:
                all_methods.add(method_name)

        table["methods"] = sorted(list(all_methods))

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
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        return str(filepath)

    def load_results(self, filepath: str):
        with open(filepath, "r") as f:
            self.results = json.load(f)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 80)

        table = self.get_comparison_table()

        header = f"{'Model':<20}"
        for method in table["methods"]:
            header += f"| {method:<25}"
        print(header)
        print("-" * len(header))

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
