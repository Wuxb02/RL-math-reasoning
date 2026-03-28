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


def run_training(model_config: str, method_config: str, output_base: str) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Training: {Path(model_config).stem} + {Path(method_config).stem}")
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
            return {
                "model": model_name,
                "method": method_name,
                "status": "skipped",
                "reason": "CoT does not require training",
            }

        if method_name == "PPO":
            dataset = get_gsm8k_dataset(split="train")
            method = PPOMethod(method_config)
            results = method.run(model, tokenizer, dataset, output_dir)
        elif method_name == "GRPO":
            dataset = get_gsm8k_dataset(split="train")
            method = GRPOMethod(method_config)
            results = method.run(model, tokenizer, dataset, output_dir)
        else:
            return {"error": f"Unknown method: {method_name}"}

        return {
            "model": model_name,
            "method": method_name,
            "checkpoint_dir": output_dir,
            "status": "completed",
        }

    except Exception as e:
        print(f"Error in training: {e}")
        return {
            "model": Path(model_config).stem,
            "method": Path(method_config).stem,
            "error": str(e),
            "status": "failed",
        }


def main():
    parser = argparse.ArgumentParser(description="Train all model+method combinations")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, help="Model config files"
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS, help="Method config files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/checkpoints",
        help="Output base directory",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project="grpo-math-comparison",
            name=f"training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    all_results = []

    experiments = list(itertools.product(args.models, args.methods))

    print(f"\nRunning {len(experiments)} training experiments:")
    for model_config, method_config in experiments:
        print(f"  - {Path(model_config).stem} + {Path(method_config).stem}")

    for i, (model_config, method_config) in enumerate(experiments, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# Training {i}/{len(experiments)}")
        print(f"{'#' * 80}")

        result = run_training(model_config, method_config, args.output)
        all_results.append(result)

        results_file = Path(args.output) / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    completed = [r for r in all_results if r["status"] == "completed"]
    skipped = [r for r in all_results if r["status"] == "skipped"]
    failed = [r for r in all_results if r["status"] == "failed"]

    print("\n\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Completed: {len(completed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed: {len(failed)}")

    if completed:
        print("\nCompleted checkpoints:")
        for r in completed:
            print(f"  - {r['model']} + {r['method']}: {r['checkpoint_dir']}")

    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  - {r['model']} + {r['method']}: {r.get('error', 'Unknown')}")

    results_file = Path(args.output) / "training_results.json"
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
