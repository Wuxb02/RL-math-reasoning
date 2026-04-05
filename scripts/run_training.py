import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import itertools
import subprocess
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env

load_env()

import wandb
from src.data.gsm8k import get_gsm8k_dataset
from src.models.loader import ModelLoader
from src.methods.cot import CoTMethod
from src.methods.rloo import RLOOMethod
from src.methods.grpo import GRPOMethod


DEFAULT_MODELS = [
    "configs/models/qwen3-0.6b.yaml",
    "configs/models/qwen3-1.7b.yaml",
    "configs/models/qwen3-4b.yaml",
]

DEFAULT_METHODS = [
    "configs/methods/cot.yaml",
    "configs/methods/rloo.yaml",
    "configs/methods/rloo-0.6b.yaml",
    "configs/methods/rloo-1.7b.yaml",
    "configs/methods/grpo.yaml",
    "configs/methods/grpo-0.6b.yaml",
    "configs/methods/grpo-1.7b.yaml",
]


def _get_model_size(model_config: str) -> float:
    """解析模型参数量（单位：B）。"""
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = cfg["model"].get("params", "0B")
    return float(params.replace("B", ""))


def _needs_distributed(model_config: str) -> bool:
    """检查模型是否需要双卡分布式训练（4B 及以上）。"""
    return _get_model_size(model_config) >= 4.0


def _get_optimized_method_config(method_config: str, model_config: str) -> str:
    """根据模型大小自动选择对应的优化配置。

    如果传入的是通用配置（grpo.yaml / rloo.yaml），
    且模型是 0.6B 或 1.7B，则自动路由到对应的加速配置。
    """
    method_name = Path(method_config).stem
    param_value = _get_model_size(model_config)

    if method_name == "grpo":
        if param_value <= 0.6:
            return "configs/methods/grpo-0.6b.yaml"
        elif param_value <= 1.7:
            return "configs/methods/grpo-1.7b.yaml"
    elif method_name == "rloo":
        if param_value <= 0.6:
            return "configs/methods/rloo-0.6b.yaml"
        elif param_value <= 1.7:
            return "configs/methods/rloo-1.7b.yaml"

    return method_config


def _run_distributed(
    model_config: str, method_config: str, output_dir: str, wandb: bool
) -> dict:
    """通过 train_distributed.sh 启动双卡训练。"""
    cmd = [
        str(Path(__file__).parent / "train_distributed.sh"),
        "--model",
        model_config,
        "--method",
        method_config,
        "--output",
        output_dir,
    ]
    if wandb:
        cmd.append("--wandb")

    print(f"[分布式] 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        return {"status": "completed", "checkpoint_dir": output_dir}
    else:
        return {
            "status": "failed",
            "error": f"train_distributed.sh 退出码 {result.returncode}",
        }


def run_training(
    model_config: str, method_config: str, output_base: str, use_wandb: bool = False
) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Training: {Path(model_config).stem} + {Path(method_config).stem}")
    print(f"{'=' * 80}\n")

    try:
        model_loader = ModelLoader(model_config)
        model, tokenizer = model_loader.load_model_and_tokenizer()

        model_name = model_loader.get_model_name()

        with open(method_config, "r", encoding="utf-8") as f:
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

        if method_name == "RLOO":
            dataset = get_gsm8k_dataset(split="train")
            method = RLOOMethod(method_config)
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

        is_distributed = _needs_distributed(model_config)

        # 自动选择加速配置（小模型单卡优化）
        effective_method_config = _get_optimized_method_config(
            method_config, model_config
        )
        if effective_method_config != method_config:
            print(f"[优化] 自动使用加速配置: {Path(effective_method_config).name}")

        if is_distributed:
            print(f"[提示] 检测到 4B+ 模型，使用双卡分布式训练...")
            with open(effective_method_config, "r", encoding="utf-8") as f:
                method_cfg = yaml.safe_load(f)
            method_name = method_cfg["method"]["name"]
            model_loader = ModelLoader(model_config)
            model_name = model_loader.get_model_name()
            output_dir = f"{args.output}/{model_name}-{method_name}"

            if method_name == "CoT":
                result = {
                    "model": model_name,
                    "method": method_name,
                    "status": "skipped",
                    "reason": "CoT does not require training",
                }
            else:
                result = _run_distributed(
                    model_config, effective_method_config, output_dir, args.wandb
                )
                result["model"] = model_name
                result["method"] = method_name
        else:
            result = run_training(
                model_config, effective_method_config, args.output, args.wandb
            )
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
