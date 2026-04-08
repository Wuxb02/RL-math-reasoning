import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env

load_env()

import wandb
from src.data.gsm8k import get_gsm8k_dataset
from src.models.loader import ModelLoader
from src.methods.cot import CoTMethod
from src.methods.rloo import RLOOMethod


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--method", type=str, required=True, help="Path to method config YAML"
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="grpo-math-comparison")

    model_loader = ModelLoader(args.model)
    model, tokenizer = model_loader.load_model_and_tokenizer()

    model_name = model_loader.get_model_name()

    import yaml

    with open(args.method, "r", encoding="utf-8") as f:
        method_cfg = yaml.safe_load(f)
    method_name = method_cfg["method"]["name"]

    if args.output is None:
        args.output = f"outputs/{model_name}-{method_name}"

    if method_name == "CoT":
        dataset = get_gsm8k_dataset(split="test")
        method = CoTMethod(args.method)
        results = method.run(model, tokenizer, dataset, args.output)
    elif method_name == "RLOO":
        dataset = get_gsm8k_dataset(split="train")
        method = RLOOMethod(args.method)
        # 1. 执行训练
        results = method.run(model, tokenizer, dataset, args.output)
        
        # 2. 唤醒评估：加载测试集并调用 evaluate
        print("\nStarting evaluation on test set...")
        test_dataset = get_gsm8k_dataset(split="test")
        eval_results = method.evaluate(model, tokenizer, test_dataset)
        
        # 3. 将评估的指标合并到 results 字典中
        results.update(eval_results)

    elif method_name == "GRPO":
        from src.methods.grpo import GRPOMethod

        dataset = get_gsm8k_dataset(split="train")
        method = GRPOMethod(args.method)
        # 1. 执行训练
        results = method.run(model, tokenizer, dataset, args.output)
        
        # 2. 唤醒评估：加载测试集并调用 evaluate
        print("\nStarting evaluation on test set...")
        test_dataset = get_gsm8k_dataset(split="test")
        eval_results = method.evaluate(model, tokenizer, test_dataset)
        
        # 3. 将评估的指标合并到 results 字典中
        results.update(eval_results)
    else:
        print(f"Unknown method: {method_name}")
        sys.exit(1)

    print(f"\nTraining completed. Results saved to {args.output}")
    if "accuracy" in results:
        print(f"Accuracy: {results['accuracy']:.2%}")
    if "format_compliance" in results:
        print(f"Format Compliance: {results['format_compliance']:.2%}")

    if args.wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
