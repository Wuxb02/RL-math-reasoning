import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_env

load_env()

from src.data.gsm8k import get_gsm8k_dataset
from src.models.loader import ModelLoader
from src.methods.cot import CoTMethod
from src.methods.ppo import PPOMethod
from src.methods.grpo import GRPOMethod


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--method", type=str, required=True, help="Method name: CoT, PPO, or GRPO"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    model_loader = ModelLoader(args.model)

    if args.checkpoint:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        model, tokenizer = model_loader.load_model_and_tokenizer()

    model_name = model_loader.get_model_name()
    test_dataset = get_gsm8k_dataset(split="test")

    if args.method == "CoT":
        method = CoTMethod()
    elif args.method == "PPO":
        method = PPOMethod()
    elif args.method == "GRPO":
        method = GRPOMethod()
    else:
        print(f"Unknown method: {args.method}")
        sys.exit(1)

    results = method.evaluate(model, tokenizer, test_dataset)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{model_name}_{args.method}_eval.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation Results for {model_name} + {args.method}:")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Format Compliance: {results['format_compliance']:.2%}")
    print(f"  Correct: {results['correct']}/{results['total']}")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
