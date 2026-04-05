# AGENTS.md — GRPO-Math Codebase Guide

## Project Overview

This project compares **CoT**, **PPO**, and **GRPO** reinforcement learning methods for math reasoning on the GSM8K benchmark using Qwen3 models (0.6B, 1.7B, 4B). It is a Python ML research project managed with `uv`.

---

## Build & Environment

**Package manager**: `uv` (not pip directly)

```bash
# Install dependencies and create virtualenv
uv sync

# Run any script via uv
uv run python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml

# Or activate the venv directly
source .venv/bin/activate
```

**Python version**: 3.12+ (see `.python-version`)

**Environment setup** (required before running):
```bash
cp .env.example .env
# Fill in WANDB_API_KEY in .env
# Optionally set HF_ENDPOINT=https://hf-mirror.com for China mirrors
```

---

## Key Commands

### Training
```bash
# Single training run (single GPU, colocate vLLM)
python scripts/train.py \
    --model configs/models/qwen3-0.6b.yaml \
    --method configs/methods/grpo.yaml \
    --wandb

# Distributed training (2 GPUs: GPU0 training + GPU1 vLLM server)
# Recommended for Qwen3-4B on 2x RTX 3090 (24GB × 2)
./scripts/train_distributed.sh \
    --model configs/models/qwen3-4b.yaml \
    --method configs/methods/grpo.yaml \
    --wandb

# Batch train all 3×3=9 combinations
python scripts/run_training.py
```

### Evaluation
```bash
# Evaluate CoT (no checkpoint needed)
python scripts/evaluate.py \
    --model configs/models/qwen3-0.6b.yaml \
    --method CoT

# Evaluate PPO/GRPO (checkpoint required)
python scripts/evaluate.py \
    --model configs/models/qwen3-0.6b.yaml \
    --checkpoint outputs/checkpoints/Qwen3-0.6B-GRPO \
    --method GRPO

# Batch evaluate all + generate comparison report
python scripts/run_evaluation.py
```

### Linting & Formatting
```bash
# Ruff is the linter/formatter (ruff cache present at .ruff_cache/)
uv run ruff check .
uv run ruff format .

# Check without fixing
uv run ruff check --no-fix .
```

### No Test Suite
There are no automated tests in this codebase. Validation is done by running the evaluation scripts and inspecting `outputs/results/`.

---

## Project Structure

```
GRPO-math/
├── configs/
│   ├── models/          # Model configs: qwen3-0.6b.yaml, qwen3-1.7b.yaml, qwen3-4b.yaml
│   └── methods/         # Method configs: cot.yaml, ppo.yaml, grpo.yaml
├── src/
│   ├── data/gsm8k.py    # GSM8K dataset loading + prompt formatting
│   ├── models/loader.py # ModelLoader: loads HF models with dtype/device detection
│   ├── methods/
│   │   ├── base.py      # BaseMethod ABC (run + evaluate abstract methods)
│   │   ├── cot.py       # CoT inference (no training)
│   │   ├── ppo.py       # PPO training via TRL
│   │   └── grpo.py      # GRPO training via TRL GRPOTrainer
│   ├── rewards/
│   │   └── math_rewards.py  # All reward functions (6 total)
│   ├── evaluation/
│   │   ├── evaluator.py     # ExperimentEvaluator orchestrates runs
│   │   └── metrics.py       # Metrics collection and reporting
│   └── utils/__init__.py    # load_env(), ASCII chart utilities
├── scripts/
│   ├── train.py         # CLI: single training run
│   ├── evaluate.py      # CLI: single evaluation
│   ├── run_training.py  # Batch training (all combos)
│   ├── run_evaluation.py# Batch evaluation + comparison report
│   └── visualize.py     # Result visualization
├── resources/           # gitignored: downloaded models + gsm8k data
├── outputs/             # gitignored: checkpoints + results
├── pyproject.toml
└── .env                 # gitignored: secrets
```

---

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local — no blank line separation enforced but consistent grouping is used
- Local imports use relative paths within `src/` (e.g., `from .base import BaseMethod`, `from ..rewards.math_rewards import ...`)
- Scripts use `sys.path.insert(0, ...)` to add project root before importing `src.*`

```python
# Standard library
import re
from typing import Dict, Any, Optional
from pathlib import Path

# Third-party
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Local (relative imports within src/)
from .base import BaseMethod
from ..rewards.math_rewards import correctness_reward_func
```

### Type Hints
- All function signatures use type hints
- Return types always annotated
- Use `typing` module: `Dict`, `Any`, `Optional`, `Tuple`, `List`
- `Optional[T]` used for nullable values (not `T | None`)

```python
def load_model_and_tokenizer(
    self, dtype: Optional[str] = None, device_map: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
```

### Docstrings
- Single-line or multi-line docstrings in Chinese for internal module documentation
- Describe the function's purpose, key parameters, and improvement notes
- No formal Sphinx/Google-style format enforced

```python
def parse_number(text: str) -> Optional[float]:
    """
    尝试将文本解析为数值，支持多种格式：
    - 整数: "42", "-42"
    - 小数: "3.14", ".5", "-0.5"
    """
```

### Naming Conventions
- **Classes**: `PascalCase` — `GRPOMethod`, `ModelLoader`, `GSM8KDataset`
- **Functions/methods**: `snake_case` — `load_model_and_tokenizer`, `extract_xml_answer`
- **Reward functions**: suffix `_reward_func` — `correctness_reward_func`, `xmlcount_reward_func`
- **Constants/module-level strings**: `UPPER_SNAKE_CASE` — `SYSTEM_PROMPT`, `XML_COT_FORMAT`
- **Config keys**: `snake_case` strings matching YAML keys exactly
- **Private methods**: single leading underscore — `_load_config`, `_detect_device`

### Class Structure
- All method implementations inherit from `BaseMethod` ABC
- Abstract methods: `run(model, tokenizer, dataset, output_dir)` and `evaluate(model, tokenizer, test_dataset)`
- Config loaded from YAML in `__init__`, stored as `self.config`

```python
class GRPOMethod(BaseMethod):
    def __init__(self, config_path: str = "configs/methods/grpo.yaml"):
        super().__init__(config_path)
        self.training_config = self.config["training"]
```

### Error Handling
- No custom exception classes; use built-in exceptions (`ValueError`, `ZeroDivisionError`)
- Numeric parsing uses `try/except ValueError` and returns `None` on failure
- Script entry points use `sys.exit(1)` for unknown method names
- Training loops wrap individual experiments in `try/except Exception as e: print(...); continue`

```python
try:
    return float(text)
except ValueError:
    return None
```

### Configuration Pattern
- All hyperparameters in YAML configs under `configs/models/` and `configs/methods/`
- Config accessed via `self.config["section"]["key"]` with `.get()` for optional values
- Never hardcode model names or hyperparameters in Python — always read from YAML

### Reward Functions Signature
All reward functions follow TRL's interface: positional args `(completions, ...)` or `(prompts, completions, answer, ...)`, return `List[float]`:

```python
def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [completion[0]["content"] for completion in completions]
    ...
    return rewards
```

### Path Handling
- Use `pathlib.Path` for all file path operations
- `Path(__file__).parent.parent` pattern to find project root from within `src/`
- Outputs always under `outputs/` directory (auto-created with `mkdir(parents=True, exist_ok=True)`)

### Device Detection
- Always auto-detect: CUDA → MPS → CPU via `torch.cuda.is_available()` and `torch.backends.mps.is_available()`
- Use `bfloat16` by default for training (set in YAML: `bf16: true`)

---

## Key Domain Concepts

- **GSM8K format**: Questions answered with `####` delimiter; answers extracted via `extract_hash_answer()`
- **Output format**: All model responses must follow XML structure:
  ```
  <reasoning>
  [step-by-step reasoning]
  </reasoning>
  <answer>
  [numeric answer]
  </answer>
  ```
- **Reward functions**: 6 functions combined — xmlcount, soft_format, strict_format, int_reward, reasoning_quality, correctness (in that order for GRPO)
- **Numeric equivalence**: `"0.5" == ".5" == "1/2" == "50%"` — always use `numeric_equivalence()` not string comparison for answer checking
- **WandB integration**: Pass `--wandb` flag to training scripts; `WANDB_API_KEY` required in `.env`

---

## Dependencies (pyproject.toml)

Core ML: `torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets`  
Utilities: `wandb`, `pyyaml`, `tqdm`, `python-dotenv`, `huggingface-hub`, `modelscope`

---

## Performance Optimization

### Dual-GPU Training (2x RTX 3090)

For training Qwen3-4B on two 24GB GPUs, use the distributed script:

```bash
./scripts/train_distributed.sh \
    --model configs/models/qwen3-4b.yaml \
    --method configs/methods/grpo.yaml
```

GPU allocation:
- **GPU 0**: Training (policy model + reference model + LoRA + optimizer)
- **GPU 1**: vLLM inference server (generation)

### Optimized Hyperparameters

The method configs have been tuned for 2x 3090:
- `num_generations: 2` (was 4) — halves generation overhead
- `max_completion_length: 128` (was 200) — covers 95%+ of GSM8K samples
- `gradient_accumulation_steps: 2` (was 4) — reduces sync overhead
- `vllm_mode: "server"` — isolates vLLM to dedicated GPU
- Flash Attention 2 enabled in model loader (RTX 3090 supported)
