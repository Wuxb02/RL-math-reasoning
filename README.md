# GRPO-Math: 数学推理强化学习对比实验

对比研究 **CoT（思维链）**、**RLOO（Reinforce Leave-One-Out）** 和 **GRPO（组相对策略优化）** 三种方法在数学推理任务上的效果差异，并测试 **Qwen3 系列不同大小模型（0.6B、1.7B、4B）** 的能力边界。

## 快速开始

### 1. 安装依赖

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync

# 或直接运行脚本
uv run python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/cot.yaml
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 WandB API Key：

```bash
# Linux/macOS
cp .env.example .env

# Windows (CMD)
copy .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
# 编辑 .env 文件，填入 WANDB_API_KEY
```

### 3. 下载模型和数据

运行 Jupyter notebook 下载所有资源到 `./resources` 目录：

```bash
jupyter notebook download_models.ipynb
```

或使用国内镜像（在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com`）。

### 4. 运行实验

#### 批量运行（推荐）

```bash
# 使用 uv 运行（推荐，跨平台）
# 0.6B / 1.7B 单卡训练 + 4B 自动切换双卡
uv run python scripts/run_training.py
uv run python scripts/run_evaluation.py

# 4B 模型手动双卡训练（GPU0 训练 + GPU1 vLLM 推理）
./scripts/train_distributed.sh \
    --model configs/models/qwen3-4b.yaml \
    --method configs/methods/grpo.yaml

# 或使用完整 Python 路径（Windows）
"D:\anaconda\python.exe" "F:\RL-math-reasoning\scripts\run_training.py"
```

训练和评估完全分离，训练输出保存在 `outputs/checkpoints/`，评估结果保存在 `outputs/results/`。

#### 单个运行

```bash
# 单个训练（使用 uv）
uv run python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/rloo.yaml --wandb

# 单个评估（CoT 无需训练，直接评估）
uv run python scripts/evaluate.py --model configs/models/qwen3-0.6b.yaml --method CoT

# 单个评估（RLOO/GRPO 需指定 checkpoint）
uv run python scripts/evaluate.py --model configs/models/qwen3-0.6b.yaml --checkpoint outputs/checkpoints/Qwen3-0.6B-GRPO --method GRPO

# Windows 完整路径版本
"D:\anaconda\python.exe" "F:\RL-math-reasoning\scripts\train.py" --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml --wandb
```

### 5. 查看结果

```bash
# Linux/macOS
cat outputs/results/comparison_report.txt
cat outputs/results/evaluation_results.json

# Windows (CMD)
type outputs\results\comparison_report.txt
type outputs\results\evaluation_results.json

# Windows (PowerShell)
Get-Content outputs\results\comparison_report.txt
Get-Content outputs\results\evaluation_results.json
```

## Windows 环境使用

本项目在 Windows 环境下运行时，需要注意以下几点：

### 1. Python 路径配置

由于项目使用 `uv` 管理依赖，在 Windows 下可以直接使用 uv 运行脚本：

```bash
# 推荐：使用 uv run（自动处理依赖）
uv run python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml --wandb

# 或使用完整 Python 路径
"D:\anaconda\python.exe" "F:\RL-math-reasoning\scripts\train.py" --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml --wandb
```

### 2. 常用命令速查

```bash
# 训练命令（0.6B / 1.7B 单卡）
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml --wandb
CUDA_VISIBLE_DEVICES=1  uv run python scripts/train.py --model configs/models/qwen3-1.7b.yaml --method configs/methods/rloo.yaml --wandb

# 训练命令（4B 双卡，Linux/macOS）
./scripts/train_distributed.sh --model configs/models/qwen3-4b.yaml --method configs/methods/grpo.yaml --wandb

# 评估命令
uv run python scripts/evaluate.py --model configs/models/qwen3-0.6b.yaml --method CoT
uv run python scripts/evaluate.py --model configs/models/qwen3-0.6b.yaml --checkpoint outputs/checkpoints/Qwen3-0.6B-GRPO --method GRPO

# 批量训练（自动检测 4B 切换双卡）
uv run python scripts/run_training.py

# 批量评估
uv run python scripts/run_evaluation.py
```

### 3. Windows Path.join 注意事项

在 Windows 环境下，路径使用反斜杠 `\`，但 Python 代码中通常使用正斜杠 `/`（跨平台兼容）。以下是兼容写法：

```python
# 推荐：使用 pathlib（跨平台兼容）
from pathlib import Path
output_dir = Path("outputs") / "checkpoints" / "Qwen3-0.6B-GRPO"

# 或使用 os.path（自动处理路径分隔符）
import os
output_dir = os.path.join("outputs", "checkpoints", "Qwen3-0.6B-GRPO")
```

### 4. 常见问题

**Q: 运行脚本报错 "No module named 'xxx'"**
```bash
# 确保依赖已安装
uv sync
```

**Q: CUDA/MPS 不可用**
```bash
# 检查 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: WandB 报错**
```bash
# 确保 .env 文件中 WandB API Key 已正确配置
# 或者在环境变量中设置
set WANDB_API_KEY=your_api_key_here
```

## 项目结构

```
GRPO-math/
├── configs/
│   ├── models/              # 模型配置
│   │   ├── qwen3-0.6b.yaml
│   │   ├── qwen3-1.7b.yaml
│   │   └── qwen3-4b.yaml
│   └── methods/             # 方法配置
│       ├── cot.yaml
│       ├── rloo.yaml
│       └── grpo.yaml
├── src/
│   ├── data/                # GSM8K 数据加载
│   │   └── gsm8k.py
│   ├── models/              # 模型加载器
│   │   └── loader.py
│   ├── methods/             # 三种方法实现
│   │   ├── cot.py           # CoT 推理基线
│   │   ├── rloo.py          # RLOO 训练
│   │   └── grpo.py          # GRPO 训练
│   ├── rewards/             # 奖励函数
│   │   └── math_rewards.py
│   ├── evaluation/          # 评估框架
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/               # 工具函数
│       └── __init__.py      # load_env(), apply_lora(), 可视化工具
├── scripts/
│   ├── train.py             # 单个模型训练
│   ├── train_distributed.sh # 双卡训练（GPU0 训练 + GPU1 vLLM）
│   ├── evaluate.py          # 单个模型评估
│   ├── run_training.py      # 批量训练（自动检测 4B 切换双卡）
│   ├── run_evaluation.py    # 批量评估 + 自动对比
│   └── visualize.py         # 结果可视化
├── resources/               # 模型和数据（gitignored）
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   ├── Qwen3-4B/
│   └── gsm8k/
├── outputs/                 # 实验输出（gitignored）
│   ├── checkpoints/         # 训练后的模型
│   └── results/             # 评估结果
├── download_models.ipynb    # 模型下载脚本
├── .env                     # API 密钥（gitignored）
├── .env.example             # 环境变量模板
└── pyproject.toml           # 项目依赖
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                              │
│  ┌───────────────────────┐                                      │
│  │ download_models.ipynb │ → resources/Qwen3-*/                 │
│  │                       │ → resources/gsm8k/                   │
│  └───────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        训练阶段                                  │
│  ┌───────────────────────┐    ┌─────────────────────────────┐   │
│  │   run_training.py     │ OR │   train.py (单个, 0.6B/1.7B)│   │
│  │   (自动检测 4B 双卡)  │    │   train_distributed.sh (4B) │   │
│  └───────────────────────┘    └─────────────────────────────┘   │
│           ↓                                                     │
│  outputs/checkpoints/                                           │
│    ├── Qwen3-0.6B-RLOO/                                         │
│    ├── Qwen3-0.6B-GRPO/                                         │
│    ├── Qwen3-1.7B-RLOO/                                         │
│    ├── Qwen3-4B-RLOO/  (双卡: GPU0 训练 + GPU1 vLLM)           │
│    └── ...                                                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        评估阶段                                  │
│  ┌───────────────────────┐    ┌─────────────────────────────┐   │
│  │  run_evaluation.py    │ OR │   evaluate.py (单个)        │   │
│  │  (批量 + 自动对比)    │    │                             │   │
│  └───────────────────────┘    └─────────────────────────────┘   │
│           ↓                                                     │
│  outputs/results/                                               │
│    ├── evaluation_results.json                                  │
│    └── comparison_report.txt                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 三种方法对比

| 方法 | 类型 | 核心思想 | 显存占用 | 训练方式 |
|------|------|----------|----------|----------|
| **CoT** | 推理 | 提示词引导逐步推理 | 低 | 无需训练 |
| **RLOO** | 训练 | REINFORCE 风格优化 + Leave-One-Out 基线 | 低 (LoRA) | 无需价值模型 |
| **GRPO** | 训练 | 组内相对奖励归一化 | 低 (LoRA) | 无需价值模型 |

### CoT（Chain-of-Thought）

- 无需训练，通过精心设计的提示词引导模型推理
- 使用 XML 格式的思维链模板（zero-shot，不含 few-shot 示例）
- 评估时使用数值等价判断（`numeric_equivalence`），与 RLOO/GRPO 保持一致
- 适合作为基线对比

**损失函数**：

CoT 是纯推理方法，不涉及训练损失。评估时使用标准语言模型的交叉熵损失：

$$L_{\text{CoT}} = -\sum_{t=1}^{T} \log P(x_t | x_{\lt t}, \text{prompt})$$

实际实现中，我们通过精心设计的 System Prompt 引导模型生成结构化的推理过程，然后从 `<answer>` 标签中提取最终答案。

### RLOO（Reinforce Leave-One-Out）

- TRL 1.0.0 推荐的 PPO 替代方案，论文证明在 RLHF 场景下性能优于 PPO
- 使用 Leave-One-Out 基线估计优势函数，无需训练价值模型
- 训练稳定，显存占用比 PPO 降低约 33%

**损失函数**：

RLOO 的核心思想是使用组内其他样本的平均奖励作为基线，计算每个样本的优势值：

$$L^{\text{RLOO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right)$$

其中：

1. **Leave-One-Out 基线**：
$$b_i = \frac{1}{G-1} \sum_{j \neq i} r_j$$
- 对每个 completion $i$，使用组内其他所有样本的奖励均值作为基线
- 优势 $A_i = r_i - b_i$

2. **重要性采样比率**：
$$\rho_i = \frac{\pi_\theta(o_i|x)}{\pi_{\theta_{\text{old}}}(o_i|x)}$$

3. **KL 散度约束**：
$$\mathcal{L}_{\text{KL}} = \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
- 使用参考模型 $\pi_{\text{ref}}$ 约束策略偏离
- $\beta$ 控制 KL 惩罚强度

**RLOO vs PPO 的关键区别**：

| 特性 | PPO | RLOO |
|------|-----|------|
| 优势估计 | 需要价值模型 + GAE | Leave-One-Out 基线 |
| 模型数量 | 3个（策略+价值+参考） | 2个（策略+参考） |
| 显存占用 | 高 | 低 (LoRA) |
| 采样效率 | 每步1个响应 | 每步G个响应 |
| 适用场景 | 通用RL任务 | 有明确奖励信号的任务 |

### GRPO（Group Relative Policy Optimization）

- DeepSeek-R1 使用的方法
- 对同一提示的多个采样结果进行组内相对奖励归一化
- 无需价值模型，显存占用降低约 33%

**损失函数**：

GRPO 的核心创新是**组内相对奖励归一化**，无需训练价值函数：

$$L^{\text{GRPO}} = -\mathbb{E}_{x \sim D} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中：

1. **组内采样**：对于每个提示 $x$，从旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 个响应 $\{o_1, o_2, ..., o_G\}$

2. **相对奖励归一化**：
$$\hat{A}_i = \frac{R(x, o_i) - \text{mean}(\{R(x, o_j)\}_{j=1}^{G})}{\text{std}(\{R(x, o_j)\}_{j=1}^{G})}$$
- $R(x, o_i)$ 是第 $i$ 个响应的奖励（由奖励函数计算）
- 使用组内均值和标准差进行归一化，无需价值模型

3. **重要性采样比率**：
$$\rho_i = \frac{\pi_\theta(o_i|x)}{\pi_{\theta_{\text{old}}}(o_i|x)}$$

4. **KL 散度约束**：
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta} \left[ \log \frac{\pi_\theta(o|x)}{\pi_{\text{ref}}(o|x)} \right]$$
- 使用参考模型 $\pi_{\text{ref}}$（通常初始化为训练前的模型）
- $\beta$ 控制 KL 惩罚强度

**GRPO vs RLOO 的关键区别**：

| 特性 | RLOO | GRPO |
|------|------|------|
| 优势估计 | Leave-One-Out 基线 | 组内均值+标准差归一化 |
| 模型数量 | 2个（策略+参考） | 2个（策略+参考） |
| 显存占用 | 低 (LoRA) | 低 (LoRA) |
| 采样效率 | 每步G个响应 | 每步G个响应 |
| 适用场景 | 通用RL任务 | 有明确奖励信号的任务 |

## 模型规格

| 模型 | 参数量 | 显存(FP16) | 上下文长度 | 单卡训练显存 | 双卡训练显存 (GPU0/GPU1) |
|------|--------|-----------|-----------|-------------|------------------------|
| Qwen3-0.6B | 0.6B | 1.5GB | 32K | ~6GB | ~4GB / ~3GB |
| Qwen3-1.7B | 1.7B | 4GB | 32K | ~12GB | ~10GB / ~6GB |
| Qwen3-4B | 4B | 9GB | 128K | ~23GB | ~18GB / ~14GB |

> **注意**：所有训练方法（RLOO/GRPO）统一使用 LoRA 微调（r=16, alpha=32），以保证不同规模模型之间的对比公平性。双卡 RTX 3090 (24GB×2) 可安全运行全部模型。0.6B/1.7B 可单卡运行，4B 建议使用双卡模式。

## 双卡训练（2x RTX 3090）

对于 Qwen3-4B 模型，推荐使用双卡分布式训练以避免单卡显存不足：

### GPU 分配

| GPU | 角色 | 内容 |
|-----|------|------|
| **GPU 0** | 训练 | 策略模型 + 参考模型 + LoRA + 优化器 |
| **GPU 1** | 推理 | vLLM 服务器（生成候选回答） |

### 使用方式

```bash
# 单个 4B 训练
./scripts/train_distributed.sh \
    --model configs/models/qwen3-4b.yaml \
    --method configs/methods/grpo.yaml \
    --wandb

# 批量运行（自动检测 4B 并切换双卡）
uv run python scripts/run_training.py
```

### 优化配置

方法配置已针对双卡 3090 优化：

| 参数 | 原值 | 优化值 | 说明 |
|------|------|--------|------|
| `num_generations` | 4 | 4 | 提供更稳定的组内基线 |
| `max_completion_length` | 200 | 1024 | 更长的生成长度 |
| `gradient_accumulation_steps` | 4 | 2 | 减少同步开销 |
| `vllm_mode` | colocate | server | 隔离 vLLM 到独立 GPU |
| `save_steps` | 100 | 200 | 减少 checkpoint I/O |

另外启用了 **Flash Attention 2**（RTX 3090 支持），额外提供 1.2-1.4x 加速。

## LoRA 配置

本项目对所有 RL 训练方法（RLOO/GRPO）统一使用 LoRA 适配器，设计决策如下：

### 为什么统一使用 LoRA？

1. **硬件限制**：4B 模型全量微调需 ~74GB 显存，远超双卡 3090 的 48GB
2. **实验公平性**：统一训练方式后，对比结果仅反映模型规模差异，而非训练方式差异
3. **显存优化**：配合 `gradient_checkpointing`、`Flash Attention 2` 和 `device_map="auto"`，三模型均可在双卡 3090 上运行

### 配置参数

LoRA 配置在 `configs/methods/grpo.yaml` 和 `configs/methods/rloo.yaml` 中定义：

```yaml
lora:
  enabled: true          # 设为 false 可切换回全量微调（仅适用于 0.6B/1.7B）
  r: 16                  # LoRA 秩
  lora_alpha: 32         # 缩放系数（alpha = 2r）
  lora_dropout: 0.05     # Dropout 率
  target_modules:        # 目标层（覆盖所有注意力 + FFN 线性层）
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

### 显存对比

| 模型 | 全量微调 | LoRA (r=16) | 节省 |
|------|----------|-------------|------|
| 0.6B | ~13 GB | ~4 GB | 69% |
| 1.7B | ~30 GB | ~8 GB | 73% |
| 4B | ~74 GB | ~23 GB | 69% |

### 技术实现

- **`src/utils/__init__.py`** — `apply_lora()` 函数，GRPO/RLOO 共用
- **`src/models/loader.py`** — `device_map="auto"` 自动多卡分配 + `gradient_checkpointing_enable()`
- **`pyproject.toml`** — 依赖 `peft>=0.18.1`

## 奖励函数设计

本项目采用 **多维度奖励函数**，从正确性、格式规范、推理质量三个维度评估模型输出。

### 奖励函数总览

| 奖励函数 | 分值范围 | 主要作用 | 优先级 |
|----------|----------|----------|--------|
| 正确性奖励 | -1.0 ~ 2.0 | 答案正确性 | 最高 |
| 数字格式奖励 | -0.1 ~ 0.1 | 答案格式检测 | 中 |
| 严格格式奖励 | 0 ~ 0.5 | XML格式规范 | 高 |
| 宽松格式奖励 | 0 ~ 0.5 | 基本格式规范 | 中 |
| XML标签计数 | -0.5 ~ 0.5 | 标签完整性 | 中 |
| 推理质量奖励 | -0.15 ~ 0.4 | 推理过程质量 | 低 |

**总奖励范围**：-1.75 ~ 4.4（各奖励函数加权求和）

---

### 1. 正确性奖励 (correctness_reward_func)

**分值**：`2.0`（正确）| `-0.5`（错误）| `-1.0`（空答案）

**核心改进**：

```python
def numeric_equivalence(answer: str, expected: str) -> bool:
    # 支持多种数值格式等价判断
    # "0.5" == ".5" == "1/2" == "50%"
    # "42" == "42.0" == "42.00"
```

**支持的数值格式**：

| 输入格式 | 示例 | 解析结果 |
|----------|------|----------|
| 整数 | `"42"`, `"-42"` | 42.0, -42.0 |
| 小数 | `"3.14"`, `".5"`, `"-0.5"` | 3.14, 0.5, -0.5 |
| 分数 | `"1/2"`, `"-3/4"` | 0.5, -0.75 |
| 千分位 | `"1,234"` | 1234.0 |
| 百分比 | `"50%"`, `"25%"` | 0.5, 0.25 |

**评分逻辑**：

```python
if not extracted:
    reward = -1.0          # 空答案，严重扣分
elif numeric_equivalence(extracted, expected):
    reward = 2.0           # 完全正确，最高奖励
else:
    reward = -0.5          # 错误答案，小扣分
```

---

### 2. 数字格式奖励 (int_reward_func)

**分值**：`0.1`（可解析为数字）| `-0.1`（非数字格式）

**作用**：检测答案是否为有效的数值格式，辅助引导模型输出数字答案。

**评分逻辑**：

```python
num = parse_number(extracted_answer)
if num is not None:
    reward = 0.1   # 任意可解析数字（整数、小数、分数、百分比均可）
else:
    reward = -0.1  # 无法解析为数字
```

**示例**：

| 输入 | 可解析 | 奖励 |
|------|--------|------|
| `"42"` | ✓ | 0.1 |
| `"-42"` | ✓ | 0.1 |
| `"3.14"` | ✓ | 0.1 |
| `"1/2"` | ✓ | 0.1 |
| `"50%"` | ✓ | 0.1 |
| `"abc"` | ✗ | -0.1 |

> **设计说明**：此函数不区分整数/小数，仅判断否为数值格式。正确性验证由 `correctness_reward_func` 负责，两者分工明确，避免奖励信号干扰。

---

### 3. 严格格式奖励 (strict_format_reward_func)

**分值**：`0.5`（匹配）| `0.0`（不匹配）

**正则表达式**：
```
<reasoning>.*?</reasoning>\s*<answer>.*?</answer>
```

**要求格式**：
```
<reasoning>
[推理内容]
</reasoning>
<answer>
[答案内容]
</answer>
```

**注意事项**：
- 使用 `re.search` 匹配，允许前缀内容
- 标签间允许任意空白（换行、空格等）
- 只要包含正确的开闭标签即可得分

---

### 4. 宽松格式奖励 (soft_format_reward_func)

**分值**：`0.5`（包含标签）| `0.0`（不包含）

**实现方式**：只检查 `<reasoning>` 和 `<answer>` 标签是否存在，不要求严格匹配。

**改进点**：
- 在训练初期提供渐进式引导
- 即使 `<reasoning>` 未闭合也能获得部分奖励
- 允许前缀内容和任意空白

**示例**（以下都符合）：
```
<thinking>思考中...</thinking><reasoning>推理</reasoning><answer>42</answer>
<reasoning>推理</reasoning>  <answer>42</answer>
```

---

### 5. XML标签计数奖励 (xmlcount_reward_func)

**分值**：`-0.5` ~ `0.5`

**检查项**：

| 检查内容 | 奖励 |
|----------|------|
| `<reasoning>` | +0.125 |
| `</reasoning>` | +0.125 |
| `<answer>` | +0.125 |
| `</answer>` | +0.125 |

**实现说明**：
- 只检查标签是否存在，移除对换行的严格要求
- 简化逻辑，提高训练初期的奖励信号覆盖率

**惩罚机制**：
- `</answer>` 之后每多一个字符扣 0.001 分
- 鼓励简洁输出，避免冗余内容

**计算公式**：
```
reward = sum(标签奖励) - len(</answer>后内容) * 0.001
reward = max(reward, -0.5)  # 限制最低分
```

---

### 6. 推理质量奖励 (reasoning_quality_reward_func)

**分值**：`-0.15` ~ `0.4`

**提取逻辑**：允许 `<reasoning>` 标签未闭合，只要存在 `<reasoning>` 就提取其后的内容进行评估。

**评估维度**：

| 维度 | 检查内容 | 奖励 |
|------|----------|------|
| 步骤数量 | ≥3 步 | +0.1 |
| 步骤数量 | ≥5 步 | +0.1 |
| 数字计算 | 包含 `+`, `-`, `*`, `/` 运算 | +0.1 |
| 计算过程 | 包含 `=` 符号 | +0.05 |
| 长度适中 | 30-150 字符 | +0.05 |
| 太短 | <20 字符 | -0.1 |
| 太长 | >180 字符 | -0.05 |

**示例**（高质量推理）：
```
<reasoning>
Let me solve this step by step.
First, I need to find the total: 15 + 27 = 42
Then, I multiply by 3: 42 * 3 = 126
Finally, I divide by 2: 126 / 2 = 63
So the answer is 63.
</reasoning>
<answer>63</answer>
```

---

### 奖励组合策略

**GRPO 训练中的奖励函数顺序**：

```python
reward_funcs=[
    xmlcount_reward_func,           # 1. 先检查格式结构
    soft_format_reward_func,        # 2. 再检查宽松格式
    strict_format_reward_func,      # 3. 再检查严格格式
    int_reward_func,                # 4. 检查答案是否为数值格式
    reasoning_quality_reward_func,  # 5. 评估推理质量
    correctness_reward_func,        # 6. 最后检查正确性（权重最高）
]
```

**设计原则**：
1. **渐进式引导**：从宽松格式到严格格式，逐步规范输出
2. **多维度评估**：同时考虑格式、内容、质量
3. **负样本惩罚**：对错误答案和空答案给予负奖励
4. **稀疏奖励平衡**：正确性奖励权重最高，但其他奖励提供稳定梯度

---

### 数值等价性实现

```python
def parse_number(text: str) -> Optional[float]:
    """支持整数、小数、分数、千分位、百分比格式"""
    text = text.strip().replace(",", "")
    
    # 百分比: "50%" -> 0.5
    if text.endswith("%"):
        return float(text[:-1]) / 100
    
    # 分数: "1/2" -> 0.5
    if "/" in text:
        parts = text.split("/")
        return float(parts[0]) / float(parts[1])
    
    # 普通数字: "3.14" -> 3.14
    return float(text)

def numeric_equivalence(answer: str, expected: str) -> bool:
    """支持多种数值格式等价判断"""
    # 精确字符串匹配（最快）
    if answer.strip() == expected.strip():
        return True
    
    # 数值解析比较
    ans_num = parse_number(answer)
    exp_num = parse_number(expected)
    
    # 相对误差比较（处理浮点精度）
    return abs(ans_num - exp_num) / abs(exp_num) < 1e-9
```

**支持的等价形式**：

| 形式1 | 形式2 | 是否等价 |
|-------|-------|----------|
| `"42"` | `"42.0"` | ✓ |
| `"0.5"` | `".5"` | ✓ |
| `"0.5"` | `"1/2"` | ✓ |
| `"-3"` | `"-3.0"` | ✓ |
| `"1,234"` | `"1234"` | ✓ |
| `"50%"` | `"0.5"` | ✓ |
| `"42"` | `"43"` | ✗ |

## 对比报告示例

运行 `run_evaluation.py` 后自动生成对比报告：

```
================================================================================
EVALUATION COMPARISON REPORT
================================================================================

## 1. 方法对比 (CoT vs RLOO vs GRPO)
------------------------------------------------------------
方法             平均准确率        平均格式合规        最佳模型
------------------------------------------------------------
CoT              45.2%            78.3%            Qwen3-4B
GRPO             62.1%            92.5%            Qwen3-4B
RLOO             58.7%            89.2%            Qwen3-4B

## 2. 模型规模对比 (0.6B vs 1.7B vs 4B)
------------------------------------------------------------
模型                 平均准确率        最佳方法           最高准确率
------------------------------------------------------------
Qwen3-0.6B          42.3%            GRPO             58.1%
Qwen3-1.7B          55.8%            GRPO             71.3%
Qwen3-4B            72.4%            GRPO             84.6%

## 4. 关键发现
------------------------------------------------------------
  最佳组合: Qwen3-4B + GRPO (84.60%)
  最差组合: Qwen3-0.6B + CoT (32.10%)
```

## 参考资料

- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - GRPO 方法来源
- [TRL](https://github.com/huggingface/trl) - RLOO 和 GRPO 实现
- [RLOO Paper](https://huggingface.co/papers/2402.14740) - Back to Basics: Revisiting REINFORCE Style Optimization
- [GSM8K](https://github.com/openai/grade-school-math) - 数学推理基准
- [Qwen3](https://qwenlm.github.io/blog/qwen3/) - 模型架构详情
