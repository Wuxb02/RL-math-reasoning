# GRPO-Math: 数学推理强化学习对比实验

对比研究 **CoT（思维链）**、**PPO（近端策略优化）** 和 **GRPO（组相对策略优化）** 三种方法在数学推理任务上的效果差异，并测试 **Qwen3 系列不同大小模型（0.6B、1.7B、4B）** 的能力边界。

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
cp .env.example .env
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
# 第一步：批量训练所有 3×3=9 个组合
python scripts/run_training.py

# 第二步：批量评估所有训练后的模型
python scripts/run_evaluation.py
```

训练和评估完全分离，训练输出保存在 `outputs/checkpoints/`，评估结果保存在 `outputs/results/`。

#### 单个运行

```bash
# 单个训练
python scripts/train.py \
    --model configs/models/qwen3-0.6b.yaml \
    --method configs/methods/grpo.yaml \
    --wandb

# 单个评估（CoT 无需训练，直接评估）
python scripts/evaluate.py \
    --model configs/models/qwen3-0.6b.yaml \
    --method CoT

# 单个评估（PPO/GRPO 需指定 checkpoint）
python scripts/evaluate.py \
    --model configs/models/qwen3-0.6b.yaml \
    --checkpoint outputs/checkpoints/Qwen3-0.6B-GRPO \
    --method GRPO
```

### 5. 查看结果

```bash
# 查看评估对比报告
cat outputs/results/comparison_report.txt

# 查看原始结果数据
cat outputs/results/evaluation_results.json
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
│       ├── ppo.yaml
│       └── grpo.yaml
├── src/
│   ├── data/                # GSM8K 数据加载
│   │   └── gsm8k.py
│   ├── models/              # 模型加载器
│   │   └── loader.py
│   ├── methods/             # 三种方法实现
│   │   ├── cot.py           # CoT 推理基线
│   │   ├── ppo.py           # PPO 训练
│   │   └── grpo.py          # GRPO 训练
│   ├── rewards/             # 奖励函数
│   │   └── math_rewards.py
│   ├── evaluation/          # 评估框架
│   │   ├── evaluator.py
│   │   └── metrics.py
│   └── utils/               # 工具函数
│       └── __init__.py
├── scripts/
│   ├── train.py             # 单个模型训练
│   ├── evaluate.py          # 单个模型评估
│   ├── run_training.py      # 批量训练（所有组合）
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
│  │   run_training.py     │ OR │   train.py (单个)           │   │
│  │   (批量 3×3=9 组合)   │    │                             │   │
│  └───────────────────────┘    └─────────────────────────────┘   │
│           ↓                                                     │
│  outputs/checkpoints/                                           │
│    ├── Qwen3-0.6B-PPO/                                          │
│    ├── Qwen3-0.6B-GRPO/                                         │
│    ├── Qwen3-1.7B-PPO/                                          │
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
| **PPO** | 训练 | 策略梯度 + 价值函数 + KL惩罚 | 高 | 需要价值模型 |
| **GRPO** | 训练 | 组内相对奖励归一化 | 中 | 无需价值模型 |

### CoT（Chain-of-Thought）

- 无需训练，通过精心设计的提示词引导模型推理
- 使用 XML 格式的思维链模板
- 适合作为基线对比

**损失函数**：

CoT 是纯推理方法，不涉及训练损失。评估时使用标准语言模型的交叉熵损失：

$$L_{\text{CoT}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}, \text{prompt})$$

实际实现中，我们通过精心设计的 System Prompt 引导模型生成结构化的推理过程，然后从 `<answer>` 标签中提取最终答案。

### PPO（Proximal Policy Optimization）

- 经典的强化学习算法，使用裁剪目标函数
- 需要策略模型、价值模型和参考模型
- 训练稳定但显存占用高

**损失函数**：

PPO 的目标函数由三部分组成：

$$L^{\text{PPO}} = L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 L^{\text{Entropy}}$$

其中：

1. **策略损失（Clipped Surrogate Objective）**：
$$L^{\text{CLIP}} = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比率
- $A_t$ 是优势函数，由 GAE（Generalized Advantage Estimation）计算
- $\epsilon$ 是裁剪参数（通常为 0.2）

2. **价值函数损失**：
$$L^{\text{VF}} = \mathbb{E}_t \left[ (V_\theta(s_t) - V_{\text{target}})^2 \right]$$
- 需要训练额外的价值模型来估计状态价值

3. **熵正则化**：
$$L^{\text{Entropy}} = \mathbb{E}_t \left[ \pi_\theta(a_t|s_t) \log \pi_\theta(a_t|s_t) \right]$$
- 鼓励探索，防止过早收敛

**KL 惩罚**：PPO 还可以添加 KL 散度惩罚来约束策略不要偏离参考模型太远：
$$L_{\text{KL}} = \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

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

**GRPO vs PPO 的关键区别**：

| 特性 | PPO | GRPO |
|------|-----|------|
| 优势估计 | 需要价值模型 + GAE | 组内相对奖励归一化 |
| 模型数量 | 3个（策略+价值+参考） | 2个（策略+参考） |
| 显存占用 | 高 | 低（~33% 减少） |
| 采样效率 | 每步1个响应 | 每步G个响应 |
| 适用场景 | 通用RL任务 | 有明确奖励信号的任务 |

## 模型规格

| 模型 | 参数量 | 显存(FP16) | 上下文长度 |
|------|--------|-----------|-----------|
| Qwen3-0.6B | 0.6B | 1.5GB | 32K |
| Qwen3-1.7B | 1.7B | 4GB | 32K |
| Qwen3-4B | 4B | 9GB | 128K |

## 奖励函数设计

多维度奖励函数：

1. **正确性奖励**（2.0分）：答案完全正确
2. **整数奖励**（0.5分）：答案是整数格式
3. **严格格式奖励**（0.5分）：完全符合 XML 格式
4. **宽松格式奖励**（0.5分）：基本符合 XML 格式
5. **XML 标签奖励**（0~0.5分）：标签位置正确

## 对比报告示例

运行 `run_evaluation.py` 后自动生成对比报告：

```
================================================================================
EVALUATION COMPARISON REPORT
================================================================================

## 1. 方法对比 (CoT vs PPO vs GRPO)
------------------------------------------------------------
方法             平均准确率        平均格式合规        最佳模型
------------------------------------------------------------
CoT              45.2%            78.3%            Qwen3-4B
GRPO             62.1%            92.5%            Qwen3-4B
PPO              58.7%            89.2%            Qwen3-4B

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
- [TRL](https://github.com/huggingface/trl) - PPO 和 GRPO 实现
- [GSM8K](https://github.com/openai/grade-school-math) - 数学推理基准
- [Qwen3](https://qwenlm.github.io/blog/qwen3/) - 模型架构详情
