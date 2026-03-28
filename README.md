# GRPO-Math: 数学推理强化学习对比实验

对比研究 **CoT（思维链）**、**PPO（近端策略优化）** 和 **GRPO（组相对策略优化）** 三种方法在数学推理任务上的效果差异，并测试 **Qwen3 系列不同大小模型（0.6B、1.7B、4B）** 的能力边界。

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
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

```bash
# 运行单个实验
python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/cot.yaml

# 运行全部9个组合（3模型 × 3方法）
python scripts/run_all.py

# 启用 WandB 日志
python scripts/train.py --model configs/models/qwen3-0.6b.yaml --method configs/methods/grpo.yaml --wandb
```

### 5. 评估和可视化

```bash
# 评估训练后的模型
python scripts/evaluate.py --model configs/models/qwen3-0.6b.yaml --checkpoint outputs/Qwen3-0.6B-GRPO --method GRPO

# 可视化结果
python scripts/visualize.py --input outputs/results/all_results.json
```

## 项目结构

```
GRPO-math/
├── configs/
│   ├── models/          # 模型配置（Qwen3-0.6B/1.7B/4B）
│   └── methods/         # 方法配置（CoT/PPO/GRPO）
├── src/
│   ├── data/            # GSM8K 数据加载
│   ├── models/          # 模型加载器
│   ├── methods/         # CoT/PPO/GRPO 实现
│   ├── rewards/         # 奖励函数
│   ├── evaluation/      # 评估框架
│   └── utils/           # 可视化工具
├── scripts/
│   ├── train.py         # 单次训练
│   ├── evaluate.py      # 评估
│   ├── run_all.py       # 批量实验
│   └── visualize.py     # 结果可视化
├── resources/           # 模型和数据（gitignored）
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   ├── Qwen3-4B/
│   └── gsm8k/
├── outputs/             # 实验输出（gitignored）
├── download_models.ipynb
├── .env                 # API密钥（gitignored）
├── .env.example         # 环境变量模板
└── pyproject.toml
```

## 三种方法对比

| 方法 | 类型 | 核心思想 | 显存占用 |
|------|------|----------|----------|
| **CoT** | 推理 | 提示词引导逐步推理 | 低 |
| **PPO** | 训练 | 策略梯度 + 价值函数 + KL惩罚 | 高（需要价值模型） |
| **GRPO** | 训练 | 组内相对奖励归一化 | 中（无需价值模型） |

### CoT（Chain-of-Thought）

- 无需训练，通过精心设计的提示词引导模型推理
- 使用 XML 格式的思维链模板
- 适合作为基线对比

### PPO（Proximal Policy Optimization）

- 经典的强化学习算法，使用裁剪目标函数
- 需要策略模型、价值模型和参考模型
- 训练稳定但显存占用高

### GRPO（Group Relative Policy Optimization）

- DeepSeek-R1 使用的方法
- 对同一提示的多个采样结果进行组内相对奖励归一化
- 无需价值模型，显存占用降低约 33%

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

## 参考资料

- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - GRPO 方法来源
- [TRL](https://github.com/huggingface/trl) - PPO 和 GRPO 实现
- [GSM8K](https://github.com/openai/grade-school-math) - 数学推理基准
- [Qwen3](https://qwenlm.github.io/blog/qwen3/) - 模型架构详情
