"""
RLOO 训练方法实现。

本模块实现 TRL 1.0.0 推荐的 RLOO（Reinforce Leave-One-Out）训练流程，
用于替代旧版 PPOTrainer。核心特点：

1. 无需价值模型（value model），只保留策略模型与参考模型，显存占用更低。
2. 对每个问题一次采样多个 completion（由 num_generations 决定），
   使用 Leave-One-Out 基线估计优势，降低奖励方差。
3. 通过 beta 控制 KL 惩罚强度，约束策略不要偏离参考模型过快。

输出格式约束统一为：
<reasoning>...</reasoning>
<answer>...</answer>
"""

import torch
import logging
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM
from trl import RLOOConfig, RLOOTrainer
from tqdm import tqdm
import wandb
from .base import BaseMethod
from ..rewards.math_rewards import (
    extract_xml_answer,
    numeric_equivalence,
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    reasoning_quality_reward_func,
)

logger = logging.getLogger(__name__)


def _make_rloo_reward_funcs():
    """
    创建适配 RLOOTrainer 的奖励函数包装器。

    RLOOTrainer 的奖励函数签名:
        reward_func(prompts, completions, completion_ids, **kwargs) -> List[float]

    其中 conversational 格式下 completions 是 List[List[dict]]，
    每个元素是 [{"content": "..."}] 形式。

    我们现有的奖励函数已经接受 (completions, ...) 格式，
    这里用闭包包装以兼容 RLOO 的调用方式。

    返回顺序与 reward_weights 一一对应：
    [xmlcount, soft_format, strict_format, int_answer, reasoning_quality, correctness]
    """

    def correctness_reward(completions, answer, **kwargs):
        """
        正确性奖励（主奖励项）。

        参数:
            completions: 模型采样输出，conversational 结构。
            answer: 数据集标准答案列表。

        返回:
            List[float]: 正确给 2.0，错误给 -0.5，空答案给 -1.0。
        """
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]

        rewards = []
        for resp, ans in zip(extracted_responses, answer):
            resp_clean = resp.strip() if resp else ""
            ans_clean = ans.strip() if ans else ""

            if not resp_clean:
                rewards.append(-1.0)
            elif numeric_equivalence(resp_clean, ans_clean):
                rewards.append(2.0)
            else:
                rewards.append(-0.5)

        return rewards

    def int_reward(completions, answer, **kwargs):
        """
        数字格式奖励（辅助奖励项）。

        该奖励不关心答案是否正确，只检查 <answer> 内容能否解析为数字，
        用于鼓励模型输出可判分的数值格式。
        """
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]

        rewards = []
        for resp in extracted_responses:
            resp_clean = resp.strip() if resp else ""
            if not resp_clean:
                rewards.append(-0.1)
                continue

            from ..rewards.math_rewards import parse_number

            num = parse_number(resp_clean)
            if num is not None:
                rewards.append(0.1)
            else:
                rewards.append(-0.1)

        return rewards

    def strict_format_reward(completions, answer, **kwargs):
        """严格格式奖励：要求包含 XML 标签结构。"""
        import re

        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward(completions, answer, **kwargs):
        """宽松格式奖励：只要求出现 reasoning/answer 标签结构。"""
        import re

        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def xmlcount_reward(completions, answer, **kwargs):
        """XML 标签计数奖励：按标签完整度与尾部冗余长度打分。"""
        from ..rewards.math_rewards import count_xml

        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]

    def reasoning_quality_reward(completions, answer, **kwargs):
        """
        推理质量奖励：鼓励清晰、分步、包含计算痕迹的 reasoning。

        评分维度包括：步骤数量、是否包含算式、长度是否适中等。
        """
        import re

        responses = [completion[0]["content"] for completion in completions]
        rewards = []

        for response in responses:
            reward = 0.0

            if "<reasoning>" in response and "</reasoning>" in response:
                reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0]
            else:
                rewards.append(0.0)
                continue

            steps = [line for line in reasoning.split("\n") if line.strip()]
            if len(steps) >= 3:
                reward += 0.1
            if len(steps) >= 5:
                reward += 0.1

            if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", reasoning):
                reward += 0.1

            if "=" in reasoning:
                reward += 0.05

            length = len(reasoning.strip())
            if 30 <= length <= 150:
                reward += 0.05
            elif length < 20:
                reward -= 0.1
            elif length > 180:
                reward -= 0.05

            rewards.append(reward)

        return rewards

    return [
        xmlcount_reward,
        soft_format_reward,
        strict_format_reward,
        int_reward,
        reasoning_quality_reward,
        correctness_reward,
    ]


class RLOOMethod(BaseMethod):
    """
    RLOO (Reinforce Leave-One-Out) 方法实现。

    TRL 1.0.0 移除了 PPOConfig/PPOTrainer，RLOO 作为官方推荐替代方案。
    RLOO 在 RLHF 场景下性能优于 PPO，且无需价值模型，显存更低。

    设计说明：
    - 本实现沿用项目统一的 6 个奖励函数，并通过 YAML 配置权重。
    - 训练阶段只做策略优化；评估阶段统一用确定性解码统计准确率。
    """

    def __init__(self, config_path: str = "configs/methods/rloo.yaml"):
        """读取方法配置并缓存训练/奖励超参数。"""
        super().__init__(config_path)
        self.training_config = self.config["training"]
        self.reward_config = self.config.get("reward", {})

    def run(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Any,
        dataset: Any,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        执行 RLOO 训练并保存 checkpoint。

        参数:
            model: 待优化的策略模型。
            tokenizer: 与模型匹配的分词器。
            dataset: 训练集（每条样本含 prompt/answer/question 字段）。
            output_dir: 模型输出目录。

        返回:
            Dict[str, Any]: 训练状态与输出目录。
        """
        # 权重顺序必须与 reward_funcs 列表严格一致。
        # 应用 LoRA 适配器（显存优化 + 实验公平性）
        from ..utils import apply_lora

        model = apply_lora(model, self.training_config)

        reward_weights = [
            self.reward_config.get("xml_count_weight", 0.5),
            self.reward_config.get("soft_format_weight", 0.5),
            self.reward_config.get("strict_format_weight", 0.5),
            self.reward_config.get("integer_weight", 0.5),
            self.reward_config.get("reasoning_quality_weight", 0.3),
            self.reward_config.get("correctness_weight", 2.0),
        ]

        rloo_config = RLOOConfig(
            output_dir=output_dir,
            run_name=f"{model.config._name_or_path.split('/')[-1]}-RLOO-gsm8k",
            learning_rate=self.training_config["learning_rate"],
            adam_beta1=self.training_config.get("adam_beta1", 0.9),
            adam_beta2=self.training_config.get("adam_beta2", 0.99),
            weight_decay=self.training_config.get("weight_decay", 0.1),
            warmup_steps=int(self.training_config.get("warmup_steps", 0.1) * 100)
            if self.training_config.get("warmup_steps", 0.1) < 1
            else self.training_config.get("warmup_steps", 0.1),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.training_config.get("logging_steps", 1),
            bf16=self.training_config.get("bf16", True) and torch.cuda.is_available(),
            per_device_train_batch_size=self.training_config.get(
                "per_device_train_batch_size", 4
            ),
            gradient_accumulation_steps=self.training_config.get(
                "gradient_accumulation_steps", 4
            ),
            # 每个 prompt 采样 G 个 completion（G 越大，组内基线越稳定，但显存/算力开销越高）。
            num_generations=self.training_config.get("num_generations", 4),
            # 单次 batch 上执行的 RLOO 更新轮数。
            num_iterations=self.training_config.get("rloo_iterations", 1),
            max_completion_length=self.training_config.get(
                "max_completion_length", 1024
            ),
            num_train_epochs=self.training_config.get("num_train_epochs", 1),
            save_steps=self.training_config.get("save_steps", 100),
            max_grad_norm=self.training_config.get("max_grad_norm", 0.1),
            # KL 惩罚系数：约束新策略与参考策略偏移，避免奖励投机。
            beta=self.training_config.get("beta", 0.05),
            reward_weights=reward_weights,
            use_vllm=self.training_config.get("use_vllm", False),
            vllm_mode=self.training_config.get("vllm_mode", "colocate"),
            vllm_gpu_memory_utilization=self.training_config.get(
                "vllm_gpu_memory_utilization", 0.3
            ),
            report_to="wandb" if wandb.run else "none",
            disable_tqdm=False,
        )

        reward_funcs = _make_rloo_reward_funcs()

        trainer = RLOOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=rloo_config,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)

        return {"status": "completed", "output_dir": output_dir}

    def evaluate(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Any,
        test_dataset: Any,
    ) -> Dict[str, float]:
        """
        在测试集上评估 RLOO 模型。

        指标说明：
        - accuracy: 数值等价准确率（支持 0.5、1/2、50% 等价）。
        - format_compliance: 是否包含 XML 结构标签的比例。
        """
        correct = 0
        total = 0
        format_correct = 0

        for item in tqdm(test_dataset, desc="Evaluating RLOO"):
            question = item["question"]
            expected_answer = item["answer"]

            messages = [
                {
                    "role": "system",
                    # 统一强制 XML 输出格式，便于奖励函数与评估提取答案。
                    "content": "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n",
                },
                {"role": "user", "content": question},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    # 评估阶段使用贪心解码，减少采样噪声。
                    **model_inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            extracted_answer = extract_xml_answer(response)

            if numeric_equivalence(extracted_answer, expected_answer):
                correct += 1
            if "<reasoning>" in response and "<answer>" in response:
                format_correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "format_compliance": format_correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
        }
