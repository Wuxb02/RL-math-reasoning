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
from typing import Dict, Any
from transformers import AutoModelForCausalLM
from trl import RLOOConfig, RLOOTrainer
from tqdm import tqdm
import wandb
from .base import BaseMethod
from ..data.gsm8k import extract_xml_answer
from ..rewards.math_rewards import (
    numeric_equivalence,
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    reasoning_quality_reward_func,
)

logger = logging.getLogger(__name__)


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
            warmup_steps=self.training_config.get("warmup_steps", 50),
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

        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            reasoning_quality_reward_func,
            correctness_reward_func,
        ]

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
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            model.config.use_cache = True

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
                    max_new_tokens=1024,
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
