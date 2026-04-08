"""GRPO 训练方法实现。

本模块实现 Group Relative Policy Optimization（GRPO）训练流程，
核心思想是：对同一问题采样多个回答，基于组内奖励做相对归一化，
从而得到优势信号（advantage）而无需价值模型。

本项目统一要求模型输出 XML 结构：
<reasoning>...</reasoning>
<answer>...</answer>
"""

import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
import wandb
from .base import BaseMethod
from ..rewards.math_rewards import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    reasoning_quality_reward_func,
    extract_xml_answer,
    numeric_equivalence,
)


class GRPOMethod(BaseMethod):
    """GRPO 方法封装。

    设计决策：
    - 奖励由 6 个子函数线性加权组成，权重来自 YAML，便于实验对比。
    - 训练与评估分离：训练期关注策略优化，评估期固定贪心解码统计指标。
    """

    def __init__(self, config_path: str = "configs/methods/grpo.yaml"):
        """加载 GRPO 配置并缓存训练超参数。"""
        super().__init__(config_path)
        self.training_config = self.config["training"]

    def run(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Any,
        dataset: Any,
        output_dir: str,
    ) -> Dict[str, Any]:
        """执行 GRPO 训练并保存模型。

        参数:
            model: 待优化的策略模型。
            tokenizer: 与模型匹配的 tokenizer。
            dataset: 训练数据集，样本至少包含 prompt/answer/question。
            output_dir: 模型保存目录。

        返回:
            Dict[str, Any]: 训练状态和输出目录。
        """
        reward_config = self.config.get("reward", {})
        # 应用 LoRA 适配器（显存优化 + 实验公平性）
        from ..utils import apply_lora

        model = apply_lora(model, self.training_config)

        # 权重顺序必须与 reward_funcs 的函数顺序保持一致。
        reward_weights = [
            reward_config.get("xml_count_weight", 0.5),
            reward_config.get("soft_format_weight", 0.5),
            reward_config.get("strict_format_weight", 0.5),
            reward_config.get("integer_weight", 0.5),
            reward_config.get("reasoning_quality_weight", 0.3),
            reward_config.get("correctness_weight", 2.0),
        ]

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            run_name=f"{model.config._name_or_path.split('/')[-1]}-GRPO-gsm8k",
            learning_rate=self.training_config["learning_rate"],
            adam_beta1=self.training_config["adam_beta1"],
            adam_beta2=self.training_config["adam_beta2"],
            weight_decay=self.training_config["weight_decay"],
            warmup_steps=self.training_config.get("warmup_steps", 50),
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            logging_steps=self.training_config["logging_steps"],
            bf16=self.training_config.get("bf16", True),
            per_device_train_batch_size=self.training_config.get(
                "per_device_train_batch_size", 2
            ),
            gradient_accumulation_steps=self.training_config.get(
                "gradient_accumulation_steps", 2
            ),
            # 每个问题采样 G 个候选回答。
            # GRPO 会基于组内奖励均值/方差做相对归一化来构造优势信号。
            num_generations=self.training_config.get("num_generations", 2),
            max_completion_length=self.training_config.get(
                "max_completion_length", 1024
            ),
            # KL 惩罚系数：约束新策略与参考策略偏移，避免奖励投机。
            beta=self.training_config.get("beta", 0.05),
            num_train_epochs=self.training_config.get("num_train_epochs", 1),
            save_steps=self.training_config.get("save_steps", 200),
            max_grad_norm=self.training_config.get("max_grad_norm", 0.1),
            log_on_each_node=False,
            use_vllm=self.training_config.get("use_vllm", False),
            vllm_mode=self.training_config.get("vllm_mode", "colocate"),
            vllm_gpu_memory_utilization=self.training_config.get(
                "vllm_gpu_memory_utilization", 0.3
            ),
            reward_weights=reward_weights,
            report_to="wandb",
            disable_tqdm=False,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                reasoning_quality_reward_func,
                correctness_reward_func,
            ],
            args=grpo_config,
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
        """在测试集上评估训练后模型。

        指标说明：
        - accuracy：答案数值等价准确率。
        - format_compliance：是否满足 XML 标签结构的比例。
        """
        correct = 0
        total = 0
        format_correct = 0

        for item in tqdm(test_dataset, desc="Evaluating GRPO"):
            question = item["question"]
            expected_answer = item["answer"]

            messages = [
                {
                    "role": "system",
                    # 强制使用统一 XML 模板，保证答案可被稳定提取和打分。
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
                    # 评估使用确定性解码，降低随机采样带来的方差。
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
