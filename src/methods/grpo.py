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
    def __init__(self, config_path: str = "configs/methods/grpo.yaml"):
        super().__init__(config_path)
        self.training_config = self.config["training"]

    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        reward_config = self.config.get("reward", {})
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
            warmup_steps=self.training_config["warmup_steps"],
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            logging_steps=self.training_config["logging_steps"],
            bf16=self.training_config["bf16"],
            per_device_train_batch_size=self.training_config[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
            num_generations=self.training_config["num_generations"],
            max_completion_length=self.training_config["max_completion_length"],
            num_train_epochs=self.training_config["num_train_epochs"],
            save_steps=self.training_config["save_steps"],
            max_grad_norm=self.training_config["max_grad_norm"],
            log_on_each_node=False,
            use_vllm=self.training_config.get("use_vllm", False),
            vllm_gpu_memory_utilization=self.training_config.get(
                "vllm_gpu_memory_utilization", 0.3
            ),
            reward_weights=reward_weights,
            report_to="wandb",
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

    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        correct = 0
        total = 0
        format_correct = 0

        for item in tqdm(test_dataset, desc="Evaluating GRPO"):
            question = item["question"]
            expected_answer = item["answer"]

            messages = [
                {
                    "role": "system",
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
                    **model_inputs, max_new_tokens=512, temperature=0.0, do_sample=False
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
