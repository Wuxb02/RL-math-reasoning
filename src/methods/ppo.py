import torch
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import wandb
from .base import BaseMethod
from ..rewards.math_rewards import (
    extract_xml_answer,
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
)


class PPOMethod(BaseMethod):
    def __init__(self, config_path: str = "configs/methods/ppo.yaml"):
        super().__init__(config_path)
        self.training_config = self.config["training"]
        self.reward_config = self.config["reward"]

    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        ppo_config = PPOConfig(
            learning_rate=self.training_config["learning_rate"],
            batch_size=self.training_config["batch_size"],
            mini_batch_size=self.training_config["mini_batch_size"],
            ppo_epochs=self.training_config["ppo_epochs"],
            kl_penalty=self.training_config["kl_penalty"],
            init_kl_coef=self.training_config["init_kl_coef"],
            adap_kl_ctrl=self.training_config["adap_kl_ctrl"],
            target=self.training_config["target"],
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
            max_grad_norm=self.training_config["max_grad_norm"],
            log_with="wandb",
            project_kwargs={"logging_dir": output_dir},
        )

        model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        def tokenize(sample):
            prompt = tokenizer.apply_chat_template(
                sample["prompt"], tokenize=False, add_generation_prompt=True
            )
            sample["input_ids"] = tokenizer.encode(prompt)
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        dataset = dataset.map(tokenize)

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value_head,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        generation_kwargs = {
            "max_new_tokens": self.training_config["max_new_tokens"],
            "temperature": self.training_config["temperature"],
            "top_p": self.training_config["top_p"],
            "do_sample": self.training_config["do_sample"],
        }

        for epoch in range(self.training_config["num_train_epochs"]):
            for batch in tqdm(ppo_trainer.dataloader, desc=f"PPO Epoch {epoch + 1}"):
                query_tensors = batch["input_ids"]

                response_tensors = ppo_trainer.generate(
                    query_tensors, return_prompt=False, **generation_kwargs
                )

                rewards = self._compute_rewards(batch, response_tensors, tokenizer)

                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)

        model_with_value_head.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return {"status": "completed", "output_dir": output_dir}

    def _compute_rewards(
        self, batch, response_tensors, tokenizer
    ) -> List[torch.Tensor]:
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        answers = batch["answer"]

        rewards = []
        for response, answer in zip(responses, answers):
            reward = 0.0
            extracted = extract_xml_answer(response)

            if extracted == answer:
                reward += self.reward_config["correctness_weight"]

            if extracted.isdigit():
                reward += self.reward_config["integer_answer_weight"]

            if "<reasoning>" in response and "<answer>" in response:
                reward += self.reward_config["xml_format_weight"]

            rewards.append(torch.tensor(reward))

        return rewards

    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        correct = 0
        total = 0
        format_correct = 0

        for item in tqdm(test_dataset, desc="Evaluating PPO"):
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

            if extracted_answer == expected_answer:
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
