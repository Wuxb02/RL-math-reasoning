import torch
from typing import Dict, Any, List
from tqdm import tqdm
from .base import BaseMethod
from ..rewards.math_rewards import extract_xml_answer


class CoTMethod(BaseMethod):
    def __init__(self, config_path: str = "configs/methods/cot.yaml"):
        super().__init__(config_path)
        self.system_prompt = self.config["system_prompt"]
        self.generation_config = self.config["generation"]
        self.few_shot_examples = self.config.get("few_shot_examples", [])

    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        return self.evaluate(model, tokenizer, dataset)

    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        correct = 0
        total = 0
        format_correct = 0
        results = []

        for item in tqdm(test_dataset, desc="Evaluating CoT"):
            question = item["question"]
            expected_answer = item["answer"]

            response = self._generate_response(model, tokenizer, question)
            extracted_answer = extract_xml_answer(response)

            is_correct = extracted_answer == expected_answer
            has_format = "<reasoning>" in response and "<answer>" in response

            if is_correct:
                correct += 1
            if has_format:
                format_correct += 1
            total += 1

            results.append(
                {
                    "question": question,
                    "expected": expected_answer,
                    "predicted": extracted_answer,
                    "full_response": response,
                    "correct": is_correct,
                    "format_ok": has_format,
                }
            )

        accuracy = correct / total if total > 0 else 0
        format_compliance = format_correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "format_compliance": format_compliance,
            "correct": correct,
            "total": total,
            "results": results,
        }

    def _generate_response(self, model, tokenizer, question: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]

        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["response"]})

        messages.append({"role": "user", "content": question})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.generation_config.get("max_new_tokens", 512),
                temperature=self.generation_config.get("temperature", 0.0),
                do_sample=self.generation_config.get("do_sample", False),
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
