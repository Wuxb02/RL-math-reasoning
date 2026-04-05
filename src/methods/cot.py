"""
CoT（Chain-of-Thought，思维链）推理基线实现。

CoT 是一种纯推理方法，不涉及模型训练。通过精心设计的 System Prompt
引导模型逐步推理并输出结构化的答案，作为强化学习方法（RLOO/GRPO）
的性能基线。

输出格式要求:
    <reasoning>
    [逐步推理过程]
    </reasoning>
    <answer>
    [最终答案]
    </answer>

支持 Few-shot 示例，可在配置文件中添加示范问答以引导模型输出格式。
"""

import torch
from typing import Dict, Any, List
from tqdm import tqdm
from .base import BaseMethod
from ..rewards.math_rewards import extract_xml_answer, numeric_equivalence


class CoTMethod(BaseMethod):
    """
    CoT（Chain-of-Thought）推理基线。

    与 RLOO/GRPO 不同，CoT 不需要训练：
    - run() 直接调用 evaluate() 执行推理
    - 使用预训练模型的零样本/少样本推理能力
    - 作为强化学习方法的对比基线

    配置项（来自 YAML）:
        - system_prompt: 系统提示词，要求模型使用 XML 格式输出
        - generation: 生成参数（max_new_tokens, temperature, do_sample）
        - few_shot_examples: 可选的示范示例列表
    """

    def __init__(self, config_path: str = "configs/methods/cot.yaml"):
        """
        初始化 CoT 方法。

        Args:
            config_path: YAML 配置文件路径，默认使用 configs/methods/cot.yaml
        """
        super().__init__(config_path)
        self.system_prompt = self.config["system_prompt"]
        self.generation_config = self.config["generation"]
        self.few_shot_examples = self.config.get("few_shot_examples", [])

    def run(self, model, tokenizer, dataset, output_dir: str) -> Dict[str, Any]:
        """
        执行 CoT 推理。

        CoT 无需训练，直接调用 evaluate() 对数据集进行推理评估。

        Args:
            model: 预训练的语言模型
            tokenizer: 对应的分词器
            dataset: 待推理的数据集
            output_dir: 输出目录（CoT 不使用，仅为接口兼容）

        Returns:
            Dict[str, Any]: 评估结果（准确率、格式合规率等）
        """
        return self.evaluate(model, tokenizer, dataset)

    def evaluate(self, model, tokenizer, test_dataset) -> Dict[str, float]:
        """
        在测试集上评估 CoT 推理能力。

        逐条处理测试样本，记录:
        - 答案正确率（精确字符串匹配）
        - 格式合规率（是否包含 reasoning 和 answer 标签）
        - 每条样本的详细结果（用于后续分析）

        Args:
            model: 预训练的语言模型
            tokenizer: 对应的分词器
            test_dataset: 测试数据集

        Returns:
            Dict[str, float]: 评估指标
                - accuracy: 答案正确率
                - format_compliance: 格式合规率
                - correct: 正确回答数量
                - total: 总样本数量
                - results: 每条样本的详细结果列表
        """
        correct = 0
        total = 0
        format_correct = 0
        results = []

        for item in tqdm(test_dataset, desc="Evaluating CoT"):
            question = item["question"]
            expected_answer = item["answer"]

            # 生成模型回答
            response = self._generate_response(model, tokenizer, question)
            # 从 XML 标签中提取答案
            extracted_answer = extract_xml_answer(response)

            # 判断是否正确和格式是否合规
            # 使用数值等价判断，与 RLOO/GRPO 保持一致
            is_correct = numeric_equivalence(extracted_answer, expected_answer)
            has_format = "<reasoning>" in response and "<answer>" in response

            if is_correct:
                correct += 1
            if has_format:
                format_correct += 1
            total += 1

            # 记录每条样本的详细结果
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
        """
        使用模型生成单条问题的回答。

        构建对话消息，支持 few-shot 示范:
        1. System Prompt（格式要求）
        2. Few-shot 示例对（可选）
        3. 当前问题

        Args:
            model: 预训练的语言模型
            tokenizer: 对应的分词器
            question: 用户问题

        Returns:
            str: 模型生成的完整回答文本
        """
        # 构建对话消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加 few-shot 示范示例（如果配置了的话）
        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["response"]})

        # 添加当前问题
        messages.append({"role": "user", "content": question})

        # 应用聊天模板，将消息列表转换为模型可接受的文本格式
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 编码输入并移动到模型设备
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成回答（无梯度计算）
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.generation_config.get("max_new_tokens", 512),
                temperature=self.generation_config.get("temperature", 0.0),
                do_sample=self.generation_config.get("do_sample", False),
            )

        # 移除 prompt 部分，只保留生成的内容
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码为文本
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
