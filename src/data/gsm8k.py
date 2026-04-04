"""
GSM8K 数据集加载与预处理模块。

GSM8K（Grade School Math 8K）是 OpenAI 发布的数学应用题数据集，
包含 8,500 道小学数学题，每道题包含逐步推理过程和最终答案。

数据格式:
    - 原始格式: question + answer（答案以 "#### 42" 结尾）
    - 处理后: prompt（对话格式）+ answer（提取后的数值）+ question

本项目要求模型输出 XML 格式:
    <reasoning>
    [逐步推理过程]
    </reasoning>
    <answer>
    [最终答案]
    </answer>
"""

import re
from typing import Optional
from datasets import load_dataset, Dataset


# XML 格式的思维链模板，用于 few-shot 示例
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 系统提示词，要求模型使用 XML 格式输出
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """
    从 XML 格式的响应中提取 <answer> 标签内容。

    用于从模型生成的回答中提取最终答案，供评估和奖励函数使用。

    Args:
        text: 模型生成的完整回答文本

    Returns:
        str: <answer> 标签内的内容，如果标签不存在则返回空字符串

    示例:
        >>> extract_xml_answer("<reasoning>...\n</reasoning>\n<answer>\n42\n</answer>")
        '42'
    """
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> Optional[str]:
    """
    从 GSM8K 原始答案中提取 "####" 后面的数值部分。

    GSM8K 数据集的答案格式为: "逐步推理过程 #### 42"
    本函数提取 "####" 后的最终答案。

    Args:
        text: GSM8K 原始答案字符串

    Returns:
        Optional[str]: 提取后的答案数值，如果不包含 "####" 则返回 None

    示例:
        >>> extract_hash_answer("First, ... #### 42")
        '42'
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


class GSM8KDataset:
    """
    GSM8K 数据集的面向对象封装。

    提供便捷的接口访问训练集和测试集，自动完成:
    1. 从本地目录或 HuggingFace Hub 加载数据
    2. 提取 "####" 后的答案数值
    3. 构建对话格式的 prompt（含 system prompt）

    使用方式:
        >>> dataset = GSM8KDataset(split="train")
        >>> for item in dataset:
        ...     print(item["prompt"], item["answer"])
    """

    def __init__(self, data_dir: str = "./resources/gsm8k", split: str = "train"):
        """
        初始化 GSM8K 数据集。

        Args:
            data_dir: 数据本地目录路径，默认 "./resources/gsm8k"
            split: 数据集划分，可选值 "train" 或 "test"
        """
        self.data_dir = data_dir
        self.split = split
        self._load_dataset()

    def _load_dataset(self):
        """
        加载并预处理数据集。

        预处理步骤:
        1. 加载原始数据（question + answer 字段）
        2. 从 answer 中提取 "####" 后的数值
        3. 构建对话格式的 prompt: [system_prompt, user_question]
        """
        raw_data = load_dataset(self.data_dir, "main")[self.split]
        self.data = raw_data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
                "question": x["question"],
            }
        )

    def get_prompts(self) -> list:
        """获取所有 prompt（对话格式的消息列表）。"""
        return self.data["prompt"]

    def get_answers(self) -> list:
        """获取所有标准答案。"""
        return self.data["answer"]

    def get_questions(self) -> list:
        """获取所有原始问题。"""
        return self.data["question"]

    def __len__(self):
        """返回数据集大小。"""
        return len(self.data)

    def __getitem__(self, idx):
        """按索引获取单条样本。"""
        return self.data[idx]


def get_gsm8k_dataset(
    split: str = "train", data_dir: str = "./resources/gsm8k"
) -> Dataset:
    """
    便捷函数：直接返回预处理后的 HuggingFace Dataset 对象。

    这是项目中最常用的数据加载方式，训练和评估脚本都使用此函数。

    Args:
        split: 数据集划分，"train"（训练集，约 7,473 条）或 "test"（测试集，约 1,319 条）
        data_dir: 数据本地目录路径

    Returns:
        Dataset: 预处理后的数据集，每条样本包含:
            - prompt: 对话格式的消息列表 [system, user]
            - answer: 提取后的答案数值（字符串）
            - question: 原始问题文本

    示例:
        >>> train_data = get_gsm8k_dataset(split="train")
        >>> test_data = get_gsm8k_dataset(split="test")
        >>> print(train_data[0]["prompt"])
        [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]
    """
    raw_data = load_dataset(data_dir, "main")[split]
    data = raw_data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
            "question": x["question"],
        }
    )
    return data
