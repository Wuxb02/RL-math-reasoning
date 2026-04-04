"""
数学推理奖励函数模块。

本模块定义了 6 个奖励函数，从格式、内容、正确性三个维度评估模型输出，
用于 RLOO 和 GRPO 强化学习训练。

奖励函数总览:
    1. correctness_reward_func  - 答案正确性（权重 2.0，核心信号）
    2. int_reward_func          - 数字格式检测（权重 0.5）
    3. strict_format_reward_func - 严格 XML 格式匹配（权重 0.5）
    4. soft_format_reward_func  - 宽松 XML 格式匹配（权重 0.5）
    5. xmlcount_reward_func     - XML 标签完整性（权重 0.5）
    6. reasoning_quality_reward_func - 推理过程质量（权重 0.3）

设计原则:
    - 渐进式引导：从宽松格式到严格格式，逐步规范输出
    - 多维度评估：同时考虑格式、内容、质量
    - 负样本惩罚：对错误答案和空答案给予负奖励
    - 稀疏奖励平衡：正确性奖励权重最高，但其他奖励提供稳定梯度

所有奖励函数遵循 TRL 接口:
    reward_func(completions, **kwargs) -> List[float]
"""

import re
from typing import List, Optional
from fractions import Fraction


def extract_xml_answer(text: str) -> str:
    """
    从 XML 格式的响应中提取 <answer> 标签内容。

    这是所有奖励函数和评估流程的基础工具函数，用于从模型生成的
    完整回答中提取最终答案。

    Args:
        text: 模型生成的完整回答文本

    Returns:
        str: <answer> 标签内的内容，去除首尾空白。
             如果标签不存在则返回空字符串。
    """
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def parse_number(text: str) -> Optional[float]:
    """
    尝试将文本解析为数值，支持多种格式。

    用于数值等价判断，使得 "0.5"、".5"、"1/2"、"50%" 等
    不同表示方式能被正确识别为相同数值。

    支持的格式:
        - 整数: "42", "-42"
        - 小数: "3.14", ".5", "-0.5"
        - 分数: "1/2", "-3/4"
        - 千分位: "1,234"
        - 百分比: "50%" → 0.5

    Args:
        text: 待解析的文本字符串

    Returns:
        Optional[float]: 解析后的浮点数，如果无法解析则返回 None
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip().replace(",", "")

    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100
        except ValueError:
            return None

    if "/" in text:
        try:
            parts = text.split("/")
            if len(parts) == 2:
                numerator = float(parts[0].strip())
                denominator = float(parts[1].strip())
                if denominator != 0:
                    return numerator / denominator
        except (ValueError, ZeroDivisionError):
            pass
        return None

    try:
        return float(text)
    except ValueError:
        return None


def numeric_equivalence(answer: str, expected: str) -> bool:
    """
    检查两个答案是否数值等价。

    这是正确性奖励的核心判断逻辑，支持多种数值格式的等价比较，
    避免因格式差异导致误判。

    支持的等价形式:
        - "42" == "42.0" == "42.00"
        - "0.5" == ".5" == "1/2" == "50%"
        - "-3" == "-3.0"

    Args:
        answer: 模型生成的答案
        expected: 标准答案

    Returns:
        bool: 如果两个答案数值等价则返回 True
    """
    # 首先尝试精确字符串匹配（最快路径）
    if answer.strip() == expected.strip():
        return True

    # 尝试数值解析比较
    answer_num = parse_number(answer)
    expected_num = parse_number(expected)

    if answer_num is None or expected_num is None:
        return False

    # 浮点数比较，使用相对误差容忍精度差异
    if expected_num == 0:
        return abs(answer_num) < 1e-9
    else:
        return abs(answer_num - expected_num) / abs(expected_num) < 1e-9


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    正确性奖励 — 最核心的奖励信号（权重 2.0）。

    从 <answer> 标签中提取答案，与标准答案进行数值等价比较。
    这是引导模型学到正确解题方法的关键信号。

    评分策略:
        - 空答案: -1.0（严重惩罚，引导模型必须输出内容）
        - 完全正确: +2.0（最高奖励，包含数值等价判断）
        - 错误答案: -0.5（小惩罚，不至于完全否定）

    Args:
        prompts: 输入的 prompt 列表（未使用，但 TRL 会传入）
        completions: 模型生成的回答列表，conversational 格式
        answer: 标准答案列表（从数据集的 answer 列传入）
        **kwargs: 其他额外参数

    Returns:
        List[float]: 每个样本的正确性奖励值
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for resp, ans in zip(extracted_responses, answer):
        resp_clean = resp.strip() if resp else ""
        ans_clean = ans.strip() if ans else ""

        if not resp_clean:
            # 空答案：严重惩罚
            rewards.append(-1.0)
        elif numeric_equivalence(resp_clean, ans_clean):
            # 完全正确（含数值等价）
            rewards.append(2.0)
        else:
            # 错误答案：小惩罚
            rewards.append(-0.5)

    return rewards


def int_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    数字格式奖励 — 引导模型输出数值型答案（权重 0.5）。

    仅检查答案是否为可解析的数字格式，不验证正确性。
    正确性由 correctness_reward_func 负责，避免信号干扰。

    设计目的:
        鼓励模型在 <answer> 标签中输出数字，而非文字描述。
        例如鼓励输出 "42" 而非 "forty-two"。

    评分策略:
        - 可解析为数字: +0.1
        - 非数字格式: -0.1

    Args:
        completions: 模型生成的回答列表
        answer: 标准答案列表
        **kwargs: 其他额外参数

    Returns:
        List[float]: 每个样本的数字格式奖励值
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for resp in extracted_responses:
        resp_clean = resp.strip() if resp else ""

        if not resp_clean:
            rewards.append(-0.1)
            continue

        num = parse_number(resp_clean)

        if num is not None:
            rewards.append(0.1)
        else:
            rewards.append(-0.1)

    return rewards


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """
    严格格式奖励 — 要求完全符合 XML 格式规范（权重 0.5）。

    期望格式（必须精确匹配）:
        <reasoning>
        [推理内容，必须有换行]
        </reasoning>
        <answer>
        [答案内容，必须有换行]
        </answer>

    使用 re.match 从字符串开头精确匹配，不允许任何前缀或后缀偏差。
    这是最高标准的格式要求。

    Args:
        completions: 模型生成的回答列表
        **kwargs: 其他额外参数

    Returns:
        List[float]: 完全匹配 +0.5，否则 +0.0
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """
    宽松格式奖励 — 只要求 XML 标签存在，不要求严格换行（权重 0.5）。

    作为严格格式的"降级奖励"，即使格式不完美也能获得部分奖励。
    使用 re.search 允许前缀内容，标签间允许任意空白。

    设计目的:
        在训练初期，模型可能无法完全符合严格格式，
        宽松奖励提供渐进式的引导信号。

    Args:
        completions: 模型生成的回答列表
        **kwargs: 其他额外参数

    Returns:
        List[float]: 包含正确标签 +0.5，否则 +0.0
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """
    XML 标签计数评分 — 检查标签完整性和位置。

    对每个标签分别评分:
        - 格式完全正确（含换行）: +0.125
        - 标签存在但格式不完美: +0.0625

    同时惩罚 </answer> 后的冗余内容（每字符 -0.001），
    鼓励简洁输出，避免模型在答案后添加多余内容。

    Args:
        text: 完整的模型回答文本

    Returns:
        float: 结构评分，范围 [-0.5, +0.5]
    """
    count = 0.0

    # 检查 <reasoning> 开始标签
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    elif text.count("<reasoning>") >= 1:
        count += 0.0625  # 有标签但格式不完全正确

    # 检查 </reasoning> 结束标签
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    elif text.count("</reasoning>") >= 1:
        count += 0.0625

    # 检查 <answer> 开始标签
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    elif text.count("<answer>") >= 1:
        count += 0.0625

    # 检查 </answer> 结束标签并扣分
    if text.count("\n</answer>\n") == 1:
        count += 0.125
        # 扣分：</answer>\n 之后的内容
        tail = text.split("\n</answer>\n")[-1]
        count -= len(tail) * 0.001
    elif text.count("\n</answer>") == 1:
        count += 0.125
        tail = text.split("\n</answer>")[-1]
        count -= len(tail) * 0.001
    elif text.count("</answer>") >= 1:
        count += 0.0625
        tail = text.split("</answer>")[-1]
        count -= len(tail) * 0.001

    return max(count, -0.5)  # 限制最低分为 -0.5


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """
    XML 标签计数奖励函数（权重 0.5）。

    包装 count_xml 函数以符合 TRL 奖励函数接口。

    Args:
        completions: 模型生成的回答列表
        **kwargs: 其他额外参数

    Returns:
        List[float]: 每个样本的 XML 结构评分
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reasoning_quality_reward_func(completions, **kwargs) -> List[float]:
    """
    推理质量奖励 — 评估 <reasoning> 标签内推理过程的质量（权重 0.3）。

    评估维度:
        1. 推理步骤数量（≥3步 +0.1，≥5步 +0.1）
           — 鼓励多步骤思考，避免跳跃式推理
        2. 是否包含算术运算（+0.1）
           — 数学题应该有计算过程
        3. 是否包含等号（+0.05）
           — 表示有推导过程
        4. 推理长度是否适中（30-150字符 +0.05）
           — 太短可能没有实质内容，太长可能冗余

    惩罚:
        - 推理太短（<20字符）-0.1
        - 推理太长（>180字符）-0.05

    Args:
        completions: 模型生成的回答列表
        **kwargs: 其他额外参数

    Returns:
        List[float]: 每个样本的推理质量评分，范围 [-0.15, +0.3]
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        reward = 0.0

        # 提取 <reasoning> 标签内的推理内容
        if "<reasoning>" in response and "</reasoning>" in response:
            reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0]
        else:
            rewards.append(0.0)
            continue

        # 1. 检查推理步骤数量（每行视为一个步骤）
        steps = [line for line in reasoning.split("\n") if line.strip()]
        if len(steps) >= 3:
            reward += 0.1  # 至少3步
        if len(steps) >= 5:
            reward += 0.1  # 至少5步

        # 2. 检查是否包含数字计算（如 "15 + 27"）
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", reasoning):
            reward += 0.1  # 包含算术运算

        # 3. 检查是否包含等号（表示有计算推导过程）
        if "=" in reasoning:
            reward += 0.05

        # 4. 推理长度检查（鼓励适中长度）
        length = len(reasoning.strip())
        if 30 <= length <= 150:
            reward += 0.05
        elif length < 20:
            reward -= 0.1
        elif length > 180:
            reward -= 0.05

        rewards.append(reward)

    return rewards
