import re
from typing import List, Optional
from fractions import Fraction


def extract_xml_answer(text: str) -> str:
    """从 XML 格式的响应中提取答案部分"""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def parse_number(text: str) -> Optional[float]:
    """
    尝试将文本解析为数值，支持多种格式：
    - 整数: "42", "-42"
    - 小数: "3.14", ".5", "-0.5"
    - 分数: "1/2", "-3/4"
    - 带逗号: "1,234" (千分位)
    - 百分比: "50%" -> 0.5
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
    检查两个答案是否数值等价

    支持的等价形式：
    - "42" == "42.0" == "42.00"
    - "0.5" == ".5" == "1/2" == "50%"
    - "-3" == "-3.0"
    """
    # 首先尝试精确字符串匹配（最快）
    if answer.strip() == expected.strip():
        return True

    # 尝试数值解析
    answer_num = parse_number(answer)
    expected_num = parse_number(expected)

    if answer_num is None or expected_num is None:
        return False

    # 浮点数比较，使用相对误差
    if expected_num == 0:
        return abs(answer_num) < 1e-9
    else:
        return abs(answer_num - expected_num) / abs(expected_num) < 1e-9


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    正确性奖励：答案完全正确得满分，否则扣分

    改进点：
    - 支持数值等价判断（"0.5" == ".5" == "1/2"）
    - 自动去除首尾空白
    - 空答案给予负奖励
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for resp, ans in zip(extracted_responses, answer):
        resp_clean = resp.strip() if resp else ""
        ans_clean = ans.strip() if ans else ""

        if not resp_clean:
            # 空答案：负奖励
            rewards.append(-1.0)
        elif numeric_equivalence(resp_clean, ans_clean):
            # 完全正确
            rewards.append(2.0)
        else:
            # 错误答案：小负奖励
            rewards.append(-0.5)

    return rewards


def int_reward_func(completions, answer, **kwargs) -> List[float]:
    """
    整数格式奖励：仅当答案是数字格式时给予小奖励，与正确性无关。

    只奖励"答案是数字"这一格式特征，避免错误整数答案获得正奖励
    而抵消正确性惩罚。
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
    严格格式奖励：完全符合 XML 格式要求

    要求格式：
    <reasoning>
    [推理内容]
    </reasoning>
    <answer>
    [答案]
    </answer>
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    """
    宽松格式奖励：只要求标签存在，不要求严格换行

    改进点：
    - 使用 re.search 而非 re.match，允许前缀内容
    - 支持标签间有任意空白
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """
    XML 标签计数奖励：检查标签出现次数和位置

    改进点：
    - 更宽松的标签检测
    - 渐进式扣分：标签后每多一个字符扣 0.001 分
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
    XML 标签计数奖励函数
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reasoning_quality_reward_func(completions, **kwargs) -> List[float]:
    """
    推理质量奖励：评估推理过程的质量

    评估维度：
    - 推理步骤数量（鼓励多步骤思考）
    - 包含数字计算（数学题应该有计算过程）
    - 推理长度适中（不要太短也不要太长）
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        reward = 0.0

        # 提取推理部分
        if "<reasoning>" in response and "</reasoning>" in response:
            reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0]
        else:
            rewards.append(0.0)
            continue

        # 1. 检查推理步骤（每行视为一个步骤）
        steps = [line for line in reasoning.split("\n") if line.strip()]
        if len(steps) >= 3:
            reward += 0.1  # 至少3步
        if len(steps) >= 5:
            reward += 0.1  # 至少5步

        # 2. 检查是否包含数字计算
        if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", reasoning):
            reward += 0.1  # 包含算术运算

        # 3. 检查是否包含等号（表示有计算过程）
        if "=" in reasoning:
            reward += 0.05

        # 4. 推理长度检查（50-500字符为最佳）
        length = len(reasoning.strip())
        if 30 <= length <= 150:
            reward += 0.05
        elif length < 20:
            reward -= 0.1
        elif length > 180:
            reward -= 0.05

        rewards.append(reward)

    return rewards
