import re
from typing import Optional
from datasets import load_dataset, Dataset


XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

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
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


class GSM8KDataset:
    def __init__(self, data_dir: str = "./resources/gsm8k", split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        self._load_dataset()

    def _load_dataset(self):
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
        return self.data["prompt"]

    def get_answers(self) -> list:
        return self.data["answer"]

    def get_questions(self) -> list:
        return self.data["question"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_gsm8k_dataset(
    split: str = "train", data_dir: str = "./resources/gsm8k"
) -> Dataset:
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
