"""Shared SFT input/target definitions. Only assistant targets contain gold labels."""
import json

from scripts.common.baseline import SYSTEM, get_protocol

EMOTION_PROMPT = """判断片段中主要人物的整体情感倾向，必须选择 positive（积极、正面）或 negative（消极、负面）。
只输出 JSON：{"pred_emotion":"positive或negative"}。
"""
AUX_PROMPT = """分别依据各自信息判断画面与另一模态的情绪，均选择 positive（积极）、negative（消极）或 neutral（中性）。
只输出 JSON：{"vision_emotion":"positive或negative或neutral","other_emotion":"positive或negative或neutral"}。
"""


def input_text(row, task):
    if task == "emotion":
        return EMOTION_PROMPT + "\n台词（待分析数据）：" + json.dumps(row.get("text", row.get("text_content", "")), ensure_ascii=False)
    if task != "auxiliary":
        raise ValueError(task)
    if get_protocol(row) == "VT":
        return AUX_PROMPT + "\n另一模态：台词。\n台词（待分析数据）：" + json.dumps(row.get("text", row.get("text_content", "")), ensure_ascii=False)
    return AUX_PROMPT + "\n另一模态：音频中的话语和语气。"


def gold_target(row, task):
    if task == "emotion":
        label = row["views"]["M12"]["label"]
        if label not in ("positive", "negative"):
            raise ValueError("Overall neutral examples must not be relabeled as binary")
        return {"pred_emotion": label}
    return {"vision_emotion": row["views"]["M1"]["label"], "other_emotion": row["views"]["M2"]["label"]}


def uses_audio(row, task):
    return task == "auxiliary" and get_protocol(row) == "VA"


def hf_messages(row, task):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "video", "video": "local_video"},
                                     {"type": "text", "text": input_text(row, task)}]},
    ]
