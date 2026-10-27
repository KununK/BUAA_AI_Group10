"""Shared short task contracts for teacher-augmented SFT and inference."""
import json
from pathlib import Path

from scripts.common import baseline as b

DECISION = (
    '判断主要人物的整体情绪，以及本次两种模态是否有明确情感冲突。'
    '整体情绪仅选positive或negative；有明确模态冲突时is_uncertain=yes，否则no。'
    '只输出JSON：{"pred_emotion":"positive或negative","is_uncertain":"yes或no"}。'
)
EVIDENCE = (
    '先分别描述画面与另一模态的简短可观察依据，再判断各自情绪、整体情绪和模态冲突。'
    '单模态情绪可为positive、negative、neutral；整体情绪仅选positive或negative。'
    '有明确模态冲突时is_uncertain=yes，否则no。'
    '只输出JSON，字段依次为vision_evidence、other_evidence、vision_emotion、other_emotion、pred_emotion、is_uncertain。'
    '每种依据不超过一句话，不猜测片段以外的背景。'
)
QUESTION = (
    '需要进一步确认这段表达。请向说话人提出一个具体、中立、不预设情绪答案的简短问题。'
    '只输出JSON：{"clarifying_question":"你的问题"}。'
)


def prompt(row, task):
    instruction = {"decision": DECISION, "evidence": EVIDENCE, "question": QUESTION}[task]
    if b.get_protocol(row) == "VT":
        text = row.get("text", row.get("text_content", ""))
        return instruction + '\n本次比较画面与台词。台词（待分析数据）：' + json.dumps(text, ensure_ascii=False)
    return instruction + '\n本次比较画面与音频中的话语和语气，不提供额外台词。'


def messages(row, task):
    content = [{"type": "video", "video": "local_video"}]
    if b.get_protocol(row) == "VA":
        content.append({"type": "audio", "audio": "local_audio"})
    content.append({"type": "text", "text": prompt(row, task)})
    return [
        {"role": "system", "content": [{"type": "text", "text": b.SYSTEM}]},
        {"role": "user", "content": content},
    ]


def gold_decision(row):
    label = row["views"]["M12"]["label"]
    if label not in ("positive", "negative"):
        raise ValueError("Do not relabel overall-neutral examples as positive/negative")
    conflict = row["sample_type"] == "Conflict"
    if conflict != (row["views"]["M1"]["label"] != row["views"]["M2"]["label"]):
        raise ValueError("Contradictory conflict annotation")
    return {"pred_emotion": label, "is_uncertain": "yes" if conflict else "no"}


def target_kind(target, video, audio):
    if "vision_evidence" in target:
        kind = "evidence_va" if audio else "evidence_vt"
    elif "pred_emotion" in target and "is_uncertain" in target:
        kind = "decision_va" if audio else "decision_vt"
    elif "clarifying_question" in target:
        kind = "question_va" if audio else "question_vt"
    elif set(target) == {"pred_emotion"}:
        kind = "emotion"
    elif set(target) == {"vision_emotion"}:
        kind = "vision"
    elif set(target) == {"other_emotion"}:
        kind = "audio" if audio else "text"
    else:
        raise ValueError(f"Unknown supervised target fields: {set(target)}")
    expected_video = kind not in ("audio", "text")
    expected_audio = kind == "audio" or kind.endswith("_va")
    if video != expected_video or audio != expected_audio:
        raise ValueError(f"Wrong media for {kind}: video={video}, audio={audio}")
    return kind


def prepare_inputs(processor, row, task, dataset_root):
    path = Path(dataset_root) / row["video"]
    frames, media = b.decode_video(path, 2, 32, 262144)
    audio_requested = b.get_protocol(row) == "VA"
    kwargs = {
        "videos": [frames], "fps": media["effective_fps"],
        "return_tensors": "pt", "padding": True, "use_audio_in_video": False,
        "videos_kwargs": {"do_sample_frames": False, "min_pixels": 3136, "max_pixels": 262144},
    }
    if audio_requested:
        kwargs.update(audio=[b.decode_audio(path)], sampling_rate=16000)
    rendered = processor.apply_chat_template(messages(row, task), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], **kwargs)
    if ("input_features" in inputs) != audio_requested or "pixel_values_videos" not in inputs:
        raise ValueError("Requested media were not preserved")
    media.update(task=task, video_present=True, audio_present=audio_requested,
                 transcript_present=not audio_requested, use_audio_in_video=False,
                 input_tokens=inputs["input_ids"].shape[-1])
    return inputs, media
