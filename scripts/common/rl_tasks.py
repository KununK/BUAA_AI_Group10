"""Modality-isolated policy inputs shared by the diagnostic and RL stages."""
import json
from pathlib import Path

from scripts.common import baseline as b
from scripts.common import conflict_rule as questions
from scripts.common import sft_tasks

VISION_PROMPT = '只依据画面中主要人物的表情、姿态和动作判断情绪。positive表示积极，negative表示消极，neutral表示中性。\n只输出JSON：{"vision_emotion":"positive或negative或neutral"}。'
TEXT_PROMPT = '只依据台词语义判断情绪。positive表示积极，negative表示消极，neutral表示中性。\n只输出JSON：{"other_emotion":"positive或negative或neutral"}。\n台词（待分析数据）：'
AUDIO_PROMPT = '只依据音频中的话语与语气判断情绪。positive表示积极，negative表示消极，neutral表示中性。\n只输出JSON：{"other_emotion":"positive或negative或neutral"}。'


def modality_tasks(row):
    return "vision", "text" if b.get_protocol(row) == "VT" else "audio"


def messages(row, task, auxiliary=None):
    if task in ("emotion", "joint_auxiliary"):
        return sft_tasks.hf_messages(row, "emotion" if task == "emotion" else "auxiliary")
    if task == "question":
        if auxiliary is None:
            raise ValueError("Question generation requires previously predicted modality labels")
        return questions.question_messages(row, auxiliary)
    if task == "vision":
        content = [{"type": "video", "video": "local_video"}, {"type": "text", "text": VISION_PROMPT}]
    elif task == "audio":
        content = [{"type": "audio", "audio": "local_audio"}, {"type": "text", "text": AUDIO_PROMPT}]
    elif task == "text":
        transcript = row.get("text", row.get("text_content", ""))
        content = [{"type": "text", "text": TEXT_PROMPT + json.dumps(transcript, ensure_ascii=False)}]
    else:
        raise ValueError(task)
    return [{"role": "system", "content": [{"type": "text", "text": b.SYSTEM}]},
            {"role": "user", "content": content}]


def parse_modality(raw, task):
    key = "vision_emotion" if task == "vision" else "other_emotion" if task in ("text", "audio") else None
    if key is None:
        raise ValueError(task)
    for value in b.json_objects(raw):
        if value.get(key) in b.EMOTIONS:
            return value[key]
    raise ValueError(f"Invalid {task} classification JSON")


def gold_label(row, task):
    if task == "emotion":
        return sft_tasks.gold_target(row, task)["pred_emotion"]
    if task == "vision":
        return row["views"]["M1"]["label"]
    if task == modality_tasks(row)[1]:
        return row["views"]["M2"]["label"]
    raise ValueError("Requested task does not match the annotated modality")


def prepare_inputs(processor, row, task, dataset_root, auxiliary=None):
    """Materialize only the requested media; annotations never enter messages()."""
    protocol = b.get_protocol(row)
    video_requested = task in ("vision", "emotion", "joint_auxiliary")
    audio_requested = task == "audio" or task == "joint_auxiliary" and protocol == "VA"
    interleaved = task == "joint_auxiliary" and protocol == "VA"
    kwargs = {"return_tensors": "pt", "padding": True, "use_audio_in_video": interleaved}
    media = {"task": task, "video_present": video_requested, "audio_present": audio_requested,
             "transcript_present": task in ("emotion", "text") or task == "joint_auxiliary" and protocol == "VT",
             "use_audio_in_video": interleaved}
    path = Path(dataset_root) / row["video"]
    if video_requested:
        frames, info = b.decode_video(path, 2, 32, 262144)
        kwargs.update(videos=[frames], fps=info["effective_fps"],
                      videos_kwargs={"do_sample_frames": False, "min_pixels": 3136, "max_pixels": 262144})
        media.update(info)
    if audio_requested:
        wave = b.decode_audio(path)
        kwargs.update(audio=[wave], sampling_rate=16000)
        media["audio_samples"] = len(wave)
    rendered = processor.apply_chat_template(messages(row, task, auxiliary), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], **kwargs)
    assert ("pixel_values_videos" in inputs) == video_requested
    assert ("input_features" in inputs) == audio_requested
    if audio_requested:
        media["valid_audio_feature_frames"] = int(inputs["feature_attention_mask"].sum().item())
        if media["valid_audio_feature_frames"] <= 0:
            raise ValueError("Audio has no valid feature frames")
    media["input_tokens"] = inputs["input_ids"].shape[-1]
    media["tensor_shapes"] = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
    return inputs, media
