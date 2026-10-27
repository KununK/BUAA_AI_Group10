"""Build gold-anchored teacher SFT/control datasets and native LF token caches."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

import yaml

from scripts.common import baseline as b
from scripts.common import distill_tasks as tasks
from scripts.data.audio import audio_path, export_audio
from scripts.data.prepare_sft import build_records

OUTPUT = b.ROOT / "outputs/distill_omni_v3"
DATA = b.ROOT / "Dataset/processed/distill_v3"
TEACHER = b.ROOT / "Dataset/codex_train_annotation_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(row, task, target, tag):
    audio = b.get_protocol(row) == "VA"
    return {
        "record_id": f"{row['sample_id']}:{tag}", "sample_id": row["sample_id"], "task": task,
        "system": b.SYSTEM,
        "messages": [
            {"role": "user", "content": "<video>" + ("<audio>" if audio else "") + tasks.prompt(row, task)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
        "videos": [str((b.ROOT / "Dataset" / row["video"]).resolve())],
        "audios": [str(audio_path(b.ROOT / "Dataset/sft", row).resolve())] if audio else [],
    }


def question_target(text):
    if not isinstance(text, str) or not text.strip() or text == "none":
        return None
    # This generic third-person meta-question is not a useful speaker-facing target.
    if text == "这段话的语气与说话人的画面表情，应如何结合理解？":
        return None
    text = re.sub(r"^说(.+)时，说话人实际想表达什么感受？$", r"你说\1时，主要想表达什么感受？", text)
    return {"clarifying_question": text}


def prepare():
    if (DATA / "preparation.json").exists():
        raise ValueError("Prepared dataset already exists; do not overwrite an active experiment")
    rows = b.read_jsonl(b.ROOT / "Dataset/train.jsonl")
    val = b.read_jsonl(b.ROOT / "Dataset/val.jsonl")
    for field in ("sample_id", "video", "split_group_id"):
        if {r[field] for r in rows} & {r[field] for r in val}:
            raise ValueError(f"Train/val overlap: {field}")
    by_id = {r["sample_id"]: r for r in rows}
    for row in rows:
        if b.get_protocol(row) == "VA":
            export_audio((b.ROOT / "Dataset" / row["video"], audio_path(b.ROOT / "Dataset/sft", row)))
    base = build_records(rows, b.ROOT / "Dataset")
    common = list(base)
    decision_rows = 0
    for row in rows:
        if row["views"]["M12"]["label"] == "neutral":
            continue
        for repeat in range(2 if row["sample_type"] == "Conflict" else 1):
            common.append(record(row, "decision", tasks.gold_decision(row), f"gold_decision:{repeat}"))
            decision_rows += 1
    labels = b.read_jsonl(TEACHER / "distillation/sft_labels_binary_filtered.jsonl")
    evidence = b.read_jsonl(TEACHER / "distillation/sft_evidence_filtered.jsonl")
    question_count = 0
    for teacher in labels:
        row = by_id[teacher["metadata"]["sample_id"]]
        target = json.loads(teacher["messages"][-1]["content"])
        if {k: target[k] for k in ("pred_emotion", "is_uncertain")} != tasks.gold_decision(row):
            raise ValueError("Filtered teacher labels differ from original gold")
        if teacher["metadata"]["needs_review"]:
            raise ValueError("Unreviewed teacher record")
        if target["is_uncertain"] == "yes":
            question = question_target(target["clarifying_question"])
            if question:
                common.append(record(row, "question", question, "teacher_question"))
                question_count += 1
    augmented, control = deepcopy(common), deepcopy(common)
    evidence_count = 0
    for teacher in evidence:
        row = by_id[teacher["metadata"]["sample_id"]]
        if row["views"]["M12"]["label"] == "neutral":
            continue
        target = json.loads(teacher["messages"][-1]["content"])
        gold = tasks.gold_decision(row)
        if any(target[k] != gold[k] for k in gold) or target["vision_emotion"] != row["views"]["M1"]["label"] or target["other_emotion"] != row["views"]["M2"]["label"]:
            raise ValueError("Unverified evidence labels")
        visual = "；".join(o["observation"] for o in target["observations"] if o["modality"] == "vision")
        other = "；".join(o["observation"] for o in target["observations"] if o["modality"] != "vision")
        if not visual or not other:
            raise ValueError("Evidence must cover both annotated modalities")
        compact = {"vision_evidence": visual, "other_evidence": other,
                   "vision_emotion": target["vision_emotion"], "other_emotion": target["other_emotion"], **gold}
        for repeat in range(4):
            augmented.append(record(row, "evidence", compact, f"teacher_evidence:{repeat}"))
            control.append(record(row, "decision", gold, f"teacher_evidence:{repeat}"))
        evidence_count += 1
    if len(augmented) != len(control):
        raise AssertionError("Control must match sample/media exposure")
    order = list(range(len(augmented)))
    random.Random(20261026).shuffle(order)
    first = next(i for i in order if augmented[i]["videos"] and augmented[i]["audios"])
    order.remove(first); order.insert(0, first)
    registry, summaries = {}, {}
    for name, records in (("distill_train", augmented), ("control_train", control)):
        records = [records[i] for i in order]
        if len({r["record_id"] for r in records}) != len(records):
            raise ValueError("Duplicate record IDs")
        counts = Counter()
        for r in records:
            target = json.loads(r["messages"][-1]["content"])
            counts[tasks.target_kind(target, bool(r["videos"]), bool(r["audios"]))] += 1
            for media in r["videos"] + r["audios"]:
                if not Path(media).is_file(): raise FileNotFoundError(media)
        b.write_json(DATA / f"{name}.json", records)
        registry[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                          "columns": {k: k for k in ("messages", "system", "videos", "audios")},
                          "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
        summaries[name] = {"records": len(records), "task_counts": dict(counts), "file_sha256": sha(DATA / f"{name}.json")}
    b.write_json(DATA / "dataset_info.json", registry)
    report = {"original_train_rows": len(rows), "original_four_task_records": len(base),
              "gold_decision_records": decision_rows, "teacher_question_records": question_count,
              "teacher_evidence_unique_binary_records": evidence_count, "evidence_repetitions": 4,
              "datasets": summaries, "control_media_and_sample_order_matched": True,
              "train_targets_from_val_or_test": False,
              "classification_targets": "Original train gold; teacher disagreements/neutral M12 excluded from evidence",
              "teacher_evidence": "Observable short evidence only; omit timestamps, unverified ASR, self-confidence and hidden thoughts",
              "source_sha256": {str(p.relative_to(b.ROOT)): sha(p) for p in (
                  b.ROOT / "Dataset/train.jsonl", b.ROOT / "Dataset/val.jsonl",
                  TEACHER / "distillation/sft_labels_binary_filtered.jsonl", TEACHER / "distillation/sft_evidence_filtered.jsonl")}}
    b.write_json(DATA / "preparation.json", report)
    b.write_json(OUTPUT / "preparation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def fingerprint(config):
    files = [DATA / "dataset_info.json", DATA / (config["dataset"] + ".json"),
             Path(__file__).resolve(), b.ROOT / "scripts/common/distill_tasks.py"]
    files += [b.ROOT / "LlamaFactory/src/llamafactory/data" / name for name in
              ("template.py", "mm_plugin.py", "converter.py", "processor/supervised.py", "collator.py")]
    keys = ("model_name_or_path", "template", "cutoff_len", "train_on_prompt", "packing", "video_fps", "video_maxlen", "video_max_pixels", "video_min_pixels", "audio_sampling_rate", "use_audio_in_video")
    return {"files": {str(p.relative_to(b.ROOT)): sha(p) for p in files},
            "preprocessing": {key: config[key] for key in keys}}


def validate_cache(config):
    path = DATA / f"{config['dataset']}_tokenization.json"
    if not path.is_file() or json.loads(path.read_text())["fingerprint"] != fingerprint(config):
        raise ValueError("Token cache missing/stale; run this dataset's tokenize stage")


def tokenize(config):
    from datasets import load_from_disk
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    fp = fingerprint(config)
    cache = Path(config["tokenized_path"])
    if cache.exists():
        validate_cache(config)
    cpu = dict(config, do_train=False, do_eval=False, bf16=False, tf32=False, use_cpu=True,
               eval_strategy="no", load_best_model_at_end=False, report_to="none")
    model_args, data_args, training_args, _, _ = get_train_args(cpu)
    module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(module["tokenizer"], data_args)
    if not cache.exists():
        get_dataset(template, model_args, data_args, training_args, "sft", **module)
    dataset = load_from_disk(str(cache))["train"]
    source = json.loads((DATA / (config["dataset"] + ".json")).read_text())
    expected = Counter(json.dumps(json.loads(r["messages"][-1]["content"]), sort_keys=True) for r in source)
    actual, counts, lengths = Counter(), Counter(), []
    for row in dataset:
        if not (len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])):
            raise AssertionError("Token/label/mask lengths differ")
        target = json.loads(module["tokenizer"].decode([t for t in row["labels"] if t != -100], skip_special_tokens=True).strip())
        actual[json.dumps(target, sort_keys=True)] += 1
        counts[tasks.target_kind(target, bool(row.get("videos")), bool(row.get("audios")))] += 1
        lengths.append(len(row["input_ids"]))
    if actual != expected or len(dataset) != len(source) or max(lengths) >= config["cutoff_len"]:
        raise ValueError("Dropped/truncated/altered targets in tokenized training data")
    result = {"fingerprint": fp, "records": len(dataset), "task_counts": dict(counts),
              "maximum_tokens": max(lengths), "p95_tokens": sorted(lengths)[int(.95*(len(lengths)-1))],
              "all_targets_match_source": True, "gpu_training_started": False}
    b.write_json(DATA / f"{config['dataset']}_tokenization.json", result)
    b.write_json(OUTPUT / f"{config['dataset']}_tokenization.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "tokenize"], required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        if not args.config: parser.error("tokenize requires --config")
        tokenize(yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
