"""Apply the cached modality-disagreement rule; generate only missing questions."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from scripts.common import baseline as b

QUESTION_SYSTEM = "你是情感澄清助手。"
QUESTION_TASK = """根据下面的模态情绪判断，向说话人提出一个简短、中立、不预设答案的澄清问题。
只输出 JSON：{"clarifying_question":"你的问题"}
"""


def rule_uncertainty(auxiliary):
    if any(auxiliary.get(k) not in b.EMOTIONS for k in ("vision_emotion", "other_emotion")):
        raise ValueError("Missing or invalid cached modality labels")
    return "yes" if auxiliary["vision_emotion"] != auxiliary["other_emotion"] else "no"


def question_messages(row, auxiliary):
    protocol = b.get_protocol(row)
    data = {"vision_emotion": auxiliary["vision_emotion"],
            "other_modality": "台词" if protocol == "VT" else "音频话语与语气",
            "other_emotion": auxiliary["other_emotion"]}
    if protocol == "VT":
        data["text"] = row.get("text", row.get("text_content", ""))
    return [
        {"role": "system", "content": [{"type": "text", "text": QUESTION_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": QUESTION_TASK + json.dumps(data, ensure_ascii=False)}]},
    ]


def parse_question(raw):
    for value in b.json_objects(raw):
        question = value.get("clarifying_question")
        if isinstance(question, str) and question.strip() and question.strip().lower() != "none":
            return question.strip()
    raise ValueError("Model did not generate a nonempty clarifying question")


def rule_record(row, source, emotion):
    auxiliary = source["auxiliary_prediction"]
    uncertain = rule_uncertainty(auxiliary)
    old = source["prediction"]
    reuse = uncertain == "yes" and old["is_uncertain"] == "yes"
    pending = uncertain == "yes" and not reuse
    record = {
        "sample_id": row["sample_id"], "video": row["video"], "protocol": b.get_protocol(row),
        "input_mode": source.get("input_mode", "protocol"), "task": "conflict_rule",
        "auxiliary_prediction": dict(auxiliary), "source_prediction": dict(old),
        "media": source.get("media", {}), "media_reused": True,
        "fixed_emotion": emotion, "rule_uncertainty": uncertain,
        "question_origin": "pending" if pending else "reused" if reuse else "not_required",
        "prediction": None if pending else {
            "pred_emotion": emotion, "is_uncertain": uncertain,
            "clarifying_question": old["clarifying_question"] if reuse else "none", "response": "",
        },
    }
    return record, pending


def finish(args, rows, records, out):
    valid, invalid = b.checked_predictions(rows, records)
    by_id = {r["sample_id"]: r for r in records}
    inconsistent = [sid for sid, p in valid.items()
                    if p["is_uncertain"] != rule_uncertainty(by_id[sid]["auxiliary_prediction"])]
    b.write_json(out / "rule_validation.json", {
        "samples": len(rows), "valid_predictions": len(valid), "inconsistent_with_rule": len(inconsistent),
        "missing_or_invalid_predictions": len(invalid), "all_rule_decisions_valid": not inconsistent and not invalid,
    })
    if inconsistent:
        raise ValueError("Saved predictions do not follow the modality-disagreement rule")
    return b.finalize(args, rows, rows, records, out)


def run(args, rows, rows_path, out):
    source_config = getattr(args, "source_config", {})
    dataset_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if source_config.get("dataset_sha256") and source_config["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset differs from the source run")
    source_path = args.source_run / "predictions.jsonl"
    sources = b.read_jsonl(source_path)
    valid, invalid = b.checked_predictions(rows, sources)
    if invalid or len(sources) != len(rows):
        raise ValueError("Source run must contain complete, valid predictions for exactly this split")
    reference = b.read_emotion_reference(args.emotion_reference, rows, b.LABEL_SETS[args.label_set])
    source_map = {r["sample_id"]: r for r in sources}
    planned, pending = [], []
    for row in rows:
        source = source_map[row["sample_id"]]
        if source["video"] != row["video"] or source["protocol"] != b.get_protocol(row):
            raise ValueError("Cached media/protocol does not match the selected dataset")
        if valid[row["sample_id"]]["pred_emotion"] != reference[row["sample_id"]]:
            raise ValueError("Source JSONL and emotion reference CSV disagree")
        record, needs_question = rule_record(row, source, reference[row["sample_id"]])
        record["label_set"] = args.label_set
        planned.append(record)
        if needs_question:
            pending.append(row)
    plan = {
        "samples": len(rows), "rule": "vision_emotion != other_emotion => yes; otherwise no",
        "uncertain_yes": sum(r["rule_uncertainty"] == "yes" for r in planned),
        "uncertain_no": sum(r["rule_uncertainty"] == "no" for r in planned),
        "reused_questions": sum(r["question_origin"] == "reused" for r in planned),
        "questions_to_generate": len(pending), "media_reinference": False,
    }
    manifest = {
        "task": "cached_conflict_rule", "split": args.split, "source_run": str(args.source_run),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "emotion_reference_sha256": hashlib.sha256(args.emotion_reference.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_hash,
        "input_mode": source_config.get("input_mode", "protocol"),
        "fps": source_config.get("fps"), "max_frames": source_config.get("max_frames"),
        "max_pixels": source_config.get("max_pixels"),
        "question_input_mode": "text_from_cached_modality_labels", "media_reinference": False,
        "model": str(Path(args.model).resolve()), "label_set": args.label_set,
        "question_system": QUESTION_SYSTEM, "question_task": QUESTION_TASK,
        "max_new_tokens": args.max_new_tokens, "attention": args.attention, "seed": args.seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
    }
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != manifest:
        raise ValueError("Output configuration differs; use a new output directory")
    b.write_json(config_path, manifest)
    b.write_json(out / "rule_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.mode == "plan":
        if pending:
            b.write_json(out / "question_example.json", question_messages(pending[0], source_map[pending[0]["sample_id"]]["auxiliary_prediction"]))
        b.write_json(out / "status.json", {"status": "planned", **plan})
        return
    prediction_path = out / "predictions.jsonl"
    if args.mode == "score":
        print(json.dumps(finish(args, rows, b.read_jsonl(prediction_path), out), ensure_ascii=False, indent=2))
        return
    previous = b.read_jsonl(prediction_path) if prediction_path.exists() else []
    if args.retry_errors:
        previous = [r for r in previous if r.get("prediction")]
        prediction_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous), encoding="utf-8")
    done = {r["sample_id"] for r in previous}
    if len(done) != len(previous) or done - source_map.keys():
        raise ValueError("Duplicate or unexpected saved prediction IDs")
    planned_map = {r["sample_id"]: r for r in planned}
    with prediction_path.open("a", encoding="utf-8", buffering=1) as f:
        for record in planned:
            if record["sample_id"] not in done and record["prediction"] is not None:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                previous.append(record)
                done.add(record["sample_id"])
    remaining = [r for r in pending if r["sample_id"] not in done]
    b.write_json(out / "status.json", {"status": "running", "questions_remaining": len(remaining)})
    if not remaining:
        result = finish(args, rows, previous, out)
        if not (out / "timing.json").exists():
            b.write_json(out / "timing.json", {"seconds_this_invocation": 0, "questions_this_invocation": 0,
                         "cached_samples": len(rows)})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    import torch
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Question generation requires a CUDA GPU with BF16; run scripts/run_conflict_rule.sh in the GPU terminal")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    config = Qwen2_5OmniConfig.from_pretrained(args.model, local_files_only=True)
    model, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=config.thinker_config, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=args.attention, local_files_only=True, output_loading_info=True,
    )
    b.write_json(out / "weight_loading.json", loading)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError("Question model weights did not load exactly")
    b.write_json(out / "environment.json", {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda})
    model.eval()
    successes, started = 0, time.monotonic()
    with prediction_path.open("a", encoding="utf-8", buffering=1) as f:
        for index, row in enumerate(remaining, 1):
            record = dict(planned_map[row["sample_id"]])
            t0 = time.monotonic()
            inputs = generated = None
            try:
                messages = question_messages(row, record["auxiliary_prediction"])
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda:0").to(torch.bfloat16)
                with torch.inference_mode():
                    generated = model.generate(**inputs, use_audio_in_video=False, do_sample=False,
                                               max_new_tokens=args.max_new_tokens,
                                               eos_token_id=processor.tokenizer.eos_token_id,
                                               pad_token_id=processor.tokenizer.pad_token_id)
                ids = generated[:, inputs["input_ids"].shape[-1]:]
                raw = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                record["raw_question_output"] = raw
                record["question_input_tokens"] = int(inputs["input_ids"].shape[-1])
                record["question_output_tokens"] = int(ids.shape[-1])
                record["prediction"] = b.parse_prediction(json.dumps({
                    "pred_emotion": record["fixed_emotion"], "is_uncertain": record["rule_uncertainty"],
                    "clarifying_question": parse_question(raw),
                }, ensure_ascii=False), b.LABEL_SETS[args.label_set])
                record["question_origin"] = "generated"
                successes += 1
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                record.update(prediction=None, question_origin="failed", error_type=type(e).__name__, error=str(e))
            finally:
                del inputs, generated
            record["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            previous.append(record)
            print(f"[{index}/{len(remaining)}] {row['sample_id']} {record.get('prediction') or record.get('error')}", flush=True)
            if index >= 3 and successes == 0:
                raise RuntimeError("First three question generations failed; inspect errors before continuing")
    result = finish(args, rows, previous, out)
    b.write_json(out / "timing.json", {"seconds_this_invocation": time.monotonic() - started,
                 "questions_this_invocation": len(remaining), "cached_samples": len(rows) - len(remaining)})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Completed conflict run containing cached auxiliary labels")
    parser.add_argument("--split", choices=["val", "val_tmp", "test"], default="val")
    parser.add_argument("--mode", choices=["infer", "plan", "score"], default="infer")
    parser.add_argument("--dataset-root", type=Path, default=b.ROOT / "Dataset")
    parser.add_argument("--model", default=b.MODEL)
    parser.add_argument("--run-name", default="conflict_rule_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    args.source_run = args.source_run.resolve()
    source_config = json.loads((args.source_run / "run_config.json").read_text())
    args.source_config = source_config
    if source_config["split"] != args.split:
        parser.error("Source run split does not match --split")
    if args.mode == "score" and (args.output_dir is None or args.split == "test"):
        parser.error("score requires --output-dir and a labeled split")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    args.label_set = source_config.get("label_set", "ternary")
    args.emotion_reference = args.source_run / "predictions.csv"
    if args.output_dir and args.output_dir.resolve() == args.source_run:
        parser.error("Use a separate output directory to preserve the source run")
    rows_path = args.dataset_root / ("test_release.jsonl" if args.split == "test" else f"{args.split}.jsonl")
    rows = b.read_jsonl(rows_path)
    out = b.output_directory(args.run_name, args.split, args.output_dir)
    with (out / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(b.Tee(sys.stdout, log)), redirect_stderr(b.Tee(sys.stderr, log)):
            print(f"Output directory: {out}", flush=True)
            run(args, rows, rows_path, out)


if __name__ == "__main__":
    main()
