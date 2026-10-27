"""Full-set evaluation of concise decisions or observable-evidence student answers."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from scripts.common import baseline as b
from scripts.common import distill_tasks as tasks
from scripts.common import conflict_rule as questions
from scripts.common.omni_policy import load_policy, generate_text
from scripts.eval.metrics import write_diagnostics


def generation_decision_scores(generated, prompt_length, tokenizer):
    """Read unprocessed next-token logits at the actual sampled yes/no position."""
    labels = ('yes', 'no')
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError('Decision labels must each be one token')
    ids = [tokens[0] for tokens in ids]
    completion = generated.sequences[0, prompt_length:].tolist()
    positions = [i for i, token in enumerate(completion) if token in ids]
    if len(positions) != 1 or generated.logits is None:
        raise ValueError('Decision class position is ambiguous; use a conditional forward')
    logits = generated.logits[positions[0]][0, ids].detach().float()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite decision class logits')
    probabilities = logits.softmax(-1)
    return {'label_logits': dict(zip(labels, logits.cpu().tolist())),
            'probabilities': dict(zip(labels, probabilities.cpu().tolist())),
            'score_source': 'raw_logits_from_actual_greedy_generation'}


def generate_scored_decision(model, processor, inputs, media):
    # Identical greedy decoding settings to generate_text; retain raw logits.
    inputs = inputs.to('cuda:0').to(torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, use_audio_in_video=media['use_audio_in_video'],
            do_sample=False, max_new_tokens=96, eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, return_dict_in_generate=True, output_logits=True)
    length = inputs['input_ids'].shape[-1]
    raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    try:
        scores = generation_decision_scores(generated, length, processor.tokenizer)
    except ValueError:
        scores = None
    return raw, scores


def parse_answer(raw, evidence=False):
    for value in b.json_objects(raw):
        if value.get("pred_emotion") not in ("positive", "negative") or value.get("is_uncertain") not in ("yes", "no"):
            continue
        if evidence and any(value.get(k) not in b.EMOTIONS for k in ("vision_emotion", "other_emotion")):
            continue
        if evidence and any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("vision_evidence", "other_evidence")):
            continue
        return value
    raise ValueError("Invalid decision/evidence JSON")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--style", choices=["decision", "evidence"], required=True)
    parser.add_argument("--conflict-rule", choices=["explicit", "disagreement"], default="explicit")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--dataset-root", type=Path, default=b.ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-source", choices=["student", "base"], default="student")
    parser.add_argument("--compare-emotion-reference", type=Path)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.conflict_rule == "disagreement" and args.style != "evidence":
        parser.error("disagreement requires evidence-style auxiliary labels")
    args.label_set, args.emotion_reference = "binary", None
    source = args.dataset_root / ("val.jsonl" if args.split == "val" else "test_release.jsonl")
    rows = b.read_jsonl(source)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = {
        "split": args.split, "adapter": str(args.adapter.resolve()), "style": args.style,
        "conflict_rule": args.conflict_rule, "question_source": args.question_source,
        "adapter_sha256": hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_ids": [r["sample_id"] for r in rows],
        "fps": 2, "max_frames": 32, "max_pixels": 262144, "audio_sampling_rate": 16000,
        "VA_external_transcript": False, "use_audio_in_video": False,
        "decision_confidence": "unprocessed yes/no logits from the actual greedy answer; conditional retry if ambiguous" if args.style == 'decision' else None,
        "source_sha256": {str(p.relative_to(b.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__).resolve(), Path(tasks.__file__), b.ROOT / "scripts/common/omni_policy.py")},
    }
    if (out / "run_config.json").exists() and json.loads((out / "run_config.json").read_text()) != config:
        raise ValueError("Evaluation configuration differs; preserve this run and use another output directory")
    b.write_json(out / "run_config.json", config)
    path = out / "predictions.jsonl"
    previous = b.read_jsonl(path) if path.exists() else []
    if args.retry_errors:
        previous = [r for r in previous if r.get("prediction")]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous))
    done = {r["sample_id"] for r in previous}
    if len(done) != len(previous) or done - {r["sample_id"] for r in rows}:
        raise ValueError("Unexpected/duplicate evaluation IDs")
    remaining = [r for r in rows if r["sample_id"] not in done]
    if remaining:
        model, processor, environment = load_policy(args.adapter)
        b.write_json(out / "environment.json", environment)
        b.write_json(out / "status.json", {"status": "running", "samples": len(rows)})
        with path.open("a", encoding="utf-8", buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                record = {"sample_id": row["sample_id"], "video": row["video"],
                          "protocol": b.get_protocol(row), "task": args.style, "label_set": "binary"}
                try:
                    inputs, media = tasks.prepare_inputs(processor, row, args.style, args.dataset_root)
                    if args.style == 'decision':
                        raw, decision_scores = generate_scored_decision(model, processor, inputs, media)
                    else:
                        raw = generate_text(model, processor, inputs, media, max_tokens=384)
                    answer = parse_answer(raw, args.style == "evidence")
                    if args.style == 'decision':
                        if decision_scores is None:
                            from scripts.eval.decision_scores import decision_prefix
                            from scripts.eval.probabilities import conditional_probabilities
                            decision_scores = conditional_probabilities(model, processor.tokenizer, inputs,
                                media, decision_prefix(answer['pred_emotion']), ('yes', 'no'))
                            decision_scores['score_source'] = 'conditional_forward_on_student_emotion'
                        record['decision_uncertainty'] = decision_scores
                    uncertain = answer["is_uncertain"]
                    auxiliary = {}
                    if args.style == "evidence":
                        auxiliary = {k: answer[k] for k in ("vision_emotion", "other_emotion")}
                        if args.conflict_rule == "disagreement":
                            uncertain = questions.rule_uncertainty(auxiliary)
                    del inputs
                    question, question_origin, raw_question = "none", "not_required", None
                    if uncertain == "yes":
                        inputs, question_media = tasks.prepare_inputs(processor, row, "question", args.dataset_root)
                        with model.disable_adapter() if args.question_source == "base" else nullcontext():
                            raw_question = generate_text(model, processor, inputs, question_media, max_tokens=128)
                        try:
                            question = questions.parse_question(raw_question)
                            question_origin = args.question_source
                        except ValueError:
                            # A genuine base-model retry; never fabricate a question to pass validation.
                            with model.disable_adapter():
                                raw_question = generate_text(model, processor, inputs, question_media, max_tokens=192)
                            question = questions.parse_question(raw_question)
                            question_origin = "base_retry"
                        del inputs
                    response = ""
                    if args.style == "evidence":
                        response = "画面：" + answer["vision_evidence"] + "；另一模态：" + answer["other_evidence"]
                    record.update(raw_output=raw, parsed_answer=answer, auxiliary_prediction=auxiliary,
                                  media=media, raw_question=raw_question, question_origin=question_origin,
                                  prediction={"pred_emotion": answer["pred_emotion"], "is_uncertain": uncertain,
                                              "clarifying_question": question, "response": response})
                except torch.OutOfMemoryError:
                    raise
                except Exception as error:
                    record.update(prediction=None, error_type=type(error).__name__, error=str(error))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                previous.append(record)
                print(f"[{index}/{len(remaining)}] {row['sample_id']} {record.get('prediction') or record.get('error')}", flush=True)
    metric = b.finalize(args, rows, rows, previous, out)
    if args.split == "val":
        if args.style == "evidence": write_diagnostics(rows, previous, out)
        if args.compare_emotion_reference:
            reference = b.read_emotion_reference(args.compare_emotion_reference, rows, b.LABEL_SETS["binary"])
            hybrid = [{**r, "prediction": {**r["prediction"], "pred_emotion": reference[r["sample_id"]]}}
                      if r.get("prediction") else r for r in previous]
            b.write_json(out / "metrics_frozen_emotion.json", b.score_rows(rows, hybrid))
    print(json.dumps(metric, ensure_ascii=False, indent=2))
    if metric["valid_predictions"] != len(rows):
        raise SystemExit("Incomplete predictions; inspect failures before reporting a score")
    if args.style == 'decision':
        if any('decision_uncertainty' not in r for r in previous):
            raise ValueError('Incomplete decision class scores')
        with (out / 'probabilities.jsonl').open('w', encoding='utf-8') as handle:
            for record in previous:
                handle.write(json.dumps({'sample_id': record['sample_id'], 'protocol': record['protocol'],
                    'decision_uncertainty': record['decision_uncertainty']}, ensure_ascii=False) + '\n')
        if args.split == 'val':
            from scripts.eval.calibration import main as calibrate
            calibrate(['--probability-run', str(out), '--emotion-run', str(out),
                       '--output-dir', str(out / 'calibration')])


if __name__ == "__main__":
    main()
