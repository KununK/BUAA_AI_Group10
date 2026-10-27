"""Video-only perception control; frozen Omni emotion/audio/text stay unchanged."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from scripts.common import baseline as b
from scripts.common import rl_tasks as tasks
from scripts.eval.evaluate import generation_label_scores


def prepare_video(processor, row, dataset_root):
    frames, media = b.decode_video(dataset_root / row['video'], 2, 32, 262144)
    indices = np.rint(np.linspace(0, media['source_frames'] - 1, len(frames))).astype(int).tolist()
    metadata = {'total_num_frames': media['source_frames'],
                'fps': media['source_frames'] / media['duration_seconds'],
                'duration': media['duration_seconds'], 'frames_indices': indices}
    text = processor.apply_chat_template(tasks.messages(row, 'vision'), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    # Qwen3-VL's video pixel limit is a TOTAL clip budget, unlike Omni's per-frame limit.
    inputs = processor(text=[text], videos=[frames], return_tensors='pt', padding=True,
        videos_kwargs={'do_sample_frames': False, 'video_metadata': [metadata],
                       'size': {'shortest_edge': 4096 * len(frames), 'longest_edge': 262144 * len(frames)}})
    if 'pixel_values_videos' not in inputs or 'input_features' in inputs:
        raise ValueError('The specialist must receive video and no audio')
    media.update(video_present=True, audio_present=False, transcript_present=False,
                 source_frame_indices=indices, source_average_fps=metadata['fps'],
                 input_tokens=inputs['input_ids'].shape[-1], video_grid_thw=inputs['video_grid_thw'].tolist())
    return inputs, media


def finish(rows, records, reference_dir, out):
    by_id = {r['sample_id']: r for r in records}
    if len(records) != len(rows) or len(by_id) != len(rows) or any(r.get('vision_emotion') not in b.EMOTIONS for r in records):
        raise ValueError('Incomplete visual predictions; inspect errors before explicitly retrying')
    matrix = {label: {value: 0 for value in b.EMOTIONS} for label in b.EMOTIONS}
    for row in rows:
        matrix[row['views']['M1']['label']][by_id[row['sample_id']]['vision_emotion']] += 1
    metrics = {'samples': len(rows), 'unique_videos': len({r['video'] for r in rows}),
        'vision_accuracy': sum(matrix[v][v] for v in b.EMOTIONS) / len(rows),
        'prediction_counts': dict(Counter(r['vision_emotion'] for r in records)), 'vision_confusion': matrix,
        'submission_ready': False, 'questions_for_new_uncertainty_generated': False,
        'comparison_scope': 'Replace M1 only; Omni M12 and protocol-specific M2 are frozen'}
    reference = b.read_jsonl(reference_dir / 'predictions.jsonl')
    if len(reference) != len(rows) or {r['sample_id'] for r in reference} != set(by_id):
        raise ValueError('Frozen reference must cover this exact full validation set')
    combined = []
    for record in reference:
        sid = record['sample_id']
        uncertain = by_id[sid]['vision_emotion'] != record['auxiliary_prediction']['other_emotion']
        combined.append({'sample_id': sid, 'prediction': {
            'pred_emotion': record['prediction']['pred_emotion'], 'is_uncertain': 'yes' if uncertain else 'no'}})
    scored = b.score_rows(rows, combined)
    metrics['frozen_Omni_with_replaced_M1'] = {k: scored[k] for k in ('macro_f1_sklearn_union',
        'conflict_recognition_rate', 'aligned_calibration_rate', 'score_sklearn_union')}
    b.write_json(out / 'vision_metrics.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--omni-reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    source = args.dataset_root / 'val.jsonl'; rows = b.read_jsonl(source)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    reference_config = json.loads((args.omni_reference / 'run_config.json').read_text())
    if reference_config['split'] != 'val' or reference_config['dataset_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Omni reference uses another dataset')
    manifest = {'split': 'val', 'model': str(args.model.resolve()),
        'model_config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
        'adapter': str(args.adapter.resolve()) if args.adapter else None,
        'adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() if args.adapter else None,
        'dataset_sha256': reference_config['dataset_sha256'], 'vision_input': 'video only, no transcript, no audio',
        'fps': 2, 'max_frames': 32, 'max_pixels_per_frame': 262144,
        'omni_reference_predictions_sha256': hashlib.sha256((args.omni_reference / 'predictions.jsonl').read_bytes()).hexdigest(),
        'source_sha256': {str(p.relative_to(b.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), b.ROOT / 'scripts/common/rl_tasks.py',
                      b.ROOT / 'scripts/common/baseline.py', b.ROOT / 'scripts/eval/evaluate.py')}}
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Preserve the previous run; its configuration differs')
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'predictions.jsonl'; previous = b.read_jsonl(path) if path.exists() else []
    if args.retry_errors:
        previous = [r for r in previous if r.get('vision_emotion') in b.EMOTIONS]
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in previous))
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - {r['sample_id'] for r in rows}:
        raise ValueError('Unexpected or duplicated sample IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        torch.cuda.set_per_process_memory_fraction(.30)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(args.model,
            dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
            local_files_only=True, output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys'):
            raise ValueError('Original specialist weights did not load exactly')
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model.eval()
        b.write_json(out / 'environment.json', {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(0), 'missing_checkpoint_keys': 0,
            'mismatched_checkpoint_keys': 0, 'process_memory_fraction': .30})
        cache = {r['video']: r for r in previous if r.get('vision_emotion') in b.EMOTIONS}
        successes = 0
        with path.open('a', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                record = {'sample_id': row['sample_id'], 'video': row['video'], 'protocol': row['protocol']}
                try:
                    if row['video'] in cache:
                        cached = cache[row['video']]
                        record.update({k: cached[k] for k in ('vision_emotion', 'raw_output', 'class_scores', 'media')})
                        record['reused_identical_video_input'] = True
                    else:
                        inputs, media = prepare_video(processor, row, args.dataset_root)
                        inputs = inputs.to('cuda:0').to(torch.bfloat16)
                        with torch.no_grad():
                            generated = model.generate(**inputs, do_sample=False, max_new_tokens=64,
                                return_dict_in_generate=True, output_logits=True,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                pad_token_id=processor.tokenizer.pad_token_id)
                        length = inputs['input_ids'].shape[-1]
                        raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
                        label = tasks.parse_modality(raw, 'vision')
                        try:
                            scores = generation_label_scores(generated, length, processor.tokenizer, b.EMOTIONS)
                        except ValueError as error:
                            scores = {'score_error': str(error)}
                        record.update(vision_emotion=label, raw_output=raw, class_scores=scores, media=media)
                        cache[row['video']] = record
                        del inputs, generated
                    successes += 1
                except torch.OutOfMemoryError:
                    raise
                except Exception as error:
                    record.update(vision_emotion=None, error_type=type(error).__name__, error=str(error))
                handle.write(json.dumps(record, ensure_ascii=False) + '\n'); previous.append(record)
                print(f'[{index}/{len(remaining)}] {row["sample_id"]} {record.get("vision_emotion") or record.get("error")}', flush=True)
                if index >= 3 and not successes:
                    raise RuntimeError('First three specialist inputs failed')
        b.write_json(out / 'resource_usage.json', {'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30})
    finish(rows, previous, args.omni_reference, out)


if __name__ == '__main__':
    main()
