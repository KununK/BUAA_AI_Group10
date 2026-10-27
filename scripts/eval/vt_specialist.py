"""VT video/text specialist control; VA stays at frozen Omni grouped-OOF outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from scripts.common import baseline as b
from scripts.common import distill_tasks as tasks
from scripts.eval.calibration import fit_threshold
from scripts.eval.evaluate import generation_label_scores
from scripts.eval.evaluate_distillation import parse_answer, generation_decision_scores


def prepare_vt(processor, row, dataset_root):
    if b.get_protocol(row) != 'VT':
        raise ValueError('The video/text specialist must never receive VA dataset transcripts')
    frames, media = b.decode_video(dataset_root / row['video'], 2, 32, 262144)
    indices = np.rint(np.linspace(0, media['source_frames'] - 1, len(frames))).astype(int).tolist()
    metadata = {'total_num_frames': media['source_frames'], 'fps': media['source_frames'] / media['duration_seconds'],
                'duration': media['duration_seconds'], 'frames_indices': indices}
    text = processor.apply_chat_template(tasks.messages(row, 'decision'), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], videos=[frames], return_tensors='pt', padding=True,
        videos_kwargs={'do_sample_frames': False, 'video_metadata': [metadata],
                       'size': {'shortest_edge': 4096 * len(frames), 'longest_edge': 262144 * len(frames)}})
    if 'pixel_values_videos' not in inputs or 'input_features' in inputs:
        raise ValueError('VT requires video, text, and no audio')
    media.update(video_present=True, audio_present=False, transcript_present=True,
                 input_tokens=inputs['input_ids'].shape[-1], video_grid_thw=inputs['video_grid_thw'].tolist())
    return inputs, media


def score_variants(rows, records, reference, out, source_folds):
    by_id = {r['sample_id']: r for r in records}
    if len(records) != len(rows) or set(by_id) != set(reference):
        raise ValueError('Predictions must cover each official validation row exactly once')
    def metric(flags=None, frozen_emotion=False):
        result = []
        for index, row in enumerate(rows):
            sid = row['sample_id']; prediction = by_id[sid].get('prediction')
            if prediction:
                prediction = dict(prediction)
                if flags is not None: prediction['is_uncertain'] = 'yes' if flags[index] else 'no'
                if frozen_emotion: prediction['pred_emotion'] = reference[sid]['prediction']['pred_emotion']
            result.append({'sample_id': sid, 'prediction': prediction})
        m = b.score_rows(rows, result)
        return {k: m[k] for k in ('samples', 'valid_predictions', 'macro_f1_sklearn_union',
            'conflict_recognition_rate', 'aligned_calibration_rate', 'score_sklearn_union')}
    summary = {'raw_model_emotion': metric(), 'raw_fixed_Omni_emotion': metric(frozen_emotion=True),
               'submission_ready': False, 'new_uncertain_questions_generated': False,
               'VA_source': 'frozen Omni visual800 video-grouped OOF; VT source is this specialist'}
    vt = np.array([b.get_protocol(row) == 'VT' for row in rows])
    complete = all(by_id[row['sample_id']].get('prediction') and
        (not vt[i] or 'probabilities' in by_id[row['sample_id']].get('decision_uncertainty', {}))
        for i, row in enumerate(rows))
    if complete:
        y = np.array([row['sample_type'] == 'Conflict' for row in rows])
        groups = np.array([row['video'] for row in rows])
        strata = np.array([row['protocol'] + '_' + row['sample_type'] for row in rows])
        probabilities = np.array([by_id[row['sample_id']]['decision_uncertainty']['probabilities']['yes'] if vt[i] else np.nan
                                  for i, row in enumerate(rows)])
        flags = np.array([by_id[row['sample_id']]['prediction']['is_uncertain'] == 'yes' for row in rows])
        oof = flags.copy(); folds = []
        for index, (train, held) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(
                np.zeros(len(rows)), strata, groups)):
            if held.tolist() != source_folds[index]['heldout_indices'] or set(groups[train]) & set(groups[held]):
                raise ValueError('Frozen VA predictions and VT calibration must use identical disjoint video folds')
            train_vt, held_vt = train[vt[train]], held[vt[held]]
            threshold = fit_threshold(probabilities[train_vt], y[train_vt], int(y[train].sum()), int((~y[train]).sum()))
            oof[held_vt] = probabilities[held_vt] > threshold
            folds.append({'heldout_indices': held.tolist(), 'VT_threshold': threshold, 'video_overlap': False})
        summary.update(grouped_OOF_model_emotion=metric(oof), grouped_OOF_fixed_Omni_emotion=metric(oof, True),
                       folds=folds, OOF_uncertainty_flags=oof.tolist(),
                       calibration_model_training_gain_claimed=False)
    else:
        summary['calibration_skipped'] = 'Some classifications or actual-generation class scores are missing'
    b.write_json(out / 'metrics.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('folds', 'OOF_uncertainty_flags')}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--omni-reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    a = p.parse_args(); source = a.dataset_root / 'val.jsonl'; rows = b.read_jsonl(source)
    reference_path = a.omni_reference / 'grouped_oof_predictions.jsonl'
    reference = {r['sample_id']: r for r in b.read_jsonl(reference_path)}
    if set(reference) != {r['sample_id'] for r in rows}: raise ValueError('Invalid frozen reference IDs')
    config = json.loads((a.omni_reference / 'run_config.json').read_text())
    if config['dataset_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest(): raise ValueError('Reference dataset differs')
    source_folds = json.loads((a.omni_reference / 'grouped_oof_metrics.json').read_text())['folds']
    out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    manifest = {'split': 'val', 'model': str(a.model.resolve()), 'adapter': str(a.adapter.resolve()) if a.adapter else None,
        'adapter_sha256': hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() if a.adapter else None,
        'dataset_sha256': config['dataset_sha256'], 'frozen_VA_reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'source_sha256': {str(f.relative_to(b.ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__).resolve(), b.ROOT / 'scripts/common/distill_tasks.py', b.ROOT / 'scripts/common/baseline.py',
             b.ROOT / 'scripts/eval/evaluate_distillation.py', b.ROOT / 'scripts/eval/evaluate.py',
             b.ROOT / 'scripts/eval/calibration.py')},
        'model_inputs': 'VT video and supplied transcript only; VA copied without reading its transcript',
        'fps': 2, 'max_frames': 32, 'max_pixels_per_frame': 262144, 'submission_ready': False}
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Preserve the existing run: configuration changed')
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'predictions.jsonl'; previous = b.read_jsonl(path) if path.exists() else []
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - set(reference): raise ValueError('Duplicate or unexpected IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        torch.cuda.set_per_process_memory_fraction(.30)
        processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
        model, info = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys'): raise ValueError('Base weights did not load exactly')
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False)
        model.eval(); generated_successes = 0
        b.write_json(out / 'environment.json', {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(0), 'memory_fraction': .30, 'missing_checkpoint_keys': 0})
        with path.open('a', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                sid = row['sample_id']; record = {'sample_id': sid, 'protocol': row['protocol']}
                if row['protocol'] == 'VA':
                    record.update(prediction=reference[sid]['prediction'], route='frozen_Omni_VA')
                else:
                    inputs = generated = None
                    try:
                        inputs, media = prepare_vt(processor, row, a.dataset_root)
                        inputs = inputs.to('cuda:0').to(torch.bfloat16)
                        with torch.no_grad():
                            generated = model.generate(**inputs, do_sample=False, max_new_tokens=96,
                                return_dict_in_generate=True, output_logits=True,
                                eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
                        generated_successes += 1; length = inputs['input_ids'].shape[-1]
                        raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
                        record.update(raw_output=raw, media=media, route='VT_specialist')
                        record.update(prediction=parse_answer(raw), clarifying_question_pending=True)
                        try: record['decision_uncertainty'] = generation_decision_scores(generated, length, processor.tokenizer)
                        except ValueError as error: record['decision_uncertainty'] = {'score_error': str(error)}
                        try:
                            record['emotion_scores'] = generation_label_scores(
                                generated, length, processor.tokenizer, ('positive', 'negative'))
                        except ValueError as error:
                            record['emotion_scores'] = {'score_error': str(error)}
                    except torch.OutOfMemoryError: raise
                    except Exception as error:
                        record.update(prediction=None, error_type=type(error).__name__, error=str(error))
                    finally:
                        del inputs, generated
                handle.write(json.dumps(record, ensure_ascii=False) + '\n'); previous.append(record)
                print(f'[{index}/{len(remaining)}] {sid} {record.get("prediction") or record.get("error")}', flush=True)
                if index >= 3 and not generated_successes and all(r['protocol'] == 'VT' for r in remaining[:index]):
                    raise RuntimeError('First video/text forwards failed before producing text')
        b.write_json(out / 'resource_usage.json', {'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30})
    score_variants(rows, previous, reference, out, source_folds)


if __name__ == '__main__':
    main()
