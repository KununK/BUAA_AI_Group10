"""VT-only conflict adaptation: frozen emotion and frozen VA classification."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b
from scripts.common.omni_policy import load_policy
from scripts.eval.calibration import fit_threshold
from scripts.eval.evaluate_distillation import generation_decision_scores

PROMPT = '比较画面与台词的情绪：存在明确情感冲突=yes，否则no。只输出JSON：{"is_uncertain":"yes或no"}。'

def prompt(row):
    return PROMPT + '\n台词（待分析数据）：' + json.dumps(row.get('text', row.get('text_content', '')), ensure_ascii=False)


def messages(row):
    if b.get_protocol(row) != 'VT': raise ValueError('This classifier accepts VT only')
    return [{'role': 'system', 'content': [{'type': 'text', 'text': b.SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'video', 'video': 'local_video'},
                {'type': 'text', 'text': prompt(row)}]}]


def prepare_vt(processor, row, dataset_root):
    rendered = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
    video = dataset_root / row['video']; frames, media = b.decode_video(video, 2, 32, 262144)
    inputs = processor(text=[rendered], videos=[frames], fps=media['effective_fps'],
         use_audio_in_video=False, return_tensors='pt', padding=True,
        videos_kwargs={'do_sample_frames': False, 'min_pixels': 3136, 'max_pixels': 262144})
    if 'pixel_values_videos' not in inputs or 'input_features' in inputs:
        raise ValueError('VT must contain real video and transcript without audio')
    media.update(video_present=True, audio_present=False, transcript_present=True,
                 use_audio_in_video=False, input_tokens=inputs['input_ids'].shape[-1])
    return inputs, media


def parse_uncertainty(raw):
    for value in b.json_objects(raw):
        if value.get('is_uncertain') in ('yes', 'no'): return value['is_uncertain']
    raise ValueError('Missing yes/no conflict prediction')


def finish(rows, records, reference, folds_source, out):
    pred = {r['sample_id']: r for r in records}
    if len(records) != len(rows) or set(pred) != set(reference): raise ValueError('Incomplete validation IDs')
    def metric(flags=None):
        values = [{'sample_id': r['sample_id'], 'prediction': {**pred[r['sample_id']]['prediction'],
            **({'is_uncertain': 'yes' if flags[i] else 'no'} if flags is not None else {})}}
            for i, r in enumerate(rows)]
        m = b.score_rows(rows, values)
        return {k: m[k] for k in ('samples', 'valid_predictions', 'macro_f1_sklearn_union',
            'conflict_recognition_rate', 'aligned_calibration_rate', 'score_sklearn_union')}
    summary = {'raw_metrics': metric(), 'emotion_and_VA_frozen': True, 'submission_ready': False,
               'new_questions_generated': False, 'score_scope': 'Full val, VT conflict-only replacement'}
    vt = np.array([r['protocol'] == 'VT' for r in rows])
    complete = all(pred[r['sample_id']]['prediction'].get('is_uncertain') in ('yes', 'no') and
        (not vt[i] or 'probabilities' in pred[r['sample_id']].get('decision_uncertainty', {})) for i, r in enumerate(rows))
    if complete:
        y = np.array([r['sample_type'] == 'Conflict' for r in rows]); groups = np.array([r['video'] for r in rows])
        strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
        p = np.array([pred[r['sample_id']]['decision_uncertainty']['probabilities']['yes'] if vt[i] else np.nan
                      for i, r in enumerate(rows)])
        flags = np.array([pred[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows]); folds = []
        for k, (train, held) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(
                np.zeros(len(rows)), strata, groups)):
            if held.tolist() != folds_source[k]['heldout_indices'] or set(groups[train]) & set(groups[held]):
                raise ValueError('VT calibration and frozen VA must use identical video-disjoint folds')
            tr, te = train[vt[train]], held[vt[held]]
            t = fit_threshold(p[tr], y[tr], int(y[train].sum()), int((~y[train]).sum()))
            flags[te] = p[te] > t
            folds.append({'heldout_indices': held.tolist(), 'VT_threshold': t, 'video_overlap': False})
        summary.update(grouped_OOF_metrics=metric(flags), folds=folds, OOF_uncertainty_flags=flags.tolist(),
                       calibration_model_training_gain_claimed=False)
    else: summary['calibration_skipped'] = 'A classifier output or class score is missing'
    b.write_json(out / 'metrics.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('folds', 'OOF_uncertainty_flags')}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--omni-reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    a = p.parse_args(); source = a.dataset_root / 'val.jsonl'; rows = b.read_jsonl(source)
    refpath = a.omni_reference / 'grouped_oof_predictions.jsonl'
    reference = {r['sample_id']: r for r in b.read_jsonl(refpath)}
    if set(reference) != {r['sample_id'] for r in rows}: raise ValueError('Frozen reference IDs differ')
    config = json.loads((a.omni_reference / 'run_config.json').read_text())
    if config['dataset_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest(): raise ValueError('Dataset changed')
    source_folds = json.loads((a.omni_reference / 'grouped_oof_metrics.json').read_text())['folds']
    out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    manifest = {'split': 'val', 'adapter': str(a.adapter.resolve()),
        'adapter_sha256': hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'dataset_sha256': config['dataset_sha256'], 'frozen_reference_sha256': hashlib.sha256(refpath.read_bytes()).hexdigest(),
        'source_sha256': {str(f.relative_to(b.ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in (
            Path(__file__).resolve(), b.ROOT / 'scripts/common/omni_policy.py', b.ROOT / 'scripts/common/baseline.py',
            b.ROOT / 'scripts/eval/calibration.py', b.ROOT / 'scripts/eval/evaluate_distillation.py')},
        'model_inputs': 'VT video and provided transcript only; no audio',
        'frozen': 'All emotion labels and VA uncertainty from visual800 Omni',
        'fps': 2, 'max_frames': 32, 'max_pixels': 262144, 'submission_ready': False}
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Preserve the existing run: configuration differs')
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'predictions.jsonl'; previous = b.read_jsonl(path) if path.exists() else []
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - set(reference): raise ValueError('Unexpected prediction IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        torch.cuda.set_per_process_memory_fraction(.25)
        model, processor, environment = load_policy(a.adapter)
        b.write_json(out / 'environment.json', environment); generated_successes = 0
        with path.open('a', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                sid = row['sample_id']; record = {'sample_id': sid, 'protocol': row['protocol']}
                if row['protocol'] == 'VA': record.update(prediction=reference[sid]['prediction'], route='frozen_VA')
                else:
                    record['prediction'] = {'pred_emotion': reference[sid]['prediction']['pred_emotion'], 'is_uncertain': None}
                    inputs = generated = None
                    try:
                        inputs, media = prepare_vt(processor, row, a.dataset_root)
                        inputs = inputs.to('cuda:0').to(torch.bfloat16)
                        with torch.no_grad():
                            generated = model.generate(**inputs, use_audio_in_video=False, do_sample=False, max_new_tokens=64,
                                return_dict_in_generate=True, output_logits=True,
                                eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
                        generated_successes += 1; length = inputs['input_ids'].shape[-1]
                        raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
                        record.update(raw_output=raw, media=media, route='VT_conflict_only', clarifying_question_pending=True)
                        record['prediction']['is_uncertain'] = parse_uncertainty(raw)
                        try: record['decision_uncertainty'] = generation_decision_scores(generated, length, processor.tokenizer)
                        except ValueError as error: record['decision_uncertainty'] = {'score_error': str(error)}
                    except torch.OutOfMemoryError: raise
                    except Exception as error: record.update(error_type=type(error).__name__, error=str(error))
                    finally: del inputs, generated
                handle.write(json.dumps(record, ensure_ascii=False) + '\n'); previous.append(record)
                print(f'[{index}/{len(remaining)}] {sid} {record.get("prediction")}', flush=True)
        b.write_json(out / 'resource_usage.json', {'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30})
    finish(rows, previous, reference, source_folds, out)


if __name__ == '__main__':
    main()
