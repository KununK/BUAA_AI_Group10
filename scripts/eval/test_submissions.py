"""Deploy the four frozen val-selected schemes on test, without fitting anything.

One Thinker is loaded lazily; independently saved LoRAs are activated one at a
time. Classification prompts, media preparation and generation reuse the val
implementations. The hybrid reuses all v4 emotions and VT decisions verbatim.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from scripts.common import baseline as b
from scripts.common import conflict_rule as questions
from scripts.common import distill_tasks as dt
from scripts.common import rl_tasks as rt
from scripts.common.omni_policy import generate_text, load_policy
from scripts.common.parsing import parse_task
from scripts.eval.decision_scores import decision_prefix
from scripts.eval.evaluate import generate_label_text
from scripts.eval.evaluate_distillation import generate_scored_decision, parse_answer
from scripts.eval.fuse import fused_yes
from scripts.eval.probabilities import conditional_probabilities
from scripts.eval.va_conflict import parse_uncertainty, prepare_va


ROOT = b.ROOT
STAGES = (
    ('01_teacher_sft800', 'outputs/distill_omni_v3/candidate_adapters/teacher_800',
     'outputs/distill_omni_v3/evaluation/teacher_800_fused'),
    ('02_control_sft800', 'outputs/distill_omni_v3/candidate_adapters/control_800',
     'outputs/distill_omni_v3/evaluation/control_800_fused'),
    ('03_v4', 'outputs/vision_omni_v4_joint/candidate_adapters/vision_frozen_language_800',
     'outputs/vision_omni_v4_joint/evaluation/vision_800_fused'),
    ('04_v4_v7', 'outputs/va_conflict_v7/train',
     'outputs/va_conflict_v7/sft_final'),
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def immutable_json(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Existing run differs; preserve it: {path}')
    b.write_json(path, value)


def public_row(row):
    # Test only has these inputs. No training/validation annotations are passed.
    return {'sample_id': row['sample_id'], 'video': row['video'],
            'text': row.get('text', row.get('text_content', '')),
            'protocol': b.get_protocol(row)}


def add_exact_adapter(model, directory, name):
    """Avoid rounding saved float32 LoRAs through a bf16 load destination."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    directory = Path(directory)
    model.load_adapter(str(directory), adapter_name=name, is_trainable=False)
    for key, value in model.named_parameters():
        if 'lora_' in key and f'.{name}.' in key:
            value.data = value.data.float()
    expected = load_file(str(directory / 'adapter_model.safetensors'))
    result = set_peft_model_state_dict(model, expected, adapter_name=name)
    if result.unexpected_keys or any(f'.{name}.' in k for k in result.missing_keys):
        raise ValueError('New adapter did not load completely')
    verify_adapter(model, directory, name)


def verify_adapter(model, directory, name):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(Path(directory) / 'adapter_model.safetensors'))
    actual = get_peft_model_state_dict(model, adapter_name=name)
    if set(expected) != set(actual) or any(
            not torch.equal(actual[k].detach().cpu(), expected[k]) for k in expected):
        raise ValueError(f'Active LoRA differs from its selected checkpoint: {name}')


class Policy:
    def __init__(self):
        self.model = self.processor = None
        self.names = {}
        self.environment = None

    def activate(self, directory):
        key = str(Path(directory).resolve())
        if self.model is None:
            self.model, self.processor, self.environment = load_policy(directory)
            self.names[key] = 'default'
            verify_adapter(self.model, directory, 'default')
        elif key not in self.names:
            name = f'adapter_{len(self.names)}'
            add_exact_adapter(self.model, directory, name)
            self.names[key] = name
        self.model.set_adapter(self.names[key])
        self.model.requires_grad_(False)
        self.model.eval()
        return self.model, self.processor


def generate_question(model, processor, row, auxiliary=None):
    if auxiliary is None:
        inputs, media = dt.prepare_inputs(processor, row, 'question', ROOT / 'Dataset')
        context, origin = nullcontext(), 'student_media_question'
    else:
        inputs, media = rt.prepare_inputs(processor, row, 'question', ROOT / 'Dataset', auxiliary)
        context, origin = model.disable_adapter(), 'base_from_predicted_modalities'
    with context:
        raw = generate_text(model, processor, inputs, media, max_tokens=128)
    try:
        question = questions.parse_question(raw)
    except ValueError:
        with model.disable_adapter():
            raw = generate_text(model, processor, inputs, media, max_tokens=192)
        question = questions.parse_question(raw)
        origin = 'base_retry'
    del inputs
    return question, origin, raw


def infer_fused(model, processor, row, threshold):
    labels, raw_heads, media_heads = {}, {}, {}
    for task in ('emotion', *rt.modality_tasks(row)):
        inputs, media = rt.prepare_inputs(processor, row, task, ROOT / 'Dataset')
        raw, _ = generate_label_text(model, processor, inputs, media, task)
        labels[task] = parse_task(raw, 'emotion')['pred_emotion'] if task == 'emotion' else rt.parse_modality(raw, task)
        raw_heads[task], media_heads[task] = raw, media
        del inputs
    inputs, media = dt.prepare_inputs(processor, row, 'decision', ROOT / 'Dataset')
    raw, scores = generate_scored_decision(model, processor, inputs, media)
    answer = parse_answer(raw)
    if scores is None:
        scores = conditional_probabilities(model, processor.tokenizer, inputs, media,
            decision_prefix(answer['pred_emotion']), ('yes', 'no'))
        scores['score_source'] = 'conditional_forward_on_student_emotion'
    del inputs
    other = 'text' if row['protocol'] == 'VT' else 'audio'
    auxiliary = {'vision_emotion': labels['vision'], 'other_emotion': labels[other]}
    direct = answer['is_uncertain'] == 'yes'
    disagree = questions.rule_uncertainty(auxiliary) == 'yes'
    uncertain = bool(fused_yes(direct, disagree, scores['probabilities']['yes'], threshold))
    question, origin, raw_question = 'none', 'not_required', None
    if uncertain:
        question, origin, raw_question = generate_question(model, processor, row,
            auxiliary=None if direct else auxiliary)
    return {'sample_id': row['sample_id'], 'protocol': row['protocol'], 'label_set': 'binary',
        'task': 'frozen_test_fusion', 'raw_emotion': raw_heads['emotion'],
        'raw_modality_heads': {k: v for k, v in raw_heads.items() if k != 'emotion'},
        'auxiliary_prediction': auxiliary, 'raw_direct': raw, 'direct_answer': answer,
        'decision_uncertainty': scores, 'threshold': threshold,
        'media': {'heads': media_heads, 'direct': media},
        'question_origin': origin, 'raw_question': raw_question,
        'prediction': {'pred_emotion': labels['emotion'], 'is_uncertain': 'yes' if uncertain else 'no',
                       'clarifying_question': question, 'response': ''}}


def hybrid_record(row, original, uncertain, question='none', origin='not_required', raw=None):
    if row['protocol'] == 'VT':
        return {**original, 'route': 'v4_VT_reused'}
    return {'sample_id': row['sample_id'], 'protocol': 'VA', 'label_set': 'binary',
        'task': 'v4_emotion_v7_VA', 'route': 'v7_raw_VA', 'raw_output': raw,
        'question_origin': origin, 'prediction': {
            'pred_emotion': original['prediction']['pred_emotion'],
            'is_uncertain': uncertain, 'clarifying_question': question, 'response': ''}}


def infer_va(model, processor, row, original):
    inputs, media = prepare_va(processor, row, ROOT / 'Dataset')
    inputs = inputs.to('cuda:0').to(torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, use_audio_in_video=False, do_sample=False, max_new_tokens=64,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    length = inputs['input_ids'].shape[-1]
    raw = processor.batch_decode(generated[:, length:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
    uncertain = parse_uncertainty(raw)
    del generated, inputs
    question, origin, raw_question = 'none', 'not_required', None
    if uncertain == 'yes':
        prior = original['prediction']
        if prior['is_uncertain'] == 'yes':
            question, origin = prior['clarifying_question'], 'reused_actual_v4_question'
        else:
            question, origin, raw_question = generate_question(model, processor, row)
    result = hybrid_record(row, original, uncertain, question, origin, raw)
    result.update(media=media, raw_question=raw_question)
    return result


def finalize(rows, records, out):
    if len(records) != len(rows):
        raise ValueError('Incomplete test inference')
    valid, invalid = b.checked_predictions(rows, records)
    if invalid or len(valid) != 734:
        raise ValueError(f'Invalid test predictions: {invalid[:3]}')
    args = SimpleNamespace(split='test', label_set='binary', emotion_reference=None, dataset_root=ROOT / 'Dataset')
    report = b.finalize(args, rows, rows, records, out)
    b.write_json(out / 'completion.json', {'finished_utc': utc(), 'rows': 734,
        'all_predictions_valid': True, 'submission_sha256': sha(out / 'submission.csv'),
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_initialized() else 0})
    return report


def run_stage(policy, rows, out, specification, common, v4=None):
    name, adapter_rel, calibration_rel = specification
    adapter, calibration = ROOT / adapter_rel, ROOT / calibration_rel / 'run_config.json'
    source = read(calibration)
    adapter_hash = sha(adapter / 'adapter_model.safetensors')
    if source['split'] != 'val' or source['adapter_sha256'] != adapter_hash:
        raise ValueError('Selected val adapter changed')
    hybrid = v4 is not None
    threshold = None if hybrid else source['threshold']
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError('Unexpected saved threshold')
    manifest = {**common, 'scheme': name, 'adapter': str(adapter), 'adapter_sha256': adapter_hash,
        'calibration_source': str(calibration), 'calibration_source_sha256': sha(calibration),
        'threshold': threshold, 'threshold_fitting_on_test': False,
        'mode': 'v4 emotion and VT frozen; v7 raw VA' if hybrid else 'same-adapter direct OR (M1!=M2 AND p_yes>saved_threshold)'}
    if hybrid:
        manifest['v4_source_sha256'] = sha(out.parent / '03_v4/predictions.jsonl')
    out.mkdir(parents=True, exist_ok=True)
    immutable_json(out / 'run_config.json', manifest)
    path = out / 'predictions.jsonl'
    previous = b.read_jsonl(path) if path.exists() else []
    seen = {r['sample_id'] for r in previous}
    if len(seen) != len(previous) or seen - {r['sample_id'] for r in rows}:
        raise ValueError('Unexpected/duplicate resumed samples')
    if previous:
        _, invalid = b.checked_predictions([r for r in rows if r['sample_id'] in seen], previous)
        if invalid:
            raise ValueError('Saved rows are invalid; preserve for diagnosis')
    if len(previous) != len(rows):
        model, processor = policy.activate(adapter)
        b.write_json(out / 'environment.json', {**policy.environment, 'active_adapter': policy.names[str(adapter.resolve())],
            'loaded_adapter_state_matches_selected_checkpoint': True, 'single_base_model': True})
        with path.open('a', encoding='utf-8', buffering=1) as handle:
            for index, row in enumerate(rows, 1):
                if row['sample_id'] in seen:
                    continue
                started = time.monotonic()
                try:
                    if hybrid:
                        original = v4[row['sample_id']]
                        record = hybrid_record(row, original, None) if row['protocol'] == 'VT' else infer_va(model, processor, row, original)
                    else:
                        record = infer_fused(model, processor, row, threshold)
                    _, invalid = b.checked_predictions([row], [record])
                    if invalid:
                        raise ValueError(str(invalid))
                except Exception as error:
                    b.write_json(out / 'failure.json', {'sample_id': row['sample_id'],
                        'type': type(error).__name__, 'message': str(error), 'utc': utc()})
                    raise
                record['seconds'] = time.monotonic() - started
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                previous.append(record)
                print(f'{name} [{index}/734] {row["sample_id"]} {record["prediction"]} ({record["seconds"]:.1f}s)', flush=True)
    return finalize(rows, previous, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '4':
        raise ValueError('This authorized test run must use physical GPU 4 only')
    rows = [public_row(r) for r in b.read_jsonl(ROOT / 'Dataset/test_release.jsonl')]
    if len(rows) != 734 or len({r['sample_id'] for r in rows}) != 734:
        raise ValueError('Unexpected test split')
    files = [Path(__file__), *[ROOT / f'scripts/{s}.py' for s in (
        'common/baseline', 'common/rl_tasks', 'common/sft_tasks', 'common/distill_tasks',
        'common/omni_policy', 'common/conflict_rule', 'eval/evaluate', 'eval/evaluate_distillation',
        'eval/fuse', 'eval/probabilities', 'eval/decision_scores', 'eval/va_conflict')]]
    common = {'split': 'test', 'dataset_sha256': sha(ROOT / 'Dataset/test_release.jsonl'),
        'template_sha256': sha(ROOT / 'Dataset/sample_submission.csv'),
        'source_sha256': {str(f.resolve().relative_to(ROOT)): sha(f) for f in files},
        'physical_GPU': 4, 'fps': 2, 'max_frames': 32, 'max_pixels': 262144,
        'audio_sampling_rate': 16000, 'emotion_input': 'all-VT',
        'conflict_input': 'VT video+text; VA video+audio without external transcript',
        'test_labels_read': False, 'thresholds_fitted_on_test': False}
    out = a.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    immutable_json(out / 'run_config.json', common)
    with (ROOT / 'outputs/.gpu_eval_4.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        torch.cuda.set_per_process_memory_fraction(.4)
        policy = Policy(); reports = {}
        for spec in STAGES:
            reference = None
            if spec[0] == '04_v4_v7':
                reference = {r['sample_id']: r for r in b.read_jsonl(out / '03_v4/predictions.jsonl')}
                if len(reference) != 734:
                    raise ValueError('v4 must complete before the hybrid')
            reports[spec[0]] = run_stage(policy, rows, out / spec[0], spec, common, reference)
            b.write_json(out / 'progress.json', {'completed': list(reports), 'updated_utc': utc()})
        b.write_json(out / 'complete.json', {'finished_utc': utc(), 'schemes': reports,
            'all_four_submissions_ready': True, 'Kaggle_uploaded': False})
        print(json.dumps({'all_four_submissions_ready': True, 'output_root': str(out)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
