"""Select one audited conflict component per input protocol, with fixed emotion.

Selection uses complete official-val results, never per-row gold-based routing.
This is model selection on val, not a claim of independent test performance.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b
from scripts.eval.calibration import fit_threshold
from scripts.eval.protocol_fusion import audit_branch, audit_reference, indexed, read, sha

STUDIES = (
    ('VA', 'va_conflict_v7', ('sft_144', 'sft_final')),
    ('VA', 'va_pairs_v9', ('sft_269', 'sft_final')),
    ('VT', 'vt_conflict_v8', ('sft_324', 'sft_final')),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused')
    p.add_argument('--emotion-adapter', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/candidate_adapters/vision_frozen_language_800')
    p.add_argument('--extra-candidates', type=Path, help='Explicit additional val runs: list of protocol, directory, job')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); rows = b.read_jsonl(b.ROOT / 'Dataset/val.jsonl')
    y = np.array([r['sample_type'] == 'Conflict' for r in rows]); groups = np.array([r['video'] for r in rows])
    protocols = np.array([r['protocol'] for r in rows]); strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(np.zeros(len(rows)), strata, groups))
    if any(set(groups[tr]) & set(groups[te]) for tr, te in splits): raise ValueError('Video leakage')
    reference, original, question_sources = audit_reference(a.reference, rows, splits)
    reference_config = read(a.reference / 'run_config.json')
    if sha(a.emotion_adapter / 'adapter_model.safetensors') != reference_config['adapter_sha256']:
        raise ValueError('Emotion adapter differs from the validated frozen reference')
    sources = [Path(__file__), a.reference / 'run_config.json', a.reference / 'grouped_oof_predictions.jsonl',
               a.reference / 'grouped_oof_metrics.json']
    candidates = {'VT': [], 'VA': []}

    def add(protocol, name, mode, flags, directory, adapter, adapter_hash):
        mask = protocols == protocol
        tp, tn = int((flags & y & mask).sum()), int((~flags & ~y & mask).sum())
        utility = Fraction(3 * tp, 10 * int(y.sum())) + Fraction(tn, 5 * int((~y).sum()))
        deployment = {'mode': mode}
        if mode == 'grouped_OOF':
            records = indexed(directory / 'predictions.jsonl', rows)
            scores = np.array([records[r['sample_id']]['decision_uncertainty']['probabilities']['yes']
                               for r in rows if r['protocol'] == protocol])
            deployment['full_val_fitted_threshold'] = fit_threshold(scores, y[mask], int(y.sum()), int((~y).sum()))
            deployment['full_val_fitted_threshold_not_used_to_report_val_score'] = True
        elif mode == 'reference':
            deployment['original_global_gate_threshold'] = reference_config['threshold']
        candidates[protocol].append({'name': name, 'protocol': protocol, 'mode': mode,
            'source_run': str(directory), 'adapter': str(adapter), 'adapter_sha256': adapter_hash,
            'TP': tp, 'TN': tn, 'utility': utility, 'deployment': deployment, '_flags': flags})

    for protocol in candidates:
        add(protocol, 'original_visual800', 'reference', original.copy(), a.reference,
            a.emotion_adapter, reference_config['adapter_sha256'])
    entries = []
    for protocol, study, stages in STUDIES:
        for stage in stages:
            directory = b.ROOT / 'outputs' / study / stage
            jobs = [directory.parent / 'jobs' / stage / 'exit.json']
            if study == 'va_conflict_v7' and stage == 'sft_final':
                jobs.append(directory.parent / 'jobs/sft_final_gpu0_retry/exit.json')
            entries.append((protocol, study + '/' + stage, directory, jobs))
    if a.extra_candidates:
        sources.append(a.extra_candidates)
        for item in read(a.extra_candidates):
            if item['protocol'] not in candidates: raise ValueError('Unsupported extra protocol')
            directory = Path(item['directory']).resolve()
            if not directory.is_relative_to(b.ROOT / 'outputs'):
                raise ValueError('Additional runs must be local project outputs')
            entries.append((item['protocol'], str(directory.relative_to(b.ROOT / 'outputs')),
                            directory, [Path(item['job']).resolve() / 'exit.json']))
    for protocol, name, directory, jobs in entries:
            successful = next((f for f in jobs if f.exists() and read(f)['exit_code'] == 0), None)
            if successful is None or not (directory / 'metrics.json').exists(): continue
            metrics = read(directory / 'metrics.json')
            if metrics.get('grouped_OOF_metrics', {}).get('valid_predictions') != 666:
                raise ValueError('A terminal candidate does not contain all valid classifications')
            config = read(directory / 'run_config.json')
            oof = audit_branch(directory, protocol, rows, reference, original, splits,
                               a.reference / 'grouped_oof_predictions.jsonl')
            records = indexed(directory / 'predictions.jsonl', rows)
            if any(r['prediction']['is_uncertain'] not in ('yes', 'no') for r in records.values()):
                raise ValueError('Invalid neural conflict class')
            raw = np.array([records[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
            for mode, flags in (('raw', raw), ('grouped_OOF', oof)):
                add(protocol, name, mode, flags, directory,
                    Path(config['adapter']), config['adapter_sha256'])
            sources.extend([successful, *(directory / f for f in ('run_config.json', 'predictions.jsonl', 'metrics.json'))])
    priority = {'reference': 2, 'raw': 1, 'grouped_OOF': 0}
    chosen = {protocol: max(values, key=lambda c: (c['utility'], priority[c['mode']]))
              for protocol, values in candidates.items()}
    records = []
    for i, row in enumerate(rows):
        sid = row['sample_id']; component = chosen[row['protocol']]
        uncertain = bool(component['_flags'][i]); question = 'none'
        if uncertain:
            question = next((origin[sid]['prediction'].get('clarifying_question')
                for origin in (reference, *question_sources)
                if origin[sid]['prediction'].get('clarifying_question') not in (None, '', 'none')), None)
        records.append({'sample_id': sid, 'protocol': row['protocol'], 'label_set': 'binary',
            'uncertainty_component': component['name'], 'uncertainty_mode': component['mode'],
            'question_pending': question is None, 'prediction': {
                'pred_emotion': reference[sid]['prediction']['pred_emotion'],
                'is_uncertain': 'yes' if uncertain else 'no', 'clarifying_question': question, 'response': ''}})
    pending = sum(r['question_pending'] for r in records)
    if not pending:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Invalid output labels or questions')
    def describe(c):
        return {**{k: v for k, v in c.items() if k not in ('utility', '_flags')}, 'utility': float(c['utility'])}
    summary = {'metrics': b.score_rows(rows, records), 'selected': {k: describe(v) for k, v in chosen.items()},
        'candidates': {k: [describe(v) for v in values] for k, values in candidates.items()},
        'emotion_adapter': str(a.emotion_adapter), 'emotion_adapter_sha256': reference_config['adapter_sha256'],
        'pending_model_questions': pending, 'submission_ready': pending == 0,
        'all_calibrated_components_refitted_on_video_disjoint_training_folds': True,
        'selection_scope': 'One fixed component per input protocol selected on complete val; no per-row gold routing; not independent test performance'}
    manifest = {'split': 'val', 'input_sha256': {str(f.resolve()): sha(f) for f in sources},
                'permitted_studies': STUDIES, 'rowwise_gold_routing': False}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # JSON round-trip normalizes tuples in the fixed candidate whitelist.
    manifest = json.loads(json.dumps(manifest))
    if (a.output_dir / 'run_config.json').exists() and read(a.output_dir / 'run_config.json') != manifest:
        raise ValueError('Use a new output directory for a changed selection universe')
    b.write_json(a.output_dir / 'run_config.json', manifest)
    b.write_json(a.output_dir / 'metrics.json', summary)
    (a.output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if not pending: b.write_prediction_csv(rows, records, a.output_dir / 'predictions.csv')
    print(json.dumps({'score': summary['metrics']['score_sklearn_union'], 'selected': summary['selected'],
                      'pending_questions': pending}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
