"""Preserve raw neural conflict decisions for one protocol and audit all sources."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b
from scripts.eval.protocol_fusion import audit_reference, audit_branch, indexed, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', choices=('VT', 'VA'), required=True)
    p.add_argument('--classifier-run', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); rows = b.read_jsonl(b.ROOT / 'Dataset/val.jsonl')
    groups = np.array([r['video'] for r in rows]); strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(np.zeros(len(rows)), strata, groups))
    if any(set(groups[tr]) & set(groups[te]) for tr, te in splits): raise ValueError('Video leakage')
    reference, flags, question_sources = audit_reference(a.reference, rows, splits)
    audit_branch(a.classifier_run, a.protocol, rows, reference, flags, splits,
                 a.reference / 'grouped_oof_predictions.jsonl')
    classifier = indexed(a.classifier_run / 'predictions.jsonl', rows); records = []
    for row in rows:
        sid = row['sample_id']; origin = classifier if row['protocol'] == a.protocol else reference
        uncertain = origin[sid]['prediction']['is_uncertain']; question = 'none'
        if uncertain == 'yes':
            question = next((source[sid]['prediction'].get('clarifying_question')
                for source in (reference, *question_sources)
                if source[sid]['prediction'].get('clarifying_question') not in (None, '', 'none')), None)
        records.append({'sample_id': sid, 'protocol': row['protocol'], 'label_set': 'binary',
            'question_pending': question is None, 'prediction': {'pred_emotion': reference[sid]['prediction']['pred_emotion'],
                'is_uncertain': uncertain, 'clarifying_question': question, 'response': ''}})
    pending = sum(r['question_pending'] for r in records); metric = b.score_rows(rows, records)
    expected = read(a.classifier_run / 'metrics.json')['raw_metrics']
    if abs(metric['score_sklearn_union'] - expected['score_sklearn_union']) > 1e-12:
        raise ValueError('Composition differs from the actual raw-classifier evaluation')
    if not pending:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Invalid combined output fields')
    sources = [Path(__file__), a.reference / 'grouped_oof_predictions.jsonl',
        a.reference / 'grouped_oof_metrics.json', a.classifier_run / 'run_config.json',
        a.classifier_run / 'predictions.jsonl', a.classifier_run / 'metrics.json']
    manifest = {'protocol': a.protocol, 'input_sha256': {str(f.resolve()): sha(f) for f in sources},
        'new_protocol_calibration': 'none: preserve actual greedy neural yes/no',
        'other_protocol_calibration': 'original gate verified from video-grouped training folds',
        'model_training_gain_claimed': False}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / 'run_config.json').exists() and read(a.output_dir / 'run_config.json') != manifest:
        raise ValueError('Preserve the existing raw composition: sources changed')
    b.write_json(a.output_dir / 'run_config.json', manifest)
    b.write_json(a.output_dir / 'metrics.json', {'metrics': metric, 'pending_model_questions': pending,
        'submission_ready': pending == 0, 'all_sources_and_thresholds_audited': True,
        'selection_scope': 'val after repeated model selection; not an independent test score'})
    (a.output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if not pending: b.write_prediction_csv(rows, records, a.output_dir / 'predictions.csv')
    print(json.dumps({'score': metric['score_sklearn_union'], 'pending_questions': pending,
                      'valid_classifications': metric['valid_predictions']}), flush=True)


if __name__ == '__main__': main()
