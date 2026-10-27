"""Test a confidence-gated fallback to the original multimodal conflict rule.

For each outer video fold, refit the original gate on that fold's training rows
before using its flags as features. Stored OOF training-row flags are deliberately
not used: their thresholds can have seen the current held fold's labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b
from scripts.eval.fuse import fit_gate, fused_yes
from scripts.eval.protocol_fusion import audit_branch, audit_reference, indexed, read, sha


def fit_fallback(new, old, probability, gold, conflicts, aligned):
    choices = np.r_[-1e-6, np.unique(probability[old & ~new]), 1.]
    def objective(threshold):
        flags = new | (old & (probability > threshold))
        tp, tn = int((flags & gold).sum()), int((~flags & ~gold).sum())
        return .3 * tp / conflicts + .2 * tn / aligned, tn, threshold
    return float(max(choices, key=objective))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', choices=('VA', 'VT'), required=True)
    p.add_argument('--classifier-run', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); rows = b.read_jsonl(b.ROOT / 'Dataset/val.jsonl')
    groups = np.array([r['video'] for r in rows]); y = np.array([r['sample_type'] == 'Conflict' for r in rows])
    selected = np.array([r['protocol'] == a.protocol for r in rows])
    strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(
        np.zeros(len(rows)), strata, groups))
    if any(set(groups[tr]) & set(groups[te]) for tr, te in splits): raise ValueError('Video leakage')
    reference, original, (independent, direct) = audit_reference(a.reference, rows, splits)
    audit_branch(a.classifier_run, a.protocol, rows, reference, original, splits,
                 a.reference / 'grouped_oof_predictions.jsonl')
    predictions = indexed(a.classifier_run / 'predictions.jsonl', rows)
    old_direct = np.array([direct[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    old_aux = np.array([independent[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    old_p = np.array([direct[r['sample_id']]['decision_uncertainty']['probabilities']['yes'] for r in rows])
    new = np.array([predictions[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    new_p = np.array([predictions[r['sample_id']]['decision_uncertainty']['probabilities']['yes']
                      if selected[i] else np.nan for i, r in enumerate(rows)])
    flags = original.copy(); folds = []
    for train, held in splits:
        old_t = fit_gate(old_direct[train], old_aux[train], old_p[train], y[train])
        old_train = fused_yes(old_direct[train], old_aux[train], old_p[train], old_t)
        old_held = fused_yes(old_direct[held], old_aux[held], old_p[held], old_t)
        if not np.array_equal(old_held, original[held]): raise ValueError('Original held predictions changed')
        tr_mask, te_mask = selected[train], selected[held]; tr, te = train[tr_mask], held[te_mask]
        new_t = fit_fallback(new[tr], old_train[tr_mask], new_p[tr], y[tr],
                             int(y[train].sum()), int((~y[train]).sum()))
        flags[te] = new[te] | (old_held[te_mask] & (new_p[te] > new_t))
        folds.append({'heldout_indices': held.tolist(), 'original_gate_threshold': old_t,
                      'fallback_threshold': new_t, 'video_overlap': False,
                      'original_gate_refitted_inside_outer_train': True})
    records = []
    for i, row in enumerate(rows):
        sid = row['sample_id']; question = 'none'
        if flags[i]:
            question = next((source[sid]['prediction'].get('clarifying_question')
                for source in (reference, independent, direct)
                if source[sid]['prediction'].get('clarifying_question') not in (None, '', 'none')), None)
        records.append({'sample_id': sid, 'protocol': row['protocol'], 'label_set': 'binary',
            'question_pending': question is None, 'prediction': {
                'pred_emotion': reference[sid]['prediction']['pred_emotion'],
                'is_uncertain': 'yes' if flags[i] else 'no', 'clarifying_question': question, 'response': ''}})
    pending = sum(r['question_pending'] for r in records)
    summary = {'metrics': b.score_rows(rows, records), 'folds': folds,
        'method': 'new_raw_yes OR (original_gate_yes AND new_p_yes > threshold)',
        'protocol': a.protocol, 'original_gate_training_features_refitted_inside_outer_train': True,
        'stored_OOF_training_row_flags_not_used_as_features': True, 'model_training_gain_claimed': False,
        'pending_model_questions': pending, 'submission_ready': pending == 0,
        'selection_scope': 'val after repeated model selection; not an independent test score'}
    sources = [Path(__file__), a.reference / 'grouped_oof_predictions.jsonl',
        a.classifier_run / 'run_config.json', a.classifier_run / 'predictions.jsonl',
        a.classifier_run / 'metrics.json']
    manifest = {'input_sha256': {str(f.resolve()): sha(f) for f in sources}, 'protocol': a.protocol}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / 'run_config.json').exists() and read(a.output_dir / 'run_config.json') != manifest:
        raise ValueError('Preserve the existing fallback run: sources changed')
    b.write_json(a.output_dir / 'run_config.json', manifest)
    b.write_json(a.output_dir / 'metrics.json', summary)
    (a.output_dir / 'grouped_oof_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if not pending:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Invalid combined labels/questions')
        b.write_prediction_csv(rows, records, a.output_dir / 'grouped_oof_predictions.csv')
    print(json.dumps({k: v for k, v in summary.items() if k != 'folds'}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
