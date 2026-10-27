"""Video-grouped calibration from three unfitted model confidence signals.

Features contain no threshold-fitted OOF flags, labels, IDs or dataset sources.
Each protocol has a fixed C=1 logistic calibrator with threshold 0.5. This is
post-hoc calibration after model selection on val, not neural training gain.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.common import baseline as b
from scripts.eval.protocol_fusion import audit_reference, audit_branch, indexed, read, sha


def logit(value):
    value = np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(value / (1 - value))


def fit(features, gold, conflicts, aligned):
    weights = np.where(gold, .3 / conflicts, .2 / aligned)
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1., max_iter=2000, tol=1e-9, random_state=20261026)
    model.fit(scaler.transform(features), gold, sample_weight=weights)
    parameters = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                  'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0])}
    return scaler, model, parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--va-run', type=Path, required=True)
    p.add_argument('--vt-run', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); rows = b.read_jsonl(b.ROOT / 'Dataset/val.jsonl')
    y = np.array([r['sample_type'] == 'Conflict' for r in rows]); groups = np.array([r['video'] for r in rows])
    protocols = np.array([r['protocol'] for r in rows]); strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(np.zeros(len(rows)), strata, groups))
    if any(set(groups[tr]) & set(groups[te]) for tr, te in splits): raise ValueError('Video leakage')
    reference, original, (independent, direct) = audit_reference(a.reference, rows, splits)
    directories = {'VT': a.vt_run, 'VA': a.va_run}; specialized = {}
    for protocol, directory in directories.items():
        audit_branch(directory, protocol, rows, reference, original, splits,
                     a.reference / 'grouped_oof_predictions.jsonl')
        specialized[protocol] = indexed(directory / 'predictions.jsonl', rows)
    values = []
    for row in rows:
        sid = row['sample_id']; protocol = row['protocol']
        old_p = direct[sid]['decision_uncertainty']['probabilities']['yes']
        new_p = specialized[protocol][sid]['decision_uncertainty']['probabilities']['yes']
        branches = independent[sid]['branches']
        vision = np.array([branches['vision']['probabilities'][c] for c in b.EMOTIONS])
        other = np.array([branches['text' if protocol == 'VT' else 'audio']['probabilities'][c] for c in b.EMOTIONS])
        if not np.isclose(vision.sum(), 1, atol=1e-5) or not np.isclose(other.sum(), 1, atol=1e-5):
            raise ValueError('Invalid unimodal distributions')
        values.append(logit([old_p, new_p, 1 - np.dot(vision, other)]))
    features = np.array(values)
    if not np.isfinite(features).all(): raise ValueError('Invalid confidence features')
    probability = np.full(len(rows), np.nan); folds = []
    for train, held in splits:
        parameters = {}
        for protocol in ('VT', 'VA'):
            tr, te = train[protocols[train] == protocol], held[protocols[held] == protocol]
            scaler, model, parameters[protocol] = fit(features[tr], y[tr], int(y[train].sum()), int((~y[train]).sum()))
            probability[te] = model.predict_proba(scaler.transform(features[te]))[:, 1]
        folds.append({'heldout_indices': held.tolist(), 'parameters': parameters, 'video_overlap': False})
    if not np.isfinite(probability).all(): raise ValueError('Incomplete heldout probabilities')
    flags = probability > .5; records = []
    for i, row in enumerate(rows):
        sid = row['sample_id']; question = 'none'
        if flags[i]:
            question = next((source[sid]['prediction'].get('clarifying_question')
                for source in (reference, independent, direct)
                if source[sid]['prediction'].get('clarifying_question') not in (None, '', 'none')), None)
        records.append({'sample_id': sid, 'protocol': row['protocol'], 'label_set': 'binary',
            'calibrated_p_yes': float(probability[i]), 'question_pending': question is None,
            'prediction': {'pred_emotion': reference[sid]['prediction']['pred_emotion'],
                'is_uncertain': 'yes' if flags[i] else 'no', 'clarifying_question': question, 'response': ''}})
    full_parameters = {}
    for protocol in ('VT', 'VA'):
        mask = protocols == protocol
        _, _, full_parameters[protocol] = fit(features[mask], y[mask], int(y.sum()), int((~y).sum()))
    pending = sum(r['question_pending'] for r in records)
    if not pending:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Invalid combined output fields')
    summary = {'metrics': b.score_rows(rows, records), 'folds': folds,
        'feature_names': ['old_direct_logit', 'specialist_logit', 'soft_unimodal_disagreement_logit'],
        'C_predeclared': 1., 'threshold_predeclared': .5, 'neural_training_gain_claimed': False,
        'pending_model_questions': pending, 'submission_ready': pending == 0,
        'no_fitted_OOF_flags_used_as_features': True,
        'full_val_deployment_parameters_not_used_for_reported_score': full_parameters,
        'selection_scope': 'Video-grouped OOF calibration after repeated model selection on val; not an independent test score'}
    sources = [Path(__file__), a.reference / 'run_config.json', a.reference / 'grouped_oof_predictions.jsonl']
    for directory in directories.values():
        sources.extend(directory / f for f in ('run_config.json', 'predictions.jsonl', 'metrics.json'))
    manifest = {'input_sha256': {str(f.resolve()): sha(f) for f in sources}, 'features_use_labels_or_IDs': False}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if (a.output_dir / 'run_config.json').exists() and read(a.output_dir / 'run_config.json') != manifest:
        raise ValueError('Preserve the previous calibration sources')
    b.write_json(a.output_dir / 'run_config.json', manifest)
    b.write_json(a.output_dir / 'metrics.json', summary)
    (a.output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if not pending: b.write_prediction_csv(rows, records, a.output_dir / 'predictions.csv')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('folds', 'full_val_deployment_parameters_not_used_for_reported_score')}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
