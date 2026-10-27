"""Diagnose uncertainty thresholds with video-grouped out-of-fold validation.

This writes score diagnostics only, not ready-to-submit predictions. Questions
must still be generated for any newly uncertain inputs. Threshold fitting is a
separate inference calibration stage and is never described as SFT/RL training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b


def conflict_scores(record):
    if 'decision_uncertainty' in record:
        probability = record['decision_uncertainty']['probabilities']['yes']
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid direct-conflict probability')
        return {'direct_conflict': float(probability)}
    def vector(task):
        probabilities = record['branches'][task]['probabilities']
        result = np.array([probabilities[k] for k in ('positive', 'negative', 'neutral')])
        if not np.isfinite(result).all() or np.any(result < 0) or not np.isclose(result.sum(), 1, atol=1e-5):
            raise ValueError('Invalid class distribution')
        return result
    visual = vector('vision')
    other = vector('text' if record['protocol'] == 'VT' else 'audio')
    return {'independent_disagreement': float(1 - np.dot(visual, other)),
            'distribution_distance': float(.5 * np.abs(visual - other).sum())}


def fit_threshold(scores, gold, total_conflicts=None, total_aligned=None):
    scores, gold = np.asarray(scores), np.asarray(gold, dtype=bool)
    if not len(scores):
        raise ValueError('No calibration samples')
    conflicts = total_conflicts if total_conflicts is not None else int(gold.sum())
    aligned = total_aligned if total_aligned is not None else int((~gold).sum())
    if not conflicts or not aligned:
        raise ValueError('Both conflict and aligned targets are needed')
    choices = np.r_[-1e-6, np.unique(scores)]
    # Strict > permits both all-yes and all-no endpoints. In an exact utility
    # tie prefer fewer false yes, then the larger threshold for determinism.
    def objective(threshold):
        predicted = scores > threshold
        tp, tn = int((predicted & gold).sum()), int((~predicted & ~gold).sum())
        return .3 * tp / conflicts + .2 * tn / aligned, tn, threshold
    return float(max(choices, key=objective))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probability-run', type=Path, required=True)
    parser.add_argument('--emotion-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    args = parser.parse_args(argv)
    probability_config = json.loads((args.probability_run / 'run_config.json').read_text())
    emotion_config = json.loads((args.emotion_run / 'run_config.json').read_text())
    for key in ('adapter_sha256', 'dataset_sha256', 'split'):
        if probability_config[key] != emotion_config[key]:
            raise ValueError('Probability and emotion runs differ: ' + key)
    if probability_config['split'] != 'val':
        raise ValueError('This diagnostic fits only on official val; never test')
    source = args.dataset_root / 'val.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != probability_config['dataset_sha256']:
        raise ValueError('Val data changed')
    rows = b.read_jsonl(source)
    probability_rows = b.read_jsonl(args.probability_run / 'probabilities.jsonl')
    emotion_rows = b.read_jsonl(args.emotion_run / 'predictions.jsonl')
    for records in (probability_rows, emotion_rows):
        if len(records) != len(rows) or {r['sample_id'] for r in records} != {r['sample_id'] for r in rows}:
            raise ValueError('Requires complete unique full-val records')
    probabilities = {r['sample_id']: r for r in probability_rows}
    emotions = {r['sample_id']: r['prediction']['pred_emotion'] for r in emotion_rows}
    gold = np.array([r['sample_type'] == 'Conflict' for r in rows])
    protocols = np.array([b.get_protocol(r) for r in rows])
    groups = np.array([r['video'] for r in rows])
    strata = np.array([p + ('_conflict' if y else '_aligned') for p, y in zip(protocols, gold)])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(np.zeros(len(rows)), strata, groups))

    def metric(predicted):
        records = [{'sample_id': row['sample_id'], 'prediction': {'pred_emotion': emotions[row['sample_id']],
                    'is_uncertain': 'yes' if predicted[index] else 'no'}} for index, row in enumerate(rows)]
        result = b.score_rows(rows, records)
        return {k: result[k] for k in ('samples', 'macro_f1_sklearn_union', 'conflict_recognition_rate',
                                       'aligned_calibration_rate', 'score_sklearn_union')}

    scores = [conflict_scores(probabilities[r['sample_id']]) for r in rows]
    results = []
    for feature in scores[0]:
        values = np.array([r[feature] for r in scores])
        for separate_protocols in (False, True):
            predictions, full_predictions = np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool)
            fold_records, full_thresholds = [], {}
            for train, heldout in folds:
                assert not (set(groups[train]) & set(groups[heldout]))
                thresholds = {}
                for protocol in ('VT', 'VA') if separate_protocols else ('all',):
                    mask = protocols[train] == protocol if protocol != 'all' else np.ones(len(train), dtype=bool)
                    threshold = fit_threshold(values[train][mask], gold[train][mask], int(gold[train].sum()), int((~gold[train]).sum()))
                    thresholds[protocol] = threshold
                    target = heldout[protocols[heldout] == protocol] if protocol != 'all' else heldout
                    predictions[target] = values[target] > threshold
                fold_records.append({'heldout_indices': heldout.tolist(), 'thresholds': thresholds,
                                     'train_videos': len(set(groups[train])), 'heldout_videos': len(set(groups[heldout])),
                                     'video_overlap': False})
            for protocol in ('VT', 'VA') if separate_protocols else ('all',):
                mask = protocols == protocol if protocol != 'all' else np.ones(len(rows), dtype=bool)
                threshold = fit_threshold(values[mask], gold[mask], int(gold.sum()), int((~gold).sum()))
                full_predictions[mask] = values[mask] > threshold
                full_thresholds[protocol] = threshold
            results.append({'feature': feature, 'separate_protocols': separate_protocols,
                            'grouped_out_of_fold_metrics': metric(predictions),
                            'full_val_fitted_apparent_metrics': metric(full_predictions),
                            'full_val_fitted_thresholds': full_thresholds, 'folds': fold_records})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    b.write_json(out / 'diagnostic.json', {
        'rows': len(rows), 'same_adapter_emotions_preserved': True, 'score_formula': '.5 F1 + .3 conflict recall + .2 aligned specificity',
        'classification_model_training_used_val': False, 'threshold_fit_dataset': 'val, video-grouped 5-fold; apparent full-fit scores separately labeled',
        'this_is_SFT_or_RL_gain': False, 'submission_ready': False, 'clarifying_questions_generated_for_new_yes': False,
        'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (args.probability_run / 'probabilities.jsonl', args.emotion_run / 'predictions.jsonl', Path(__file__))},
        'results': results})
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'folds'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
