"""Validate a same-adapter conflict rule using real predictions from both heads.

yes = direct_yes OR (isolated_modality_disagreement AND direct_score > threshold).
The threshold is the only fitted parameter; video-grouped out-of-fold scores
are reported separately from predictions using the threshold fitted on full val.
All final yes questions already exist in at least one genuine model generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b


def verify_confidence_source(score_run, decision_run, score_config, direct_records, score_records):
    if Path(score_run).resolve() == Path(decision_run).resolve():
        # New evaluations save the generation logits alongside each decision.
        reference = {r['sample_id']: r.get('decision_uncertainty') for r in direct_records}
        if any(r.get('decision_uncertainty') is None or
               r['decision_uncertainty'] != reference.get(r['sample_id']) for r in score_records):
            raise ValueError('Inline scores differ from the actual generated decision scores')
    elif score_config.get('reference_prediction_sha256') != hashlib.sha256(
            (Path(decision_run) / 'predictions.jsonl').read_bytes()).hexdigest():
        raise ValueError('Direct scores were conditioned on another prediction run')


def fused_yes(direct, independent, scores, threshold):
    direct, independent, scores = np.asarray(direct, dtype=bool), np.asarray(independent, dtype=bool), np.asarray(scores)
    if not (direct.shape == independent.shape == scores.shape) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid same-adapter conflict features')
    return direct | (independent & (scores > threshold))


def fit_gate(direct, independent, scores, gold):
    direct, independent = np.asarray(direct, dtype=bool), np.asarray(independent, dtype=bool)
    scores, gold = np.asarray(scores, dtype=float), np.asarray(gold, dtype=bool)
    candidates = np.r_[-1e-6, np.unique(scores[independent & ~direct]), 1.]
    positive, negative = int(gold.sum()), int((~gold).sum())
    if not positive or not negative:
        raise ValueError('Both gold conflict types are required for fitting')
    def objective(threshold):
        prediction = fused_yes(direct, independent, scores, threshold)
        tp, tn = int((prediction & gold).sum()), int((~prediction & ~gold).sum())
        return .3 * tp / positive + .2 * tn / negative, tn, threshold
    return float(max(candidates, key=objective))


def make_predictions(rows, independent, direct, flags):
    if len(flags) != len(rows):
        raise ValueError('Prediction flags must cover every source row')
    records = []
    for row, uncertain in zip(rows, flags):
        sid = row['sample_id']
        if uncertain:
            # The rule guarantees every yes has a genuine generated question.
            origin = direct[sid] if direct[sid]['prediction']['is_uncertain'] == 'yes' else independent[sid]
            if origin['prediction']['is_uncertain'] != 'yes':
                raise ValueError('Rule produced yes without an existing model question')
            question = origin['prediction']['clarifying_question']
            question_source = ('direct_' + origin.get('question_origin', 'model')) if origin is direct[sid] else 'independent_base'
        else:
            question, question_source = 'none', 'not_required'
        records.append({'sample_id': sid, 'protocol': b.get_protocol(row), 'label_set': 'binary',
                        'task': 'same_adapter_conflict_fusion', 'question_source': question_source,
                        'prediction': {'pred_emotion': independent[sid]['prediction']['pred_emotion'],
                                       'is_uncertain': 'yes' if uncertain else 'no',
                                       'clarifying_question': question, 'response': ''}})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-run', type=Path, required=True)
    parser.add_argument('--decision-run', type=Path, required=True)
    parser.add_argument('--decision-scores', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    args = parser.parse_args()
    directories = (args.independent_run, args.decision_run, args.decision_scores)
    configs = [json.loads((d / 'run_config.json').read_text()) for d in directories]
    for key in ('adapter_sha256', 'dataset_sha256', 'split'):
        if len({c[key] for c in configs}) != 1:
            raise ValueError('All three runs must use the same ' + key)
    if configs[0]['split'] != 'val' or configs[0].get('auxiliary_mode') != 'independent' or configs[1].get('style') != 'decision':
        raise ValueError('Requires the complete independent/direct official-val runs')
    source = args.dataset_root / 'val.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != configs[0]['dataset_sha256']:
        raise ValueError('Val data changed')
    prediction_paths = [args.independent_run / 'predictions.jsonl', args.decision_run / 'predictions.jsonl',
                        args.decision_scores / 'probabilities.jsonl']
    rows = b.read_jsonl(source)
    loaded = [b.read_jsonl(p) for p in prediction_paths]
    ids = {r['sample_id'] for r in rows}
    for records in loaded:
        if len(records) != len(rows) or {r['sample_id'] for r in records} != ids:
            raise ValueError('Requires complete unique source IDs')
    for records in loaded[:2]:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Source contains invalid labels or questions')
    verify_confidence_source(args.decision_scores, args.decision_run, configs[2], loaded[1], loaded[2])
    independent, direct, score_records = [{r['sample_id']: r for r in records} for records in loaded]
    dy = np.array([direct[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    iy = np.array([independent[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    scores = np.array([score_records[r['sample_id']]['decision_uncertainty']['probabilities']['yes'] for r in rows])
    gold = np.array([r['sample_type'] == 'Conflict' for r in rows])
    groups = np.array([r['video'] for r in rows])
    strata = np.array([b.get_protocol(r) + '_' + r['sample_type'] for r in rows])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026)
    oof = np.zeros(len(rows), dtype=bool); fold_records=[]
    for train, heldout in folds.split(np.zeros(len(rows)), strata, groups):
        assert not (set(groups[train]) & set(groups[heldout]))
        threshold = fit_gate(dy[train], iy[train], scores[train], gold[train])
        oof[heldout] = fused_yes(dy[heldout], iy[heldout], scores[heldout], threshold)
        fold_records.append({'heldout_indices': heldout.tolist(), 'threshold': threshold, 'video_overlap': False})
    threshold = fit_gate(dy, iy, scores, gold)
    predicted = fused_yes(dy, iy, scores, threshold)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = {'split': 'val', 'adapter_sha256': configs[0]['adapter_sha256'],
                'dataset_sha256': configs[0]['dataset_sha256'], 'method': __doc__, 'threshold': threshold,
                'threshold_fit_dataset': 'official val; apparent full-fit predictions, grouped OOF reported separately',
                'features_use_gold_or_ids': False, 'model_training_gain_claimed': False,
                'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (*prediction_paths, Path(__file__))}}
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Fusion configuration changed; preserve the existing run')
    b.write_json(out / 'run_config.json', manifest)
    records = make_predictions(rows, independent, direct, predicted)
    oof_records = make_predictions(rows, independent, direct, oof)
    (out / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    metric = b.finalize(SimpleNamespace(split='val', label_set='binary', emotion_reference=None), rows, rows, records, out)
    b.write_json(out / 'grouped_oof_metrics.json', {'folds': fold_records, 'metrics': b.score_rows(rows, oof_records),
                 'questions_all_generated_by_models': True, 'same_video_never_in_both_fold_parts': True})
    print(json.dumps({'full_val_fitted_score': metric['score_sklearn_union'],
                      'grouped_OOF_score': b.score_rows(rows, oof_records)['score_sklearn_union'], 'threshold': threshold}), flush=True)


if __name__ == '__main__':
    main()
