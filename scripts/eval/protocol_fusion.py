"""Audit and combine protocol-specific conflict classifiers on complete val only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.common import baseline as b
from scripts.eval.calibration import fit_threshold
from scripts.eval.fuse import fit_gate, fused_yes, verify_confidence_source


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(path, rows):
    records = b.read_jsonl(path)
    result = {r['sample_id']: r for r in records}
    if len(records) != len(rows) or set(result) != {r['sample_id'] for r in rows}:
        raise ValueError(f'Incomplete or duplicate IDs: {path}')
    return result


def audit_reference(directory, rows, splits):
    path = directory / 'grouped_oof_predictions.jsonl'
    reference = indexed(path, rows)
    config = read(directory / 'run_config.json')
    if config['dataset_sha256'] != sha(b.ROOT / 'Dataset/val.jsonl'):
        raise ValueError('Reference validation data changed')
    folders = {}
    for name, expected in config['source_sha256'].items():
        source = Path(name)
        if sha(source) != expected: raise ValueError(f'Reference source changed: {source}')
        if source.suffix == '.jsonl':
            cfg = read(source.parent / 'run_config.json')
            if cfg['adapter_sha256'] != config['adapter_sha256']:
                raise ValueError('Reference heads used different adapters')
            folders['scores' if source.name == 'probabilities.jsonl' else
                    'independent' if cfg.get('auxiliary_mode') == 'independent' else 'direct'] = source
    independent, direct, scores = [indexed(folders[k], rows) for k in ('independent', 'direct', 'scores')]
    verify_confidence_source(folders['scores'].parent, folders['direct'].parent,
        read(folders['scores'].parent / 'run_config.json'), list(direct.values()), list(scores.values()))
    dy = np.array([direct[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    iy = np.array([independent[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    p = np.array([scores[r['sample_id']]['decision_uncertainty']['probabilities']['yes'] for r in rows])
    y = np.array([r['sample_type'] == 'Conflict' for r in rows])
    expected = np.array([reference[r['sample_id']]['prediction']['is_uncertain'] == 'yes' for r in rows])
    folds = read(directory / 'grouped_oof_metrics.json')['folds']
    for (train, held), fold in zip(splits, folds, strict=True):
        threshold = fit_gate(dy[train], iy[train], p[train], y[train])
        if held.tolist() != fold['heldout_indices'] or threshold != fold['threshold']:
            raise ValueError('Reference threshold cannot be reproduced from its training fold')
        if not np.array_equal(fused_yes(dy[held], iy[held], p[held], threshold), expected[held]):
            raise ValueError('Reference OOF predictions cannot be reproduced')
    return reference, expected, (independent, direct)


def audit_branch(directory, protocol, rows, reference, original_flags, splits, reference_path):
    config = read(directory / 'run_config.json')
    if (config['split'] != 'val' or config['dataset_sha256'] != sha(b.ROOT / 'Dataset/val.jsonl')
            or config['frozen_reference_sha256'] != sha(reference_path)
            or config['adapter_sha256'] != sha(Path(config['adapter']) / 'adapter_model.safetensors')):
        raise ValueError('Branch data, frozen reference or weights changed')
    predictions = indexed(directory / 'predictions.jsonl', rows)
    metric = read(directory / 'metrics.json')
    flags = original_flags.copy(); y = np.array([r['sample_type'] == 'Conflict' for r in rows])
    selected = np.array([r['protocol'] == protocol for r in rows]); p = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        r = predictions[row['sample_id']]
        if r['prediction']['pred_emotion'] != reference[row['sample_id']]['prediction']['pred_emotion']:
            raise ValueError('A supposedly frozen emotion prediction changed')
        if selected[i]:
            prob = r['decision_uncertainty']['probabilities']
            if not np.isfinite(list(prob.values())).all() or not np.isclose(sum(prob.values()), 1):
                raise ValueError('Invalid classifier probabilities')
            p[i] = prob['yes']
            if not 0 <= p[i] <= 1: raise ValueError('Invalid yes probability')
        elif (r['prediction']['is_uncertain'] == 'yes') != original_flags[i]:
            raise ValueError('The other protocol was not frozen')
    for (train, held), fold in zip(splits, metric['folds'], strict=True):
        tr, te = train[selected[train]], held[selected[held]]
        threshold = fit_threshold(p[tr], y[tr], int(y[train].sum()), int((~y[train]).sum()))
        if held.tolist() != fold['heldout_indices'] or threshold != fold[protocol + '_threshold']:
            raise ValueError('Branch threshold cannot be reproduced from its training fold')
        flags[te] = p[te] > threshold
    if not np.array_equal(flags, np.array(metric['OOF_uncertainty_flags'], dtype=bool)):
        raise ValueError('Branch OOF flags cannot be reproduced')
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused')
    parser.add_argument('--va-run', type=Path)
    parser.add_argument('--vt-run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    a = parser.parse_args(); rows = b.read_jsonl(b.ROOT / 'Dataset/val.jsonl')
    groups = np.array([r['video'] for r in rows])
    strata = np.array([r['protocol'] + '_' + r['sample_type'] for r in rows])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(
        np.zeros(len(rows)), strata, groups))
    if any(set(groups[train]) & set(groups[held]) for train, held in splits):
        raise ValueError('Video leakage across calibration folds')
    reference, original, question_sources = audit_reference(a.reference, rows, splits)
    flags = original.copy()
    for protocol, directory in (('VA', a.va_run), ('VT', a.vt_run)):
        if directory:
            candidate = audit_branch(directory, protocol, rows, reference, original, splits,
                                     a.reference / 'grouped_oof_predictions.jsonl')
            mask = np.array([r['protocol'] == protocol for r in rows]); flags[mask] = candidate[mask]
    records = []
    for i, row in enumerate(rows):
        sid = row['sample_id']; question = 'none'
        if flags[i]:
            question = next((origin[sid]['prediction'].get('clarifying_question')
                for origin in (reference, *question_sources)
                if origin[sid]['prediction'].get('clarifying_question') not in (None, '', 'none')), None)
        records.append({'sample_id': sid, 'protocol': row['protocol'], 'label_set': 'binary',
            'question_pending': question is None, 'prediction': {
                'pred_emotion': reference[sid]['prediction']['pred_emotion'],
                'is_uncertain': 'yes' if flags[i] else 'no', 'clarifying_question': question, 'response': ''}})
    pending = sum(r['question_pending'] for r in records)
    if not pending:
        _, invalid = b.checked_predictions(rows, records)
        if invalid: raise ValueError('Combined labels or question fields are invalid')
    summary = {'metrics': b.score_rows(rows, records), 'all_thresholds_refitted_on_training_folds': True,
        'all_saved_OOF_flags_reproduced': True, 'video_disjoint_folds': True, 'M12_frozen': True,
        'VA_run': str(a.va_run), 'VT_run': str(a.vt_run), 'pending_model_questions': pending,
        'submission_ready': pending == 0, 'selection_scope': 'val; repeated model selection, not an independent test score'}
    sources = [b.ROOT / 'Dataset/val.jsonl', a.reference / 'run_config.json',
               a.reference / 'grouped_oof_predictions.jsonl', a.reference / 'grouped_oof_metrics.json']
    for directory in (a.va_run, a.vt_run):
        if directory:
            sources.extend(directory / name for name in ('run_config.json', 'predictions.jsonl', 'metrics.json'))
    manifest = {'split': 'val', 'source_sha256': sha(Path(__file__)),
                'input_sha256': {str(p.resolve()): sha(p) for p in sources},
                'calibration_scope': 'video-grouped OOF; held videos never used to fit their thresholds'}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = a.output_dir / 'run_config.json'
    if config_path.exists() and read(config_path) != manifest:
        raise ValueError('Preserve the existing composition: inputs or implementation changed')
    b.write_json(config_path, manifest)
    (a.output_dir / 'grouped_oof_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if not pending:
        b.write_prediction_csv(rows, records, a.output_dir / 'grouped_oof_predictions.csv')
    b.write_json(a.output_dir / 'metrics.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
