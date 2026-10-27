"""Mine starting-policy errors on original TRAIN inputs for an RL curriculum."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from scripts.common import baseline as b
from scripts.common import rl_tasks as tasks
from scripts.common import distill_tasks
from scripts.common.omni_policy import load_policy, generate_text
from scripts.common.parsing import parse_task, parse_decision


def select_rows(rows, limit, seed):
    if limit < 1:
        raise ValueError('Mining limit must be positive')
    # Keep all rare conflicts when the limit permits; distribute aligned examples
    # across protocol, emotion, and generated/real strata without using val labels.
    eligible = [r for r in rows if r['views']['M12']['label'] in ('positive', 'negative')]
    rng = random.Random(seed)
    conflicts = [r for r in eligible if r['sample_type'] == 'Conflict']
    rng.shuffle(conflicts)
    selected = conflicts[:limit]
    pools = defaultdict(list)
    for row in eligible:
        if row['sample_type'] == 'Aligned':
            pools[(b.get_protocol(row), row['views']['M12']['label'], b.is_generated(row))].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    while len(selected) < min(limit, len(eligible)) and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(selected) < limit:
                selected.append(pools[key].pop())
    rng.shuffle(selected)
    return selected


def prediction_errors(row, prediction):
    errors = {'emotion': prediction['pred_emotion'] != row['views']['M12']['label']}
    if 'vision_emotion' in prediction and 'other_emotion' in prediction:
        errors.update(vision=prediction['vision_emotion'] != row['views']['M1']['label'],
                      other=prediction['other_emotion'] != row['views']['M2']['label'],
                      conflict=(prediction['vision_emotion'] != prediction['other_emotion']) != (row['sample_type'] == 'Conflict'))
    if 'direct_is_uncertain' in prediction:
        errors.update(direct_emotion=prediction['direct_pred_emotion'] != row['views']['M12']['label'],
                      direct_conflict=(prediction['direct_is_uncertain'] == 'yes') != (row['sample_type'] == 'Conflict'))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    parser.add_argument('--limit', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--include-decision', action='store_true', help='Also mine the trained direct conflict head')
    parser.add_argument('--decision-only', action='store_true', help='Mine only overall emotion and direct decisions for decision-only RL')
    args = parser.parse_args()
    if args.decision_only and not args.include_decision:
        parser.error('--decision-only requires --include-decision')
    path = args.dataset_root / 'train.jsonl'
    rows = select_rows(b.read_jsonl(path), args.limit, args.seed)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = {'train_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                'source_sha256': {str(p.relative_to(b.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), Path(tasks.__file__), Path(distill_tasks.__file__))},
                'sample_ids': [r['sample_id'] for r in rows], 'seed': args.seed,
                'validation_or_test_used': False, 'input_mode': 'independent', 'include_decision': args.include_decision,
                'decision_only': args.decision_only}
    config = out / 'run_config.json'
    if config.exists() and json.loads(config.read_text()) != manifest:
        raise ValueError('Mining config changed; preserve the existing run')
    b.write_json(config, manifest)
    trace = out / 'predictions.jsonl'
    previous = b.read_jsonl(trace) if trace.exists() else []
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - set(manifest['sample_ids']):
        raise ValueError('Unexpected/duplicate training prediction IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        model, processor, environment = load_policy(args.adapter)
        b.write_json(out / 'environment.json', environment)
        with trace.open('a', encoding='utf-8', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                prediction, raw_outputs, media_inputs = {}, {}, {}
                requested = (('emotion', 'decision') if args.decision_only else
                             ('emotion', *tasks.modality_tasks(row)) + (('decision',) if args.include_decision else ()))
                for task in requested:
                    prepare = distill_tasks.prepare_inputs if task == 'decision' else tasks.prepare_inputs
                    inputs, media = prepare(processor, row, task, args.dataset_root)
                    raw = generate_text(model, processor, inputs, media)
                    if task == 'decision':
                        direct = parse_decision(raw)
                        prediction.update(direct_pred_emotion=direct['pred_emotion'], direct_is_uncertain=direct['is_uncertain'])
                    elif task == 'emotion':
                        prediction['pred_emotion'] = parse_task(raw, task)['pred_emotion']
                    else:
                        prediction['vision_emotion' if task == 'vision' else 'other_emotion'] = tasks.parse_modality(raw, task)
                    raw_outputs[task], media_inputs[task] = raw, media
                    del inputs
                record = {'sample_id': row['sample_id'], 'protocol': b.get_protocol(row),
                          'prediction': prediction, 'errors': prediction_errors(row, prediction),
                          'raw_outputs': raw_outputs, 'media': media_inputs}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                previous.append(record)
                print(f'[{index}/{len(remaining)}] {row["sample_id"]} {record["errors"]}', flush=True)
    by_id = {r['sample_id']: r for r in previous}
    if set(by_id) != set(manifest['sample_ids']):
        raise ValueError('Incomplete mining results')
    counts = Counter(k for r in previous for k, wrong in r['errors'].items() if wrong)
    curriculum = {**manifest, 'completed': True, 'records': [by_id[r['sample_id']] for r in rows],
                  'error_counts': dict(counts), 'rows': len(rows)}
    b.write_json(out / 'curriculum.json', curriculum)
    print(json.dumps({'rows': len(rows), 'error_counts': dict(counts)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
