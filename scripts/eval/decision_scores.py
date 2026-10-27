"""Score direct conflict yes/no using the student's own generated emotion prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.common import baseline as b
from scripts.common import distill_tasks as tasks
from scripts.common.omni_policy import load_policy
from scripts.eval.probabilities import conditional_probabilities


def decision_prefix(predicted_emotion):
    if predicted_emotion not in ('positive', 'negative'):
        raise ValueError('Expected the student binary emotion, never a gold label')
    return '{"pred_emotion":"' + predicted_emotion + '","is_uncertain":"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    args = parser.parse_args()
    config = json.loads((args.decision_run / 'run_config.json').read_text())
    if config.get('style') != 'decision':
        raise ValueError('Requires a completed direct-decision run')
    adapter = Path(config['adapter'])
    source = args.dataset_root / ('val.jsonl' if config['split'] == 'val' else 'test_release.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['dataset_sha256']:
        raise ValueError('Source data changed')
    if hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() != config['adapter_sha256']:
        raise ValueError('Student adapter changed')
    rows = b.read_jsonl(source)
    reference = b.read_jsonl(args.decision_run / 'predictions.jsonl')
    valid, invalid = b.checked_predictions(rows, reference)
    if invalid or len(reference) != len(rows) or len(valid) != len(rows):
        raise ValueError('Requires complete valid student decisions')
    manifest = {'split': config['split'], 'adapter': str(adapter.resolve()),
                'adapter_sha256': config['adapter_sha256'], 'dataset_sha256': config['dataset_sha256'],
                'sample_ids': [r['sample_id'] for r in rows], 'input_mode': 'protocol_direct',
                'conditioning_emotion': 'Student-generated pred_emotion from the completed decision run; no gold labels',
                'calibrated': False, 'labels_used_for_forward': False,
                'reference_prediction_sha256': hashlib.sha256((args.decision_run / 'predictions.jsonl').read_bytes()).hexdigest(),
                'source_sha256': {str(p.relative_to(b.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), Path(tasks.__file__),
                                            b.ROOT / 'scripts/eval/probabilities.py', b.ROOT / 'scripts/common/omni_policy.py')}}
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Scoring config changed; preserve the previous run')
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'probabilities.jsonl'
    previous = b.read_jsonl(path) if path.exists() else []
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - set(manifest['sample_ids']):
        raise ValueError('Unexpected/duplicate decision score IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        model, processor, environment = load_policy(adapter)
        b.write_json(out / 'environment.json', environment)
        with path.open('a', encoding='utf-8', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                sid = row['sample_id']
                predicted_emotion = valid[sid]['pred_emotion']
                inputs, media = tasks.prepare_inputs(processor, row, 'decision', args.dataset_root)
                inputs = inputs.to('cuda:0').to(torch.bfloat16)
                scores = conditional_probabilities(model, processor.tokenizer, inputs, media,
                                                   decision_prefix(predicted_emotion), ('yes', 'no'))
                record = {'sample_id': sid, 'protocol': b.get_protocol(row), 'media': media,
                          'predicted_emotion_condition': predicted_emotion, 'decision_uncertainty': scores}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                previous.append(record)
                del inputs
                # Release temporary forward buffers while sharing the inference card.
                torch.cuda.empty_cache()
                print(f'[{index}/{len(remaining)}] {sid}', flush=True)
    if len(previous) != len(rows):
        raise ValueError('Incomplete decision scores')
    b.write_json(out / 'status.json', {'status': 'complete', 'rows': len(rows), 'calibrated': False,
                 'probabilities_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
