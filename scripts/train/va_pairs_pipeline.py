"""Evaluate the two preselected train-only VA pairing SFT checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.common import baseline as b
from scripts.train.vt_conflict_pipeline import read, status, wait

OUT = b.ROOT / 'outputs/va_pairs_v9'
REFERENCE = b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused'


def run(name, command, gpu=''):
    job = OUT / 'jobs' / name
    if not (job / 'launch.json').exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
                   OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
        subprocess.run([sys.executable, '-m', 'scripts.common.jobs', '--job-dir', str(job),
                        '--', *map(str, command)], cwd=b.ROOT, env=env, check=True)
    wait(job)


def evaluate(name, adapter):
    destination = OUT / name
    run(name, [sys.executable, '-u', '-m', 'scripts.eval.va_conflict', '--adapter', adapter,
        '--omni-reference', REFERENCE, '--output-dir', destination], gpu='0')
    metric = read(destination / 'metrics.json')
    if metric['grouped_OOF_metrics']['valid_predictions'] != 666:
        raise RuntimeError('Incomplete VA pairing evaluation')
    run(name + '_refit', [sys.executable, '-u', '-m', 'scripts.eval.protocol_fusion',
        '--va-run', destination, '--output-dir', destination / 'refit_audit'])
    run(name + '_raw', [sys.executable, '-u', '-m', 'scripts.eval.raw_protocol_fusion',
        '--protocol', 'VA', '--classifier-run', destination, '--output-dir', destination / 'raw_combination'])
    print(json.dumps({'stage': name, 'raw': metric['raw_metrics'],
                      'grouped_OOF': metric['grouped_OOF_metrics']}), flush=True)
    return metric


def main():
    train = OUT / 'jobs/train'; checkpoint = OUT / 'train/checkpoint-269'
    while not all((checkpoint / name).exists() for name in
                  ('trainer_state.json', 'adapter_model.safetensors', 'adapter_config.json')):
        if status(train) == 'complete':
            raise RuntimeError('Training ended without the preselected paired-data checkpoint')
        time.sleep(20)
    if read(checkpoint / 'trainer_state.json')['global_step'] != 269:
        raise RuntimeError('Unexpected VA pairing checkpoint step')
    midpoint = evaluate('sft_269', checkpoint)
    wait(train)
    audit = read(OUT / 'train/input_and_weight_audit.json')
    if audit['global_step'] != 537 or audit['frozen_visual_tensors_bitwise_preserved'] != 64:
        raise RuntimeError('Native paired-data training audit failed')
    final = evaluate('sft_final', OUT / 'train')
    temporary = OUT / '.evaluation_complete.tmp.json'
    b.write_json(temporary, {'native_optimizer_steps': 537, 'preselected_steps': [269, 537],
        'source_baseline': 'outputs/va_conflict_v7/sft_final', 'sft_269': midpoint,
        'sft_final': final, 'training_audit': audit, 'test_labels_used': False,
        'synthetic_M12_labels_created': False, 'all_emotion_and_VT_predictions_frozen': True})
    temporary.replace(OUT / 'evaluation_complete.json')


if __name__ == '__main__': main()
