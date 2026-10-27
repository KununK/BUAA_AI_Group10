"""Evaluate the two preselected VA-only SFT checkpoints without restarting jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.common import baseline as b
from scripts.common.jobs import identity

OUT = b.ROOT / 'outputs/va_conflict_v7'
REFERENCE = b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused'


def read(path):
    return json.loads(path.read_text())


def status(name):
    job = OUT / 'jobs' / name
    if (job / 'exit.json').exists():
        code = read(job / 'exit.json')['exit_code']
        if code: raise RuntimeError(f'{name} exited with {code}; inspect console.log')
        return 'complete'
    marker = job / 'worker.json'
    if not marker.exists(): marker = job / 'launch.json'
    value = read(marker)
    pid = value.get('pid', value.get('supervisor_pid'))
    if identity(pid) != value['kernel_start_time']:
        # Allow the supervisor to persist its exit record after the worker ends.
        time.sleep(2)
        if (job / 'exit.json').exists(): return status(name)
        raise RuntimeError(f'{name} has no live recorded process or terminal exit record')
    return 'running'


def wait(name):
    while status(name) != 'complete': time.sleep(20)


def evaluate(name, adapter):
    job = OUT / 'jobs' / name
    if not (job / 'launch.json').exists():
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1',
                           OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
        subprocess.run([sys.executable, '-m', 'scripts.common.jobs', '--job-dir', str(job), '--',
            sys.executable, '-u', '-m', 'scripts.eval.va_conflict', '--adapter', str(adapter),
            '--omni-reference', str(REFERENCE), '--output-dir', str(OUT / name)],
            env=environment, cwd=b.ROOT, check=True)
    wait(name)
    metric = read(OUT / name / 'metrics.json')
    if metric['grouped_OOF_metrics']['samples'] != 666:
        raise RuntimeError('Incomplete VA checkpoint evaluation')
    print(json.dumps({'checkpoint': name, 'raw': metric['raw_metrics'],
                      'grouped_OOF': metric['grouped_OOF_metrics']}), flush=True)
    return metric


def main():
    # Each checkpoint sees the same frozen VT/M12 reference and the same folds.
    wait('baseline_init')
    checkpoint = OUT / 'train/checkpoint-144'
    while not all((checkpoint / f).exists() for f in
                  ('trainer_state.json', 'adapter_config.json', 'adapter_model.safetensors')):
        if status('train') == 'complete':
            raise RuntimeError('Native training ended without the preselected checkpoint 144')
        time.sleep(20)
    # Trainer writes trainer_state after all checkpoint weights and optimizer data.
    if read(checkpoint / 'trainer_state.json')['global_step'] != 144:
        raise RuntimeError('Unexpected checkpoint step')
    midpoint = evaluate('sft_144', checkpoint)
    wait('train')
    state = read(OUT / 'train/trainer_state.json')
    if state['global_step'] != 287: raise RuntimeError('Incomplete native SFT run')
    final = evaluate('sft_final', OUT / 'train')
    b.write_json(OUT / 'evaluation_complete.json', {
        'training_optimizer_updates': state['global_step'], 'preselected_checkpoint_steps': [144, 287],
        'baseline': read(OUT / 'baseline_init/metrics.json'), 'sft_144': midpoint, 'sft_final': final,
        'emotion_and_VT_frozen': True, 'test_labels_used': False,
        'question_generation_required_before_submission': True})


if __name__ == '__main__': main()
