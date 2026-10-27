"""Native VT conflict SFT and preselected evaluations, preserving the VA study."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.common import baseline as b
from scripts.common.jobs import identity

OUT = b.ROOT / 'outputs/vt_conflict_v8'
VA_STUDY = b.ROOT / 'outputs/va_conflict_v7'
REFERENCE = b.ROOT / 'outputs/vision_omni_v4_joint/evaluation/vision_800_fused'
INITIAL = b.ROOT / 'outputs/vision_omni_v4_joint/candidate_adapters/vision_frozen_language_800'


def read(path):
    return json.loads(path.read_text())


def status(job):
    if (job / 'exit.json').exists():
        result = read(job / 'exit.json')
        if result['exit_code'] != 0:
            raise RuntimeError(f'{job}: actual exit {result["exit_code"]}; inspect console.log')
        return 'complete'
    marker = job / 'worker.json'
    if not marker.exists(): marker = job / 'launch.json'
    value = read(marker)
    if identity(value.get('pid', value.get('supervisor_pid'))) != value['kernel_start_time']:
        time.sleep(2)
        if (job / 'exit.json').exists(): return status(job)
        raise RuntimeError(f'{job}: no live process or authoritative terminal record')
    return 'running'


def wait(job):
    while status(job) != 'complete': time.sleep(20)


def launch(name, command, gpu='1'):
    job = OUT / 'jobs' / name
    if not (job / 'launch.json').exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                   OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
        subprocess.run([sys.executable, '-m', 'scripts.common.jobs', '--job-dir', str(job),
                        '--', *map(str, command)], env=env, cwd=b.ROOT, check=True)
    return job


def evaluation(name, adapter):
    job = launch(name, [sys.executable, '-u', '-m', 'scripts.eval.vt_conflict',
        '--adapter', adapter, '--omni-reference', REFERENCE, '--output-dir', OUT / name], gpu='0')
    wait(job)
    metric = read(OUT / name / 'metrics.json')
    if metric['grouped_OOF_metrics']['valid_predictions'] != 666:
        raise RuntimeError('Requires all 666 valid validation classifications')
    print(json.dumps({'checkpoint': name, 'raw_metrics': metric['raw_metrics'],
                      'grouped_OOF_metrics': metric['grouped_OOF_metrics']}), flush=True)
    return metric


def wait_va_study():
    completion = VA_STUDY / 'evaluation_complete.json'
    while not completion.exists():
        recovery = VA_STUDY / 'jobs/evaluation_recovery'
        job = recovery if (recovery / 'launch.json').exists() else VA_STUDY / 'jobs/evaluation_pipeline'
        if status(job) == 'complete' and not completion.exists():
            raise RuntimeError('VA orchestration ended without its complete evaluation artifact')
        time.sleep(20)
    result = read(completion)
    if result['training_optimizer_updates'] != 287 or result['sft_final']['raw_metrics']['valid_predictions'] != 666:
        raise RuntimeError('Incomplete recovered VA study')


def main():
    baseline = evaluation('baseline_init', INITIAL)
    # Keep GPU1 free of this training job during the same-prompt baseline.
    # At most one VA/VT evaluator will later share GPU1 with the native SFT.
    while not (OUT / 'jobs/train/launch.json').exists():
        free = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 60000: break
        print(f'Waiting for GPU1 training capacity: {free} MiB free', flush=True)
        time.sleep(30)
    train = launch('train', [b.ROOT / '.venv/bin/llamafactory-cli', 'train',
                             b.ROOT / 'configs/sft/vt_conflict_v8.yaml'])
    checkpoint = OUT / 'train/checkpoint-324'
    while not all((checkpoint / f).exists() for f in
                  ('trainer_state.json', 'adapter_config.json', 'adapter_model.safetensors')):
        if status(train) == 'complete':
            raise RuntimeError('Training ended without preselected checkpoint 324')
        time.sleep(20)
    if read(checkpoint / 'trainer_state.json')['global_step'] != 324:
        raise RuntimeError('Unexpected VT midpoint step')
    # GPU0 evaluates after VA recovery; GPU1 continues the existing native SFT.
    wait_va_study()
    midpoint = evaluation('sft_324', checkpoint)
    wait(train)
    state = read(OUT / 'train/trainer_state.json')
    if state['global_step'] != 647: raise RuntimeError('Incomplete VT native SFT')
    final = evaluation('sft_final', OUT / 'train')
    b.write_json(OUT / 'evaluation_complete.json', {
        'training_optimizer_updates': state['global_step'], 'preselected_checkpoint_steps': [324, 647],
        'baseline': baseline, 'sft_324': midpoint, 'sft_final': final,
        'emotion_and_VA_frozen': True, 'test_labels_used': False,
        'question_generation_required_before_submission': True})


if __name__ == '__main__': main()
