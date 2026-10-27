"""Run recorded native SFT jobs and full-val candidate evaluations on GPUs 1/0.

This driver never restarts a vanished worker or a failed job automatically.
Every child retains its own console, command, PID identity and terminal exit code.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.common import baseline as b
from scripts.common.jobs import identity

ROOT = b.ROOT
OUT = ROOT / 'outputs/distill_omni_v3'
PYTHON = str(ROOT / '.venv/bin/python')


def state(name):
    directory = OUT / 'jobs' / name
    terminal = directory / 'exit.json'
    if terminal.exists():
        result = json.loads(terminal.read_text())
        if result['exit_code']:
            raise RuntimeError(f'{name} failed with exit {result["exit_code"]}; inspect its console.log')
        return 'complete'
    for filename, pid_key in [('worker.json', 'pid'), ('launch.json', 'supervisor_pid')]:
        file = directory / filename
        if file.exists():
            saved = json.loads(file.read_text())
            if identity(saved[pid_key]) == saved['kernel_start_time']:
                return 'running'
    # A worker may have exited just before its supervisor writes exit.json.
    time.sleep(1)
    if terminal.exists():
        return state(name)
    raise RuntimeError(f'{name} has no live matching process and no terminal result; do not restart blindly')


def wait(name):
    while state(name) != 'complete':
        time.sleep(10)


def launch(name, module, arguments, gpu):
    command = [PYTHON, '-u', '-m', module, *map(str, arguments)]
    directory = OUT / 'jobs' / name
    if (directory / 'launch.json').exists():
        old = json.loads((directory / 'launch.json').read_text())
        if old['command'] != command or old['CUDA_VISIBLE_DEVICES'] != str(gpu):
            raise RuntimeError(f'Existing job {name} has a different command')
        state(name)
        return name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    subprocess.run([PYTHON, '-m', 'scripts.common.jobs', '--job-dir', str(directory), '--', *command],
                   cwd=ROOT, env=env, check=True)
    return name


def summary(stage):
    results = []
    for path in sorted((OUT / 'evaluation').glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        if metrics.get('valid_predictions') != 666:
            continue
        results.append({'run': path.parent.name, 'metrics_path': str(path.relative_to(ROOT)), **metrics})
    b.write_json(OUT / 'comparison.json', {'stage': stage, 'results': results,
                 'target_score': .95, 'test_labels_used': False, 'RL_completed_in_this_pipeline': False})
    b.write_json(OUT / 'status.json', {'stage': stage, 'pipeline_pid': os.getpid(),
                 'completed_full_val_runs': [r['run'] for r in results], 'target_achieved': False})


def evaluate(variant, step, style):
    name = f'{variant}_{step}_{style}'
    adapter = OUT / 'candidate_adapters' / f'{variant}_{step}'
    destination = OUT / 'evaluation' / name
    if style == 'independent':
        module = 'scripts.eval.evaluate'
        arguments = ['--adapter', adapter, '--auxiliary-mode', 'independent', '--output-dir', destination]
    else:
        module = 'scripts.eval.evaluate_distillation'
        arguments = ['--adapter', adapter, '--style', style, '--output-dir', destination]
    wait(launch('eval_' + name, module, arguments, 0))
    metrics = json.loads((destination / 'metrics.json').read_text())
    if metrics['valid_predictions'] != 666:
        raise RuntimeError(f'{name} has incomplete val predictions')
    summary(f'evaluated_{name}')


def train_variant(variant, steps, stop_after=None):
    wait('tokenize_teacher_host' if variant == 'teacher' else 'tokenize_control_host')
    config = ROOT / f'configs/sft/distill_v3_{variant}.yaml'
    arguments = ['--config', config, '--max-hours', 7]
    if stop_after is not None:
        arguments += ['--stop-after', stop_after]
    name = launch('sft_' + variant, 'scripts.train.distill_sft', arguments, 1)
    summary('training_' + variant)
    completed = set()
    while True:
        terminal = state(name) == 'complete'
        candidates = []
        for step in steps:
            if (OUT / 'candidate_adapters' / f'{variant}_{step}' / 'adapter_model.safetensors').exists():
                candidates.append(step)
        if terminal:
            final = OUT / f'sft_{variant}'
            audit = json.loads((final / 'input_and_gradient_audit.json').read_text())
            if not audit['initial_adapter_exact_match'] or audit['finite_gradient_steps_this_invocation'] < 2:
                raise RuntimeError(f'{variant} did not pass actual input/gradient checks')
            final_step = audit['global_step']
            candidate = OUT / 'candidate_adapters' / f'{variant}_{final_step}'
            if not candidate.exists():
                temp = candidate.with_name('.' + candidate.name + '.tmp')
                temp.mkdir(parents=True, exist_ok=True)
                for filename in ('adapter_config.json', 'adapter_model.safetensors'):
                    shutil.copy2(final / filename, temp / filename)
                b.write_json(temp / 'candidate.json', {'variant': variant, 'global_step': final_step,
                             'native_source': str(final), 'completed_epoch': audit['completed_epoch']})
                temp.rename(candidate)
            candidates.append(final_step)
        for step in sorted(set(candidates)):
            for style in ('decision', 'independent'):
                if (step, style) not in completed:
                    evaluate(variant, step, style)
                    completed.add((step, style))
        if terminal:
            break
        time.sleep(10)


def main():
    # Standalone two-update trial artifacts are disposable. Formal training
    # performs its own initialization, input and gradient checks in distill_sft.
    wait('baseline_independent')
    summary('baseline_reproduced')
    train_variant('teacher', [200, 400, 800, 1200, 2013])
    # Stop the label control at step 800 while retaining the exact teacher LR
    # schedule. The valid evidence ablation is teacher_800 vs control_800.
    train_variant('control', [400, 800], stop_after=800)
    summary('SFT_comparisons_complete_RL_pending')


if __name__ == '__main__':
    main()
