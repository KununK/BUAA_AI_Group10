"""Complete the VA study after the recorded GPU1 evaluation OOM and GPU0 retry."""
from pathlib import Path

from scripts.common import baseline as b
from scripts.train.va_conflict_pipeline import OUT, read, wait


def main():
    for name in ('baseline_init', 'sft_144', 'train', 'sft_final_gpu0_retry'):
        wait(name)
    state = read(OUT / 'train/trainer_state.json')
    if state['global_step'] != 287: raise RuntimeError('VA native SFT is incomplete')
    values = {name: read(OUT / name / 'metrics.json') for name in ('baseline_init', 'sft_144', 'sft_final')}
    if any(m['grouped_OOF_metrics']['valid_predictions'] != 666 for m in values.values()):
        raise RuntimeError('Requires all validation classifications')
    result = {'training_optimizer_updates': state['global_step'], 'preselected_checkpoint_steps': [144, 287],
        'baseline': values['baseline_init'], 'sft_144': values['sft_144'], 'sft_final': values['sft_final'],
        'emotion_and_VT_frozen': True, 'test_labels_used': False,
        'question_generation_required_before_submission': True,
        'recovery': {'original_final_job': read(OUT / 'jobs/sft_final/exit.json'),
            'successful_retry': read(OUT / 'jobs/sft_final_gpu0_retry/exit.json'),
            'change': 'Move final evaluation to physical GPU0; identical model, code and inputs'}}
    temporary = OUT / '.evaluation_complete.tmp.json'
    b.write_json(temporary, result)
    temporary.replace(OUT / 'evaluation_complete.json')
    print('VA study complete, including the successful GPU0 recovery', flush=True)


if __name__ == '__main__': main()
