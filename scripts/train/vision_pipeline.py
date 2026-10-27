"""Conditional visual-only LoRA training after the recorded language-SFT study.

Select using complete raw/OOF validation scores, never apparent full-val fits.
GPU1 trains vision; GPU0 evaluates one visual checkpoint at a time. The main SFT
pipeline has ended before this starts, leaving room for the one RL evaluator or
RL training process already on GPU0. Existing jobs are inspected, not restarted.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

from scripts.common import baseline as b
from scripts.train.distill_pipeline import OUT as STUDY, ROOT, launch, state, wait

OUT = ROOT / 'outputs/vision_omni_v4_joint'
CONFIG = ROOT / 'configs/sft/vision_replay.yaml'
TARGET = .95


def load(path):
    return json.loads(Path(path).read_text())


def completed_metrics(directory):
    path = directory / 'metrics.json'
    if not path.exists():
        return None
    result = load(path)
    return result if result.get('valid_predictions') == 666 else None


def run(name, module, arguments, gpu):
    wait(launch(name, module, arguments, gpu))


def parent_scores():
    """Return completed SFT parents and scores that can legitimately select them."""
    parents, observations, actual_hashes = {}, [], {}
    for directory in sorted((STUDY / 'evaluation').iterdir()):
        if not completed_metrics(directory) or not (directory / 'run_config.json').exists():
            continue
        config = load(directory / 'run_config.json')
        if config.get('split') != 'val' or not config.get('adapter') or config.get('threshold_fit_dataset'):
            continue
        adapter = Path(config['adapter']).resolve()
        if adapter.parent != (STUDY / 'candidate_adapters').resolve() or not adapter.name.startswith(('teacher_', 'control_')):
            continue
        if adapter not in actual_hashes:
            actual_hashes[adapter] = hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
        key = config['adapter_sha256']
        if actual_hashes[adapter] != key:
            raise ValueError('Candidate SFT weights changed')
        parents[key] = {'adapter': str(adapter), 'adapter_sha256': key}
        metric = completed_metrics(directory)
        observations.append({'adapter_sha256': key, 'score': metric['score_sklearn_union'],
                             'kind': 'raw', 'path': str(directory / 'metrics.json')})
        calibration = directory / 'calibration/diagnostic.json'
        if calibration.exists():
            for result in load(calibration)['results']:
                observations.append({'adapter_sha256': key,
                    'score': result['grouped_out_of_fold_metrics']['score_sklearn_union'],
                    'kind': 'grouped_OOF_threshold', 'path': str(calibration)})
    for directory in sorted((STUDY / 'evaluation').iterdir()):
        path = directory / 'grouped_oof_metrics.json'
        if not path.exists() or not completed_metrics(directory):
            continue
        config = load(directory / 'run_config.json')
        if config.get('adapter_sha256') in parents:
            observations.append({'adapter_sha256': config['adapter_sha256'],
                'score': load(path)['metrics']['score_sklearn_union'],
                'kind': 'grouped_OOF_fusion', 'path': str(path)})
    if not observations:
        raise ValueError('No complete, verified SFT parent scores')
    best = max(observations, key=lambda value: value['score'])
    return {**parents[best['adapter_sha256']], 'selection_score': best}, observations


def observed_goal_candidate():
    scores = []
    for directory in sorted((STUDY / 'evaluation').iterdir()):
        metric = completed_metrics(directory)
        if not metric or not (directory / 'run_config.json').exists():
            continue
        config = load(directory / 'run_config.json')
        if config.get('split') != 'val':
            continue
        if not config.get('threshold_fit_dataset') and (config.get('style') in ('decision', 'evidence') or config.get('auxiliary_mode') in ('joint', 'independent')):
            scores.append({'score': metric['score_sklearn_union'], 'kind': 'raw', 'path': str(directory / 'metrics.json')})
        path = directory / 'grouped_oof_metrics.json'
        if path.exists():
            scores.append({'score': load(path)['metrics']['score_sklearn_union'],
                           'kind': 'grouped_OOF_fusion', 'path': str(path)})
        path = directory / 'calibration/diagnostic.json'
        if path.exists():
            for result in load(path)['results']:
                scores.append({'score': result['grouped_out_of_fold_metrics']['score_sklearn_union'],
                               'kind': 'grouped_OOF_threshold', 'path': str(path)})
    return max(scores, key=lambda value: value['score']) if scores else None


def ensure_sft_fusions():
    for decision in sorted((STUDY / 'evaluation').glob('*_decision')):
        prefix = decision.name.removesuffix('_decision')
        if not prefix.startswith(('teacher_', 'control_')) or not completed_metrics(decision):
            continue
        independent = decision.with_name(prefix + '_independent')
        destination = decision.with_name(prefix + '_fused')
        scores = decision if (decision / 'probabilities.jsonl').exists() else decision.with_name(prefix + '_decision_scores')
        if (destination / 'grouped_oof_metrics.json').exists() or not completed_metrics(independent) or not (scores / 'probabilities.jsonl').exists():
            continue
        run('vision_parent_fuse_' + prefix, 'scripts.eval.fuse',
            ['--independent-run', independent, '--decision-run', decision,
             '--decision-scores', scores, '--output-dir', destination], '')


def evaluate(step):
    adapter = OUT / 'candidate_adapters' / f'vision_frozen_language_{step}'
    for style in ('decision', 'independent'):
        destination = OUT / 'evaluation' / f'vision_{step}_{style}'
        if style == 'decision':
            module = 'scripts.eval.evaluate_distillation'
            arguments = ['--adapter', adapter, '--style', style, '--output-dir', destination]
        else:
            module = 'scripts.eval.evaluate'
            arguments = ['--adapter', adapter, '--auxiliary-mode', style, '--output-dir', destination]
        run(f'vision_joint_eval_{step}_{style}', module, arguments, 0)
        if not completed_metrics(destination):
            raise ValueError('Incomplete full-val visual checkpoint evaluation')
    decision, independent = [OUT / 'evaluation' / f'vision_{step}_{style}' for style in ('decision', 'independent')]
    fused = OUT / 'evaluation' / f'vision_{step}_fused'
    run(f'vision_joint_fuse_{step}', 'scripts.eval.fuse',
        ['--independent-run', independent, '--decision-run', decision,
         '--decision-scores', decision, '--output-dir', fused], '')
    calibration = load(decision / 'calibration/diagnostic.json')
    return {'step': step, 'adapter': str(adapter),
        'direct_raw_score': completed_metrics(decision)['score_sklearn_union'],
        'independent_raw_score': completed_metrics(independent)['score_sklearn_union'],
        'best_direct_grouped_OOF_score': max(x['grouped_out_of_fold_metrics']['score_sklearn_union'] for x in calibration['results']),
        'fusion_grouped_OOF_score': load(fused / 'grouped_oof_metrics.json')['metrics']['score_sklearn_union'],
        'fused_apparent_full_val_score': completed_metrics(fused)['score_sklearn_union']}


def verified_training_end():
    audit = load(OUT / 'train/input_and_gradient_audit.json')
    gradient = load(OUT / 'train/training_completion.json')
    if (not audit['initial_adapter_exact_match'] or not audit['language_lora_frozen_and_bitwise_preserved']
            or not gradient['visual_parameter_update_verified'] or gradient['visual_nonzero_gradient_steps'] < 2):
        raise ValueError('Formal visual input/gradient/frozen-language audit failed')
    return audit


def main():
    wait('sft_pipeline')
    wait('eval_best_teacher_evidence_overlap')
    ensure_sft_fusions()
    selected, scores = parent_scores()
    b.write_json(OUT / 'source_selection.json', {'selected': selected, 'scores': scores,
        'apparent_full_val_fits_excluded': True, 'selection_never_uses_test': True})
    existing = observed_goal_candidate()
    if existing and existing['score'] >= TARGET:
        b.write_json(OUT / 'conditional_decision.json', {'status': 'skipped_existing_SFT_or_RL_reaches_target',
                     'selected_SFT_parent': selected, 'existing_target_candidate': existing,
                     'new_visual_training_performed': False})
        return
    initialization = OUT / 'initialization'
    run('vision_joint_initialize', 'scripts.train.vision_adapter',
        ['--stage', 'initialize', '--source-adapter', selected['adapter'], '--output-dir', initialization], '')
    run('vision_joint_verify_initialization', 'scripts.train.vision_adapter',
        ['--stage', 'verify', '--output-dir', initialization], 1)
    training = launch('vision_joint_train', 'scripts.train.distill_sft',
        ['--config', CONFIG, '--adapter', initialization, '--train-vision', '--freeze-language-lora', '--max-hours', 4], 1)
    b.write_json(OUT / 'conditional_decision.json', {'status': 'formal_visual_training_started',
                 'selected': selected, 'new_visual_parameters': 655360, 'frozen_language_lora': True})
    done, results = set(), []
    while True:
        terminal = state(training) == 'complete'
        candidates = [step for step in (200, 400, 800, 1065)
                      if (OUT / f'candidate_adapters/vision_frozen_language_{step}/adapter_model.safetensors').exists()]
        if terminal:
            audit = verified_training_end(); final_step = audit['global_step']
            final = OUT / f'candidate_adapters/vision_frozen_language_{final_step}'
            if not final.exists():
                temporary = final.with_name('.' + final.name + '.tmp')
                temporary.mkdir(parents=True)
                for filename in ('adapter_config.json', 'adapter_model.safetensors'):
                    shutil.copy2(OUT / 'train' / filename, temporary / filename)
                b.write_json(temporary / 'candidate.json', {'global_step': final_step, 'native_source': str(OUT / 'train')})
                temporary.rename(final)
            if final_step not in candidates:
                candidates.append(final_step)
        for step in sorted(set(candidates) - done):
            result = evaluate(step); results.append(result); done.add(step)
            b.write_json(OUT / 'comparison.json', {'source_selection': selected, 'results': results,
                         'target': TARGET, 'apparent_scores_separately_labeled': True})
            # Check the requested goal using genuine raw or held-out calibration
            # scores. Never use the full-val-fitted apparent metric for this gate.
            if max(result[k] for k in ('direct_raw_score', 'independent_raw_score',
                   'best_direct_grouped_OOF_score', 'fusion_grouped_OOF_score')) >= TARGET:
                wait(training); verified_training_end()
                b.write_json(OUT / 'evaluation_complete.json', {'status': 'validated_target_reached',
                             'validated_candidate': result, 'native_training_audit_passed': True,
                             'later_optional_candidates_skipped_after_target': True})
                return
        if terminal:
            b.write_json(OUT / 'evaluation_complete.json', {'status': 'planned_candidates_evaluated',
                         'results': results, 'native_training_audit_passed': True})
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
