"""Online GRPO with TRL configuration/math utilities and complete Omni media forwards.

This project training loop deliberately replaces TRL 0.24's image-only input plumbing.
It is not an unmodified GRPOTrainer or LLaMAFactory PPO invocation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from trl import GRPOConfig
from trl.trainer.utils import disable_dropout_in_model, selective_log_softmax

from scripts.common import baseline as b
from scripts.common.omni_policy import load_policy
from scripts.common import rl_tasks as tasks
from scripts.common import distill_tasks
from scripts.train.rewards import (completion_reward, group_advantages, grpo_loss,
                                   modality_branch_rewards, branch_advantages)


def activate(model, name):
    model.set_adapter(name)
    for key, value in model.named_parameters():
        value.requires_grad_('lora_' in key and '.default.' in key and name == 'default')


def parameter_digest(model, adapter):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if 'lora_' in name and f'.{adapter}.' in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def response_logps(model, inputs, completion, media, temperature):
    """Score only sampled assistant tokens, with the exact original video/audio inputs."""
    length = completion.shape[-1]
    packed = dict(inputs)
    packed['input_ids'] = torch.cat((inputs['input_ids'], completion), dim=-1)
    packed['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(completion)), dim=-1)
    packed['use_audio_in_video'] = media['use_audio_in_video']
    # Avoid allocating prompt_length * vocabulary logits. This keeps the full causal transformer forward.
    head = model.get_base_model().lm_head
    hook = head.register_forward_pre_hook(lambda _module, args: (args[0][:, -(length+1):-1, :],))
    try:
        output = model(**packed, use_cache=False, return_dict=True)
    finally:
        hook.remove()
    assert output.logits.shape[1] == length
    return selective_log_softmax(output.logits.float() / temperature, completion)


def rollout(model, processor, inputs, media, config):
    model.eval()
    with torch.no_grad():
        result = model.generate(**inputs, use_audio_in_video=media['use_audio_in_video'],
            do_sample=True, temperature=config.temperature, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=config.max_completion_length,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True)
    completion = result[:, inputs['input_ids'].shape[-1]:].detach()
    # Batch size is one: generation already stops at EOS and has no right padding.
    text = decode_completion(processor, completion)
    return completion, text


def decode_completion(processor, completion):
    try:
        return processor.batch_decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except TypeError:
        unknown = [token for token in completion[0].tolist()
                   if processor.tokenizer.convert_ids_to_tokens(token) is None]
        if not unknown:
            raise
        # Keep the exact sampled IDs for likelihood/gradients. Undecodable answers
        # receive the predeclared invalid-format reward instead of aborting training.
        return 'undecodable_token_ids:' + json.dumps(unknown)


def choose_plan(rows, groups, seed, smoke, curriculum=None, decision_fraction=0.):
    if not 0 <= decision_fraction <= 1:
        raise ValueError('Decision fraction must be in [0, 1]')
    rng = random.Random(seed)
    plan = []
    pools = {}
    emotion = {}
    seen = set()
    mined = {r['sample_id']: r for r in curriculum['records']} if curriculum else {}

    def pick(pool, task):
        if not curriculum:
            return rng.choice(pool)
        keys = ('emotion',) if task == 'emotion' else ('direct_emotion', 'direct_conflict') if task == 'decision' else ('vision', 'other', 'conflict')
        hard = [row for row in pool if any(mined.get(row['sample_id'], {}).get('errors', {}).get(key, False) for key in keys)]
        # Prefer real starting-policy failures while retaining a 30% global
        # anchor. Within a hard pool, visual/conflict errors receive more weight.
        if hard and rng.random() < .7:
            pool = hard
        weights = []
        for row in pool:
            errors = mined.get(row['sample_id'], {}).get('errors', {})
            weight = 1 + (4 * errors.get('emotion', False) if task == 'emotion' else
                          4 * errors.get('direct_conflict', False) + 2 * errors.get('direct_emotion', False) if task == 'decision' else
                          3 * errors.get('vision', False) + errors.get('other', False)
                          + 2 * errors.get('conflict', False))
            weights.append(weight)
        return rng.choices(pool, weights=weights, k=1)[0]
    for row in rows:
        gold_conflict = row['views']['M1']['label'] != row['views']['M2']['label']
        assert gold_conflict == (row['sample_type'] == 'Conflict')
        pools.setdefault((row['sample_type'], b.get_protocol(row), b.is_generated(row)), []).append(row)
        label = row['views']['M12']['label']
        key = row.get('split_group_id', row['video'])
        if label in ('positive', 'negative') and key not in seen:
            emotion.setdefault(label, []).append(row)
            seen.add(key)
    if smoke:
        specifications = [('auxiliary', 'Conflict', 'VA', False), ('auxiliary', 'Conflict', 'VT', False),
                          ('auxiliary', 'Aligned', 'VA', False), ('emotion', None, None, None)]
        for i in range(groups):
            task, kind, protocol, generated = specifications[i % len(specifications)]
            if task == 'auxiliary' and decision_fraction and (decision_fraction == 1 or i % 2 == 0):
                task = 'decision'
            pool = emotion['positive'] if task == 'emotion' else pools[(kind, protocol, generated)]
            if task == 'decision': pool = [r for r in pool if r['views']['M12']['label'] in ('positive', 'negative')]
            row = rng.choice(pool)
            plan.append({'sample_id': row['sample_id'], 'task': task})
        return plan
    for _ in range(groups):
        if rng.random() < .25:
            row = pick(emotion[rng.choice(['positive', 'negative'])], 'emotion')
            task = 'emotion'
        else:
            kind = 'Conflict' if rng.random() < .60 else 'Aligned'
            protocol = rng.choice(['VT', 'VA'])
            # Half of conflict sampling is VA; real/generated are selected evenly when both exist.
            keys = [k for k in pools if k[:2] == (kind, protocol)]
            task = 'decision' if decision_fraction and rng.random() < decision_fraction else 'auxiliary'
            eligible = {k: [r for r in pools[k] if task != 'decision' or r['views']['M12']['label'] in ('positive', 'negative')] for k in keys}
            keys = [k for k in keys if eligible[k]]
            row = pick(eligible[rng.choice(keys)], task)
        plan.append({'sample_id': row['sample_id'], 'task': task})
    return plan


def save_checkpoint(model, optimizer, out, progress, config, reference_digest):
    step = progress['groups_completed']
    target = out / f'checkpoint-{step}'
    if target.exists():
        return
    temporary = out / f'.checkpoint-{step}.tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    activate(model, 'default')
    model.save_pretrained(temporary, selected_adapters=['default'], safe_serialization=True,
                          save_embedding_layers=False)
    torch.save({'optimizer': optimizer.state_dict(), 'progress': progress,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'reference_digest': reference_digest}, temporary / 'trainer_state.pt')
    b.write_json(temporary / 'progress.json', progress)
    b.write_json(temporary / 'grpo_config.json', config.to_dict())
    temporary.rename(target)
    b.write_json(out / 'latest_checkpoint.json', {'path': str(target.resolve()), **progress})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, default=b.ROOT / 'outputs/independent_omni_v2/selected_sft_adapter')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    p.add_argument('--groups', type=int, default=192)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.5)
    p.add_argument('--learning-rate', type=float, default=3e-6)
    p.add_argument('--beta', type=float, default=.04)
    p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--save-every', type=int, default=48)
    p.add_argument('--max-hours', type=float, default=5)
    p.add_argument('--input-mode', choices=['joint', 'independent'], default='joint')
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--curriculum', type=Path, help='Verified original-train errors mined by the starting SFT')
    p.add_argument('--branch-credit', action='store_true', help='Normalize independent modality rewards per branch')
    p.add_argument('--target-updates', type=int, help='Stop after this many real optimizer updates, subject to group/time cap')
    p.add_argument('--decision-fraction', type=float, default=0.,
                   help='Fraction of non-emotion groups that optimize the trained direct conflict task')
    a = p.parse_args()
    if a.group_size < 2 or a.groups < 1 or a.iterations < 1:
        p.error('GRPO requires at least two sampled completions per group')
    if a.branch_credit and a.input_mode != 'independent':
        p.error('--branch-credit requires independent modality inputs')
    if a.target_updates is not None and a.target_updates < 1:
        p.error('--target-updates must be positive')
    if not 0 <= a.decision_fraction <= 1:
        p.error('--decision-fraction must be in [0, 1]')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = b.read_jsonl(a.dataset_root / 'train.jsonl')
    by_id = {r['sample_id']: r for r in rows}
    curriculum = json.loads(a.curriculum.read_text()) if a.curriculum else None
    if curriculum:
        if (not curriculum.get('completed') or curriculum['train_sha256'] != hashlib.sha256((a.dataset_root / 'train.jsonl').read_bytes()).hexdigest()
                or curriculum['adapter_sha256'] != hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
                or len({r['sample_id'] for r in curriculum['records']}) != len(curriculum['records'])
                or any(r['sample_id'] not in by_id for r in curriculum['records'])):
            raise ValueError('Curriculum is incomplete, duplicated, not from this SFT, or not original train')
        if a.decision_fraction and any('direct_conflict' not in r['errors'] for r in curriculum['records']):
            raise ValueError('Direct-decision RL requires a curriculum mined with --include-decision')
    plan = choose_plan(rows, a.groups, a.seed, a.smoke, curriculum, a.decision_fraction)
    source_files = [Path(__file__).resolve(), Path(tasks.__file__).resolve(),
                    Path(distill_tasks.__file__).resolve(), *[b.ROOT / 'scripts' / name for name in
                      ('train/rewards.py','common/omni_policy.py','common/sft_tasks.py','common/parsing.py','common/conflict_rule.py','common/baseline.py')]]
    existing = json.loads((out/'experiment_config.json').read_text()) if (out/'experiment_config.json').exists() else None
    initial_resume = existing.get('initial_resume') if existing else None
    if not existing and a.resume:
        initial_resume = {'checkpoint':str(a.resume.resolve()),
            'adapter_sha256':hashlib.sha256((a.resume/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'progress':json.loads((a.resume/'progress.json').read_text())}
    manifest = {'algorithm': 'GRPO, project Omni multimodal loop using TRL configuration and utilities',
        'initial_resume':initial_resume,
        'arguments': {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items() if k != 'resume'},
        'train_sha256': hashlib.sha256((a.dataset_root / 'train.jsonl').read_bytes()).hexdigest(),
        'adapter_sha256': hashlib.sha256((a.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'source_sha256': {str(f.relative_to(b.ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        'plan': plan, 'media': {'fps': 2, 'max_frames': 32, 'max_pixels': 262144, 'audio_sampling_rate': 16000},
        'rewards': {'invalid_json': -1, 'emotion': {'valid_format': .05, 'correct': .95},
                    'auxiliary': {'valid_format': .05, 'vision_correct': .2, 'other_correct': .2, 'conflict_correct': .55},
                    'decision': {'valid_format': .05, 'emotion_correct': .45, 'conflict_correct': .5}},
        'question_training': False, 'reference_policy': f'frozen starting adapter: {a.adapter.resolve()}',
        'sampling': f'25% emotion, 75% stratified auxiliary/direct; direct fraction={a.decision_fraction}; non-emotion 60% Conflict/40% Aligned and 50% VA/50% VT',
        'validation_labels_used_for_training': False, 'test_labels_used': False}
    manifest.update(curriculum_sha256=hashlib.sha256(a.curriculum.read_bytes()).hexdigest() if a.curriculum else None,
                    branch_credit=a.branch_credit,
                    branch_reward={'invalid': -1, 'format': .05, 'own_modality_correct': .5, 'conflict_correct': .45}
                    if a.branch_credit else None,
                    hard_sampling_probability=.7 if curriculum else None,
                    dynamic_sampling='Zero-signal groups have no update; continue to next prompt until target updates or declared cap')
    old_path = out / 'experiment_config.json'
    if old_path.exists():
        if json.loads(old_path.read_text()) != manifest:
            raise ValueError('Experiment configuration differs; use another directory')
        if not a.resume:
            raise ValueError('An existing experiment requires an explicit checkpoint resume')
    b.write_json(old_path, manifest)
    config = GRPOConfig(output_dir=str(out), per_device_train_batch_size=a.group_size,
        num_generations=a.group_size, max_prompt_length=None, max_completion_length=64,
        learning_rate=a.learning_rate, beta=a.beta, temperature=a.temperature, top_p=1, top_k=0,
        epsilon=.2, num_iterations=a.iterations, loss_type='grpo', scale_rewards='group',
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        disable_dropout=True, report_to='none', max_grad_norm=1.0, seed=a.seed)
    model, processor, environment = load_policy(a.resume or a.adapter, trainable=True)
    model.load_adapter(str(a.adapter), adapter_name='reference', is_trainable=False)
    # Adding an adapter to a BF16 base rounds its checkpoint into BF16 before PEFT
    # promotes it to FP32. Reload after promotion so reference equals the SFT file.
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict
    for name, value in model.named_parameters():
        if 'lora_' in name and '.reference.' in name:
            value.data = value.data.float()
    loaded = set_peft_model_state_dict(model, load_file(str(a.adapter / 'adapter_model.safetensors')),
                                      adapter_name='reference')
    if loaded.missing_keys or loaded.unexpected_keys:
        # The loader also lists the intentionally absent base and actor weights.
        missing_reference = [k for k in loaded.missing_keys if '.reference.' in k]
        if missing_reference or loaded.unexpected_keys:
            raise ValueError('Reference adapter did not reload exactly')
    if not a.resume:
        parameters = dict(model.named_parameters())
        for name, value in parameters.items():
            if 'lora_' in name and '.default.' in name:
                if not torch.equal(value, parameters[name.replace('.default.', '.reference.')]):
                    raise AssertionError('Initial actor and frozen reference weights differ')
    activate(model, 'default')
    trainable = [(name, value) for name,value in model.named_parameters() if value.requires_grad]
    if not trainable or any('lora_' not in n or '.default.' not in n for n,_ in trainable):
        raise AssertionError('Only the actor LoRA is trainable')
    reference_digest = parameter_digest(model, 'reference')
    initial_digest = parameter_digest(model, 'default')
    disable_dropout_in_model(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    phase = ['initialization']
    media_forwards = {}
    def count_media(kind):
        def hook(_module, _inputs, _output):
            key = f'{phase[0]}/{kind}'
            media_forwards[key] = media_forwards.get(key, 0)+1
        return hook
    model.get_base_model().visual.register_forward_hook(count_media('video'))
    model.get_base_model().audio_tower.register_forward_hook(count_media('audio'))
    optimizer = torch.optim.AdamW([value for _,value in trainable], lr=a.learning_rate, weight_decay=0)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    progress = {'groups_completed': 0, 'optimizer_steps': 0, 'groups_with_reward_variation': 0,
                'zero_variance_groups': 0, 'completions_sampled': 0, 'gradient_checks': 0,
                'finite_gradients': True, 'weight_updates_verified': 0}
    if a.resume:
        state = torch.load(a.resume / 'trainer_state.pt', map_location='cpu', weights_only=False)
        if state['reference_digest'] != reference_digest:
            raise AssertionError('Reference adapter changed')
        optimizer.load_state_dict(state['optimizer'])
        progress = state['progress']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        # Any groups after the last atomic checkpoint must be sampled again with
        # restored RNG/optimizer state; remove their incomplete trace tail first.
        for filename in ('rollouts.jsonl', 'training_log.jsonl'):
            log_path = out / filename
            if log_path.exists():
                kept = [r for r in b.read_jsonl(log_path) if r['group'] <= progress['groups_completed']]
                log_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
    environment.update(trainable_parameters=sum(v.numel() for _,v in trainable),
                       trainable_tensors=len(trainable), reference_digest=reference_digest)
    b.write_json(out / 'environment.json', environment)
    b.write_json(out / 'status.json', {'status': 'running', **progress})
    started = time.monotonic()
    with (out / 'rollouts.jsonl').open('a', buffering=1) as traces, (out / 'training_log.jsonl').open('a', buffering=1) as logs:
        for group_index in range(progress['groups_completed'], len(plan)):
            spec = plan[group_index]
            row, task = by_id[spec['sample_id']], spec['task']
            input_tasks = ['decision'] if task == 'decision' else ['emotion'] if task == 'emotion' else ['joint_auxiliary'] if a.input_mode == 'joint' else list(tasks.modality_tasks(row))
            activate(model, 'default')
            prepared = []
            for subtask in input_tasks:
                prepare = distill_tasks.prepare_inputs if subtask == 'decision' else tasks.prepare_inputs
                inputs, media = prepare(processor, row, subtask, a.dataset_root)
                prepared.append((inputs.to('cuda:0').to(torch.bfloat16), media))
            rewards, details, completions, branch_scores = [], [], [], []
            phase[0] = 'rollout'
            for sample_index in range(a.group_size):
                sample, raw = [], []
                for inputs, media in prepared:
                    tokens, text = rollout(model, processor, inputs, media, config)
                    sample.append(tokens)
                    raw.append(text)
                if len(input_tasks) > 1:
                    try:
                        parsed = {'vision_emotion': tasks.parse_modality(raw[0], input_tasks[0]),
                                  'other_emotion': tasks.parse_modality(raw[1], input_tasks[1])}
                        reward_raw = json.dumps(parsed)
                    except ValueError:
                        reward_raw = 'invalid'
                else:
                    reward_raw = raw[0]
                reward, components = completion_reward(row, task, reward_raw)
                if a.branch_credit and task == 'auxiliary':
                    scores, branch_detail = modality_branch_rewards(row, raw)
                    branch_scores.append(scores)
                    components.update(branch_detail=branch_detail, branch_rewards=scores)
                rewards.append(reward)
                details.append({'raw': raw, 'reward': reward, 'components': components,
                                'completion_tokens': [t.shape[-1] for t in sample],
                                'completion_token_ids': [t[0].tolist() for t in sample]})
                completions.append(sample)
            advantages = (branch_advantages(branch_scores) if branch_scores else
                          group_advantages(rewards).unsqueeze(-1).expand(-1, len(prepared)))
            has_variance = advantages.abs().max().item() > 0
            trace = {'group': group_index+1, 'sample_id': row['sample_id'], 'protocol': b.get_protocol(row),
                'sample_type': row['sample_type'], 'source': row.get('source_dataset'), 'task': task,
                'media': [media for _,media in prepared], 'samples': details, 'advantages': advantages.tolist()}
            traces.write(json.dumps(trace, ensure_ascii=False)+'\n')
            stats = {'group': group_index+1, 'mean_reward': sum(rewards)/len(rewards), 'reward_variation': has_variance,
                     'task': task, 'protocol': b.get_protocol(row), 'sample_type': row['sample_type']}
            if branch_scores:
                stats.update(joint_reward_diagnostic=stats['mean_reward'],
                             mean_reward=sum(sum(scores) for scores in branch_scores)/(a.group_size*len(prepared)),
                             branch_mean_rewards=[sum(scores[i] for scores in branch_scores)/a.group_size
                                                  for i in range(len(prepared))])
            if has_variance:
                progress['groups_with_reward_variation'] += 1
                # Score the same sampled completions under the old actor and frozen SFT reference.
                model.eval()
                with torch.no_grad():
                    phase[0] = 'old_policy'
                    old = [[response_logps(model, inp, tokens, media, config.temperature).detach()
                            for (inp,media),tokens in zip(prepared,sample)] for sample in completions]
                    activate(model, 'reference')
                    phase[0] = 'reference_policy'
                    ref = [[response_logps(model, inp, tokens, media, config.temperature).detach()
                            for (inp,media),tokens in zip(prepared,sample)] for sample in completions]
                activate(model, 'default')
                if progress['optimizer_steps'] == 0:
                    mismatch = max((x-y).abs().max().item() for xs,ys in zip(old,ref) for x,y in zip(xs,ys))
                    if mismatch > 1e-5:
                        raise AssertionError(f'Initial actor/reference mismatch {mismatch}')
                    stats['initial_actor_reference_logp_max_diff'] = mismatch
                for iteration in range(a.iterations):
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    phase[0] = 'policy_gradient'
                    metrics = []
                    for sample_index,sample in enumerate(completions):
                        for branch,((inp,media),tokens) in enumerate(zip(prepared,sample)):
                            logps = response_logps(model, inp, tokens, media, config.temperature)
                            loss, metric = grpo_loss(logps, old[sample_index][branch], ref[sample_index][branch],
                                advantages[sample_index, branch].to('cuda:0'), config.beta, config.epsilon)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite GRPO loss')
                            (loss / (a.group_size*len(prepared))).backward()
                            metric['loss'] = loss.detach().item()
                            metrics.append(metric)
                    grad_norm = torch.nn.utils.clip_grad_norm_([v for _,v in trainable], 1.0)
                    if not torch.isfinite(grad_norm) or grad_norm.item() <= 0:
                        raise FloatingPointError(f'Invalid gradient norm: {grad_norm}')
                    progress['gradient_checks'] += 1
                    probe_name, probe = next((n,v) for n,v in trainable if v.grad is not None and v.grad.abs().max() > 0)
                    before = probe.detach().clone()
                    optimizer.step()
                    difference = (probe.detach()-before).abs().max().item()
                    if difference <= 0:
                        raise AssertionError('Optimizer failed to change the actor parameter probe')
                    progress['weight_updates_verified'] += 1
                    progress['optimizer_steps'] += 1
                    stats.update(gradient_norm=grad_norm.item(), weight_update_max=difference,
                                 weight_update_probe=probe_name, **{key:sum(m[key] for m in metrics)/len(metrics) for key in metrics[0]})
                del old, ref
            else:
                progress['zero_variance_groups'] += 1
            progress['completions_sampled'] += a.group_size * len(prepared)
            progress['groups_completed'] = group_index+1
            stats.update(optimizer_steps=progress['optimizer_steps'], seconds=time.monotonic()-started,
                         max_cuda_memory_gb=torch.cuda.max_memory_allocated()/2**30)
            logs.write(json.dumps(stats, ensure_ascii=False)+'\n')
            print(json.dumps(stats, ensure_ascii=False), flush=True)
            b.write_json(out / 'status.json', {'status': 'running', **progress, 'last_group': stats})
            del prepared, completions
            if progress['groups_completed'] % a.save_every == 0:
                save_checkpoint(model, optimizer, out, progress, config, reference_digest)
            if time.monotonic()-started > a.max_hours*3600:
                break
            if a.target_updates is not None and progress['optimizer_steps'] >= a.target_updates:
                break
    save_checkpoint(model, optimizer, out, progress, config, reference_digest)
    reference_after = parameter_digest(model, 'reference')
    final_digest = parameter_digest(model, 'default')
    audit = {'completed_utc': datetime.now(timezone.utc).isoformat(), **progress,
        'seconds_this_invocation': time.monotonic()-started, 'reference_unchanged': reference_digest == reference_after,
        'initial_actor_digest': initial_digest, 'final_actor_digest': final_digest,
        'actor_changed_this_invocation': initial_digest != final_digest,
        'target_updates': a.target_updates,
        'target_updates_achieved': a.target_updates is not None and progress['optimizer_steps'] >= a.target_updates,
        'budget_limited': progress['groups_completed'] < a.groups and not
                          (a.target_updates is not None and progress['optimizer_steps'] >= a.target_updates),
        'all_planned_groups_completed': progress['groups_completed'] == a.groups,
        'actual_reward_driven_updates': progress['optimizer_steps'] > 0,
        'media_forward_calls_this_invocation': media_forwards,
        'max_cuda_memory_gb': torch.cuda.max_memory_allocated()/2**30}
    b.write_json(out / 'training_audit.json', audit)
    b.write_json(out / 'status.json', {'status': 'complete' if audit['all_planned_groups_completed'] or audit['target_updates_achieved'] else 'budget_stopped', **audit})
    if not audit['reference_unchanged'] or not audit['actual_reward_driven_updates']:
        raise RuntimeError('RL audit failed: missing genuine updates or frozen reference changed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import sys
        if '--output-dir' in sys.argv:
            failed_out = Path(sys.argv[sys.argv.index('--output-dir')+1])
            failed_path = failed_out / 'status.json'
            prior = json.loads(failed_path.read_text()) if failed_path.exists() else {}
            b.write_json(failed_path, {**prior, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
