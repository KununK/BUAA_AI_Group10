"""Verifiable training rewards and GRPO math; no validation or test labels are used."""
import torch

from scripts.common import baseline as b
from scripts.common.parsing import parse_task, parse_decision
from scripts.common import rl_tasks as tasks


def completion_reward(row, task, raw):
    try:
        if task == 'decision':
            parsed = parse_decision(raw)
            emotion = parsed['pred_emotion'] == tasks.gold_label(row, 'emotion')
            conflict = row['sample_type'] == 'Conflict'
            if conflict != (row['views']['M1']['label'] != row['views']['M2']['label']):
                raise AssertionError('Contradictory training annotation')
            correct = (parsed['is_uncertain'] == 'yes') == conflict
            return .05 + .45 * emotion + .5 * correct, {
                'format': 1, 'emotion_correct': int(emotion), 'conflict_correct': int(correct), 'parsed': parsed}
        if task == 'emotion':
            parsed = parse_task(raw, 'emotion')
            correct = parsed['pred_emotion'] == tasks.gold_label(row, task)
            return .05 + .95 * correct, {'format': 1, 'emotion_correct': int(correct), 'parsed': parsed}
        parsed = parse_task(raw, 'auxiliary')
        vision = parsed['vision_emotion'] == row['views']['M1']['label']
        other = parsed['other_emotion'] == row['views']['M2']['label']
        predicted_conflict = parsed['vision_emotion'] != parsed['other_emotion']
        gold_conflict = row['sample_type'] == 'Conflict'
        if gold_conflict != (row['views']['M1']['label'] != row['views']['M2']['label']):
            raise AssertionError('Contradictory training annotation')
        conflict_correct = predicted_conflict == gold_conflict
        return .05 + .2 * vision + .2 * other + .55 * conflict_correct, {
            'format': 1, 'vision_correct': int(vision), 'other_correct': int(other),
            'conflict_correct': int(conflict_correct), 'predicted_conflict': predicted_conflict, 'parsed': parsed}
    except ValueError:
        return -1.0, {'format': 0, 'invalid': True}


def group_advantages(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    # Population std is well defined for identical rewards; keep zero-variance groups exactly zero.
    return (r - r.mean()) / (r.std(unbiased=False) + 1e-4)


def modality_branch_rewards(row, raw_branches):
    """Separate credit for each independently sampled modality plus joint conflict.

    A malformed other branch does not erase valid supervision for this branch.
    Both branches must be valid before their disagreement earns conflict credit.
    """
    if len(raw_branches) != 2:
        raise ValueError('Expected visual and text/audio completions')
    subtasks = tasks.modality_tasks(row)
    predicted = []
    for raw, subtask in zip(raw_branches, subtasks):
        try:
            predicted.append(tasks.parse_modality(raw, subtask))
        except ValueError:
            predicted.append(None)
    gold = [row['views'][key]['label'] for key in ('M1', 'M2')]
    conflict = row['sample_type'] == 'Conflict'
    if conflict != (gold[0] != gold[1]):
        raise ValueError('Contradictory training conflict target')
    conflict_correct = None not in predicted and ((predicted[0] != predicted[1]) == conflict)
    rewards = [(-1.0 if label is None else .05 + .5 * (label == target) + .45 * conflict_correct)
               for label, target in zip(predicted, gold)]
    return rewards, {'branch_labels': predicted, 'branch_correct': [p == g for p, g in zip(predicted, gold)],
                     'conflict_correct': bool(conflict_correct)}


def branch_advantages(rewards):
    """Normalize sampled rewards per branch, never across unrelated branches."""
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError('Expected generations x branches rewards')
    return (values - values.mean(dim=0)) / (values.std(dim=0, unbiased=False) + 1e-4)


def grpo_loss(logps, old_logps, reference_logps, advantage, beta=.04, epsilon=.2):
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    delta = reference_logps - logps
    kl = delta.exp() - delta - 1
    loss = (-surrogate + beta * kl).mean()
    return loss, {'kl': kl.detach().mean().item(), 'ratio_mean': ratio.detach().mean().item(),
                  'clip_fraction': ((ratio-1).abs() > epsilon).float().detach().mean().item()}
