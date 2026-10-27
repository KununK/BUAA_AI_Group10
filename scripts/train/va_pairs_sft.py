"""Native LLaMAFactory SFT with frozen visual LoRA and a bounded GPU allocator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, TrainerCallback
import yaml

from scripts.common import baseline as b


class PairAudit(TrainerCallback):
    def __init__(self, config, preparation):
        self.config, self.preparation = config, preparation
        self.counts = Counter(); self.probe_steps = 0; self.nonzero_probe_steps = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], local_files_only=True, use_fast=False)
        self.hook = None

    def on_init_end(self, args, state, control, model=None, **kwargs):
        # Resuming a PEFT adapter does not apply LF's target-module filters again.
        for name, parameter in model.named_parameters():
            if 'lora_' in name: parameter.requires_grad_('.visual.' not in name)
            elif parameter.requires_grad: raise ValueError('Only language LoRA may be trainable')
        self.trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.trainable != 40370176: raise ValueError('Unexpected language-only LoRA parameter count')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        self.initial = Path(self.config['adapter_name_or_path']) / 'adapter_model.safetensors'
        expected = load_file(str(self.initial)); actual = get_peft_model_state_dict(model)
        if expected.keys() != actual.keys() or any(not torch.equal(v, actual[k].detach().cpu()) for k, v in expected.items()):
            raise ValueError('Native LF did not initialize exactly from the selected VA adapter')
        self.visual = {k: v.clone() for k, v in expected.items() if '.visual.' in k}
        if len(self.visual) != 64: raise ValueError('Unexpected frozen visual adapter')
        self.probe = next(p for n, p in model.named_parameters() if p.requires_grad and 'lora_B' in n)
        self.hook = model.register_forward_pre_hook(self.batch, with_kwargs=True)
        print(json.dumps({'native_LLaMAFactory_SFT': True, 'initial_adapter_exact_match': True,
                          'trainable_language_lora_parameters': self.trainable, 'frozen_visual_tensors': 64}), flush=True)

    def batch(self, model, args, kwargs):
        labels = kwargs.get('labels')
        if labels is None: return
        if labels.shape[0] != 1: raise ValueError('Input audit expects microbatch one')
        attended = kwargs['attention_mask'][0] > 0; ids = kwargs['input_ids'][0]
        if torch.any(labels[0][~attended] != -100): raise ValueError('Padding entered supervised loss')
        for token in ('<|VIDEO|>', '<|AUDIO|>'):
            if not ((ids == self.tokenizer.convert_tokens_to_ids(token)) & attended).any():
                raise ValueError('VA training must retain both real modalities')
        if 'pixel_values_videos' not in kwargs or 'input_features' not in kwargs:
            raise ValueError('Missing real visual or audio features')
        target = json.loads(self.tokenizer.decode(labels[0][labels[0] != -100].tolist(), skip_special_tokens=True).strip())
        if set(target) != {'is_uncertain'} or target['is_uncertain'] not in ('yes', 'no'):
            raise ValueError('Only the derived/official conflict target may be supervised')
        self.counts[target['is_uncertain']] += 1

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grad = self.probe.grad
        if grad is None or not torch.isfinite(grad).all(): raise ValueError('Invalid language-LoRA gradient probe')
        self.probe_steps += 1; self.nonzero_probe_steps += int(grad.detach().abs().max().item() > 0)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        if self.hook: self.hook.remove()
        actual = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
        original = load_file(str(self.initial))
        if any(not torch.isfinite(v).all() for v in actual.values()): raise ValueError('Nonfinite final weights')
        if any(not torch.equal(actual[k], v) for k, v in self.visual.items()): raise ValueError('Frozen visual LoRA changed')
        changed = [k for k in actual if not torch.equal(actual[k], original[k])]
        if not changed or any('.visual.' in k for k in changed): raise ValueError('Invalid parameter update scope')
        if state.global_step != 537 or dict(self.counts) != self.preparation['assistant_targets']:
            raise ValueError('Native training did not cover the complete declared pair/replay data')
        b.write_json(Path(args.output_dir) / 'input_and_weight_audit.json', {
            'native_LLaMAFactory_SFT': True, 'initial_adapter_exact_match': True,
            'global_step': state.global_step, 'epoch': state.epoch, 'actual_target_counts': dict(self.counts),
            'all_training_forwards_have_video_and_audio': True, 'padding_excluded_from_loss': True,
            'frozen_visual_tensors_bitwise_preserved': len(self.visual), 'changed_language_tensors': len(changed),
            'all_final_weights_finite': True, 'finite_gradient_probe_steps': self.probe_steps,
            'nonzero_gradient_probe_steps': self.nonzero_probe_steps, 'all_parameter_gradients_audited': False,
            'trainable_parameters': self.trainable, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
            'allocator_fraction': .40, 'source_adapter_sha256': hashlib.sha256(self.initial.read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    a = parser.parse_args(); config = yaml.safe_load(a.config.read_text())
    preparation = json.loads((Path(config['dataset_dir']) / 'preparation.json').read_text())
    for name, expected in preparation['sha256'].items():
        if hashlib.sha256((b.ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Paired-data source changed: ' + name)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.40); torch.set_num_threads(8)
    from llamafactory.train.tuner import run_exp
    run_exp(config, callbacks=[PairAudit(config, preparation)])


if __name__ == '__main__': main()
