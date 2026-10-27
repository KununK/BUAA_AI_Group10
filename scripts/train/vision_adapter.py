"""Extend one language LoRA with zero-initialized last-block visual attention LoRA.

Initialization never changes source weights or model outputs. Verification uses
the native LLaMAFactory loader, then compares both adapters on identical real
inputs in a shared backbone. No new base model or environment is installed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from scripts.common import baseline as b


def expanded_config(original, depth=32, blocks=8, rank=8):
    from peft import LoraConfig
    if not 0 < blocks <= depth or rank < 1:
        raise ValueError('Invalid last visual blocks/rank')
    config = LoraConfig(**original)
    if not isinstance(config.target_modules, set):
        raise ValueError('This initializer requires the existing explicit target-module list')
    modules = [f'visual.blocks.{i}.attn.{name}' for i in range(depth-blocks, depth) for name in ('q', 'k', 'v', 'proj')]
    config.target_modules = set(config.target_modules) | set(modules)
    config.rank_pattern = {**config.rank_pattern, **{re.escape(name): rank for name in modules}}
    config.alpha_pattern = {**config.alpha_pattern, **{re.escape(name): 2*rank for name in modules}}
    return config, modules


def initialize(source, output, blocks, rank, seed):
    from accelerate import init_empty_weights
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniThinkerForConditionalGeneration
    if output.exists():
        raise ValueError('Preserve the existing initialized adapter; use a new output directory')
    original = json.loads((source / 'adapter_config.json').read_text())
    base_config = Qwen2_5OmniConfig.from_pretrained(b.MODEL, local_files_only=True)
    config, modules = expanded_config(original, base_config.thinker_config.vision_config.depth, blocks, rank)
    with init_empty_weights(include_buffers=True):
        model = Qwen2_5OmniThinkerForConditionalGeneration(base_config.thinker_config)
        model = get_peft_model(model, config)
    shapes = {key: tuple(value.shape) for key, value in get_peft_model_state_dict(model).items()}
    old = load_file(str(source / 'adapter_model.safetensors'))
    if set(old) - set(shapes):
        raise ValueError('Initialization lost existing language LoRA tensors')
    generator = torch.Generator().manual_seed(seed)
    initialized = {}; visual_parameters = 0
    for key, shape in shapes.items():
        if key in old:
            if old[key].shape != shape: raise ValueError('Existing language rank changed')
            initialized[key] = old[key]
        else:
            if '.visual.blocks.' not in key or not any(f'.{name}.lora_' in key for name in modules):
                raise ValueError('Unexpected new trainable module: ' + key)
            if key.endswith('.lora_A.weight'):
                if shape[0] != rank: raise ValueError('Visual rank pattern was not applied')
                bound = 1/math.sqrt(shape[1])
                initialized[key] = torch.empty(shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
            elif key.endswith('.lora_B.weight'):
                if shape[1] != rank: raise ValueError('Visual rank pattern was not applied')
                initialized[key] = torch.zeros(shape, dtype=torch.float32)
            else:
                raise ValueError('Unexpected visual tensor: ' + key)
            visual_parameters += math.prod(shape)
    if len(initialized)-len(old) != 2*len(modules):
        raise ValueError('Visual target coverage mismatch')
    output.mkdir(parents=True)
    config.base_model_name_or_path = original['base_model_name_or_path']
    config.save_pretrained(output)
    save_file(initialized, str(output / 'adapter_model.safetensors'))
    assert all(torch.equal(initialized[k], v) for k, v in old.items())
    b.write_json(output / 'initialization.json', {'source_adapter': str(source.resolve()),
        'source_sha256': hashlib.sha256((source / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'initialized_sha256': hashlib.sha256((output / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'base_model': b.MODEL, 'language_tensors_bitwise_preserved': True, 'all_new_B_exactly_zero': True,
        'last_visual_blocks': blocks, 'visual_rank': rank, 'visual_modules': modules,
        'new_visual_parameters': visual_parameters, 'seed': seed, 'dense_base_weights_loaded': False,
        'trained': False})
    print(json.dumps({'new_visual_parameters': visual_parameters, 'old_tensors': len(old),
                      'new_tensors': len(initialized), 'zero_output_initialization': True}), flush=True)


def verify(output):
    from peft import get_peft_model_state_dict
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file
    import yaml
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_model, load_tokenizer
    from scripts.common import distill_tasks, rl_tasks
    from scripts.eval.probabilities import label_probabilities, conditional_probabilities
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for real native-loader verification')
    torch.set_num_threads(8)
    initialization = json.loads((output / 'initialization.json').read_text())
    source = Path(initialization['source_adapter'])
    config = yaml.safe_load((b.ROOT / 'configs/sft/distill_v3_teacher.yaml').read_text())
    config.update(adapter_name_or_path=str(output.resolve()), freeze_vision_tower=False,
                  output_dir=str((output / 'verification_output').resolve()), report_to='none')
    model_args, data_args, _, finetuning_args, _ = get_train_args(config)
    module = load_tokenizer(model_args)
    model = load_model(module['tokenizer'], model_args, finetuning_args, is_trainable=True).to('cuda:0')
    actual, expected = get_peft_model_state_dict(model), load_file(str(output / 'adapter_model.safetensors'))
    if set(actual) != set(expected) or any(not torch.equal(actual[k].detach().cpu(), expected[k]) for k in actual):
        raise ValueError('Native LF did not load the expanded adapter exactly')
    trainable = {name: value.numel() for name, value in model.named_parameters() if value.requires_grad}
    if any('lora_' not in name or 'audio_tower.' in name for name in trainable):
        raise ValueError('Unexpected non-LoRA or audio training parameters')
    visual = sum(n for name, n in trainable.items() if 'visual.' in name)
    if visual != initialization['new_visual_parameters']:
        raise ValueError('Native LF froze/lost the visual adapters')
    model.load_adapter(str(source), adapter_name='language_reference', is_trainable=False)
    for name, value in model.named_parameters():
        if 'lora_' in name and '.language_reference.' in name: value.data=value.data.float()
    result = set_peft_model_state_dict(model, load_file(str(source / 'adapter_model.safetensors')), adapter_name='language_reference')
    if result.unexpected_keys or any('.language_reference.' in k for k in result.missing_keys):
        raise ValueError('Reference language adapter failed to load exactly')
    model.eval(); representatives={}
    for row in b.read_jsonl(b.ROOT / 'Dataset/train.jsonl'):
        representatives.setdefault((b.get_protocol(row), b.is_generated(row)), row)
    comparisons=[]
    for row in representatives.values():
        for task in ('vision', 'decision'):
            prepare = rl_tasks.prepare_inputs if task == 'vision' else distill_tasks.prepare_inputs
            inputs, media = prepare(module['processor'], row, task, b.ROOT / 'Dataset')
            inputs=inputs.to('cuda:0').to(torch.bfloat16)
            scores=[]
            for adapter in ('language_reference', 'default'):
                model.set_adapter(adapter)
                value = (label_probabilities(model, module['tokenizer'], inputs, media, 'vision') if task=='vision'
                         else conditional_probabilities(model, module['tokenizer'], inputs, media,
                                                        '{"pred_emotion":"', ('positive','negative')))
                scores.append(value['label_logits'])
            difference=max(abs(scores[0][k]-scores[1][k]) for k in scores[0])
            if difference != 0: raise ValueError(f'Zero-B visual initialization changed outputs: {difference}')
            comparisons.append({'protocol': b.get_protocol(row), 'generated': b.is_generated(row),
                                'task': task, 'max_label_logit_difference': difference,
                                'audio_present': media.get('audio_present',False)})
            del inputs
    b.write_json(output / 'native_verification.json', {'native_LLaMAFactory_loader': True,
        'all_adapter_weights_exact': True, 'trainable_parameters': sum(trainable.values()),
        'visual_trainable_parameters': visual, 'language_trainable_parameters': sum(trainable.values())-visual,
        'audio_adapters_added': False, 'new_B_zero_output_difference': True,
        'comparisons': comparisons, 'training_performed': False})
    print(json.dumps({'native_load_passed':True,'visual_parameters':visual,'zero_output_comparisons':len(comparisons)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['initialize','verify'],required=True)
    parser.add_argument('--source-adapter',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--blocks',type=int,default=8)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20261026)
    args=parser.parse_args()
    if args.stage=='initialize':
        if not args.source_adapter: parser.error('--source-adapter required')
        initialize(args.source_adapter.resolve(),args.output_dir.resolve(),args.blocks,args.rank,args.seed)
    else: verify(args.output_dir.resolve())


if __name__=='__main__': main()
