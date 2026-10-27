"""Continue the original SFT LoRA in native LLaMAFactory with isolated task audits."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
import yaml

from scripts.common import baseline as b
from scripts.data.prepare_sft import CONFIG,OUTPUT,validate_cache
from scripts.train.callbacks import AuditCallback


class IndependentAudit(AuditCallback):
    def __init__(self,seconds,config,resumed):
        super().__init__(seconds)
        self.config,self.resumed=config,resumed
        self.tokenizer=AutoTokenizer.from_pretrained(config['model_name_or_path'],local_files_only=True,use_fast=False)
        self.counts=Counter();self.gradient_steps=0;self.hook=None

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        super().on_train_begin(args,state,control,model=model,**kwargs)
        if not self.resumed:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual=get_peft_model_state_dict(model)
            expected=load_file(str(Path(self.config['adapter_name_or_path'])/'adapter_model.safetensors'))
            if set(actual)!=set(expected) or any(not torch.equal(actual[k].detach().cpu(),expected[k]) for k in actual):
                raise RuntimeError('Native LF did not initialize the actor exactly from SFT500')
        self.initial_source_match=not self.resumed
        self.hook=model.register_forward_pre_hook(self.audit_input,with_kwargs=True)

    def audit_input(self,module,args,kwargs):
        labels=kwargs.get('labels')
        if labels is None:
            return
        if labels.shape[0]!=1:
            raise RuntimeError('This input audit requires microbatch 1')
        target=json.loads(self.tokenizer.decode(labels[0][labels[0]!=-100].tolist(),skip_special_tokens=True).strip())
        ids=kwargs['input_ids'][0];mask=kwargs['attention_mask'][0]>0
        if mask.dim()!=1:
            raise RuntimeError('Unexpected packed or 4D attention mask')
        video=bool(((ids==self.tokenizer.convert_tokens_to_ids('<|VIDEO|>'))&mask).any())
        audio=bool(((ids==self.tokenizer.convert_tokens_to_ids('<|AUDIO|>'))&mask).any())
        image=bool(((ids==self.tokenizer.convert_tokens_to_ids('<|IMAGE|>'))&mask).any())
        task='emotion' if 'pred_emotion' in target else 'vision' if 'vision_emotion' in target else 'audio' if audio else 'text'
        if video!=(task in ('emotion','vision')) or audio!=(task=='audio') or image:
            raise RuntimeError(f'Unexpected attended media for {task}')
        if torch.any(labels[0][~mask]!=-100):
            raise RuntimeError('Dummy media or padding entered supervised loss')
        if video and 'pixel_values_videos' not in kwargs or audio and 'input_features' not in kwargs:
            raise RuntimeError('Real media features were dropped before the model forward')
        self.counts[task]+=1

    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        super().on_pre_optimizer_step(args,state,control,model=model,**kwargs)
        self.gradient_steps+=1

    def on_train_end(self,args,state,control,model=None,**kwargs):
        super().on_train_end(args,state,control,model=model,**kwargs)
        if self.hook:
            self.hook.remove()
        expected=json.loads((Path(self.config['dataset_dir'])/'preparation_report.json').read_text())['task_counts']
        completed=not self.stopped_for_budget and state.epoch is not None and state.epoch>=.999
        if completed and not self.resumed and dict(self.counts)!=expected:
            raise RuntimeError(f'Native training did not cover the full prepared epoch: {dict(self.counts)} != {expected}')
        b.write_json(Path(args.output_dir)/'input_and_gradient_audit.json',{
            'initial_adapter_exact_match':self.initial_source_match,'resumed':self.resumed,
            'actual_forward_counts_this_invocation':dict(self.counts),'expected_epoch_task_counts':expected,
            'all_attended_media_match_task':True,'dummy_tokens_excluded_from_loss':True,
            'finite_gradient_steps_this_invocation':self.gradient_steps,'completed_epoch':completed,
            'global_step':state.global_step,'epoch':state.epoch,'native_LLaMAFactory_SFT':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-hours',type=float,default=7)
    a=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA in this executor')
    torch.set_num_threads(8)
    config=yaml.safe_load(CONFIG.read_text())
    validate_cache(config)
    if a.resume:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint=get_last_checkpoint(config['output_dir'])
        if not checkpoint:
            raise RuntimeError('No native checkpoint available for resume')
        config['resume_from_checkpoint']=checkpoint
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    b.write_json(out/'effective_config.json',config)
    environment={'python':os.sys.executable,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,
        'packages':{n:version(n) for n in ('torch','transformers','peft','trl','llamafactory')},
        'starting_adapter_sha256':hashlib.sha256((Path(config['adapter_name_or_path'])/'adapter_model.safetensors').read_bytes()).hexdigest()}
    b.write_json(out/'environment.json',environment)
    files=[Path(__file__).resolve(),CONFIG,b.ROOT/'scripts/data/prepare_sft.py',b.ROOT/'scripts/train/callbacks.py',b.ROOT/'scripts/common/rl_tasks.py']
    b.write_json(out/'source_manifest.json',{str(f.relative_to(b.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    b.write_json(out/'process.json',{'pid':os.getpid(),'started_at_unix':time.time(),'status':'running'})
    from llamafactory.train.tuner import run_exp
    run_exp(config,callbacks=[IndependentAudit(a.max_hours*3600,config,a.resume)])
    b.write_json(out/'process.json',{'pid':os.getpid(),'status':'complete','finished_at_unix':time.time()})


if __name__=='__main__':
    main()
