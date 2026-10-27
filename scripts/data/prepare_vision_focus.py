"""Prepare native-LF visual emotion supervision plus overall-emotion replay.

All targets are original train labels. The language adapter can remain frozen;
there are no audio/text-only training records without trainable visual inputs.
This registry/cache is separate from the running teacher/control experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import yaml

from scripts.common import baseline as b
from scripts.common.distill_tasks import target_kind
from scripts.data.prepare_sft import build_records

DATA=b.ROOT/'Dataset/processed/vision_v4'
OUT=b.ROOT/'outputs/vision_omni_v4'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if DATA.exists(): raise ValueError('Do not overwrite an existing focused dataset')
    rows=b.read_jsonl(b.ROOT/'Dataset/train.jsonl')
    val=b.read_jsonl(b.ROOT/'Dataset/val.jsonl')
    for key in ('sample_id','video','split_group_id'):
        if {r[key] for r in rows}&{r[key] for r in val}: raise ValueError('Train/val overlap')
    original=build_records(rows,b.ROOT/'Dataset')
    records=[r for r in original if r['task']=='vision' or r['task']=='emotion' and r['record_id'].endswith(':0')]
    random.Random(20261026).shuffle(records)
    if len({r['record_id'] for r in records})!=len(records): raise ValueError('Duplicate record IDs')
    for r in records:
        if not r['videos'] or r['audios']: raise ValueError('Focus inputs must be visual with no audio')
    b.write_json(DATA/'vision_focus.json',records)
    b.write_json(DATA/'dataset_info.json',{'vision_focus':{'file_name':'vision_focus.json','formatting':'sharegpt',
        'columns':{k:k for k in ('messages','system','videos','audios')},
        'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}})
    counts=dict(Counter(r['task'] for r in records))
    report={'source':'original train only','source_train_sha256':sha(b.ROOT/'Dataset/train.jsonl'),
            'datasets':{'vision_focus':{'records':len(records),'task_counts':counts,'file_sha256':sha(DATA/'vision_focus.json')}},
            'pure_visual_M1_with_existing_conflict_weights':True,'M12_replay_one_copy_per_video_text':True,
            'new_teacher_labels_added':False,'val_test_used_for_training':False,
            'intended_training':'Only explicitly added late visual LoRA; freeze existing language LoRA'}
    b.write_json(DATA/'preparation.json',report);b.write_json(OUT/'preparation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def fingerprint(config):
    if Path(config['dataset_dir']).resolve()!=DATA: raise ValueError('Unexpected vision dataset directory')
    sources=[Path(__file__).resolve(),b.ROOT/'scripts/data/prepare_sft.py',b.ROOT/'scripts/common/rl_tasks.py',
             b.ROOT/'scripts/common/sft_tasks.py',b.ROOT/'scripts/common/baseline.py',
             DATA/'vision_focus.json',DATA/'dataset_info.json']
    sources += [b.ROOT/'LlamaFactory/src/llamafactory/data'/p for p in
                ('template.py','mm_plugin.py','converter.py','processor/supervised.py','collator.py')]
    keys=('model_name_or_path','template','cutoff_len','train_on_prompt','packing','video_fps','video_maxlen',
          'video_max_pixels','video_min_pixels','audio_sampling_rate','use_audio_in_video')
    return {'files':{str(p.relative_to(b.ROOT)):sha(p) for p in sources},
            'preprocessing':{k:config[k] for k in keys}}


def validate_cache(config):
    meta=DATA/'vision_focus_tokenization.json'
    if not meta.exists() or json.loads(meta.read_text())['fingerprint']!=fingerprint(config):
        raise ValueError('Focused visual cache missing or stale')


def tokenize(config):
    from datasets import load_from_disk
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    cache=Path(config['tokenized_path']);fp=fingerprint(config)
    if cache.exists(): validate_cache(config)
    cpu=dict(config,do_train=False,do_eval=False,use_cpu=True,bf16=False,tf32=False,report_to='none')
    model_args,data_args,training_args,_,_=get_train_args(cpu)
    module=load_tokenizer(model_args);template=get_template_and_fix_tokenizer(module['tokenizer'],data_args)
    if not cache.exists(): get_dataset(template,model_args,data_args,training_args,'sft',**module)
    dataset=load_from_disk(str(cache))['train']
    source=json.loads((DATA/'vision_focus.json').read_text())
    expected=Counter(json.dumps(json.loads(r['messages'][-1]['content']),sort_keys=True) for r in source)
    actual=Counter();counts=Counter();lengths=[]
    for row in dataset:
        target=json.loads(module['tokenizer'].decode([x for x in row['labels'] if x!=-100],skip_special_tokens=True).strip())
        actual[json.dumps(target,sort_keys=True)]+=1
        counts[target_kind(target,bool(row.get('videos')),bool(row.get('audios')))]+=1
        lengths.append(len(row['input_ids']))
        if len(row['input_ids'])!=len(row['labels']): raise ValueError('Token/label length mismatch')
    if actual!=expected or len(dataset)!=len(source) or max(lengths)>=config['cutoff_len']:
        raise ValueError('Focused records were dropped or truncated')
    report={'fingerprint':fp,'records':len(dataset),'task_counts':dict(counts),'maximum_tokens':max(lengths),
            'all_targets_match_source':True,'all_inputs_visual_without_audio':True}
    b.write_json(DATA/'vision_focus_tokenization.json',report);b.write_json(OUT/'tokenization.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','tokenize'],required=True)
    p.add_argument('--config',type=Path,default=b.ROOT/'configs/sft/vision_focus.yaml');args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:tokenize(yaml.safe_load(args.config.read_text()))


if __name__=='__main__':main()
