"""Prepare and verify isolated-modality SFT using the installed LLaMAFactory pipeline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import yaml

from scripts.common import baseline as b
from scripts.common import rl_tasks as tasks
from scripts.data.audio import audio_path, export_audio

CONFIG = b.ROOT/'configs/sft/omni_lora.yaml'
OUTPUT = b.ROOT/'outputs/sft_next'


def repetitions(rows, task):
    """Use train annotations only to balance the supervised tasks, never as input text."""
    if task == 'emotion':
        return 2
    if task == 'vision':
        real_va = any(b.get_protocol(r)=='VA' and r['sample_type']=='Conflict' and not b.is_generated(r) for r in rows)
        va = any(b.get_protocol(r)=='VA' and r['sample_type']=='Conflict' for r in rows)
        conflict = any(r['sample_type']=='Conflict' for r in rows)
        count = 5 if real_va else 3 if va else 2 if conflict else 1
        return max(count,3 if rows[0]['views']['M1']['label']=='neutral' else 1)
    row = rows[0]
    if task != tasks.modality_tasks(row)[1]:
        raise ValueError('The M2 target must match its annotated modality')
    conflict = row['sample_type']=='Conflict'
    count = (6 if not b.is_generated(row) else 3) if task=='audio' and conflict else 2 if conflict else 1
    return max(count,3 if row['views']['M2']['label']=='neutral' else 1)


def make_record(row, task, dataset_root, repeat=0):
    chat = tasks.messages(row,task)
    content=[]
    for part in chat[1]['content']:
        content.append('<video>' if part['type']=='video' else '<audio>' if part['type']=='audio' else part['text'])
    key = 'pred_emotion' if task=='emotion' else 'vision_emotion' if task=='vision' else 'other_emotion'
    target = {key:tasks.gold_label(row,task)}
    return {'record_id':f"{row['sample_id']}:{task}:{repeat}",'sample_id':row['sample_id'],'task':task,
        'source_dataset':row.get('source_dataset'),'sample_type':row['sample_type'],
        'system':b.SYSTEM,'messages':[{'role':'user','content':''.join(content)},
            {'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))}],
        'videos':[str((dataset_root/row['video']).resolve())] if task in ('emotion','vision') else [],
        'audios':[str(audio_path(dataset_root/'sft',row).resolve())] if task=='audio' else []}


def build_records(rows,dataset_root):
    groups=defaultdict(list)
    for row in rows:
        groups[row['video']].append(row)
    records=[]
    for group in groups.values():
        if len({r['views']['M1']['label'] for r in group})!=1:
            raise ValueError('Conflicting M1 labels for identical visual input')
        row=group[0]
        for repeat in range(repetitions(group,'vision')):
            records.append(make_record(row,'vision',dataset_root,repeat))
    seen={}
    for row in rows:
        key=(row['video'],row.get('text',row.get('text_content','')))
        label=row['views']['M12']['label']
        if key in seen:
            if seen[key]!=label:
                raise ValueError('Contradictory overall targets for identical input')
        else:
            seen[key]=label
            if label!='neutral':
                records.extend(make_record(row,'emotion',dataset_root,i) for i in range(repetitions([row],'emotion')))
        task=tasks.modality_tasks(row)[1]
        records.extend(make_record(row,task,dataset_root,i) for i in range(repetitions([row],task)))
    random.Random(42).shuffle(records)
    # The first Arrow write batch must contain actual lists for both media columns.
    prefix=[]
    for task in ('audio','vision','text','emotion'):
        index=next(i for i,r in enumerate(records) if r['task']==task)
        prefix.append(records.pop(index))
    return prefix+records


def prepare(config):
    source=b.ROOT/'Dataset'
    rows=b.read_jsonl(source/'train.jsonl');val=b.read_jsonl(source/'val.jsonl')
    for field in ('sample_id','video','split_group_id'):
        if {r[field] for r in rows}&{r[field] for r in val}:
            raise ValueError(f'Train/val overlap: {field}')
    for row in rows:
        if (row['sample_type']=='Conflict')!=(row['views']['M1']['label']!=row['views']['M2']['label']):
            raise ValueError('Inconsistent conflict annotation')
        if not (source/row['video']).is_file():
            raise FileNotFoundError(row['video'])
    records=build_records(rows,source)
    audio_jobs={row['video']:audio_path(source/'sft',row) for row in rows if b.get_protocol(row)=='VA'}
    for video,target in audio_jobs.items():
        export_audio((source/video,target))
    directory=Path(config['dataset_dir']);directory.mkdir(parents=True,exist_ok=True)
    registry={}
    for name,data in [('independent_train',records)]:
        if len({r['record_id'] for r in data})!=len(data):
            raise ValueError('Duplicate record IDs')
        b.write_json(directory/f'{name}.json',data)
        registry[name]={'file_name':f'{name}.json','formatting':'sharegpt',
            'columns':{'messages':'messages','system':'system','videos':'videos','audios':'audios'},
            'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}
    b.write_json(directory/'dataset_info.json',registry)
    detail=Counter((r['task'],r['source_dataset'],r['sample_type']) for r in records)
    report={'source_train_rows':len(rows),'source_val_rows':len(val),'training_records':len(records),
        'unique_source_video_paths':len({r['video'] for r in rows}),
        'task_counts':dict(Counter(r['task'] for r in records)),
        'weighted_distribution':{'/'.join(k):v for k,v in detail.items()},
        'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('train.jsonl','val.jsonl')},
        'starting_adapter':config['adapter_name_or_path'],
        'notes':['All training targets derive from original train only; official val is selection only.',
            'M1 deduplicates identical videos, M12 deduplicates identical video/text; duplicates must have consistent targets.',
            'Overall neutral is excluded from binary M12 training; all auxiliary neutral targets are retained.',
            'Two copies of each overall-emotion example preserve that task while adding isolated modality tasks.',
            'Real VA conflict: audio repeated 6 times, visual 5; generated VA conflict: audio/visual 3 times.',
            'Other conflict auxiliary examples repeat twice; auxiliary neutral has at least 3 copies.',
            'Only messages/system/videos/audios map to model inputs; IDs, sample type and source never do.',
            'Reuse local 16kHz mono PCM16 sidecars from the first SFT; originals are unchanged.']}
    b.write_json(directory/'preparation_report.json',report)
    b.write_json(OUTPUT/'preparation_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def fingerprint(config):
    files=[CONFIG,Path(__file__).resolve(),b.ROOT/'scripts/common/rl_tasks.py',b.ROOT/'scripts/common/sft_tasks.py',b.ROOT/'scripts/common/baseline.py',b.ROOT/'scripts/data/audio.py']
    files += [Path(config['dataset_dir'])/f'{name}.json' for name in ('independent_train','dataset_info','preparation_report')]
    files += [b.ROOT/'LlamaFactory/src/llamafactory/data'/name for name in ('template.py','mm_plugin.py','converter.py','processor/supervised.py','collator.py')]
    return {str(p.relative_to(b.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_cache(config):
    p=Path(config['dataset_dir'])/'tokenization_manifest.json'
    if not p.exists() or json.loads(p.read_text())['fingerprint']!=fingerprint(config):
        raise ValueError('Independent SFT cache missing or stale; preprocess this version first')


def tokenize(config):
    from datasets import load_from_disk
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    fp=fingerprint(config);manifest=Path(config['dataset_dir'])/'tokenization_manifest.json'
    if Path(config['tokenized_path']).exists():
        validate_cache(config)
    else:
        cpu=dict(config,do_train=False,do_eval=False,bf16=False,tf32=False,use_cpu=True,report_to='none',preprocessing_num_workers=1)
        model_args,data_args,training_args,_,_=get_train_args(cpu)
        module=load_tokenizer(model_args);template=get_template_and_fix_tokenizer(module['tokenizer'],data_args)
        get_dataset(template,model_args,data_args,training_args,'sft',**module)
    data=load_from_disk(config['tokenized_path'])['train']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['model_name_or_path'],local_files_only=True,use_fast=False)
    source=json.loads((Path(config['dataset_dir'])/'independent_train.json').read_text())
    expected=Counter((r['task'],json.dumps(json.loads(r['messages'][1]['content']),sort_keys=True)) for r in source)
    actual=Counter();lengths=[]
    for row in data:
        assert len(row['input_ids'])==len(row['labels'])==len(row['attention_mask'])
        target=json.loads(tokenizer.decode([x for x in row['labels'] if x!=-100],skip_special_tokens=True).strip())
        task='emotion' if 'pred_emotion' in target else 'vision' if 'vision_emotion' in target else 'audio' if row.get('audios') else 'text'
        actual[(task,json.dumps(target,sort_keys=True))]+=1
        lengths.append(len(row['input_ids']))
    if actual!=expected or len(data)!=len(source) or max(lengths)>=config['cutoff_len']:
        raise ValueError('Lost, altered or truncated supervised targets')
    result={'fingerprint':fp,'records':len(data),'all_targets_match_source':True,
        'maximum_tokens':max(lengths),'p95_tokens':sorted(lengths)[int(.95*(len(lengths)-1))],
        'target_histogram':{task+'/'+target:n for (task,target),n in actual.items()},'gpu_training_started':False}
    b.write_json(manifest,result);b.write_json(OUTPUT/'tokenization_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)




def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['prepare','tokenize','preprocess'],required=True)
    a=p.parse_args();config=yaml.safe_load(CONFIG.read_text())
    if a.mode=='preprocess':
        tokenize(config)
        return
    {'prepare':prepare,'tokenize':tokenize}[a.mode](config)


if __name__=='__main__':
    main()
