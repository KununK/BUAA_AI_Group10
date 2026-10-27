"""Continue the frozen four-scheme test run on user-authorized GPUs 0, 1, 4.

The original inference implementation and its fingerprints remain unchanged.
Only scheduling changes: v7 VA predictions can precede the v4 emotion results.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from scripts.eval import test_submissions as core


def common_config(root, gpu):
    common = core.read(root / 'run_config.json')
    for relative, expected in common['source_sha256'].items():
        if core.sha(core.ROOT / relative) != expected:
            raise ValueError('Original inference source changed: ' + relative)
    if core.sha(core.ROOT / 'Dataset/test_release.jsonl') != common['dataset_sha256']:
        raise ValueError('Test inputs changed')
    return {**common, 'physical_GPU': gpu}


def infer_v7(model, processor, row):
    inputs, media = core.prepare_va(processor, row, core.ROOT / 'Dataset')
    inputs = inputs.to('cuda:0').to(torch.bfloat16)
    with torch.no_grad():
        output = model.generate(**inputs, use_audio_in_video=False, do_sample=False, max_new_tokens=64,
            eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    raw = processor.batch_decode(output[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
    uncertain = core.parse_uncertainty(raw)
    del output, inputs
    question, origin, raw_question = 'none', 'not_required', None
    if uncertain == 'yes':
        question, origin, raw_question = core.generate_question(model, processor, row)
    return {'sample_id': row['sample_id'], 'protocol': 'VA', 'is_uncertain': uncertain,
        'raw_output': raw, 'clarifying_question': question, 'question_origin': origin,
        'raw_question': raw_question, 'media': media}


def cache_v7(root, rows, common):
    out = root / 'v7_VA_cache'; out.mkdir(exist_ok=True)
    adapter = core.ROOT / core.STAGES[3][1]
    config = core.read(core.ROOT / core.STAGES[3][2] / 'run_config.json')
    digest = core.sha(adapter / 'adapter_model.safetensors')
    if digest != config['adapter_sha256']:
        raise ValueError('Selected v7 weights changed')
    core.immutable_json(out / 'run_config.json', {**common, 'adapter': str(adapter),
        'adapter_sha256': digest, 'purpose': 'VA-only raw classifier cache for the fourth submission',
        'source_sha256_parallel_driver': core.sha(Path(__file__)), 'no_emotion_labels_created': True})
    selected = [r for r in rows if r['protocol'] == 'VA']
    path = out / 'classifications.jsonl'
    records = core.b.read_jsonl(path) if path.exists() else []
    seen = {r['sample_id'] for r in records}
    if len(seen) != len(records) or seen - {r['sample_id'] for r in selected}:
        raise ValueError('Invalid resumed VA cache')
    if len(records) != len(selected):
        policy = core.Policy(); model, processor = policy.activate(adapter)
        core.b.write_json(out / 'environment.json', policy.environment)
        with path.open('a', encoding='utf-8', buffering=1) as handle:
            for i, row in enumerate(selected, 1):
                if row['sample_id'] in seen:
                    continue
                record = infer_v7(model, processor, row)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n'); records.append(record)
                print(f'v7 VA [{i}/{len(selected)}] {row["sample_id"]} {record["is_uncertain"]}', flush=True)
                # Keep caching fragmentation below the tight GPU1 allocation cap.
                torch.cuda.empty_cache()
    if len(records) != 347:
        raise ValueError('VA cache must cover all 347 test VA records')
    core.b.write_json(out / 'complete.json', {'rows': len(records), 'finished_utc': core.utc(),
        'classifications_sha256': core.sha(path), 'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3})


def collect(root, rows):
    for name in ('parallel_control_gpu4', 'parallel_v4_gpu0', 'parallel_v7_gpu1'):
        job = root / 'jobs' / name
        while not (job / 'exit.json').exists():
            time.sleep(15)
        if core.read(job / 'exit.json')['exit_code'] != 0:
            raise RuntimeError(f'Parallel inference did not succeed: {name}')
    reference = {r['sample_id']: r for r in core.b.read_jsonl(root/'03_v4/predictions.jsonl')}
    cache_path = root/'v7_VA_cache/classifications.jsonl'
    cached = {r['sample_id']: r for r in core.b.read_jsonl(cache_path)}
    if set(cached) != {r['sample_id'] for r in rows if r['protocol']=='VA'} or len(reference)!=734:
        raise ValueError('Incomplete or mismatched parallel source predictions')
    out = root/'04_v4_v7';out.mkdir(exist_ok=True)
    cfg = core.read(root/'03_v4/run_config.json')
    manifest = {**cfg, 'scheme':'04_v4_v7', 'physical_GPU':None, 'physical_GPUs':{'v4':0,'v7':1},
        'adapter':str(core.ROOT/core.STAGES[3][1]),
        'adapter_sha256':core.sha(core.ROOT/core.STAGES[3][1]/'adapter_model.safetensors'),
        'mode':'v4 emotion and VT frozen; v7 raw VA', 'threshold':None,
        'VT_threshold':cfg['threshold'], 'v4_source_sha256':core.sha(root/'03_v4/predictions.jsonl'),
        'v7_cache_sha256':core.sha(cache_path), 'source_sha256_parallel_driver':core.sha(Path(__file__))}
    core.immutable_json(out/'run_config.json',manifest)
    records=[]
    for row in rows:
        original=reference[row['sample_id']]
        if row['protocol']=='VT':
            records.append(core.hybrid_record(row,original,None));continue
        source=cached[row['sample_id']];question,origin='none','not_required'
        if source['is_uncertain']=='yes':
            if original['prediction']['is_uncertain']=='yes':
                question,origin=original['prediction']['clarifying_question'],'reused_actual_v4_question'
            else:question,origin=source['clarifying_question'],'cached_actual_v7_question'
        record=core.hybrid_record(row,original,source['is_uncertain'],question,origin,source['raw_output'])
        record.update(media=source['media'],raw_question=source['raw_question']);records.append(record)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    core.finalize(rows,records,out)
    reports={name:core.read(root/name/'summary.json') for name,_,_ in core.STAGES}
    core.b.write_json(root/'complete.json',{'finished_utc':core.utc(),'schemes':reports,
        'all_four_submissions_ready':True,'Kaggle_uploaded':False,'parallel_GPUs':[0,1,4],
        'original_serial_job_intentionally_interrupted':'jobs/inference/parallel_handoff.json'})
    print(json.dumps({'all_four_submissions_ready':True,'output_root':str(root)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--stage',choices=['control','v4','v7','collect'],required=True)
    a=p.parse_args();root=a.output_root.resolve()
    rows=[core.public_row(r) for r in core.b.read_jsonl(core.ROOT/'Dataset/test_release.jsonl')]
    if a.stage=='collect':
        common_config(root,None);collect(root,rows);return
    expected={'control':'4','v4':'0','v7':'1'}[a.stage]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:
        raise ValueError('Wrong physical GPU for the recorded parallel plan')
    torch.cuda.set_per_process_memory_fraction(.25 if a.stage=='v7' else .4)
    common=common_config(root,int(expected))
    with (root/f'.stage_{a.stage}.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if a.stage=='v7':cache_v7(root,rows,common)
        else:
            spec=core.STAGES[1 if a.stage=='control' else 2]
            core.run_stage(core.Policy(),rows,root/spec[0],spec,common)


if __name__=='__main__':main()
