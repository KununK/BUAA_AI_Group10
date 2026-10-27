"""Evaluate an Omni RL adapter on the same full-set metrics and SFT task contracts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.common import baseline as b
from scripts.common import conflict_rule as questions
from scripts.eval.metrics import write_diagnostics
from scripts.common.parsing import parse_task
from scripts.common.omni_policy import load_policy, generate_text
from scripts.common import rl_tasks as tasks


def generation_label_scores(generated, prompt_length, tokenizer, labels):
    """Read unprocessed class logits at the actual generated label token."""
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(value) != 1 for value in ids):
        raise ValueError('Class labels must each be one token')
    ids = [value[0] for value in ids]
    completion = generated.sequences[0, prompt_length:].tolist()
    positions = [i for i, token in enumerate(completion) if token in ids]
    if len(positions) != 1 or generated.logits is None:
        raise ValueError('Generated class position is ambiguous')
    logits = generated.logits[positions[0]][0, ids].detach().float()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite class logits')
    probabilities = logits.softmax(-1)
    return {'probabilities': dict(zip(labels, probabilities.cpu().tolist())),
            'label_logits': dict(zip(labels, logits.cpu().tolist())),
            'predicted_label': labels[int(probabilities.argmax())],
            'score_source': 'raw_logits_from_actual_greedy_generation'}


def generate_label_text(model, processor, inputs, media, task, max_tokens=64, device='cuda:0'):
    """Preserve greedy generation settings; retain its class distributions."""
    inputs = inputs.to(device).to(torch.bfloat16)
    labels = ('positive', 'negative') if task == 'emotion' else b.EMOTIONS
    with torch.no_grad():
        generated = model.generate(**inputs, use_audio_in_video=media['use_audio_in_video'],
            do_sample=False, max_new_tokens=max_tokens, eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id, return_dict_in_generate=True, output_logits=True)
    length = inputs['input_ids'].shape[-1]
    raw = processor.batch_decode(generated.sequences[:, length:], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    try:
        scores = generation_label_scores(generated, length, processor.tokenizer, labels)
    except ValueError:
        from scripts.eval.probabilities import label_probabilities
        try:
            scores = label_probabilities(model, processor.tokenizer, inputs, media, task)
            scores['score_source'] = 'canonical_prefix_forward_fallback'
        except ValueError as error:
            # Optional diagnostics must not invalidate an otherwise valid answer.
            scores = {'score_error': str(error)}
    return raw, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--auxiliary-mode', choices=['joint','independent'], required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--dataset-root', type=Path, default=b.ROOT/'Dataset')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--compare-emotion-reference', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    a = p.parse_args()
    a.label_set, a.emotion_reference = 'binary', None
    source = a.dataset_root/('val.jsonl' if a.split == 'val' else 'test_release.jsonl')
    all_rows = b.read_jsonl(source)
    rows = all_rows
    if a.smoke:
        representatives = {}
        for row in rows:
            representatives.setdefault((b.get_protocol(row), b.is_generated(row)),row)
        rows = list(representatives.values())
    out = a.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    files = [Path(__file__).resolve(),Path(tasks.__file__).resolve(),b.ROOT/'scripts/common/omni_policy.py',
             b.ROOT/'scripts/common/sft_tasks.py',b.ROOT/'scripts/common/conflict_rule.py',
             b.ROOT/'scripts/eval/probabilities.py']
    manifest = {'split':a.split,'adapter':str(a.adapter.resolve()),'auxiliary_mode':a.auxiliary_mode,
        'adapter_sha256':hashlib.sha256((a.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sample_ids':[r['sample_id'] for r in rows],
        'source_sha256':{str(f.relative_to(b.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'fps':2,'max_frames':32,'max_pixels':262144,'audio_sampling_rate':16000,
        'label_set':'binary','question_adapter_enabled':False,
        'emotion_source':'this RL adapter, all-VT','uncertainty_rule':'vision_emotion != other_emotion',
        'class_probabilities':'actual greedy label logits; declared canonical-prefix fallback'}
    config_path = out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != manifest:
        raise ValueError('Existing evaluation differs; use another directory')
    b.write_json(config_path,manifest)
    path = out/'predictions.jsonl'
    previous = b.read_jsonl(path) if path.exists() else []
    if a.retry_errors:
        previous = [r for r in previous if r.get('prediction')]
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in previous))
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done-{r['sample_id'] for r in rows}:
        raise ValueError('Unexpected or duplicate prediction IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        model,processor,environment = load_policy(a.adapter)
        b.write_json(out/'environment.json',environment)
        b.write_json(out/'status.json',{'status':'running','samples':len(rows)})
        successes = 0
        with path.open('a',buffering=1) as handle:
            for index,row in enumerate(remaining,1):
                record = {'sample_id':row['sample_id'],'video':row['video'],'protocol':b.get_protocol(row),
                          'task':'rl_multitask','label_set':'binary','input_mode':a.auxiliary_mode}
                try:
                    inputs,media = tasks.prepare_inputs(processor,row,'emotion',a.dataset_root)
                    raw_emotion, emotion_scores = generate_label_text(model,processor,inputs,media,'emotion')
                    emotion = parse_task(raw_emotion,'emotion')['pred_emotion']
                    scores, score_media = {'emotion':emotion_scores},{'emotion':media}
                    del inputs
                    auxiliary,raw_aux,aux_media = {},{},{}
                    branches = ['joint_auxiliary'] if a.auxiliary_mode == 'joint' else tasks.modality_tasks(row)
                    for subtask in branches:
                        inputs,media = tasks.prepare_inputs(processor,row,subtask,a.dataset_root)
                        if subtask == 'joint_auxiliary':
                            raw = generate_text(model,processor,inputs,media)
                            auxiliary = parse_task(raw,'auxiliary')
                        else:
                            raw, scores[subtask] = generate_label_text(model,processor,inputs,media,subtask)
                            score_media[subtask] = media
                            auxiliary['vision_emotion' if subtask == 'vision' else 'other_emotion'] = tasks.parse_modality(raw,subtask)
                        raw_aux[subtask],aux_media[subtask] = raw,media
                        del inputs
                    uncertain = questions.rule_uncertainty(auxiliary)
                    question,raw_question = 'none',None
                    if uncertain == 'yes':
                        inputs,media = tasks.prepare_inputs(processor,row,'question',a.dataset_root,auxiliary)
                        with model.disable_adapter():
                            raw_question = generate_text(model,processor,inputs,media,max_tokens=128)
                        question = questions.parse_question(raw_question)
                        del inputs
                    record.update(raw_emotion=raw_emotion,raw_auxiliary=raw_aux,raw_question=raw_question,
                        auxiliary_prediction=auxiliary,media=aux_media,branches=scores,probability_media=score_media,prediction={
                        'pred_emotion':emotion,'is_uncertain':uncertain,'clarifying_question':question,'response':''})
                    successes += 1
                except torch.OutOfMemoryError:
                    raise
                except Exception as error:
                    record.update(prediction=None,error_type=type(error).__name__,error=str(error))
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                previous.append(record)
                print(f"[{index}/{len(remaining)}] {row['sample_id']} {record.get('prediction') or record.get('error')}",flush=True)
                if index >= 3 and successes == 0:
                    raise RuntimeError('First three evaluations all failed')
    metrics = b.finalize(a,rows,all_rows,previous,out)
    probability_records = [{'sample_id':r['sample_id'],'protocol':r['protocol'],
        'branches':r.get('branches',{}),'media':r.get('probability_media',{})} for r in previous]
    (out/'probabilities.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probability_records))
    complete = a.auxiliary_mode == 'independent' and len(probability_records) == len(rows) and all(
        all('probabilities' in r['branches'].get(task,{}) for task in ('emotion',*tasks.modality_tasks(r)))
        for r in probability_records)
    b.write_json(out/'probability_status.json',{'complete':complete,'rows':len(probability_records),
        'generation_settings_changed':False,'probabilities_are_calibrated':False})
    if a.split == 'val':
        write_diagnostics(rows,previous,out)
        if a.compare_emotion_reference:
            reference = b.read_emotion_reference(a.compare_emotion_reference,all_rows,b.LABEL_SETS['binary'])
            hybrid = [{**r,'prediction':{**r['prediction'],'pred_emotion':reference[r['sample_id']]}}
                      if r.get('prediction') else r for r in previous]
            b.write_json(out/'metrics_frozen_emotion.json',b.score_rows(rows,hybrid))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    if metrics['valid_predictions'] != len(rows):
        raise SystemExit('Evaluation incomplete; inspect errors and retry explicitly')


if __name__ == '__main__':
    main()
