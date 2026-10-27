"""Select a verified native token-cache subset for visual-LoRA adaptation.

Keep gold M1 examples, one M12 replay per video/text, and all protocol-specific
gold joint decisions. Existing text/audio-only branches have no visual gradient.
No teacher evidence, questions, val or test records are used in this subset.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml

from scripts.common import baseline as b
from scripts.common.distill_tasks import target_kind
from scripts.data.prepare_distillation import validate_cache as validate_source

DATA = b.ROOT / 'Dataset/processed/vision_replay_v4'
OUT = b.ROOT / 'outputs/vision_omni_v4_joint'
SOURCE_CONFIG = b.ROOT / 'configs/sft/distill_v3_teacher.yaml'
KEYS = ('model_name_or_path', 'template', 'cutoff_len', 'train_on_prompt', 'packing',
        'video_fps', 'video_maxlen', 'video_max_pixels', 'video_min_pixels',
        'audio_sampling_rate', 'use_audio_in_video')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(config):
    if Path(config['dataset_dir']).resolve() != DATA or config['dataset'] != 'vision_replay':
        raise ValueError('Unexpected visual replay dataset')
    source = yaml.safe_load(SOURCE_CONFIG.read_text())
    validate_source(source)
    files = [Path(__file__).resolve(), SOURCE_CONFIG,
             Path(source['dataset_dir']) / 'distill_train_tokenization.json',
             DATA / 'vision_replay.json', DATA / 'dataset_info.json']
    return {'files': {str(p.relative_to(b.ROOT)): sha(p) for p in files},
            'preprocessing': {k: config[k] for k in KEYS}}


def validate_cache(config):
    p = DATA / 'vision_replay_tokenization.json'
    if not p.exists() or json.loads(p.read_text())['fingerprint'] != fingerprint(config):
        raise ValueError('Visual replay subset cache missing or stale')


def main():
    from datasets import DatasetDict, load_from_disk
    from transformers import AutoTokenizer
    config = yaml.safe_load((b.ROOT / 'configs/sft/vision_replay.yaml').read_text())
    source = yaml.safe_load(SOURCE_CONFIG.read_text())
    validate_source(source)
    if DATA.exists():
        raise ValueError('Preserve the existing visual replay subset')
    if any(config[k] != source[k] for k in KEYS):
        raise ValueError('Subset reuse requires identical native preprocessing')
    records = json.loads((Path(source['dataset_dir']) / 'distill_train.json').read_text())
    original = load_from_disk(source['tokenized_path'])['train']
    if len(original) != len(records):
        raise ValueError('Source cache lost/reordered examples')
    indices = [i for i, r in enumerate(records) if r['task'] in ('vision', 'decision')
               or r['task'] == 'emotion' and r['record_id'].endswith(':0')]
    selected = [records[i] for i in indices]
    subset = original.select(indices)
    tokenizer = AutoTokenizer.from_pretrained(source['model_name_or_path'], local_files_only=True, use_fast=False)
    train_ids = {r['sample_id'] for r in b.read_jsonl(b.ROOT / 'Dataset/train.jsonl')}
    forbidden = {r['sample_id'] for name in ('val.jsonl', 'test_release.jsonl')
                 for r in b.read_jsonl(b.ROOT / 'Dataset' / name)}
    counts, lengths = Counter(), []
    for raw, encoded in zip(selected, subset):
        if raw['sample_id'] not in train_ids or raw['sample_id'] in forbidden:
            raise ValueError('Non-train example in visual replay')
        for field in ('videos', 'audios'):
            if (encoded.get(field) or []) != raw[field]:
                raise ValueError('Native cache order/media differs from source')
        target = json.loads(tokenizer.decode([x for x in encoded['labels'] if x != -100], skip_special_tokens=True).strip())
        if target != json.loads(raw['messages'][-1]['content']):
            raise ValueError('Native cache order/targets differ from source')
        prompt = raw['messages'][0]['content'].replace('<video>', '').replace('<audio>', '')
        if prompt not in tokenizer.decode(encoded['input_ids'], skip_special_tokens=False):
            raise ValueError('Source prompt and selected native tokens disagree')
        if not raw['videos'] or len(encoded['input_ids']) != len(encoded['labels']):
            raise ValueError('Missing visual inputs or malformed labels')
        counts[target_kind(target, True, bool(raw['audios']))] += 1
        lengths.append(len(encoded['input_ids']))
    if max(lengths) >= config['cutoff_len']:
        raise ValueError('Truncated visual replay example')
    b.write_json(DATA / 'vision_replay.json', selected)
    registry = json.loads((Path(source['dataset_dir']) / 'dataset_info.json').read_text())['distill_train']
    b.write_json(DATA / 'dataset_info.json', {'vision_replay': {**registry, 'file_name': 'vision_replay.json'}})
    report = {'source': 'Original train gold tasks; exact subset of existing native LF cache',
              'datasets': {'vision_replay': {'records': len(subset), 'task_counts': dict(counts)}},
              'selected_source_indices': indices, 'source_cache': source['tokenized_path'],
              'val_test_used_for_training': False, 'every_selected_prompt_media_target_verified': True,
              'freeze_existing_language_lora': True}
    b.write_json(DATA / 'preparation.json', report)
    DatasetDict(train=subset).save_to_disk(config['tokenized_path'])
    manifest = {'fingerprint': fingerprint(config), 'records': len(subset), 'task_counts': dict(counts),
                'maximum_tokens': max(lengths), 'all_targets_match_source': True,
                'every_source_prompt_and_media_matches_native_tokens': True, 'new_media_decoding': False}
    b.write_json(DATA / 'vision_replay_tokenization.json', manifest)
    b.write_json(OUT / 'preparation.json', report)
    b.write_json(OUT / 'tokenization.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'fingerprint'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
