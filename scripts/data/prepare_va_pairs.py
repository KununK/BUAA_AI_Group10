"""Train-only VA emotion comparisons with balanced matching/mismatching donors."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import wave

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Dataset/processed/va_pairs_v9'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train_path = ROOT / 'Dataset/train.jsonl'
    rows = [json.loads(line) for line in train_path.open()]
    va = {r['sample_id']: r for r in rows if r['protocol'] == 'VA'}
    originals_path = ROOT / 'Dataset/processed/va_conflict_v7/va_conflict_gold.json'
    originals = json.loads(originals_path.read_text())
    by_id = {r['sample_id']: r for r in originals}
    if set(by_id) != set(va): raise ValueError('Original replay does not cover exactly original-train VA')
    reserved = [json.loads(line) for name in ('val.jsonl', 'test_release.jsonl')
                for line in (ROOT / 'Dataset' / name).open()]
    if ({r['sample_id'] for r in reserved} & set(va)
            or {r['video'] for r in reserved} & {r['video'] for r in va.values()}):
        raise ValueError('Train/validation/test source overlap')
    prompt_source = ROOT / 'scripts/eval/va_conflict.py'
    prompt = next(ast.literal_eval(n.value) for n in ast.parse(prompt_source.read_text()).body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROMPT' for t in n.targets))
    anchors = {label: [] for label in ('positive', 'negative')}
    donors = {label: [] for label in ('positive', 'negative')}
    for sid, row in va.items():
        audio = Path(by_id[sid]['audios'][0])
        with wave.open(str(audio)) as wav:
            if wav.getframerate() != 16000: raise ValueError('Unexpected audio sampling rate')
            duration = wav.getnframes() / wav.getframerate()
        if not .5 <= duration <= 15: continue
        for view, groups in (('M1', anchors), ('M2', donors)):
            annotation = row['views'][view]
            if annotation['is_clear'] and annotation['label'] in groups:
                groups[annotation['label']].append(sid)
    if any(len(group) < 20 for group in (*anchors.values(), *donors.values())):
        raise ValueError('Insufficient clearly labeled train-only anchor/donor pools')
    rng = random.Random(20261026); pairs = []; seen = set(); cell_counts = Counter()
    for visual_label in anchors:
        for audio_label in donors:
            added = 0
            while added < 500:
                anchor = rng.choice(anchors[visual_label]); donor = rng.choice(donors[audio_label])
                if va[anchor]['video'] == va[donor]['video'] or (anchor, donor) in seen: continue
                seen.add((anchor, donor)); added += 1
                conflict = visual_label != audio_label
                record = {'sample_id': f'va_pair_v9:{anchor}::{donor}', 'record_id': f'pair:{len(pairs)}',
                    'system': originals[0]['system'], 'messages': [
                        {'role': 'user', 'content': '<video><audio>' + prompt},
                        {'role': 'assistant', 'content': json.dumps({'is_uncertain': 'yes' if conflict else 'no'}, separators=(',', ':'))}],
                    'videos': by_id[anchor]['videos'], 'audios': by_id[donor]['audios'],
                    'augmentation': {'visual_source_train_id': anchor, 'audio_source_train_id': donor,
                        'M1_label': visual_label, 'M2_label': audio_label,
                        'derived_sample_type': 'Conflict' if conflict else 'Aligned'}}
                if record['videos'] == by_id[donor]['videos']: raise ValueError('Donor must be a different video')
                pairs.append(record); cell_counts[visual_label + '__' + audio_label] += 1
    rng.shuffle(pairs); data = [*originals, *pairs]
    targets = Counter(json.loads(r['messages'][-1]['content'])['is_uncertain'] for r in data)
    if len(data) != 4294 or targets != {'yes': 1792, 'no': 2502}: raise ValueError('Unexpected pair/replay exposure counts')
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'va_pairs_gold.json'; content = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if target.exists() and target.read_text() != content: raise ValueError('Preserve the existing paired dataset')
    target.write_text(content)
    registry = {'va_pairs_gold': {'file_name': target.name, 'formatting': 'sharegpt',
        'columns': {'messages': 'messages', 'system': 'system', 'videos': 'videos', 'audios': 'audios'},
        'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}}}
    (OUT / 'dataset_info.json').write_text(json.dumps(registry, indent=2) + '\n')
    audit = {'records': len(data), 'original_replay_exposures': len(originals), 'paired_exposures': len(pairs),
        'pair_cells': dict(cell_counts), 'assistant_targets': dict(targets), 'unique_pairs': len(seen),
        'anchor_pools': {k: len(v) for k, v in anchors.items()}, 'audio_donor_pools': {k: len(v) for k, v in donors.items()},
        'derived_labels': 'Original-train VA M1 vision label differs from donor VA M2 audio label',
        'both_synthetic_classes_use_different_video_audio_sources': True, 'synthetic_M12_labels_created': False,
        'no_reserved_ID_or_video_overlap': True, 'no_dataset_transcript_in_prompts': True,
        'expected_optimizer_updates': 537, 'preselected_steps': [269, 537],
        'sha256': {str(p.relative_to(ROOT)): digest(p) for p in (train_path, originals_path, prompt_source, target)}}
    (OUT / 'preparation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
