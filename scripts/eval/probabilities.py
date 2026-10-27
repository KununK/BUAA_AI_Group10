"""Collect restricted-label conditional scores from actual Omni media forwards.

These are uncalibrated probabilities conditional on a canonical assistant JSON
prefix and allowed labels, not calibrated confidence or a new trained model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.common import baseline as b
from scripts.common import rl_tasks as tasks
from scripts.common.omni_policy import load_policy


def label_probabilities(model, tokenizer, inputs, media, task):
    key = 'pred_emotion' if task == 'emotion' else 'vision_emotion' if task == 'vision' else 'other_emotion'
    labels = ('positive', 'negative') if task == 'emotion' else ('positive', 'negative', 'neutral')
    prefix = '{"' + key + '":"'
    return conditional_probabilities(model, tokenizer, inputs, media, prefix, labels)


def conditional_probabilities(model, tokenizer, inputs, media, prefix, labels):
    """Score single-token allowed labels after a declared assistant prefix."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    label_ids = []
    for label in labels:
        full = tokenizer.encode(prefix + label, add_special_tokens=False)
        if full[:len(prefix_ids)] != prefix_ids or len(full) != len(prefix_ids) + 1:
            raise ValueError('This scorer requires each label to be exactly one token after its JSON prefix')
        label_ids.append(full[-1])
    packed = dict(inputs)
    extra = torch.tensor([prefix_ids], dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device)
    packed['input_ids'] = torch.cat((inputs['input_ids'], extra), dim=-1)
    packed['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(extra)), dim=-1)
    head = model.get_base_model().lm_head
    hook = head.register_forward_pre_hook(lambda _module, args: (args[0][:, -1:, :],))
    try:
        with torch.no_grad():
            output = model(**packed, use_audio_in_video=media['use_audio_in_video'], use_cache=False, return_dict=True)
            if output.logits.shape[1] != 1:
                raise ValueError('Expected logits only at the next-label position')
            selected = output.logits[0, -1, label_ids].float()
            probabilities = torch.softmax(selected, dim=-1)
    finally:
        hook.remove()
    if not torch.isfinite(probabilities).all():
        raise ValueError('Nonfinite label probabilities')
    return {'probabilities': dict(zip(labels, probabilities.cpu().tolist())),
            'label_logits': dict(zip(labels, selected.cpu().tolist())),
            'predicted_label': labels[int(probabilities.argmax().item())],
            'assistant_prefix': prefix, 'label_token_ids': dict(zip(labels, label_ids))}


def cropped_vision_inputs(processor, row, dataset_root, fraction):
    # Crop BEFORE shrinking the full frame, preserving original face detail.
    # This is a fixed central crop, not a face/speaker detector.
    frames, media = b.decode_video(Path(dataset_root) / row['video'], 2, 32, 2097152)
    height, width = frames.shape[1:3]
    crop_height, crop_width = max(1, round(height*fraction)), max(1, round(width*fraction))
    top, left = (height-crop_height)//2, (width-crop_width)//2
    frames = frames[:, top:top+crop_height, left:left+crop_width, :].copy()
    rendered = processor.apply_chat_template(tasks.messages(row, 'vision'), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], videos=[frames], fps=media['effective_fps'], return_tensors='pt',
                       padding=True, use_audio_in_video=False,
                       videos_kwargs={'do_sample_frames': False, 'min_pixels': 3136, 'max_pixels': 262144})
    if 'input_features' in inputs or 'pixel_values_videos' not in inputs:
        raise ValueError('Cropped visual branch must contain video and no audio')
    media.update(task='vision', video_present=True, audio_present=False, transcript_present=False,
                 use_audio_in_video=False, input_tokens=inputs['input_ids'].shape[-1],
                 crop_type='fixed_center', crop_fraction=fraction, source_sampled_size=[height, width],
                 crop_box_xyxy=[left, top, left+crop_width, top+crop_height])
    return inputs, media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--dataset-root', type=Path, default=b.ROOT / 'Dataset')
    parser.add_argument('--vision-center-crop', type=float, default=1.)
    parser.add_argument('--reuse-probabilities', type=Path,
                        help='Reuse the same adapter full-frame run emotion/other scores; only recompute vision')
    args = parser.parse_args()
    if not 0 < args.vision_center_crop <= 1:
        parser.error('--vision-center-crop must be in (0, 1]')
    source = args.dataset_root / ('val.jsonl' if args.split == 'val' else 'test_release.jsonl')
    rows = b.read_jsonl(source)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = {'split': args.split, 'adapter': str(args.adapter.resolve()),
                'adapter_sha256': hashlib.sha256((args.adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                'dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'sample_ids': [r['sample_id'] for r in rows], 'input_mode': 'independent',
                'emotion_mode': 'all-VT', 'calibrated': False, 'labels_used_for_forward': False,
                'vision_center_crop': args.vision_center_crop,
                'source_sha256': {str(p.relative_to(b.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), Path(tasks.__file__))}}
    reference = None
    if args.reuse_probabilities:
        prior = json.loads((args.reuse_probabilities / 'run_config.json').read_text())
        for key in ('split', 'adapter_sha256', 'dataset_sha256', 'sample_ids'):
            if prior[key] != manifest[key]:
                raise ValueError('Cannot reuse probability scores from different ' + key)
        if prior.get('vision_center_crop', 1.) != 1.:
            raise ValueError('Reference must be the original full-frame probability run')
        records = b.read_jsonl(args.reuse_probabilities / 'probabilities.jsonl')
        if len(records) != len(rows) or {r['sample_id'] for r in records} != set(manifest['sample_ids']):
            raise ValueError('Probability reference must be complete and unique')
        reference = {r['sample_id']: r for r in records}
        manifest['reference_probabilities_sha256'] = hashlib.sha256((args.reuse_probabilities / 'probabilities.jsonl').read_bytes()).hexdigest()
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('Probability collection configuration changed; use another output directory')
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'probabilities.jsonl'
    previous = b.read_jsonl(path) if path.exists() else []
    done = {r['sample_id'] for r in previous}
    if len(done) != len(previous) or done - set(manifest['sample_ids']):
        raise ValueError('Unexpected/duplicate probability record IDs')
    remaining = [r for r in rows if r['sample_id'] not in done]
    if remaining:
        model, processor, environment = load_policy(args.adapter)
        b.write_json(out / 'environment.json', environment)
        with path.open('a', encoding='utf-8', buffering=1) as handle:
            for index, row in enumerate(remaining, 1):
                prior = reference[row['sample_id']] if reference else None
                branches = dict(prior['branches']) if prior else {}
                input_media = dict(prior['media']) if prior else {}
                for task in ('vision',) if reference else ('emotion', *tasks.modality_tasks(row)):
                    if task == 'vision' and args.vision_center_crop < 1:
                        inputs, media = cropped_vision_inputs(processor, row, args.dataset_root, args.vision_center_crop)
                    else:
                        inputs, media = tasks.prepare_inputs(processor, row, task, args.dataset_root)
                    inputs = inputs.to('cuda:0').to(torch.bfloat16)
                    branches[task] = label_probabilities(model, processor.tokenizer, inputs, media, task)
                    input_media[task] = media
                    del inputs
                record = {'sample_id': row['sample_id'], 'protocol': b.get_protocol(row),
                          'branches': branches, 'media': input_media}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                previous.append(record)
                print(f'[{index}/{len(remaining)}] {row["sample_id"]}', flush=True)
    if len(previous) != len(rows):
        raise ValueError('Incomplete probability collection')
    b.write_json(out / 'status.json', {'status': 'complete', 'rows': len(rows),
                 'all_original_media_preserved': True, 'calibrated': False,
                 'probabilities_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
