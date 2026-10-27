"""Per-modality and source metrics for validation predictions."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.common import baseline as b

def write_diagnostics(rows, records, out):
    by_id = {r["sample_id"]: r for r in records}
    diagnostics = {}
    for protocol in ("all", "VT", "VA"):
        subset = [row for row in rows if protocol == "all" or b.get_protocol(row) == protocol]
        diagnostics[protocol] = {"samples": len(subset)}
        for view, field in (("M1", "vision_emotion"), ("M2", "other_emotion")):
            matrix = {gold: {pred: 0 for pred in (*b.EMOTIONS, "missing")} for gold in b.EMOTIONS}
            for row in subset:
                label = by_id.get(row["sample_id"], {}).get("auxiliary_prediction", {}).get(field, "missing")
                matrix[row["views"][view]["label"]][label] += 1
            diagnostics[protocol][view] = {"confusion": matrix,
                "accuracy": sum(matrix[label][label] for label in b.EMOTIONS) / len(subset) if subset else None}
    b.write_json(out / "auxiliary_metrics.json", diagnostics)
    source_metrics = {}
    for source in sorted({r.get("source_dataset", "unknown") for r in rows}):
        subset = [r for r in rows if r.get("source_dataset", "unknown") == source]
        ids = {r["sample_id"] for r in subset}
        source_metrics[source] = b.score_rows(subset, [r for r in records if r["sample_id"] in ids])
    b.write_json(out / "metrics_by_source.json", source_metrics)


def combine_same_adapter(emotion_run, uncertainty_run, out, dataset_root=None):
    """Reuse this same adapter's all-VT emotion and protocol-aware conflict calls.

    The function joins complete predictions, never selects samples using gold.
    This is an inference variant, not a new checkpoint or an RL training gain.
    """
    emotion_run, uncertainty_run, out = map(Path, (emotion_run, uncertainty_run, out))
    ec = json.loads((emotion_run / 'run_config.json').read_text())
    uc = json.loads((uncertainty_run / 'run_config.json').read_text())
    for key in ('split', 'adapter_sha256', 'dataset_sha256'):
        if ec[key] != uc[key]:
            raise ValueError(f'Cannot combine different {key}')
    if ec.get('emotion_source') != 'this RL adapter, all-VT':
        raise ValueError('Emotion run must actually use this adapter and the all-VT emotion task')
    source = (Path(dataset_root) if dataset_root else b.ROOT / 'Dataset') / ('val.jsonl' if ec['split'] == 'val' else 'test_release.jsonl')
    if hashlib.sha256(source.read_bytes()).hexdigest() != ec['dataset_sha256']:
        raise ValueError('Dataset changed since evaluation')
    rows = b.read_jsonl(source)
    emotion_rows = b.read_jsonl(emotion_run / 'predictions.jsonl')
    uncertain_rows = b.read_jsonl(uncertainty_run / 'predictions.jsonl')
    expected_ids = {r['sample_id'] for r in rows}
    for records in (emotion_rows, uncertain_rows):
        if len(records) != len(rows) or {r['sample_id'] for r in records} != expected_ids or any(not r.get('prediction') for r in records):
            raise ValueError('Both source runs must have complete, valid, unique predictions')
        _, invalid = b.checked_predictions(rows, records)
        if invalid:
            raise ValueError('A source run contains invalid labels or clarification questions')
    emotions = {r['sample_id']: r['prediction']['pred_emotion'] for r in emotion_rows}
    combined = [{**record, 'prediction': {**record['prediction'], 'pred_emotion': emotions[record['sample_id']]}}
                for record in uncertain_rows]
    manifest = {'split': ec['split'], 'adapter_sha256': ec['adapter_sha256'], 'dataset_sha256': ec['dataset_sha256'],
                'inference_variant': 'same-adapter all-VT emotion plus protocol-aware direct conflict',
                'emotion_run': str(emotion_run.resolve()), 'uncertainty_run': str(uncertainty_run.resolve()),
                'source_prediction_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (emotion_run / 'predictions.jsonl', uncertainty_run / 'predictions.jsonl')},
                'gold_used_to_select_predictions': False}
    if (out / 'run_config.json').exists() and json.loads((out / 'run_config.json').read_text()) != manifest:
        raise ValueError('A different combined run already exists')
    out.mkdir(parents=True, exist_ok=True)
    b.write_json(out / 'run_config.json', manifest)
    path = out / 'predictions.jsonl'
    with path.open('w', encoding='utf-8') as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    args = SimpleNamespace(split=ec['split'], label_set='binary', emotion_reference=None, dataset_root=source.parent)
    metrics = b.finalize(args, rows, rows, combined, out)
    if metrics['valid_predictions'] != len(rows):
        raise ValueError('Combined outputs failed full-set validation')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a same-adapter inference combination; no fitting or gold-based selection')
    parser.add_argument('--emotion-run', type=Path, required=True)
    parser.add_argument('--uncertainty-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(combine_same_adapter(args.emotion_run, args.uncertainty_run, args.output_dir), ensure_ascii=False, indent=2))
