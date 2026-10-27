"""JSON label contracts shared by training and evaluation."""
from scripts.common import baseline as b


def parse_decision(raw):
    for value in b.json_objects(raw):
        if value.get('pred_emotion') in ('positive', 'negative') and value.get('is_uncertain') in ('yes', 'no'):
            return {k: value[k] for k in ('pred_emotion', 'is_uncertain')}
    raise ValueError('Invalid direct-decision JSON')

def parse_task(raw, task):
    fields = {"pred_emotion": ("positive", "negative")} if task == "emotion" else {
        "vision_emotion": b.EMOTIONS, "other_emotion": b.EMOTIONS}
    for value in b.json_objects(raw):
        if all(value.get(k) in options for k, options in fields.items()):
            return {k: value[k] for k in fields}
    raise ValueError(f"Invalid {task} JSON")
