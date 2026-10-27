"""Load the exact local Thinker and its LoRA without changing the SFT environment."""
from importlib.metadata import version
import os
import sys

import torch

from scripts.common import baseline as b


def load_policy(adapter, trainable=False):
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this executor")
    torch.manual_seed(42)
    torch.set_num_threads(8)
    processor = Qwen2_5OmniProcessor.from_pretrained(b.MODEL, local_files_only=True, use_fast=False)
    config = Qwen2_5OmniConfig.from_pretrained(b.MODEL, local_files_only=True)
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        b.MODEL, config=config.thinker_config, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError("Original Thinker checkpoint did not load exactly")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    model.eval()
    environment = {"python": sys.executable, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "gpu": torch.cuda.get_device_name(0), "trainable": trainable,
                   "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "llamafactory")},
                   "missing_checkpoint_keys": 0, "mismatched_checkpoint_keys": 0}
    return model, processor, environment


def generate_text(model, processor, inputs, media, max_tokens=64, do_sample=False, temperature=1.0):
    inputs = inputs.to("cuda:0").to(torch.bfloat16)
    kwargs = {"do_sample": do_sample, "max_new_tokens": max_tokens,
              "use_audio_in_video": media["use_audio_in_video"],
              "eos_token_id": processor.tokenizer.eos_token_id,
              "pad_token_id": processor.tokenizer.pad_token_id}
    if do_sample:
        kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
    with torch.no_grad():
        generated = model.generate(**inputs, **kwargs)
    response = generated[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
