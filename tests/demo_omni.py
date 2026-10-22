"""Qwen2.5-Omni: text -> text, or text -> text + speech. No video generation."""

import argparse
import os
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/data/vlm/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00"
SYSTEM = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="请用中文简短介绍你自己，不超过三句话。")
    parser.add_argument("--audio", action="store_true", help="Generate speech as well as the text response")
    parser.add_argument("--gpu", default="1", help="Physical GPU index; default: 1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/omni_demo/response.wav")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import torch
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this script in your GPU-enabled terminal.")

    # Thinker understands/generates text. Talker + token2wav generate speech.
    config = Qwen2_5OmniConfig.from_pretrained(args.model, local_files_only=True)
    config.enable_audio_output = args.audio
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)

    # Use the official system prompt, which is required for reliable speech output.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, return_tensors="pt").to("cuda:0")

    with torch.inference_mode():
        result = model.generate(
            **inputs, return_audio=args.audio,
            thinker_max_new_tokens=128, thinker_do_sample=False,
        )
    text_ids, audio = result if args.audio else (result, None)
    reply = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    )[0]
    print("\n模型文字回答：\n" + reply)

    if audio is not None:
        # Save a 24 kHz mono WAV using the standard library; no soundfile needed.
        samples = audio.detach().float().cpu().numpy().reshape(-1)
        if not np.isfinite(samples).all() or samples.size == 0:
            raise RuntimeError("Model returned an invalid audio waveform")
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(args.output), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(pcm.tobytes())
        print(f"语音文件：{args.output.resolve()}（{len(samples) / 24000:.2f} 秒）")


if __name__ == "__main__":
    main()
