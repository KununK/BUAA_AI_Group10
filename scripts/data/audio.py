"""Reusable extraction of 16 kHz mono training audio."""
import hashlib
import wave
import numpy as np
from scripts.common import baseline as b

def audio_path(directory, row):
    return directory / "audio" / (hashlib.sha256(row["video"].encode()).hexdigest()[:24] + ".wav")

def export_audio(pair):
    video, target = pair
    if target.exists():
        with wave.open(str(target)) as f:
            if f.getframerate() == 16000 and f.getnchannels() == 1 and f.getnframes() > 0:
                return
        raise ValueError(f"Invalid existing audio: {target}")
    audio = b.decode_audio(video)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    with wave.open(str(temporary), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    temporary.replace(target)
