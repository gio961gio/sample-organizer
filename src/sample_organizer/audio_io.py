"""Audio I/O operations with pure-Python fallbacks."""
import os
import wave
import struct
import math
from typing import Tuple, Optional
import numpy as np


def load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load a WAV audio file into a normalized mono float32 numpy array [-1.0, 1.0]."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit unsigned PCM
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit signed PCM
        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        reshaped = raw_arr.reshape(-1, 3)
        padded = np.pad(reshaped, ((0, 0), (1, 0)), mode="constant")
        unpacked = np.frombuffer(padded.tobytes(), dtype="<i4").astype(np.float32)
        data = unpacked / 2147483648.0
    elif sampwidth == 4:
        # 32-bit signed or float
        try:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
        except Exception:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    # Reshape multichannel and downmix to mono
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        data = np.mean(data, axis=1)

    return data, sample_rate


def save_audio(file_path: str, signal: np.ndarray, sample_rate: int = 44100):
    """Save a mono float32 numpy array to a 16-bit PCM WAV file."""
    clipped = np.clip(signal, -1.0, 1.0)
    int_data = (clipped * 32767.0).astype(np.int16)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_data.tobytes())
