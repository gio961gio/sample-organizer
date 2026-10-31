"""Audio I/O operations using soundfile backend with polyphase resampling."""
import os
import math
from typing import Tuple, Optional
import numpy as np
import scipy.signal
import soundfile as sf


def load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file (WAV, AIFF, FLAC) into a normalized mono float32 numpy array [-1.0, 1.0].

    Parameters
    ----------
    file_path : str
        Path to audio file.
    target_sr : int, optional
        Target sample rate for resampling. If None or equal to source rate, no resampling is applied.

    Returns
    -------
    data : np.ndarray
        1D float32 normalized mono audio signal.
    sample_rate : int
        Sample rate of the returned signal (target_sr if resampled, else original sample rate).

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file contains no frames, non-finite values, or target_sr <= 0.
    RuntimeError
        If decoding fails due to corrupted file or unsupported codec.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to read audio file '{file_path}': {exc}") from exc

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate ({sample_rate}) in '{file_path}'")

    if data.shape[0] == 0:
        raise ValueError(f"Audio file '{file_path}' contains no audio frames")

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Audio file '{file_path}' contains non-finite values (NaN or Inf)")

    # Downmix multichannel to mono via equal-weight averaging
    if data.shape[1] > 1:
        mono_signal = np.mean(data, axis=1)
    else:
        mono_signal = data[:, 0]

    mono_signal = np.ascontiguousarray(mono_signal, dtype=np.float32)

    # Optional resampling
    if target_sr is not None:
        if not isinstance(target_sr, (int, np.integer)) or target_sr <= 0:
            raise ValueError(f"target_sr must be a positive integer, got {target_sr}")
        if target_sr != sample_rate:
            gcd = math.gcd(int(target_sr), int(sample_rate))
            up = int(target_sr) // gcd
            down = int(sample_rate) // gcd
            mono_signal = scipy.signal.resample_poly(mono_signal, up, down).astype(np.float32)
            sample_rate = int(target_sr)

    return mono_signal, sample_rate


def save_audio(
    file_path: str,
    signal: np.ndarray,
    sample_rate: int = 44100,
    subtype: Optional[str] = "PCM_16",
):
    """
    Save a mono float32 numpy array to an audio file (default: 16-bit PCM WAV).

    Parameters
    ----------
    file_path : str
        Destination path.
    signal : np.ndarray
        1D audio array [-1.0, 1.0].
    sample_rate : int, default 44100
        Audio sample rate.
    subtype : str, optional
        Format subtype (e.g. 'PCM_16', 'PCM_24', 'FLOAT').

    Raises
    ------
    ValueError
        If sample_rate <= 0, signal is empty, or signal contains non-finite values.
    TypeError
        If signal is not a numpy array.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got: {sample_rate}")

    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be a numpy ndarray, got {type(signal)}")

    if signal.size == 0:
        raise ValueError("Cannot save an empty audio array")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Cannot save audio containing non-finite values (NaN or Inf)")

    signal_1d = np.squeeze(signal)
    if signal_1d.ndim != 1:
        raise ValueError(f"save_audio expects 1D mono signal, got shape {signal.shape}")

    clipped = np.clip(signal_1d, -1.0, 1.0).astype(np.float32)

    try:
        sf.write(file_path, clipped, sample_rate, subtype=subtype)
    except Exception as exc:
        raise RuntimeError(f"Failed to write audio file '{file_path}': {exc}") from exc
