"""Harmonic Key Detection using Chromagram and Krumhansl-Schmuckler profiles."""
import numpy as np
from typing import Dict, Any

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler Key Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def compute_chroma(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a 12-dimensional pitch chroma vector using discrete Fourier analysis."""
    if len(signal) < 1024:
        return np.ones(12) / 12.0

    window_size = 4096
    hop_size = 2048
    num_frames = (len(signal) - window_size) // hop_size
    if num_frames < 1:
        window_size = len(signal)
        num_frames = 1
        hop_size = 1

    window = np.hanning(window_size)
    chroma = np.zeros(12)

    # Note center frequencies from C2 (65.4 Hz) to B6 (1975.5 Hz)
    freq_bins = np.fft.rfftfreq(window_size, 1.0 / sample_rate)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start : start + window_size] * window
        spectrum = np.abs(np.fft.rfft(frame))

        # Map frequencies to 12 pitch classes
        for pitch_idx in range(12):
            for octave in range(2, 7):
                midi_note = (octave + 1) * 12 + pitch_idx
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                if freq < freq_bins[-1]:
                    bin_idx = int(np.argmin(np.abs(freq_bins - freq)))
                    chroma[pitch_idx] += spectrum[bin_idx]

    total = np.sum(chroma)
    if total > 0:
        chroma = chroma / total
    return chroma


def estimate_key(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Estimate the musical key of an audio signal."""
    chroma = compute_chroma(signal, sample_rate)
    chroma_norm = chroma - np.mean(chroma)
    chroma_std = np.std(chroma)
    if chroma_std == 0:
        chroma_std = 1.0
    chroma_norm = chroma_norm / chroma_std

    best_score = -1.0
    best_key = "C Major"
    best_root = "C"
    best_scale = "Major"

    for i in range(12):
        # Correlate with shifted Major profile
        maj_shift = np.roll(MAJOR_PROFILE, i)
        maj_norm = (maj_shift - np.mean(maj_shift)) / np.std(maj_shift)
        corr_maj = float(np.corrcoef(chroma_norm, maj_norm)[0, 1])

        if corr_maj > best_score:
            best_score = corr_maj
            best_root = PITCH_CLASSES[i]
            best_scale = "Major"
            best_key = f"{best_root} Major"

        # Correlate with shifted Minor profile
        min_shift = np.roll(MINOR_PROFILE, i)
        min_norm = (min_shift - np.mean(min_shift)) / np.std(min_shift)
        corr_min = float(np.corrcoef(chroma_norm, min_norm)[0, 1])

        if corr_min > best_score:
            best_score = corr_min
            best_root = PITCH_CLASSES[i]
            best_scale = "Minor"
            best_key = f"{best_root} Minor"

    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))

    return {
        "key": best_key,
        "root": best_root,
        "scale": best_scale,
        "confidence": round(confidence, 3),
        "chroma": [round(float(c), 4) for c in chroma],
    }
