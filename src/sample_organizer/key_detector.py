"""Harmonic Key Detection using Chromagram and Krumhansl-Schmuckler cognitive profiles."""
import numpy as np
from typing import Dict, Any

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler Cognitive Key Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def compute_chroma(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute a 12-dimensional pitch chroma vector using short-time Fourier analysis."""
    if not isinstance(signal, np.ndarray) or len(signal) == 0 or not np.all(np.isfinite(signal)):
        return np.zeros(12, dtype=np.float32)

    if np.max(np.abs(signal)) < 1e-5:
        return np.zeros(12, dtype=np.float32)

    window_size = 4096
    hop_size = 2048

    if len(signal) < window_size:
        padded = np.pad(signal, (0, window_size - len(signal)), mode="constant")
    else:
        padded = signal

    num_frames = max(1, (len(padded) - window_size) // hop_size + 1)
    window = np.hanning(window_size)
    chroma = np.zeros(12, dtype=np.float64)

    freq_bins = np.fft.rfftfreq(window_size, 1.0 / sample_rate)

    for i in range(num_frames):
        start = i * hop_size
        frame = padded[start : start + window_size]
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)), mode="constant")
        spectrum = np.abs(np.fft.rfft(frame * window))

        # Map frequencies to 12 pitch classes across octaves 2 to 6
        for pitch_idx in range(12):
            for octave in range(2, 7):
                midi_note = (octave + 1) * 12 + pitch_idx
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                if freq < freq_bins[-1]:
                    bin_idx = int(np.argmin(np.abs(freq_bins - freq)))
                    chroma[pitch_idx] += float(spectrum[bin_idx])

    total = np.sum(chroma)
    if total > 1e-8:
        chroma = chroma / total
    else:
        chroma = np.zeros(12, dtype=np.float32)

    return chroma.astype(np.float32)


def estimate_key(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Estimate musical key using Krumhansl-Schmuckler correlation.

    Returns 'Unknown' if the audio is silent, non-tonal, or ambiguous.
    """
    default_unknown = {
        "key": "Unknown",
        "root": "Unknown",
        "scale": "Unknown",
        "confidence": 0.0,
        "chroma": [0.0] * 12,
    }

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got: {sample_rate}")

    if not isinstance(signal, np.ndarray) or len(signal) < 1024:
        return default_unknown

    if not np.all(np.isfinite(signal)) or np.max(np.abs(signal)) < 1e-4:
        return default_unknown

    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-4:  # roughly -80 dBFS
        return default_unknown

    chroma = compute_chroma(signal, sample_rate)
    chroma_std = float(np.std(chroma))
    chroma_peak_diff = float(np.max(chroma) - np.min(chroma))

    if chroma_std < 0.02 or chroma_peak_diff < 0.05:
        # Flat or non-tonal chroma vector (noise, ambient rumble, or silence)
        return default_unknown

    chroma_norm = (chroma - np.mean(chroma)) / chroma_std

    candidates = []

    for i in range(12):
        # Major
        maj_shift = np.roll(MAJOR_PROFILE, i)
        maj_norm = (maj_shift - np.mean(maj_shift)) / np.std(maj_shift)
        corr_maj = float(np.dot(chroma_norm, maj_norm) / len(chroma_norm))
        candidates.append((corr_maj, PITCH_CLASSES[i], "Major", f"{PITCH_CLASSES[i]} Major"))

        # Minor
        min_shift = np.roll(MINOR_PROFILE, i)
        min_norm = (min_shift - np.mean(min_shift)) / np.std(min_shift)
        corr_min = float(np.dot(chroma_norm, min_norm) / len(chroma_norm))
        candidates.append((corr_min, PITCH_CLASSES[i], "Minor", f"{PITCH_CLASSES[i]} Minor"))

    # Sort descending by correlation
    candidates.sort(key=lambda c: c[0], reverse=True)

    best_score, best_root, best_scale, best_key = candidates[0]
    second_score = candidates[1][0]
    margin = best_score - second_score

    # Rejection criteria for noise / non-tonal sound
    if best_score < 0.25 or margin < 0.02:
        return default_unknown

    # Heuristic confidence calculation [0.0, 1.0]
    conf_base = max(0.0, min(1.0, (best_score + 0.2) / 1.2))
    conf_margin = max(0.0, min(1.0, margin * 4.0))
    confidence = float(np.clip(0.6 * conf_base + 0.4 * conf_margin, 0.0, 1.0))

    return {
        "key": best_key,
        "root": best_root,
        "scale": best_scale,
        "confidence": round(confidence, 3),
        "chroma": [round(float(c), 4) for c in chroma],
    }
