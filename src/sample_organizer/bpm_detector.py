"""BPM and tempo estimation via energy flux and onset autocorrelation."""
import numpy as np
from typing import Dict, Any


def estimate_bpm(signal: np.ndarray, sample_rate: int, min_bpm: float = 60.0, max_bpm: float = 180.0) -> Dict[str, Any]:
    """Estimate the BPM (tempo) of rhythmic audio loops."""
    duration = len(signal) / sample_rate
    if duration < 0.5:
        return {"bpm": 120.0, "confidence": 0.0, "is_rhythmic": False}

    frame_size = int(0.02 * sample_rate)  # 20ms frames
    hop_size = int(0.01 * sample_rate)    # 10ms hop
    num_frames = (len(signal) - frame_size) // hop_size

    if num_frames < 10:
        return {"bpm": 120.0, "confidence": 0.0, "is_rhythmic": False}

    # Compute frame energy envelope
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start : start + frame_size]
        energy[i] = np.sum(frame ** 2)

    # First difference for onset strength
    onsets = np.diff(energy)
    onsets = np.maximum(0, onsets)  # Half-wave rectification
    onsets = onsets - np.mean(onsets)

    # Autocorrelation of onsets
    autocorr = np.correlate(onsets, onsets, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    # Map lag in frames to BPM
    frame_rate = sample_rate / hop_size
    min_lag = int(frame_rate * (60.0 / max_bpm))
    max_lag = int(frame_rate * (60.0 / min_bpm))

    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    search_region = autocorr[min_lag : max_lag + 1]
    if len(search_region) == 0:
        return {"bpm": 120.0, "confidence": 0.0, "is_rhythmic": False}

    peak_lag_idx = int(np.argmax(search_region)) + min_lag
    peak_val = autocorr[peak_lag_idx]
    mean_val = np.mean(search_region)

    period_sec = peak_lag_idx / frame_rate
    estimated_bpm = 60.0 / period_sec if period_sec > 0 else 120.0

    confidence = float((peak_val - mean_val) / (np.max(autocorr) + 1e-6))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "bpm": round(float(estimated_bpm), 1),
        "confidence": round(confidence, 3),
        "is_rhythmic": bool(confidence > 0.15),
    }
