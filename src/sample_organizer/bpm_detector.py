"""BPM and tempo estimation via energy flux and FFT-based onset autocorrelation."""
import numpy as np
from typing import Dict, Any, Optional


def estimate_bpm(
    signal: np.ndarray,
    sample_rate: int,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
) -> Dict[str, Any]:
    """
    Estimate the tempo (BPM) of rhythmic audio signals.

    Parameters
    ----------
    signal : np.ndarray
        1D mono audio buffer.
    sample_rate : int
        Audio sample rate in Hz.
    min_bpm : float, default 60.0
        Minimum allowable tempo.
    max_bpm : float, default 180.0
        Maximum allowable tempo.

    Returns
    -------
    dict with keys:
        bpm : float or None
            Estimated beats per minute, or None if non-rhythmic or undetectable.
        confidence : float
            Heuristic confidence score [0.0, 1.0].
        is_rhythmic : bool
            True if a periodic tempo was detected with sufficient confidence.
    """
    default_non_rhythmic = {"bpm": None, "confidence": 0.0, "is_rhythmic": False}

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got: {sample_rate}")
    if min_bpm <= 0 or max_bpm <= min_bpm:
        raise ValueError(f"Invalid BPM range: min={min_bpm}, max={max_bpm}")

    if not isinstance(signal, np.ndarray) or len(signal) == 0:
        return default_non_rhythmic

    if not np.all(np.isfinite(signal)) or np.max(np.abs(signal)) < 1e-4:
        # Silence or non-finite
        return default_non_rhythmic

    duration = len(signal) / sample_rate
    if duration < 0.6:
        # Too short to reliably infer periodic rhythm
        return default_non_rhythmic

    # 20ms analysis window, 10ms hop
    frame_size = max(64, int(0.02 * sample_rate))
    hop_size = max(32, int(0.01 * sample_rate))
    num_frames = (len(signal) - frame_size) // hop_size + 1

    if num_frames < 30:
        return default_non_rhythmic

    # Compute short-time frame energy envelope
    squared = signal ** 2
    energy = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        st = i * hop_size
        energy[i] = float(np.sum(squared[st : st + frame_size]))

    # Half-wave rectified first difference (onset strength)
    onsets = np.diff(energy)
    onsets = np.maximum(0.0, onsets)

    onset_std = float(np.std(onsets))
    onset_max = float(np.max(onsets))
    if onset_max < 1e-6 or onset_std < 1e-6:
        # Constant tone or silence
        return default_non_rhythmic

    # FFT-based autocorrelation
    n = len(onsets)
    onsets_norm = onsets - np.mean(onsets)
    fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    f_onsets = np.fft.rfft(onsets_norm, n=fft_size)
    autocorr = np.fft.irfft(f_onsets * np.conj(f_onsets))[:n]

    frame_rate = float(sample_rate) / float(hop_size)
    min_lag = max(1, int(frame_rate * (60.0 / max_bpm)))
    max_lag = min(n - 1, int(frame_rate * (60.0 / min_bpm)))

    if max_lag <= min_lag:
        return default_non_rhythmic

    search_region = autocorr[min_lag : max_lag + 1]
    if len(search_region) == 0:
        return default_non_rhythmic

    # Score candidates with a gentle tempo prior (centered around 115 BPM) to break octave ambiguities
    candidates = []
    mean_val = float(np.mean(search_region))
    std_val = float(np.std(search_region)) + 1e-8

    for lag_rel, val in enumerate(search_region):
        actual_lag = lag_rel + min_lag
        period_sec = actual_lag / frame_rate
        tempo = 60.0 / period_sec

        # Gaussian prior around 115 BPM (std = 45 BPM)
        prior = np.exp(-0.5 * ((tempo - 115.0) / 45.0) ** 2)
        score = (val - mean_val) * (0.8 + 0.4 * prior)
        candidates.append((score, actual_lag, val, tempo))

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_lag, peak_val, best_tempo = candidates[0]

    # Periodicity significance
    z_score = (peak_val - mean_val) / std_val

    # Half-time / double-time preference check
    # If the half-lag (double tempo) has a strong peak, prefer it if within limits
    half_lag = best_lag // 2
    if half_lag >= min_lag:
        half_val = autocorr[half_lag]
        if half_val > 0.88 * peak_val and 80.0 <= (60.0 / (half_lag / frame_rate)) <= 150.0:
            best_lag = half_lag
            best_tempo = 60.0 / (best_lag / frame_rate)

    if z_score < 1.75 or peak_val <= 0:
        return default_non_rhythmic

    confidence = float(np.clip((z_score - 1.5) / 3.0, 0.0, 1.0))
    is_rhythmic = confidence >= 0.20

    if not is_rhythmic:
        return default_non_rhythmic

    return {
        "bpm": round(float(best_tempo), 1),
        "confidence": round(confidence, 3),
        "is_rhythmic": True,
    }
