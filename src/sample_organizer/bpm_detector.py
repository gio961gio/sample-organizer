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
    Estimate the tempo (BPM) of rhythmic audio signals without arbitrary tempo priors.

    Evaluates both half-time and double-time interpretations using objective
    harmonic summation across integer lag multiples, then recalculates confidence
    based on the definitive selected tempo.

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
            Recalculated heuristic confidence score [0.0, 1.0].
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
        return default_non_rhythmic

    duration = len(signal) / sample_rate
    if duration < 0.6:
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

    mean_energy = float(np.mean(energy))
    onset_std = float(np.std(onsets))
    onset_max = float(np.max(onsets))

    # Reject silence or continuous stationary tones with no rhythmic beat onsets
    if onset_max < 1e-5 or onset_std < 1e-6:
        return default_non_rhythmic

    if mean_energy > 0 and (onset_std / mean_energy) < 0.05:
        # Stationary continuous tone (e.g. constant sine or pad without onsets)
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

    mean_val = float(np.mean(search_region))
    std_val = float(np.std(search_region)) + 1e-8

    # Find prominent local peaks in search region
    candidate_lags = []
    for lag in range(min_lag + 1, max_lag):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]:
            if autocorr[lag] > mean_val:
                candidate_lags.append(lag)

    if not candidate_lags:
        # Fallback to argmax in search region
        best_lag = int(np.argmax(search_region)) + min_lag
        candidate_lags = [best_lag]

    # Evaluate each candidate using unbiased harmonic summation:
    # Harmonic score S(L) = autocorr[L] + 0.5 * autocorr[2L] + 0.25 * autocorr[3L]
    scored_candidates = []
    for lag in candidate_lags:
        h_score = autocorr[lag]
        if lag * 2 < len(autocorr):
            h_score += 0.5 * autocorr[lag * 2]
        if lag * 3 < len(autocorr):
            h_score += 0.25 * autocorr[lag * 3]

        scored_candidates.append((h_score, lag))

    scored_candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, selected_lag = scored_candidates[0]

    # Recalculate confidence definitively on the selected lag
    peak_val = autocorr[selected_lag]
    z_score = (peak_val - mean_val) / std_val

    # Peak contrast with adjacent valleys
    valley_window = max(2, int(frame_rate * 0.05))  # 50ms window
    left_bound = max(min_lag, selected_lag - valley_window)
    right_bound = min(max_lag, selected_lag + valley_window)
    surrounding = autocorr[left_bound : right_bound + 1]
    valley_val = float(np.min(surrounding)) if len(surrounding) > 0 else mean_val
    peak_prominence = (peak_val - valley_val) / (std_val + 1e-8)

    if z_score < 1.75 or peak_val <= 0 or peak_prominence < 0.5:
        return default_non_rhythmic

    confidence_z = np.clip((z_score - 1.5) / 3.0, 0.0, 1.0)
    confidence_prom = np.clip(peak_prominence / 3.0, 0.0, 1.0)
    confidence = float(np.clip(0.6 * confidence_z + 0.4 * confidence_prom, 0.0, 1.0))

    is_rhythmic = confidence >= 0.20
    if not is_rhythmic:
        return default_non_rhythmic

    estimated_bpm = 60.0 / (selected_lag / frame_rate)

    return {
        "bpm": round(float(estimated_bpm), 1),
        "confidence": round(confidence, 3),
        "is_rhythmic": True,
    }
