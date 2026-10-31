"""BPM and tempo estimation via energy flux, FFT onset autocorrelation, and octave evaluation."""
import numpy as np
from typing import Dict, Any, Optional, List


def estimate_bpm(
    signal: np.ndarray,
    sample_rate: int,
    min_bpm: float = 60.0,
    max_bpm: float = 180.0,
) -> Dict[str, Any]:
    """
    Estimate the tempo (BPM) of rhythmic audio signals without arbitrary tempo priors.

    Evaluates candidate periods using parabolic sub-sample peak refinement and
    balanced octave (half-time / double-time) harmonic analysis. Confidently
    disambiguates rhythmic octaves without systematically favoring longer lags,
    and scales confidence according to peak prominence, candidate margins, and
    octave ambiguity.

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
    target_min_lag = frame_rate * (60.0 / max_bpm)
    target_max_lag = frame_rate * (60.0 / min_bpm)

    # Pad search window slightly to allow boundary peak detection and parabolic interpolation
    search_min = max(1, int(np.floor(target_min_lag)) - 2)
    search_max = min(n - 2, int(np.ceil(target_max_lag)) + 2)

    if search_max <= search_min:
        return default_non_rhythmic

    search_region = autocorr[search_min : search_max + 1]
    if len(search_region) == 0:
        return default_non_rhythmic

    mean_val = float(np.mean(search_region))
    std_val = float(np.std(search_region)) + 1e-8

    # Detect local peaks with 3-point parabolic interpolation for sub-sample accuracy
    peaks: List[Dict[str, Any]] = []
    for lag in range(search_min + 1, search_max):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]:
            if autocorr[lag] > mean_val:
                y0, y1, y2 = float(autocorr[lag - 1]), float(autocorr[lag]), float(autocorr[lag + 1])
                denom = y0 - 2.0 * y1 + y2
                if abs(denom) > 1e-8:
                    delta = 0.5 * (y0 - y2) / denom
                    delta = float(np.clip(delta, -1.0, 1.0))
                    r_lag = lag + delta
                    r_val = y1 - 0.25 * (y0 - y2) * delta
                else:
                    r_lag = float(lag)
                    r_val = y1

                bpm = 60.0 / (r_lag / frame_rate)
                # Allow a tiny tolerance around target boundaries before clipping
                if (min_bpm - 2.0) <= bpm <= (max_bpm + 2.0):
                    peaks.append({
                        "lag": r_lag,
                        "val": max(0.0, r_val),
                        "bpm": bpm,
                        "int_lag": lag,
                    })

    if not peaks:
        best_i = int(np.argmax(search_region)) + search_min
        r_lag = float(best_i)
        r_val = max(0.0, float(autocorr[best_i]))
        peaks.append({
            "lag": r_lag,
            "val": r_val,
            "bpm": 60.0 / (r_lag / frame_rate),
            "int_lag": best_i,
        })

    # Sort peaks descending by value
    peaks.sort(key=lambda p: p["val"], reverse=True)

    # Disambiguate octave relationships (half-time vs double-time):
    # A candidate at lag L is a subharmonic if an octave peak exists near L/2 with ratio >= 0.40.
    candidates = []
    for p in peaks:
        half_lag = p["lag"] / 2.0
        half_match = None
        for other in peaks:
            if abs(other["lag"] - half_lag) <= max(1.5, 0.06 * p["lag"]):
                half_match = other
                break

        is_subharmonic = False
        octave_ambiguity = 0.0

        if half_match is not None:
            ratio = half_match["val"] / (p["val"] + 1e-8)
            if ratio >= 0.40:
                # Clear onset presence at half the lag: fundamental beat is the faster tempo
                is_subharmonic = True
            elif ratio >= 0.20:
                # Genuine octave ambiguity: record ambiguity factor to adjust confidence
                octave_ambiguity = ratio

        candidates.append({
            "peak": p,
            "is_subharmonic": is_subharmonic,
            "octave_ambiguity": octave_ambiguity,
            "score": p["val"],
        })

    # Filter out subharmonics whose fundamental faster tempo is represented
    viable = [c for c in candidates if not c["is_subharmonic"]]
    if not viable:
        viable = candidates

    viable.sort(key=lambda c: c["score"], reverse=True)
    best = viable[0]
    p_best = best["peak"]
    selected_lag = int(round(p_best["lag"]))
    selected_bpm = float(np.clip(p_best["bpm"], min_bpm, max_bpm))

    # Definitive confidence calculation
    peak_val = p_best["val"]
    z_score = (peak_val - mean_val) / std_val

    # Contrast with surrounding valleys
    valley_window = max(2, int(frame_rate * 0.05))  # 50ms window
    left_bound = max(search_min, selected_lag - valley_window)
    right_bound = min(search_max, selected_lag + valley_window)
    surrounding = autocorr[left_bound : right_bound + 1]
    valley_val = float(np.min(surrounding)) if len(surrounding) > 0 else mean_val
    peak_prominence = (peak_val - valley_val) / std_val

    if z_score < 1.75 or peak_val <= 0 or peak_prominence < 0.5:
        return default_non_rhythmic

    # Compute margin to the runner-up distinct tempo candidate
    margin = 1.0
    for c in viable[1:]:
        if abs(c["peak"]["bpm"] - p_best["bpm"]) > 4.0:
            margin = max(0.0, (best["score"] - c["score"]) / (best["score"] + 1e-6))
            break

    conf_z = np.clip((z_score - 1.5) / 3.0, 0.0, 1.0)
    conf_prom = np.clip(peak_prominence / 3.0, 0.0, 1.0)
    conf_margin = np.clip(margin, 0.3, 1.0)

    raw_conf = float(0.5 * conf_z + 0.3 * conf_prom + 0.2 * conf_margin)

    # Reduce confidence when rhythmic octave is ambiguous
    if best["octave_ambiguity"] > 0:
        raw_conf *= float(1.0 - 0.4 * best["octave_ambiguity"])

    confidence = float(np.clip(raw_conf, 0.0, 1.0))
    is_rhythmic = confidence >= 0.20

    if not is_rhythmic:
        return default_non_rhythmic

    return {
        "bpm": round(float(selected_bpm), 1),
        "confidence": round(confidence, 3),
        "is_rhythmic": True,
    }
