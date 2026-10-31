"""Audio metrics computation: RMS, peak, crest factor, ZCR, and aggregated spectral power features."""
import numpy as np
from typing import Dict, Any, Tuple


def calculate_metrics(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Compute standard acoustic dynamics and aggregated spectral power metrics.

    Spectral metrics are evaluated across the full duration of the signal using
    overlapping Hann windows and power spectral aggregation. For long signals,
    uniform sampling is augmented with energy-aware pooling to ensure brief
    high-frequency transients and localized acoustic events are never dropped.

    Parameters
    ----------
    signal : np.ndarray
        1D audio array.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    dict with keys:
        duration_sec, peak_linear, peak_db, rms_linear, rms_db, crest_factor_db, zcr,
        spectral_centroid, low_energy_ratio, high_energy_ratio
    """
    default_empty = {
        "duration_sec": 0.0,
        "peak_linear": 0.0,
        "peak_db": -120.0,
        "rms_linear": 0.0,
        "rms_db": -120.0,
        "crest_factor_db": 0.0,
        "zcr": 0.0,
        "spectral_centroid": 0.0,
        "low_energy_ratio": 0.0,
        "high_energy_ratio": 0.0,
    }

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got: {sample_rate}")

    if not isinstance(signal, np.ndarray) or len(signal) == 0:
        return default_empty

    if not np.all(np.isfinite(signal)):
        return default_empty

    duration = float(len(signal)) / float(sample_rate)
    peak = float(np.max(np.abs(signal)))
    peak_db = 20.0 * np.log10(max(peak, 1e-6))

    rms = float(np.sqrt(np.mean(signal ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))

    crest_factor = peak_db - rms_db if rms > 1e-6 else 0.0

    # Zero Crossing Rate (ZCR)
    if len(signal) > 1:
        signs = np.signbit(signal)
        zero_crossings = int(np.count_nonzero(signs[:-1] != signs[1:]))
        zcr = float(zero_crossings) / float(len(signal) - 1)
    else:
        zcr = 0.0

    # Aggregated Spectral Features over entire signal with overlapping windows
    win_size = min(2048, len(signal))
    if win_size >= 64:
        hop_size = win_size // 2  # 50% overlap
        total_frames = max(1, (len(signal) - win_size) // hop_size + 1)

        # Frame selection: full coverage for <= 128 frames; uniform + energy-pooled for long files
        if total_frames <= 128:
            selected_indices = [i * hop_size for i in range(total_frames)]
        else:
            uniform_indices = np.linspace(0, len(signal) - win_size, num=128, dtype=int).tolist()

            # Fast vectorized frame energy via cumulative sum of squared samples
            cum_sq = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64) ** 2)))
            all_st = np.arange(total_frames) * hop_size
            all_en = all_st + win_size
            frame_energies = cum_sq[all_en] - cum_sq[all_st]

            # Top 64 energetic frames guarantee capturing brief transient bursts anywhere in the file
            top_energies_idx = np.argsort(frame_energies)[-64:]
            top_st = (top_energies_idx * hop_size).tolist()

            selected_indices = sorted(set(uniform_indices + top_st))

        window = np.hanning(win_size)
        freqs = np.fft.rfftfreq(win_size, 1.0 / sample_rate)

        # Compute power spectrum for each selected frame: P(f) = |X(f)|^2
        accumulated_power = np.zeros(len(freqs), dtype=np.float64)

        for start_idx in selected_indices:
            frame = signal[start_idx : start_idx + win_size]
            if len(frame) < win_size:
                frame = np.pad(frame, (0, win_size - len(frame)), mode="constant")
            fft_mag = np.abs(np.fft.rfft(frame * window))
            frame_power = fft_mag ** 2
            accumulated_power += frame_power

        total_power = float(np.sum(accumulated_power))

        if total_power > 1e-12:
            spectral_centroid = float(np.sum(freqs * accumulated_power) / total_power)
            low_mask = freqs <= 250.0
            high_mask = freqs >= 4000.0
            low_energy_ratio = float(np.sum(accumulated_power[low_mask]) / total_power)
            high_energy_ratio = float(np.sum(accumulated_power[high_mask]) / total_power)
        else:
            spectral_centroid = 0.0
            low_energy_ratio = 0.0
            high_energy_ratio = 0.0
    else:
        spectral_centroid = 0.0
        low_energy_ratio = 0.0
        high_energy_ratio = 0.0

    return {
        "duration_sec": round(duration, 3),
        "peak_linear": round(peak, 4),
        "peak_db": round(peak_db, 2),
        "rms_linear": round(rms, 4),
        "rms_db": round(rms_db, 2),
        "crest_factor_db": round(crest_factor, 2),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(spectral_centroid, 1),
        "low_energy_ratio": round(low_energy_ratio, 3),
        "high_energy_ratio": round(high_energy_ratio, 3),
    }


def detect_silence(signal: np.ndarray, threshold_db: float = -60.0) -> Tuple[int, int]:
    """
    Return the start and end sample indices of active audio above threshold.

    Parameters
    ----------
    signal : np.ndarray
        Audio signal.
    threshold_db : float, default -60.0
        Threshold in dB below full scale.

    Returns
    -------
    (start_index, end_index) : Tuple[int, int]
    """
    if not isinstance(signal, np.ndarray) or len(signal) == 0:
        return 0, 0
    if not np.all(np.isfinite(signal)):
        return 0, 0

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    active = np.where(np.abs(signal) >= threshold_linear)[0]
    if len(active) == 0:
        return 0, 0
    return int(active[0]), int(active[-1])
