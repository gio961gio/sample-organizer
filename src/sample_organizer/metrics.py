"""Audio metrics computation: RMS, peak, crest factor, ZCR, and spectral features."""
import numpy as np
from typing import Dict, Any, Tuple


def calculate_metrics(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Compute standard acoustic dynamics and spectral metrics from an audio buffer.

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

    # Spectral features via rFFT
    n_fft = min(2048, len(signal))
    if n_fft >= 64:
        fft_mag = np.abs(np.fft.rfft(signal[:n_fft] * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        sum_mag = np.sum(fft_mag)

        if sum_mag > 1e-8:
            spectral_centroid = float(np.sum(freqs * fft_mag) / sum_mag)
            low_mask = freqs <= 250.0
            high_mask = freqs >= 4000.0
            low_energy_ratio = float(np.sum(fft_mag[low_mask]) / sum_mag)
            high_energy_ratio = float(np.sum(fft_mag[high_mask]) / sum_mag)
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
