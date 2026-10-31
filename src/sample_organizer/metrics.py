"""Audio metrics computation: RMS, peak, crest factor, and silence trim."""
import numpy as np
from typing import Dict, Any, Tuple


def calculate_metrics(signal: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Compute standard acoustic metrics from an audio buffer."""
    if len(signal) == 0:
        return {
            "duration_sec": 0.0,
            "peak_linear": 0.0,
            "peak_db": -120.0,
            "rms_linear": 0.0,
            "rms_db": -120.0,
            "crest_factor_db": 0.0,
            "zcr": 0.0,
        }

    duration = float(len(signal)) / float(sample_rate)
    peak = float(np.max(np.abs(signal)))
    peak_db = 20.0 * np.log10(max(peak, 1e-6))

    rms = float(np.sqrt(np.mean(signal ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))

    crest_factor = peak_db - rms_db if rms > 1e-6 else 0.0

    # Zero Crossing Rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    zcr = float(zero_crossings) / float(len(signal))

    return {
        "duration_sec": round(duration, 3),
        "peak_linear": round(peak, 4),
        "peak_db": round(peak_db, 2),
        "rms_linear": round(rms, 4),
        "rms_db": round(rms_db, 2),
        "crest_factor_db": round(crest_factor, 2),
        "zcr": round(zcr, 4),
    }


def detect_silence(signal: np.ndarray, threshold_db: float = -60.0) -> Tuple[int, int]:
    """Return the start and end sample indices of active audio above threshold."""
    if len(signal) == 0:
        return 0, 0
    threshold_linear = 10.0 ** (threshold_db / 20.0)
    active = np.where(np.abs(signal) >= threshold_linear)[0]
    if len(active) == 0:
        return 0, 0
    return int(active[0]), int(active[-1])
