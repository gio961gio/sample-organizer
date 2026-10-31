"""Dedicated tests for BPM detection (Phase 3)."""
import numpy as np
import pytest
from sample_organizer.bpm_detector import estimate_bpm


def make_rhythmic_pulses(sr: int, duration_sec: float, bpm: float, noise_level: float = 0.0) -> np.ndarray:
    """Generate periodic audio pulses at specified BPM."""
    total_samples = int(sr * duration_sec)
    signal = np.zeros(total_samples, dtype=np.float32)

    beat_interval_samples = int(sr * (60.0 / bpm))
    pulse_len = min(int(sr * 0.03), beat_interval_samples // 4)

    for start in range(0, total_samples - pulse_len, beat_interval_samples):
        t_pulse = np.linspace(0, 1.0, pulse_len)
        signal[start : start + pulse_len] = 0.8 * np.sin(np.pi * t_pulse)

    if noise_level > 0:
        np.random.seed(42)
        signal += (np.random.uniform(-noise_level, noise_level, total_samples)).astype(np.float32)

    return signal


@pytest.mark.parametrize("target_bpm", [70.0, 90.0, 120.0, 140.0])
def test_synthetic_bpm_pulses(target_bpm):
    """Test BPM estimation on synthetic pulse trains across tempos."""
    sr = 44100
    duration = 4.0
    signal = make_rhythmic_pulses(sr, duration, target_bpm)

    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    assert abs(res["bpm"] - target_bpm) <= 3.0
    assert res["confidence"] > 0.3


def test_rhythm_with_noise():
    """Test rhythm detection with added noise."""
    sr = 44100
    signal = make_rhythmic_pulses(sr, 4.0, 120.0, noise_level=0.1)
    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    assert abs(res["bpm"] - 120.0) <= 3.5


def test_non_rhythmic_signals():
    """Test silence, constant tones, and short signals return bpm=None."""
    sr = 44100

    # 1. Pure silence
    res_sil = estimate_bpm(np.zeros(sr * 2, dtype=np.float32), sr)
    assert res_sil["bpm"] is None
    assert res_sil["is_rhythmic"] is False
    assert res_sil["confidence"] == 0.0

    # 2. Constant sine tone (no onsets)
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    constant_tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    res_tone = estimate_bpm(constant_tone, sr)
    assert res_tone["bpm"] is None
    assert res_tone["is_rhythmic"] is False

    # 3. Short signal (< 0.6s)
    short_signal = np.array([0.5, -0.5] * 1000, dtype=np.float32)
    res_short = estimate_bpm(short_signal, sr)
    assert res_short["bpm"] is None
    assert res_short["is_rhythmic"] is False


def test_bpm_parameter_validation():
    """Test validation of sample rate and min/max BPM parameters."""
    sr = 44100
    sig = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        estimate_bpm(sig, 0)

    with pytest.raises(ValueError, match="Invalid BPM range"):
        estimate_bpm(sig, sr, min_bpm=-10, max_bpm=120)

    with pytest.raises(ValueError, match="Invalid BPM range"):
        estimate_bpm(sig, sr, min_bpm=140, max_bpm=100)
