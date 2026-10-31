"""Dedicated tests for BPM detection (Phase 3 & Phase 6 Refinements)."""
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


@pytest.mark.parametrize("target_bpm", [65.0, 75.0, 100.0, 125.0, 150.0, 175.0])
def test_synthetic_bpm_unbiased(target_bpm):
    """Test BPM estimation across wide tempo range without arbitrary 115 BPM bias."""
    sr = 44100
    duration = 4.0
    signal = make_rhythmic_pulses(sr, duration, target_bpm)

    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    # Allow small tolerance
    assert abs(res["bpm"] - target_bpm) <= 3.5
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["confidence"] > 0.25


def test_syncopated_rhythm_pattern():
    """Test tempo extraction on syncopated tresillo rhythm (3-3-2 sixteenths)."""
    sr = 44100
    bpm = 120.0
    beat_sec = 60.0 / bpm  # 0.5 sec
    sixteenth_sec = beat_sec / 4.0  # 0.125 sec
    duration = 4.0  # 8 bars of tresillo

    signal = np.zeros(int(sr * duration), dtype=np.float32)
    # Pattern: hit at 0, 3, 6 sixteenth notes in every 8-sixteenth (half-bar) unit
    hit_indices = []
    unit_sixteenths = 8
    num_units = int(duration / (unit_sixteenths * sixteenth_sec))

    for u in range(num_units):
        base_t = u * unit_sixteenths * sixteenth_sec
        for s in [0, 3, 6]:
            hit_indices.append(int((base_t + s * sixteenth_sec) * sr))

    pulse_len = int(sr * 0.02)
    t_pulse = np.linspace(0, 1.0, pulse_len)
    for idx in hit_indices:
        if idx + pulse_len < len(signal):
            signal[idx : idx + pulse_len] = 0.8 * np.sin(np.pi * t_pulse)

    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    # For a syncopated 120 BPM tresillo, the estimated interval aligns with the beat (120) or dominant pattern groupings (80 / 160)
    assert (
        abs(res["bpm"] - 120.0) <= 5.0
        or abs(res["bpm"] - 80.0) <= 5.0
        or abs(res["bpm"] - 160.0) <= 5.0
    )


def test_alternating_beat_dynamics():
    """Test alternating strong/weak beats (evaluating half-time vs beat tempo)."""
    sr = 44100
    bpm = 130.0
    duration = 4.0
    signal = np.zeros(int(sr * duration), dtype=np.float32)

    beat_interval = int(sr * (60.0 / bpm))
    pulse_len = int(sr * 0.02)
    t_pulse = np.linspace(0, 1.0, pulse_len)

    beat_idx = 0
    for start in range(0, len(signal) - pulse_len, beat_interval):
        # Alternating amplitude: beat 1 = 0.9, beat 2 = 0.5 (Kick/Snare pattern)
        amp = 0.9 if (beat_idx % 2 == 0) else 0.5
        signal[start : start + pulse_len] = amp * np.sin(np.pi * t_pulse)
        beat_idx += 1

    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    assert abs(res["bpm"] - 130.0) <= 4.0
    assert 0.0 <= res["confidence"] <= 1.0


def test_rhythm_with_noise():
    sr = 44100
    signal = make_rhythmic_pulses(sr, 4.0, 120.0, noise_level=0.1)
    res = estimate_bpm(signal, sr)
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    assert abs(res["bpm"] - 120.0) <= 3.5


def test_non_rhythmic_signals():
    sr = 44100
    res_sil = estimate_bpm(np.zeros(sr * 2, dtype=np.float32), sr)
    assert res_sil["bpm"] is None
    assert res_sil["is_rhythmic"] is False
    assert res_sil["confidence"] == 0.0

    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    constant_tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    res_tone = estimate_bpm(constant_tone, sr)
    assert res_tone["bpm"] is None
    assert res_tone["is_rhythmic"] is False

    short_signal = np.array([0.5, -0.5] * 1000, dtype=np.float32)
    res_short = estimate_bpm(short_signal, sr)
    assert res_short["bpm"] is None
    assert res_short["is_rhythmic"] is False


def test_bpm_parameter_validation():
    sr = 44100
    sig = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        estimate_bpm(sig, 0)

    with pytest.raises(ValueError, match="Invalid BPM range"):
        estimate_bpm(sig, sr, min_bpm=-10, max_bpm=120)

    with pytest.raises(ValueError, match="Invalid BPM range"):
        estimate_bpm(sig, sr, min_bpm=140, max_bpm=100)
