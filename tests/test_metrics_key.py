"""Dedicated tests for Metrics and Key Detection (Phase 2, Phase 6 & Phase 7 Spectral Metrics)."""
import time
import numpy as np
import pytest
from sample_organizer.metrics import calculate_metrics, detect_silence
from sample_organizer.key_detector import compute_chroma, estimate_key


def test_calculate_metrics_valid():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)

    m = calculate_metrics(signal, sr)
    assert m["duration_sec"] == 1.0
    assert m["peak_linear"] == pytest.approx(0.5, abs=0.01)
    assert m["peak_db"] == pytest.approx(-6.02, abs=0.1)
    assert m["rms_db"] == pytest.approx(-9.03, abs=0.1)
    assert m["crest_factor_db"] == pytest.approx(3.01, abs=0.2)
    assert m["zcr"] > 0.003
    assert m["spectral_centroid"] > 50.0
    assert m["low_energy_ratio"] > 0.8


def test_calculate_metrics_edge_cases():
    sr = 44100
    empty_m = calculate_metrics(np.array([], dtype=np.float32), sr)
    assert empty_m["duration_sec"] == 0.0
    assert empty_m["peak_db"] == -120.0

    nan_m = calculate_metrics(np.array([0.1, np.nan, np.inf]), sr)
    assert nan_m["duration_sec"] == 0.0

    with pytest.raises(ValueError):
        calculate_metrics(np.array([0.1]), -44100)


def test_detect_silence():
    signal = np.zeros(1000, dtype=np.float32)
    signal[100:600] = 0.5
    start, end = detect_silence(signal, threshold_db=-20.0)
    assert start == 100
    assert end == 599

    start_sil, end_sil = detect_silence(np.zeros(500), -40.0)
    assert start_sil == 0 and end_sil == 0
    assert detect_silence(np.array([])) == (0, 0)


def test_spectral_metrics_across_full_signal():
    """Verify spectral aggregation covers the full audio file beyond the first 2048 samples."""
    sr = 44100
    total_samples = sr * 2  # 2 seconds
    signal = np.zeros(total_samples, dtype=np.float32)

    # First second: 100 Hz low-frequency bass
    t_low = np.linspace(0, 1.0, sr, endpoint=False)
    signal[:sr] = 0.5 * np.sin(2 * np.pi * 100.0 * t_low)

    # Second second: 6000 Hz high-frequency sheen (well beyond sample 2048!)
    t_high = np.linspace(0, 1.0, sr, endpoint=False)
    signal[sr:] = 0.5 * np.sin(2 * np.pi * 6000.0 * t_high)

    m = calculate_metrics(signal, sr)
    # Both low and high power must be represented
    assert m["low_energy_ratio"] > 0.15
    assert m["high_energy_ratio"] > 0.15
    # Centroid should be balanced between the low and high sections
    assert 1000.0 < m["spectral_centroid"] < 5000.0


def test_spectral_power_efficiency_long_file():
    """Verify performance and numerical stability on long audio files."""
    sr = 44100
    # 30 seconds of audio
    signal = np.random.uniform(-0.1, 0.1, sr * 30).astype(np.float32)

    start_t = time.perf_counter()
    m = calculate_metrics(signal, sr)
    elapsed = time.perf_counter() - start_t

    # Must complete in well under 0.25 seconds
    assert elapsed < 0.25
    assert m["duration_sec"] == 30.0
    assert np.isfinite(m["spectral_centroid"])
    assert 0.0 <= m["low_energy_ratio"] <= 1.0
    assert 0.0 <= m["high_energy_ratio"] <= 1.0


@pytest.mark.parametrize("position_sec", [0.5, 12.345, 27.89])
def test_high_frequency_transients_in_different_positions(position_sec):
    """
    Verify that a brief high-frequency transient in a 30s file is reliably captured
    regardless of its temporal position (beginning, middle, or end).
    """
    sr = 44100
    duration = 30.0
    signal = np.zeros(int(sr * duration), dtype=np.float32)

    # Low-level background rumble (100 Hz)
    t = np.linspace(0, duration, len(signal), endpoint=False)
    signal += (0.01 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)

    # Insert a 15ms 6500 Hz transient burst at position_sec
    burst_len = int(sr * 0.015)
    t_burst = np.linspace(0, 0.015, burst_len, endpoint=False)
    burst = (0.8 * np.sin(2 * np.pi * 6500.0 * t_burst)).astype(np.float32)
    start_idx = int(position_sec * sr)
    signal[start_idx : start_idx + burst_len] += burst

    m = calculate_metrics(signal, sr)
    # Energy pooling ensures the burst is captured
    assert m["high_energy_ratio"] > 0.02
    assert m["spectral_centroid"] > 200.0


def test_long_silence_with_brief_events():
    """Verify that long silence does not prevent capturing short high-energy events."""
    sr = 44100
    duration = 20.0
    signal = np.zeros(int(sr * duration), dtype=np.float32)

    # Place a single 20ms burst at second 14.5 surrounded by 19.98s of complete silence
    burst_len = int(sr * 0.02)
    t_burst = np.linspace(0, 0.02, burst_len, endpoint=False)
    burst = (0.9 * np.sin(2 * np.pi * 5000.0 * t_burst)).astype(np.float32)
    start_idx = int(14.5 * sr)
    signal[start_idx : start_idx + burst_len] = burst

    m = calculate_metrics(signal, sr)
    assert m["duration_sec"] == 20.0
    assert m["high_energy_ratio"] > 0.60
    assert m["spectral_centroid"] > 3500.0


def test_distributed_brief_events():
    """Verify aggregated spectral metrics capture multiple brief events distributed across time."""
    sr = 44100
    duration = 25.0
    signal = np.zeros(int(sr * duration), dtype=np.float32)

    event_times = [2.0, 7.5, 13.0, 19.5, 23.0]
    burst_len = int(sr * 0.02)
    t_burst = np.linspace(0, 0.02, burst_len, endpoint=False)

    for i, t_pos in enumerate(event_times):
        freq = 6000.0 if i % 2 == 0 else 150.0
        burst = (0.7 * np.sin(2 * np.pi * freq * t_burst)).astype(np.float32)
        idx = int(t_pos * sr)
        signal[idx : idx + burst_len] = burst

    m = calculate_metrics(signal, sr)
    assert m["duration_sec"] == 25.0
    assert m["low_energy_ratio"] > 0.15
    assert m["high_energy_ratio"] > 0.15
    assert 500.0 < m["spectral_centroid"] < 5000.0


def test_key_detection_triads():
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)

    c_maj = (
        0.3 * np.sin(2 * np.pi * 261.63 * t) +
        0.3 * np.sin(2 * np.pi * 329.63 * t) +
        0.3 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)

    res_c = estimate_key(c_maj, sr)
    assert res_c["key"] == "C Major"
    assert res_c["root"] == "C"
    assert res_c["scale"] == "Major"
    assert res_c["confidence"] > 0.5

    a_min = (
        0.3 * np.sin(2 * np.pi * 440.00 * t) +
        0.3 * np.sin(2 * np.pi * 523.25 * t) +
        0.3 * np.sin(2 * np.pi * 659.25 * t)
    ).astype(np.float32)

    res_a = estimate_key(a_min, sr)
    assert res_a["key"] == "A Minor"
    assert res_a["root"] == "A"
    assert res_a["scale"] == "Minor"
    assert res_a["confidence"] > 0.5


def test_key_detection_non_tonal_and_silence():
    sr = 44100
    res_sil = estimate_key(np.zeros(2048, dtype=np.float32), sr)
    assert res_sil["key"] == "Unknown"
    assert res_sil["confidence"] == 0.0

    res_short = estimate_key(np.array([0.1, -0.1] * 50, dtype=np.float32), sr)
    assert res_short["key"] == "Unknown"

    np.random.seed(42)
    noise = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)
    res_noise = estimate_key(noise, sr)
    assert res_noise["key"] == "Unknown" or res_noise["confidence"] < 0.2
