"""Dedicated tests for Metrics and Key Detection (Phase 2)."""
import numpy as np
import pytest
from sample_organizer.metrics import calculate_metrics, detect_silence
from sample_organizer.key_detector import compute_chroma, estimate_key


def test_calculate_metrics_valid():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sine wave with amplitude 0.5
    signal = (0.5 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)

    m = calculate_metrics(signal, sr)
    assert m["duration_sec"] == 1.0
    assert m["peak_linear"] == pytest.approx(0.5, abs=0.01)
    assert m["peak_db"] == pytest.approx(-6.02, abs=0.1)
    assert m["rms_db"] == pytest.approx(-9.03, abs=0.1)
    assert m["crest_factor_db"] == pytest.approx(3.01, abs=0.2)
    assert m["zcr"] > 0.003
    assert m["spectral_centroid"] > 50.0
    assert m["low_energy_ratio"] > 0.8  # 100 Hz tone is below 250 Hz


def test_calculate_metrics_edge_cases():
    sr = 44100
    # Empty array
    empty_m = calculate_metrics(np.array([], dtype=np.float32), sr)
    assert empty_m["duration_sec"] == 0.0
    assert empty_m["peak_db"] == -120.0

    # Non-finite values
    nan_m = calculate_metrics(np.array([0.1, np.nan, np.inf]), sr)
    assert nan_m["duration_sec"] == 0.0

    # Invalid sample rate
    with pytest.raises(ValueError):
        calculate_metrics(np.array([0.1]), -44100)


def test_detect_silence():
    signal = np.zeros(1000, dtype=np.float32)
    signal[100:600] = 0.5
    start, end = detect_silence(signal, threshold_db=-20.0)
    assert start == 100
    assert end == 599

    # All silence
    start_sil, end_sil = detect_silence(np.zeros(500), -40.0)
    assert start_sil == 0 and end_sil == 0

    # Empty
    assert detect_silence(np.array([])) == (0, 0)


def test_key_detection_triads():
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)

    # 1. C Major Triad (C4=261.63, E4=329.63, G4=392.00)
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

    # 2. A Minor Triad (A4=440.00, C5=523.25, E5=659.25)
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
    # Pure silence
    res_sil = estimate_key(np.zeros(2048, dtype=np.float32), sr)
    assert res_sil["key"] == "Unknown"
    assert res_sil["confidence"] == 0.0

    # Short signal
    res_short = estimate_key(np.array([0.1, -0.1] * 50, dtype=np.float32), sr)
    assert res_short["key"] == "Unknown"

    # White noise (non-tonal, ambiguous)
    np.random.seed(42)
    noise = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)
    res_noise = estimate_key(noise, sr)
    assert res_noise["key"] == "Unknown" or res_noise["confidence"] < 0.2
