"""Unit tests for Sample Organizer audio DSP and organization engine."""
import os
import shutil
import tempfile
import numpy as np
import pytest
from sample_organizer.audio_io import save_audio, load_audio
from sample_organizer.metrics import calculate_metrics, detect_silence
from sample_organizer.key_detector import estimate_key
from sample_organizer.bpm_detector import estimate_bpm
from sample_organizer.organizer import analyze_file, organize_samples, classify_sample


@pytest.fixture
def temp_audio_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_audio_save_load(temp_audio_dir):
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    wav_path = os.path.join(temp_audio_dir, "test_440.wav")
    save_audio(wav_path, signal, sr)

    loaded_signal, loaded_sr = load_audio(wav_path)
    assert loaded_sr == sr
    assert len(loaded_signal) == sr
    assert np.allclose(loaded_signal, signal, atol=1e-3)


def test_calculate_metrics():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.8 * np.sin(2 * np.pi * 100.0 * t).astype(np.float32)

    metrics = calculate_metrics(signal, sr)
    assert metrics["duration_sec"] == 1.0
    assert metrics["peak_linear"] == pytest.approx(0.8, abs=0.01)
    assert metrics["crest_factor_db"] > 2.0


def test_detect_silence():
    signal = np.zeros(1000, dtype=np.float32)
    signal[200:800] = 0.5
    start, end = detect_silence(signal, threshold_db=-20.0)
    assert start == 200
    assert end == 799


def test_key_detector():
    sr = 44100
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # Generate A4 (440 Hz) + C#5 (554.37 Hz) + E5 (659.25 Hz) -> A Major triad
    signal = (
        0.3 * np.sin(2 * np.pi * 440.0 * t) +
        0.3 * np.sin(2 * np.pi * 554.37 * t) +
        0.3 * np.sin(2 * np.pi * 659.25 * t)
    ).astype(np.float32)

    res = estimate_key(signal, sr)
    assert res["root"] == "A"
    assert res["scale"] == "Major"
    assert res["confidence"] > 0.5


def test_bpm_detector():
    sr = 44100
    duration = 3.0
    signal = np.zeros(int(sr * duration), dtype=np.float32)
    # Create 120 BPM pulses (every 0.5 seconds)
    pulse_interval = int(sr * 0.5)
    for i in range(0, len(signal) - 1000, pulse_interval):
        signal[i : i + 500] = 0.9

    res = estimate_bpm(signal, sr)
    assert abs(res["bpm"] - 120.0) <= 2.0
    assert res["is_rhythmic"] is True


def test_organize_samples(temp_audio_dir):
    sr = 44100
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    kick_signal = (0.9 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)

    sample_path = os.path.join(temp_audio_dir, "kick_01.wav")
    save_audio(sample_path, kick_signal, sr)

    analysis = analyze_file(sample_path)
    out_dir = os.path.join(temp_audio_dir, "organized")
    res = organize_samples([analysis], out_dir, mode="copy")

    assert res["processed"] == 1
    assert os.path.exists(res["operations"][0]["destination"])
