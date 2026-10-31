"""Dedicated tests for Organizer engine (Phase 4, Phase 6 & Phase 7 File Safety & Classification)."""
import os
import tempfile
import hashlib
import numpy as np
import pytest
from sample_organizer.audio_io import save_audio
from sample_organizer.organizer import (
    scan_directory,
    organize_samples,
    classify_sample,
    analyze_file,
    compute_sha256,
    copy_file_exclusive,
)


@pytest.fixture
def sample_env():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "source")
        dst = os.path.join(tmp, "destination")
        os.makedirs(src, exist_ok=True)
        yield src, dst


def test_dry_run_non_destructive(sample_env):
    """Verify that dry-run does not create folders or write any files."""
    src, dst = sample_env
    sr = 44100
    save_audio(os.path.join(src, "kick.wav"), np.zeros(2000, dtype=np.float32), sr)

    results = scan_directory(src)
    assert len(results) == 1

    summary = organize_samples(results, dst, mode="dry-run")
    assert summary["planned"] == 1
    assert summary["processed"] == 0
    assert summary["errors"] == 0
    assert len(summary["operations"]) == 1
    assert summary["operations"][0]["status"] == "planned"
    assert not os.path.exists(dst)


def test_copy_source_integrity_and_sha256(sample_env):
    """Verify copy operation leaves source intact and validates destination SHA-256."""
    src, dst = sample_env
    sr = 44100
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    original_audio = (0.75 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

    src_file = os.path.join(src, "pristine_source.wav")
    save_audio(src_file, original_audio, sr)

    src_hash = compute_sha256(src_file)
    src_size = os.path.getsize(src_file)

    results = scan_directory(src)
    summary = organize_samples(results, dst, mode="copy")

    assert summary["processed"] == 1
    assert summary["errors"] == 0

    dest_file = summary["operations"][0]["destination"]
    assert os.path.exists(dest_file)

    # SHA-256 and size verification
    assert os.path.exists(src_file)
    assert compute_sha256(src_file) == src_hash
    assert os.path.getsize(src_file) == src_size
    assert compute_sha256(dest_file) == src_hash
    assert os.path.getsize(dest_file) == src_size


def test_exclusive_creation_and_collision_handling(sample_env):
    """Verify exclusive file creation avoids overwriting and resolves collisions atomically."""
    src, dst = sample_env
    sr = 44100

    # 1. Pre-create a file in destination
    kicks_dest_dir = os.path.join(dst, "Kicks")
    os.makedirs(kicks_dest_dir, exist_ok=True)
    existing_dest_file = os.path.join(kicks_dest_dir, "kick.wav")
    save_audio(existing_dest_file, np.zeros(1000, dtype=np.float32), sr)
    existing_hash = compute_sha256(existing_dest_file)

    # 2. Put a new kick in source with the same name but different content
    src_kick = os.path.join(src, "kick.wav")
    save_audio(src_kick, np.ones(3000, dtype=np.float32) * 0.5, sr)
    src_hash = compute_sha256(src_kick)

    results = scan_directory(src)
    results[0]["category"] = "Kicks"

    summary = organize_samples(results, dst, mode="copy")
    assert summary["processed"] == 1
    assert summary["errors"] == 0

    new_dest_file = summary["operations"][0]["destination"]
    assert new_dest_file != existing_dest_file
    assert os.path.basename(new_dest_file) == "kick_001.wav"

    # Pre-existing file must remain completely unaltered
    assert compute_sha256(existing_dest_file) == existing_hash
    # New file must match source hash
    assert compute_sha256(new_dest_file) == src_hash


def test_partial_file_cleanup_on_error(sample_env, monkeypatch):
    """Verify that partial files are cleanly removed if an error occurs mid-copy."""
    src, dst = sample_env
    sr = 44100
    src_file = os.path.join(src, "error_test.wav")
    save_audio(src_file, np.ones(5000, dtype=np.float32) * 0.5, sr)

    target_candidate = os.path.join(dst, "error_test.wav")

    # Simulate an I/O write error during chunk streaming
    original_write = os.write

    def failing_write(fd, data):
        # Allow small initial write then fail
        raise IOError("Disk full or network interruption")

    # Directly test copy_file_exclusive failure cleanup
    with pytest.raises(Exception):
        with monkeypatch.context() as m:
            m.setattr("os.fsync", lambda fd: failing_write(fd, b""))
            copy_file_exclusive(src_file, target_candidate)

    # Destination candidate must not exist as a partial / orphaned file
    assert not os.path.exists(target_candidate)
    # Source file must remain intact
    assert os.path.exists(src_file)


def test_move_operation_verifies_dest_and_removes_source(sample_env):
    """Verify move operation confirms destination hash and deletes source file."""
    src, dst = sample_env
    sr = 44100
    src_file = os.path.join(src, "sample_to_move.wav")
    save_audio(src_file, np.array([0.2, -0.2] * 200, dtype=np.float32), sr)
    orig_hash = compute_sha256(src_file)

    results = scan_directory(src)
    summary = organize_samples(results, dst, mode="move")

    assert summary["processed"] == 1
    dest_path = summary["operations"][0]["destination"]
    assert not os.path.exists(src_file)
    assert os.path.exists(dest_path)
    assert compute_sha256(dest_path) == orig_hash


def test_reject_same_source_and_destination_directory(sample_env):
    """Verify that organizing with source and destination as the same directory is rejected."""
    src, _ = sample_env
    sr = 44100
    save_audio(os.path.join(src, "test.wav"), np.zeros(1000, dtype=np.float32), sr)
    results = scan_directory(src)

    # 1. Rejection when source_dir parameter is explicitly matching
    with pytest.raises(ValueError, match="cannot be identical"):
        organize_samples(results, src, mode="copy", source_dir=src)

    # 2. Per-sample directory safety rejection when trying to organize into itself
    summary = organize_samples(results, src, mode="copy")
    assert summary["errors"] == 1
    assert summary["operations"][0]["status"] == "error"
    assert "matches destination" in summary["operations"][0]["error_msg"]


def test_invalid_mode(sample_env):
    """Verify invalid mode raises ValueError."""
    src, dst = sample_env
    with pytest.raises(ValueError, match="Invalid mode"):
        organize_samples([], dst, mode="invalid_mode")


def test_scan_directory_exclusion(sample_env):
    """Verify that nested excluded directories are skipped."""
    src, _ = sample_env
    sr = 44100
    save_audio(os.path.join(src, "top.wav"), np.zeros(1000, dtype=np.float32), sr)

    nested_out = os.path.join(src, "nested_out")
    os.makedirs(nested_out, exist_ok=True)
    save_audio(os.path.join(nested_out, "nested.wav"), np.zeros(1000, dtype=np.float32), sr)

    results = scan_directory(src, exclude_dirs=[nested_out])
    filenames = [r["filename"] for r in results]
    assert "top.wav" in filenames
    assert "nested.wav" not in filenames


def test_error_distinction_and_preservation(sample_env):
    """Verify distinction of read errors and preservation of file paths."""
    src, dst = sample_env

    # 1. Valid sample
    sr = 44100
    save_audio(os.path.join(src, "valid.wav"), np.array([0.5, -0.5] * 500, dtype=np.float32), sr)

    # 2. Corrupt sample
    corrupt_file = os.path.join(src, "corrupted.wav")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_WAV_HEADER_CORRUPTED_BYTES")

    results = scan_directory(src)
    assert len(results) == 2

    errors = [r for r in results if "error" in r]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "read"
    assert errors[0]["file_path"] == corrupt_file

    summary = organize_samples(results, dst, mode="copy")
    assert summary["processed"] == 1
    assert summary["errors"] == 1

    failed_ops = [op for op in summary["operations"] if op["status"] == "error"]
    assert len(failed_ops) == 1
    assert failed_ops[0]["error_type"] == "read"
    assert failed_ops[0]["source"] == corrupt_file


# =========================================================================
# Phase 7: Real Synthetic Signal End-to-End Classification Regression Tests
# (Audio I/O -> Metrics -> Key -> BPM -> Classify)
# =========================================================================

def test_e2e_kick_with_transient(tmp_path):
    """End-to-end regression: Kick with fast pitch decay transient."""
    sr = 44100
    dur = 0.35
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freq = 160.0 * np.exp(-t / 0.04) + 45.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    signal = (np.sin(phase) * np.exp(-t / 0.07)).astype(np.float32)
    # Add click transient
    signal[:int(sr * 0.005)] += np.linspace(0.8, 0.0, int(sr * 0.005))

    path = str(tmp_path / "kick.wav")
    save_audio(path, signal, sr)

    res = analyze_file(path)
    assert res["category"] == "Kicks"
    assert res["metrics"]["crest_factor_db"] > 10.5
    assert res["metrics"]["spectral_centroid"] < 800.0


def test_e2e_hihat(tmp_path):
    """End-to-end regression: Hi-hat with high frequency content and rapid decay."""
    sr = 44100
    dur = 0.08
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    np.random.seed(42)
    noise = np.random.uniform(-1.0, 1.0, len(t)).astype(np.float32) * np.exp(-t / 0.015)
    high_sine = 0.5 * np.sin(2 * np.pi * 8000.0 * t).astype(np.float32) * np.exp(-t / 0.02)
    signal = noise + high_sine

    path = str(tmp_path / "hihat.wav")
    save_audio(path, signal, sr)

    res = analyze_file(path)
    assert res["category"] == "Hats_Cymbals"
    assert res["metrics"]["spectral_centroid"] > 5000.0 or res["metrics"]["high_energy_ratio"] > 0.35


def test_e2e_drum_loop_with_tonal_resonances(tmp_path):
    """End-to-end regression: Drum loop at 120 BPM with a resonant pitched 110 Hz 808 bass."""
    sr = 44100
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    signal = np.zeros(len(t), dtype=np.float32)

    # Resonant pitched tone at 110 Hz (A2)
    signal += 0.25 * np.sin(2 * np.pi * 110.0 * t).astype(np.float32)

    # Periodic percussive beats at 120 BPM
    pulse_len = int(sr * 0.05)
    tp = np.linspace(0, 1.0, pulse_len)
    for st in range(0, len(signal) - pulse_len, int(sr * 0.5)):
        signal[st : st + pulse_len] += 0.8 * np.sin(np.pi * tp)

    path = str(tmp_path / "drum_loop_tonal.wav")
    save_audio(path, signal, sr)

    res = analyze_file(path)
    # Must be categorized as Drum_Loops despite the tonal resonance!
    assert res["category"] == "Drum_Loops"
    assert res["is_rhythmic"] is True
    assert res["bpm"] is not None
    assert res["metrics"]["crest_factor_db"] >= 9.5


def test_e2e_melodic_loop_with_percussion(tmp_path):
    """End-to-end regression: Rich melodic chord progression with light drum accompaniment."""
    sr = 44100
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # C Major chord (C4=261.63, E4=329.63, G4=392.00)
    c_chord = (
        0.35 * np.sin(2 * np.pi * 261.63 * t) +
        0.35 * np.sin(2 * np.pi * 329.63 * t) +
        0.35 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)

    # Light percussion taps
    pulse_len = int(sr * 0.03)
    tp = np.linspace(0, 1.0, pulse_len)
    for st in range(0, len(c_chord) - pulse_len, int(sr * 0.5)):
        c_chord[st : st + pulse_len] += 0.20 * np.sin(np.pi * tp)

    path = str(tmp_path / "melodic_loop.wav")
    save_audio(path, c_chord, sr)

    res = analyze_file(path)
    assert res["category"] == "Melodic_Loops"
    assert res["key"] == "C Major"
    assert res["key_confidence"] >= 0.50


def test_e2e_synth_pad_non_rhythmic(tmp_path):
    """End-to-end regression: Non-rhythmic ambient chord pad with slow attack and release."""
    sr = 44100
    dur = 2.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # A Major chord pad
    pad = (
        0.35 * np.sin(2 * np.pi * 440.0 * t) +
        0.35 * np.sin(2 * np.pi * 554.37 * t) +
        0.35 * np.sin(2 * np.pi * 659.25 * t)
    ).astype(np.float32)
    # Smooth bell-shaped envelope (no sharp onsets)
    pad *= np.sin(np.pi * t / dur).astype(np.float32)

    path = str(tmp_path / "synth_pad.wav")
    save_audio(path, pad, sr)

    res = analyze_file(path)
    assert res["category"] in ("Melodic_Loops", "FX_Textures")
    assert res["is_rhythmic"] is False
    assert res["key"] == "A Major"


def test_e2e_unclassifiable_sample(tmp_path):
    """End-to-end regression: Extremely low amplitude / silence audio."""
    sr = 44100
    dur = 1.5
    signal = np.zeros(int(sr * dur), dtype=np.float32)

    path = str(tmp_path / "silence.wav")
    save_audio(path, signal, sr)

    res = analyze_file(path)
    assert res["category"] == "Unknown"
    assert res["is_rhythmic"] is False
    assert res["key"] == "Unknown"
