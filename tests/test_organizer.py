"""Dedicated tests for Organizer engine (Phase 4 & Phase 6 Refinements)."""
import os
import tempfile
import numpy as np
import pytest
from sample_organizer.audio_io import save_audio
from sample_organizer.organizer import scan_directory, organize_samples, classify_sample


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


def test_copy_source_integrity(sample_env):
    """Verify that copy operation leaves the original source file completely intact."""
    src, dst = sample_env
    sr = 44100
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    original_audio = (0.75 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

    src_file = os.path.join(src, "pristine_source.wav")
    save_audio(src_file, original_audio, sr)

    src_size = os.path.getsize(src_file)
    with open(src_file, "rb") as f:
        src_bytes = f.read()

    results = scan_directory(src)
    summary = organize_samples(results, dst, mode="copy")

    assert summary["processed"] == 1
    assert summary["errors"] == 0

    dest_file = summary["operations"][0]["destination"]
    assert os.path.exists(dest_file)

    # Source file integrity verification
    assert os.path.exists(src_file)
    assert os.path.getsize(src_file) == src_size
    with open(src_file, "rb") as f:
        assert f.read() == src_bytes

    # Destination file verification
    assert os.path.getsize(dest_file) == src_size


def test_collision_with_preexisting_file(sample_env):
    """Verify that copy mode does not overwrite a file that already exists in destination."""
    src, dst = sample_env
    sr = 44100

    # 1. Pre-create a file in destination
    kicks_dest_dir = os.path.join(dst, "Kicks")
    os.makedirs(kicks_dest_dir, exist_ok=True)
    existing_dest_file = os.path.join(kicks_dest_dir, "kick.wav")
    save_audio(existing_dest_file, np.zeros(1000, dtype=np.float32), sr)
    existing_size = os.path.getsize(existing_dest_file)

    # 2. Put a new kick in source with the same name but different content
    src_kick = os.path.join(src, "kick.wav")
    save_audio(src_kick, np.ones(3000, dtype=np.float32) * 0.5, sr)
    new_src_size = os.path.getsize(src_kick)

    results = scan_directory(src)
    # Ensure it routes to Kicks
    results[0]["category"] = "Kicks"

    summary = organize_samples(results, dst, mode="copy")
    assert summary["processed"] == 1
    assert summary["errors"] == 0

    new_dest_file = summary["operations"][0]["destination"]
    assert new_dest_file != existing_dest_file
    assert os.path.basename(new_dest_file) == "kick_001.wav"

    # Pre-existing file must remain intact
    assert os.path.getsize(existing_dest_file) == existing_size
    assert os.path.getsize(new_dest_file) == new_src_size


def test_move_operation(sample_env):
    """Verify move operation removes source and writes destination."""
    src, dst = sample_env
    sr = 44100
    src_file = os.path.join(src, "sample_to_move.wav")
    save_audio(src_file, np.array([0.2, -0.2] * 200, dtype=np.float32), sr)

    results = scan_directory(src)
    summary = organize_samples(results, dst, mode="move")

    assert summary["processed"] == 1
    assert not os.path.exists(src_file)
    assert os.path.exists(summary["operations"][0]["destination"])


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


def test_classification_categories_multi_feature():
    """Verify robust categorization combining spectral, harmonic, and rhythmic dynamics."""
    # 1. Kick: short, high crest factor, low frequency, low centroid
    kick_m = {"duration_sec": 0.35, "crest_factor_db": 13.0, "low_energy_ratio": 0.50, "high_energy_ratio": 0.02, "spectral_centroid": 120.0, "peak_db": -0.5}
    assert classify_sample(kick_m, {"is_rhythmic": False}) == "Kicks"

    # 2. Snare: short, high crest factor, mid centroid
    snare_m = {"duration_sec": 0.40, "crest_factor_db": 11.5, "low_energy_ratio": 0.10, "high_energy_ratio": 0.20, "spectral_centroid": 1500.0, "peak_db": -1.0}
    assert classify_sample(snare_m, {"is_rhythmic": False}) == "Snares_Claps"

    # 3. HiHat: short, high high_energy_ratio and high ZCR
    hat_m = {"duration_sec": 0.08, "crest_factor_db": 8.0, "high_energy_ratio": 0.60, "zcr": 0.25, "spectral_centroid": 7000.0, "peak_db": -3.0}
    assert classify_sample(hat_m, {"is_rhythmic": False}) == "Hats_Cymbals"

    # 4. Percussion: short, moderate crest factor
    perc_m = {"duration_sec": 0.25, "crest_factor_db": 7.5, "low_energy_ratio": 0.20, "high_energy_ratio": 0.10, "spectral_centroid": 1200.0, "peak_db": -6.0}
    assert classify_sample(perc_m, {"is_rhythmic": False}) == "Percussion"

    # 5. Drum Loop: duration >= 1.2s, rhythmic, high crest factor (even with mild key resonance)
    drum_loop_m = {"duration_sec": 4.0, "crest_factor_db": 11.0, "peak_db": -1.0}
    bpm_info = {"bpm": 124.0, "is_rhythmic": True}
    mild_key = {"key": "A Minor", "confidence": 0.38}  # Mild resonance should NOT make it Melodic!
    assert classify_sample(drum_loop_m, bpm_info, key_info=mild_key) == "Drum_Loops"

    # 6. Melodic Loop: duration >= 1.2s, high key confidence, moderate crest factor
    melodic_m = {"duration_sec": 4.0, "crest_factor_db": 7.5, "peak_db": -2.0}
    strong_key = {"key": "C Major", "confidence": 0.75}
    assert classify_sample(melodic_m, bpm_info, key_info=strong_key) == "Melodic_Loops"

    # 7. FX Texture: duration >= 1.2s, non-rhythmic, non-tonal, dynamic
    fx_m = {"duration_sec": 3.0, "crest_factor_db": 5.0, "peak_db": -14.0}
    assert classify_sample(fx_m, {"is_rhythmic": False}, key_info={"key": "Unknown", "confidence": 0.0}) == "FX_Textures"

    # 8. Unknown: low level silence
    silent_m = {"duration_sec": 0.5, "crest_factor_db": 0.0, "peak_db": -65.0}
    assert classify_sample(silent_m, {"is_rhythmic": False}) == "Unknown"


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
