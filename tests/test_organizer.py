"""Dedicated tests for Organizer engine (Phase 4)."""
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
    # Create sample in src
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

    # Destination directory MUST NOT exist!
    assert not os.path.exists(dst)


def test_copy_and_collision_handling(sample_env):
    """Verify that copy mode resolves collisions deterministically without overwriting."""
    src, dst = sample_env
    sr = 44100
    # Two different files that would resolve to the same name
    save_audio(os.path.join(src, "hit_a.wav"), np.array([0.5, -0.5] * 500, dtype=np.float32), sr)
    save_audio(os.path.join(src, "hit_b.wav"), np.array([0.3, -0.3] * 500, dtype=np.float32), sr)

    results = scan_directory(src)
    # Force both to have the same destination filename
    for r in results:
        r["filename"] = "hit.wav"

    summary = organize_samples(results, dst, mode="copy")
    assert summary["processed"] == 2
    assert summary["errors"] == 0

    op1 = summary["operations"][0]["destination"]
    op2 = summary["operations"][1]["destination"]
    assert op1 != op2
    assert os.path.exists(op1)
    assert os.path.exists(op2)


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

    # Scan with exclusion
    results = scan_directory(src, exclude_dirs=[nested_out])
    filenames = [r["filename"] for r in results]
    assert "top.wav" in filenames
    assert "nested.wav" not in filenames


def test_unknown_metadata_clean_names():
    """Verify that Unknown key or None bpm does not pollute file paths."""
    metrics = {"duration_sec": 0.5, "crest_factor_db": 5.0, "peak_db": -6.0}
    bpm_info = {"bpm": None, "confidence": 0.0, "is_rhythmic": False}
    key_info = {"key": "Unknown", "root": "Unknown", "scale": "Unknown", "confidence": 0.0}

    cat = classify_sample(metrics, bpm_info, key_info)
    assert cat != "Melodic_Loops"
