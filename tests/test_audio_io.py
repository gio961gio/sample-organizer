"""Dedicated tests for Audio I/O (Phase 1)."""
import os
import tempfile
import numpy as np
import pytest
import soundfile as sf
from sample_organizer.audio_io import load_audio, save_audio


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_roundtrip_subtypes(temp_dir):
    """Test round-trip across PCM 8, 16, 24, 32 and FLOAT formats."""
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    signal = (0.7 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    subtypes = [
        ("PCM_U8", 0.02),
        ("PCM_16", 1e-3),
        ("PCM_24", 1e-4),
        ("PCM_32", 1e-4),
        ("FLOAT", 1e-5),
    ]

    for subtype, tol in subtypes:
        path = os.path.join(temp_dir, f"test_{subtype}.wav")
        save_audio(path, signal, sr, subtype=subtype)
        loaded, loaded_sr = load_audio(path)
        assert loaded_sr == sr
        assert loaded.shape == signal.shape
        assert np.all(np.isfinite(loaded))
        assert np.allclose(loaded, signal, atol=tol)


def test_multiformat_support(temp_dir):
    """Test reading and writing WAV, AIFF, and FLAC (mono and stereo)."""
    sr = 44100
    n_frames = int(sr * 0.2)
    t = np.linspace(0, 0.2, n_frames, endpoint=False)
    mono = (0.6 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    stereo = np.column_stack([mono, 0.5 * mono])

    # WAV mono
    wav_path = os.path.join(temp_dir, "test.wav")
    save_audio(wav_path, mono, sr)
    data, s_sr = load_audio(wav_path)
    assert s_sr == sr and len(data) == n_frames

    # AIFF stereo
    aiff_path = os.path.join(temp_dir, "test.aiff")
    sf.write(aiff_path, stereo, sr, format="AIFF")
    aiff_data, aiff_sr = load_audio(aiff_path)
    assert aiff_sr == sr
    assert aiff_data.shape == (n_frames,)
    expected_downmix = (stereo[:, 0] + stereo[:, 1]) / 2.0
    assert np.allclose(aiff_data, expected_downmix, atol=1e-3)

    # FLAC mono and stereo
    flac_mono_path = os.path.join(temp_dir, "test_mono.flac")
    sf.write(flac_mono_path, mono, sr, format="FLAC")
    flac_data, flac_sr = load_audio(flac_mono_path)
    assert flac_sr == sr
    assert np.allclose(flac_data, mono, atol=1e-3)

    flac_stereo_path = os.path.join(temp_dir, "test_stereo.flac")
    sf.write(flac_stereo_path, stereo, sr, format="FLAC")
    fstereo_data, fstereo_sr = load_audio(flac_stereo_path)
    assert fstereo_sr == sr
    assert fstereo_data.shape == (n_frames,)
    assert np.all(np.isfinite(fstereo_data))


def test_resampling(temp_dir):
    """Test polyphase resampling functionality and precision."""
    sr = 44100
    target_sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    path = os.path.join(temp_dir, "resample_in.wav")
    save_audio(path, signal, sr)

    resampled, resampled_sr = load_audio(path, target_sr=target_sr)
    assert resampled_sr == target_sr
    assert abs(len(resampled) - target_sr) <= 1
    assert np.all(np.isfinite(resampled))

    # Same target_sr should return unmodified
    same_sr_data, same_sr = load_audio(path, target_sr=sr)
    assert same_sr == sr
    assert len(same_sr_data) == sr


def test_invalid_inputs(temp_dir):
    """Test boundary conditions, invalid parameters, and corrupted inputs."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        load_audio(os.path.join(temp_dir, "does_not_exist.wav"))

    # 2. Target sr <= 0
    valid_wav = os.path.join(temp_dir, "dummy.wav")
    save_audio(valid_wav, np.array([0.1, -0.1], dtype=np.float32), 44100)

    with pytest.raises(ValueError, match="target_sr must be a positive integer"):
        load_audio(valid_wav, target_sr=0)

    with pytest.raises(ValueError, match="target_sr must be a positive integer"):
        load_audio(valid_wav, target_sr=-44100)

    # 3. Empty / Corrupted file
    empty_file = os.path.join(temp_dir, "empty.wav")
    with open(empty_file, "wb") as f:
        f.write(b"")
    with pytest.raises(RuntimeError):
        load_audio(empty_file)

    corrupt_file = os.path.join(temp_dir, "corrupt.wav")
    with open(corrupt_file, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00randomgarbage")
    with pytest.raises(RuntimeError):
        load_audio(corrupt_file)

    # 4. Save invalid audio
    with pytest.raises(ValueError, match="Cannot save an empty audio array"):
        save_audio(os.path.join(temp_dir, "empty_out.wav"), np.array([]), 44100)

    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        save_audio(os.path.join(temp_dir, "bad_sr.wav"), np.array([0.5]), 0)

    with pytest.raises(ValueError, match="Cannot save audio containing non-finite values"):
        save_audio(os.path.join(temp_dir, "nan.wav"), np.array([0.5, np.nan]), 44100)

    with pytest.raises(TypeError):
        save_audio(os.path.join(temp_dir, "bad_type.wav"), [0.1, 0.2], 44100)  # type: ignore
