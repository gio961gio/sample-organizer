# Sample Organizer

[![CI](https://github.com/gio961gio/sample-organizer/actions/workflows/ci.yml/badge.svg)](https://github.com/gio961gio/sample-organizer/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Audio Sample Analysis, Harmonic Key Detection & Safe Library Organization Engine**

Sample Organizer is a modular Python command-line tool and DSP library designed for audio producers, sound designers, and sample archivists. It ingests unorganized audio collections (WAV, AIFF, FLAC), computes acoustic and spectral dynamics, detects musical key and tempo (BPM), and sorts audio samples into structured folders with atomic exclusive file creation and zero data loss.

---

## Key Features

- **Multi-Format Audio I/O**: High-fidelity ingestion of WAV (PCM 8/16/24/32-bit and IEEE float), AIFF/AIFF-C, and FLAC using `soundfile` (`libsndfile`). Downmixes multichannel buffers to mono via equal-weight channel averaging and provides exact polyphase resampling via `scipy.signal.resample_poly`.
- **Harmonic Key Estimation**: Computes a 12-dimensional pitch chroma vector and correlates against cognitive **Krumhansl-Schmuckler** Major and Minor profiles. Returns `Unknown` for silence, non-tonal noise, or ambiguous distributions to prevent false positive classifications.
- **Accurate BPM & Octave Detection**: Estimates loop tempo using energy flux onset envelopes, FFT autocorrelation, and 3-point parabolic sub-sample peak refinement. Resolves half-time and double-time ambiguities across 60–180 BPM without arbitrary tempo priors, adjusting confidence according to candidate margins and rhythmic ambiguity.
- **Acoustic & Full-Duration Spectral Metrics**: Computes Peak (dBFS), RMS (dBFS), Crest Factor (transient dynamics), Zero Crossing Rate (ZCR), Spectral Centroid (Hz), Low Energy Ratio (< 250 Hz), and High Energy Ratio (> 4000 Hz). Uses overlapping Hann windows with energy-aware pooling to ensure brief high-frequency transients anywhere in long files are captured.
- **Multi-Feature Classification**: Categorizes samples into `Kicks`, `Snares_Claps`, `Hats_Cymbals`, `Percussion`, `Drum_Loops`, `Melodic_Loops`, `FX_Textures`, or `Unknown` combining spectral dynamics, harmonic confidence, and rhythmic periodicity.
- **Safe File Organization**:
  - **Non-destructive Dry-Run**: `--mode dry-run` evaluates and displays the operation plan without writing files or creating directories.
  - **Atomic Exclusive Creation**: Destination files are created using atomic exclusive creation (`'xb'`) to prevent race conditions or concurrent overwrites.
  - **Deterministic Collision Handling**: Automatically resolves filename collisions deterministically (`sample_001.wav`, `sample_002.wav`).
  - **SHA-256 Verification & Cleanup**: Verifies byte integrity between source and destination with SHA-256, automatically deleting partial files if an operation fails.
  - **Directory Collision Protection**: Rejects operations where source and destination paths are identical using canonical path resolution (`os.path.realpath`).

---

## Project Structure

```
sample-organizer/
├── src/
│   └── sample_organizer/
│       ├── __init__.py          # Package version and metadata
│       ├── audio_io.py          # Soundfile backend I/O & polyphase resampling
│       ├── metrics.py           # Dynamics, ZCR & spectral features
│       ├── key_detector.py      # Chromagram & Krumhansl-Schmuckler key engine
│       ├── bpm_detector.py      # FFT onset autocorrelation & octave evaluation
│       ├── organizer.py         # Directory scanner, classifier & atomic file safety
│       └── cli.py               # Subcommands (analyze, scan, organize)
├── tests/
│   ├── test_audio_io.py         # Audio I/O format and resampling tests
│   ├── test_metrics_key.py      # Acoustic metrics, transients & key rejection tests
│   ├── test_bpm_detector.py     # BPM pulse, boundary & octave ambiguity tests
│   ├── test_organizer.py        # File safety, collision, SHA-256 & E2E regression tests
│   └── test_sample_organizer.py # Integrated suite
├── .github/
│   └── workflows/ci.yml         # GitHub Actions CI matrix
├── pyproject.toml               # Package build configuration
├── requirements.txt             # Runtime & test dependencies
├── ROADMAP.md                   # Long-term development plan
└── LICENSE                      # MIT License
```

---

## Installation

```bash
git clone https://github.com/gio961gio/sample-organizer.git
cd sample-organizer
pip install -r requirements.txt
pip install -e .
```

---

## CLI Usage

### 1. Analyze an audio file
```bash
sample-organizer analyze path/to/sample.wav
```
Output:
```
Analyzing: synth_loop_c_maj.wav...

--- Audio Overview ---
  File:            synth_loop_c_maj.wav
  Category:        Melodic_Loops
  Sample Rate:     44100 Hz
  Duration:        4.0 s

--- Musical Analysis ---
  Key:             C Major (conf: 0.88)
  BPM:             120.0 BPM (conf: 0.72)
  Rhythmic:        Yes

--- Dynamics & Signal ---
  Peak:            -2.5 dBFS
  RMS:             -12.9 dBFS
  Crest Factor:    10.4 dB
  ZCR:             0.0152
  Centroid:        512.3 Hz
  Low/High Energy: 0.38 / 0.04
```

### 2. Scan a sample library folder
```bash
sample-organizer scan path/to/unorganized_samples/
```

### 3. Organize samples safely

Preview changes without modifying the filesystem:
```bash
sample-organizer organize path/to/source/ path/to/destination/ --mode dry-run
```

Copy samples into sorted category folders:
```bash
sample-organizer organize path/to/source/ path/to/destination/ --mode copy
```

Move samples:
```bash
sample-organizer organize path/to/source/ path/to/destination/ --mode move
```

---

## Running the Test Suite

Run the full automated test suite with pytest:
```bash
pytest -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
