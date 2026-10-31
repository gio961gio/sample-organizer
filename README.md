# Sample Organizer

[![CI](https://github.com/gio961gio/sample-organizer/actions/workflows/ci.yml/badge.svg)](https://github.com/gio961gio/sample-organizer/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Audio Sample Analysis, Harmonic Key Detection & Safe Library Organization Engine**

Sample Organizer is a modular Python command-line tool and DSP library designed for audio producers, sound designers, and sample archivists. It ingests unorganized audio collections (WAV, AIFF, FLAC), computes acoustic and spectral dynamics, detects musical key and tempo (BPM), and sorts audio samples into structured folders with deterministic collision handling and zero data loss.

---

## Key Features

- **Multi-Format Audio I/O**: High-fidelity ingestion of WAV (PCM 8/16/24/32-bit and IEEE float), AIFF/AIFF-C, and FLAC using `soundfile` (`libsndfile`). Downmixes multichannel buffers to mono via equal-weight channel averaging and provides exact polyphase resampling via `scipy.signal.resample_poly`.
- **Harmonic Key Estimation**: Computes a 12-dimensional pitch chroma vector and correlates against cognitive **Krumhansl-Schmuckler** Major and Minor profiles. Returns `Unknown` for silence, non-tonal noise, or ambiguous distributions to prevent false positive classifications.
- **FFT Onset BPM Detection**: Estimates loop tempo using energy flux onset envelopes and FFT-accelerated autocorrelation within 60–180 BPM. Evaluates periodicity significance ($z$-score) and incorporates a gentle Gaussian tempo prior to resolve half-time/double-time ambiguities; returns `None` for non-rhythmic audio.
- **Acoustic & Spectral Metrics**: Computes Peak (dBFS), RMS (dBFS), Crest Factor (transient dynamics), Zero Crossing Rate (ZCR), Spectral Centroid (Hz), Low Energy Ratio (< 250 Hz), and High Energy Ratio (> 4000 Hz).
- **Heuristic Classification**: Categorizes samples into `Kicks`, `Snares_Claps`, `Hats_Cymbals`, `Percussion`, `Drum_Loops`, `Melodic_Loops`, `FX_Textures`, or `Unknown`.
- **Safe File Organization**:
  - **Non-destructive Dry-Run**: `--mode dry-run` evaluates and displays the operation plan without writing files or creating directories.
  - **Collision Prevention**: Automatically resolves filename collisions deterministically (`sample_001.wav`, `sample_002.wav`) without overwriting existing files.
  - **Loop Exclusion**: Automatically skips target destination directories when nested inside the source folder.

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
│       ├── bpm_detector.py      # FFT onset autocorrelation & tempo prior
│       ├── organizer.py         # Directory scanner, classifier & collision resolver
│       └── cli.py               # Subcommands (analyze, scan, organize)
├── tests/
│   ├── test_audio_io.py         # Audio I/O format and resampling tests
│   ├── test_metrics_key.py      # Acoustic metrics & key rejection tests
│   ├── test_bpm_detector.py     # BPM pulse & non-rhythmic tests
│   ├── test_organizer.py        # Safe organization & dry-run tests
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
