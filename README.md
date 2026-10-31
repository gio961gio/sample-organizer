# Sample Organizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **Audio Sample Analysis, Harmonic Key Detection & Automated Library Organization Engine**

Sample Organizer is a lightweight, pure-Python command-line tool and DSP library designed for music producers, sound designers, and audio engineers. It scans cluttered sample directories, computes acoustic dynamics, detects musical key and tempo (BPM), and automatically sorts audio samples into structured harmonic folders.

---

## Key Features

- **Harmonic Key Estimation**: Computes a 12-dimensional pitch chroma vector and correlates against cognitive **Krumhansl-Schmuckler** Major and Minor profiles.
- **BPM & Rhythm Detection**: Estimates loop tempo using energy flux onset envelopes and lag autocorrelation within 60–180 BPM.
- **Acoustic Dynamics**: Calculates Peak (dBFS), RMS (dBFS), Crest Factor (transient dynamics), Zero Crossing Rate (ZCR), and active audio boundaries.
- **Automated Classification**: Automatically tags one-shots (Kicks, Snares, Hats, Percussion) versus loops (Drum Loops, Melodic Loops) based on duration, spectral density, and rhythmicity.
- **Smart File Organization**: Restructures messy sample folders into clean category directories formatted with root key and BPM metadata.

---

## Project Architecture

```
sample-organizer/
├── src/
│   └── sample_organizer/
│       ├── __init__.py          # Version & package metadata
│       ├── audio_io.py          # PCM WAV file reader/writer (8, 16, 24, 32-bit)
│       ├── metrics.py           # RMS, Peak, Crest factor, silence detection
│       ├── key_detector.py      # Pitch class profile & Krumhansl-Schmuckler
│       ├── bpm_detector.py      # Onset flux & autocorrelation tempo tracker
│       ├── organizer.py         # Recursive scanner, classifier & sorting engine
│       └── cli.py               # Command-line interface subcommands
├── tests/
│   └── test_sample_organizer.py # Comprehensive unit test suite
├── pyproject.toml               # Package build configuration
├── requirements.txt             # Project dependencies
├── ROADMAP.md                   # 25+ hours of planned development
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

## Quickstart & CLI Usage

### 1. Analyze a single sample
```bash
sample-organizer analyze path/to/sample.wav
```
Output:
```
--- Audio Overview ---
  File:       synth_lead_c.wav
  Category:   Melodic_Loops
  Sample Rate:44100 Hz
  Duration:   4.0 s

--- Musical Analysis ---
  Key:        C Major (conf: 0.82)
  BPM:        124.0 BPM (conf: 0.74)
  Rhythmic:   Yes

--- Dynamics & Signal ---
  Peak:       -0.5 dBFS
  RMS:        -14.2 dBFS
  Crest:      13.7 dB
  ZCR:        0.0482
```

### 2. Scan a sample pack directory
```bash
sample-organizer scan path/to/unorganized_samples/
```

### 3. Organize samples into structured folders
```bash
sample-organizer organize path/to/source_samples/ path/to/organized_library/ --mode copy
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.