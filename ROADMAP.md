# Sample Organizer - Development Roadmap

This document outlines the planned future development for **Sample Organizer**. Over **30 hours** of engineering work are scheduled across four major phases.

---

## Phase 1: Core DSP & Ingestion (Completed - v0.1.0)
- [x] PCM WAV audio I/O for 8/16/24/32-bit files without external binary dependencies.
- [x] Dynamics metrics: Peak, RMS, Crest Factor, Zero Crossing Rate.
- [x] Cognitive key detection with Krumhansl-Schmuckler harmonic profiles.
- [x] BPM onset autocorrelation estimator.
- [x] Basic heuristic classifier and directory sorting engine.
- [x] Unit test suite covering signal processing and file handling.

---

## Phase 2: High-Performance Batch Processing & SQLite Caching (~10 Hours)
- [ ] **Parallel Processing**: Implement `concurrent.futures.ProcessPoolExecutor` for multithreaded scanning of massive (50GB+) sample collections.
- [ ] **SQLite Metadata Cache**: Create a local SQLite schema (`samples.db`) storing audio fingerprints (MD5/SHA256), acoustic metrics, and tags to avoid re-analyzing unchanged files.
- [ ] **Incremental Sync**: Add `--update` flag to only analyze newly added files in monitored directories.
- [ ] **Flac & Ogg Vorbis Support**: Integrate native FLAC decoder support.

---

## Phase 3: DAW Metadata & Broadcast WAV Integration (~8 Hours)
- [ ] **RIFF INFO & BEXT Chunks**: Write key, BPM, and artist metadata directly into the WAV RIFF header.
- [ ] **Ableton Live `.asd` File Generation**: Generate companion `.asd` warp files with pre-calculated BPM markers.
- [ ] **FL Studio Sampler Loop Points**: Add support for writing loop start/end cue points into WAV chunks.
- [ ] **Duplicate Detection**: Identify exact and acoustic duplicate samples based on spectral similarity.

---

## Phase 4: Desktop UI & Waveform Visualizer (~12 Hours)
- [ ] **Modern Desktop GUI**: Build a native desktop GUI using PyQt6 or Tkinter with drag-and-drop sample import.
- [ ] **Waveform Display**: Interactive audio waveform canvas with zoom, pan, and transient marker overlays.
- [ ] **Audio Playback Engine**: Integrated low-latency playback preview via `sounddevice` or native audio APIs.
- [ ] **One-Click Export**: Drag organized samples directly from the UI into any digital audio workstation.