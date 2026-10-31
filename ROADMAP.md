# Sample Organizer - Development Roadmap

This roadmap documents completed milestones and future engineering phases for **Sample Organizer**. Over **30 hours** of future development are scheduled.

---

## Completed Milestones (v0.1.0 – v0.2.0)
- [x] **Multi-Format Audio I/O**: High-fidelity ingestion (`WAV`, `AIFF`, `FLAC`) using `soundfile` with polyphase resampling (`scipy.signal.resample_poly`).
- [x] **Cognitive Key Detection**: Krumhansl-Schmuckler profiles with strict non-tonal and noise rejection (`Unknown`).
- [x] **Acoustic & Full-Duration Spectral Metrics**: Peak, RMS, Crest Factor, ZCR, Spectral Centroid, and power ratios across overlapping Hann windows with transient energy pooling.
- [x] **Accurate BPM & Octave Disambiguation**: Sub-sample parabolic peak interpolation, unbiased octave harmonic analysis across 60–180 BPM, boundary detection, and candidate margin confidence scoring.
- [x] **Multi-Feature Classification**: Combinatorial classification balancing spectral crest factors, harmonic key confidence, and rhythmic periodicity.
- [x] **Atomic File Safety & Integrity**: Exclusive creation (`'xb'`), SHA-256 byte verification on copy/move, partial file cleanup on error, and canonical path checks preventing destination overwriting.
- [x] **CLI Subcommands**: `analyze`, `scan`, `organize` (with `--mode dry-run`, `copy`, and `move`).
- [x] **Comprehensive Test Suite**: Automated integration and regression test matrix on GitHub Actions CI.

---

## Phase 2: High-Performance Batch Ingestion & SQLite Caching (~10 Hours)
- [ ] **Multiprocessing Engine**: Parallel scanning and feature extraction using `concurrent.futures.ProcessPoolExecutor` for 50GB+ libraries.
- [ ] **SQLite Metadata Cache**: Local schema (`samples.db`) storing audio checksums (SHA-256), acoustic metrics, and tags to eliminate re-analysis of unchanged files.
- [ ] **Incremental Sync**: CLI flag `--update` to process only newly added or modified sample assets.
- [ ] **Ogg Vorbis / MP3 Reading**: Optional format decoders for compressed sample library archives.

---

## Phase 3: DAW Metadata & Broadcast Chunks (~8 Hours)
- [ ] **RIFF INFO & BEXT Chunks**: Write key, BPM, and sound designer metadata directly into WAV RIFF headers.
- [ ] **Ableton Live `.asd` File Generation**: Generate companion `.asd` warp files with pre-calculated BPM markers.
- [ ] **FL Studio Sampler Cue Points**: Add support for writing loop start/end points into WAV sampler chunks.
- [ ] **Acoustic Duplicate Detection**: Identify duplicate or clipped audio assets using spectral difference fingerprints.

---

## Phase 4: Desktop UI & Waveform Canvas (~12 Hours)
- [ ] **Native Desktop GUI**: Standalone PyQt6 desktop application with drag-and-drop ingestion.
- [ ] **Interactive Waveform Display**: Canvas visualizer with zoom, pan, and transient marker overlays.
- [ ] **Low-Latency Audio Playback**: Audition audio clips instantly using `sounddevice`.
- [ ] **DAW Drag-Out**: Drag organized samples directly from the UI into digital audio workstations.
