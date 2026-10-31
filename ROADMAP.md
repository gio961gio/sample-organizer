# Sample Organizer - Development Roadmap

This roadmap documents completed milestones and future engineering phases for **Sample Organizer**. Over **30 hours** of future development are scheduled.

---

## Phase 1: Core Engine & Audio I/O Stabilization (Completed - v0.1.0)
- [x] High-fidelity multi-format I/O (`WAV`, `AIFF`, `FLAC`) using `soundfile` backend.
- [x] Polyphase resampling via `scipy.signal.resample_poly` with exact rational factor reduction.
- [x] Dynamics & spectral metrics: Peak, RMS, Crest Factor, Zero Crossing Rate, Spectral Centroid, Low/High energy ratios.
- [x] Cognitive key detection using Krumhansl-Schmuckler profiles with non-tonal and noise rejection (`Unknown`).
- [x] Robust BPM estimation via FFT autocorrelation, periodicity significance ($z$-score), and Gaussian tempo prior.
- [x] Non-destructive `--mode dry-run` with zero disk modifications.
- [x] Deterministic name collision resolution (`_001.wav`, `_002.wav`) preventing accidental overwrites.
- [x] CLI commands (`analyze`, `scan`, `organize`) and automated GitHub Actions CI workflow.

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
