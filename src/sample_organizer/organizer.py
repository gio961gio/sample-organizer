"""Sample organization engine: recursive scanning, classification, and safe file sorting."""
import os
import shutil
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from .audio_io import load_audio
from .metrics import calculate_metrics
from .key_detector import estimate_key
from .bpm_detector import estimate_bpm

SUPPORTED_EXTENSIONS = [".wav", ".aif", ".aiff", ".flac"]


def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def copy_file_exclusive(
    src_path: str,
    desired_target: str,
    reserved_paths: Optional[Set[str]] = None,
) -> Tuple[str, str]:
    """
    Atomically copy src_path to a non-colliding destination using exclusive creation ('xb').

    Guarantees:
    - Never overwrites an existing file, even under concurrent operations.
    - If a collision occurs, increments counter (_001, _002, ...) until an exclusive slot is acquired.
    - Automatically removes partial files if writing or verification fails.
    - Verifies byte-level integrity between source and destination using SHA-256.
    - Preserves file metadata timestamps.

    Returns
    -------
    (final_destination_path, file_sha256) : Tuple[str, str]
    """
    real_src = os.path.realpath(src_path)
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file does not exist: '{src_path}'")

    src_hash = compute_sha256(src_path)
    src_size = os.path.getsize(src_path)

    base_dir, filename = os.path.split(desired_target)
    stem, ext = os.path.splitext(filename)
    os.makedirs(base_dir, exist_ok=True)

    counter = 0
    while True:
        if counter == 0:
            candidate = os.path.normpath(desired_target)
        else:
            candidate = os.path.normpath(os.path.join(base_dir, f"{stem}_{counter:03d}{ext}"))
        counter += 1

        if reserved_paths is not None and candidate in reserved_paths:
            continue

        real_candidate = os.path.realpath(candidate)
        if real_candidate == real_src:
            raise ValueError(f"Source and destination paths are identical: '{src_path}'")

        # Atomic exclusive creation ('xb' uses O_CREAT|O_EXCL on POSIX, CREATE_NEW on Windows)
        try:
            with open(candidate, "xb") as dst_f:
                dst_hash = hashlib.sha256()
                try:
                    with open(src_path, "rb") as src_f:
                        while chunk := src_f.read(65536):
                            dst_f.write(chunk)
                            dst_hash.update(chunk)
                    dst_f.flush()
                    os.fsync(dst_f.fileno())
                except Exception:
                    # Clean up partial file on write failure
                    try:
                        dst_f.close()
                        if os.path.exists(candidate):
                            os.remove(candidate)
                    except Exception:
                        pass
                    raise

            # Verify written destination SHA-256 and size
            if dst_hash.hexdigest() != src_hash or os.path.getsize(candidate) != src_size:
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                except Exception:
                    pass
                raise RuntimeError(f"Integrity check failed: SHA-256 mismatch copying '{src_path}' to '{candidate}'")

            # Preserve modification metadata
            try:
                shutil.copystat(src_path, candidate)
            except Exception:
                pass

            if reserved_paths is not None:
                reserved_paths.add(candidate)

            return candidate, src_hash

        except FileExistsError:
            # File exists or was created concurrently; try next candidate suffix
            continue


def classify_sample(
    metrics: Dict[str, Any],
    bpm_info: Dict[str, Any],
    key_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Robust heuristic classification combining spectral, harmonic, and rhythmic dynamics.

    Categories:
    - Kicks, Snares_Claps, Hats_Cymbals, Percussion (one-shots)
    - Drum_Loops, Melodic_Loops, FX_Textures (loops & long audio)
    - Unknown (insufficient evidence, ambiguous, or silence)
    """
    duration = metrics.get("duration_sec", 0.0)
    crest_factor = metrics.get("crest_factor_db", 0.0)
    zcr = metrics.get("zcr", 0.0)
    peak_db = metrics.get("peak_db", -120.0)
    low_ratio = metrics.get("low_energy_ratio", 0.0)
    high_ratio = metrics.get("high_energy_ratio", 0.0)
    centroid = metrics.get("spectral_centroid", 0.0)

    if peak_db < -55.0:
        return "Unknown"

    # Short percussive hits (one-shots: duration < 1.2s)
    if duration < 1.2:
        if high_ratio > 0.35 or zcr > 0.15 or centroid > 5000.0:
            return "Hats_Cymbals"
        if low_ratio > 0.35 and crest_factor > 10.5 and centroid < 800.0:
            return "Kicks"
        if crest_factor > 9.0 and (centroid > 800.0 or high_ratio > 0.10):
            return "Snares_Claps"
        if crest_factor > 5.5:
            return "Percussion"
        return "Unknown"

    # Loops & Sustained Audio (duration >= 1.2s)
    key_conf = key_info.get("confidence", 0.0) if key_info else 0.0
    has_tonal_key = (
        key_info is not None
        and key_info.get("key") != "Unknown"
        and key_conf >= 0.35
    )
    is_rhythmic = bpm_info.get("is_rhythmic", False)

    # Multi-feature disambiguation between Drum Loops and Melodic Loops
    if is_rhythmic:
        # High harmonic confidence + moderate crest factor indicates melodic instrumentation
        if key_conf >= 0.65:
            return "Melodic_Loops"
        if key_conf >= 0.45 and crest_factor < 10.0:
            return "Melodic_Loops"

        # Strong percussive crest factor or weak tonal confidence indicates drum loop
        if crest_factor >= 9.5 or key_conf < 0.45:
            return "Drum_Loops"

        return "Drum_Loops"

    # Non-rhythmic long audio
    if has_tonal_key and key_conf >= 0.45:
        return "Melodic_Loops"
    if crest_factor > 3.5:
        return "FX_Textures"

    return "Unknown"


def analyze_file(file_path: str) -> Dict[str, Any]:
    """Run full acoustic analysis on a single audio sample."""
    signal, sr = load_audio(file_path)
    metrics = calculate_metrics(signal, sr)
    key_info = estimate_key(signal, sr)
    bpm_info = estimate_bpm(signal, sr)
    category = classify_sample(metrics, bpm_info, key_info=key_info)

    filename = os.path.basename(file_path)
    stem, ext = os.path.splitext(filename)

    return {
        "file_path": str(file_path),
        "filename": filename,
        "extension": ext.lower(),
        "sample_rate": sr,
        "category": category,
        "metrics": metrics,
        "key": key_info["key"],
        "root": key_info["root"],
        "scale": key_info["scale"],
        "key_confidence": key_info["confidence"],
        "bpm": bpm_info["bpm"],
        "bpm_confidence": bpm_info["confidence"],
        "is_rhythmic": bpm_info["is_rhythmic"],
    }


def scan_directory(
    directory_path: str,
    supported_exts: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan a folder recursively and analyze all audio files.

    Distinguishes and preserves read vs analysis errors with full file paths.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: '{directory_path}'")

    if supported_exts is None:
        supported_exts = SUPPORTED_EXTENSIONS
    supported_exts_lower = {ext.lower() for ext in supported_exts}

    excluded_canonical = set()
    if exclude_dirs:
        for ed in exclude_dirs:
            if os.path.exists(ed):
                excluded_canonical.add(os.path.realpath(ed))

    results = []
    for root, dirnames, files in os.walk(directory_path):
        dirnames[:] = [
            d for d in dirnames
            if os.path.realpath(os.path.join(root, d)) not in excluded_canonical
        ]

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts_lower:
                full_path = os.path.join(root, f)
                # First attempt reading
                try:
                    signal, sr = load_audio(full_path)
                except Exception as read_exc:
                    results.append({
                        "file_path": full_path,
                        "filename": f,
                        "error": str(read_exc),
                        "error_type": "read",
                    })
                    continue

                # Then attempt analysis
                try:
                    metrics = calculate_metrics(signal, sr)
                    key_info = estimate_key(signal, sr)
                    bpm_info = estimate_bpm(signal, sr)
                    category = classify_sample(metrics, bpm_info, key_info=key_info)

                    results.append({
                        "file_path": full_path,
                        "filename": f,
                        "extension": ext,
                        "sample_rate": sr,
                        "category": category,
                        "metrics": metrics,
                        "key": key_info["key"],
                        "root": key_info["root"],
                        "scale": key_info["scale"],
                        "key_confidence": key_info["confidence"],
                        "bpm": bpm_info["bpm"],
                        "bpm_confidence": bpm_info["confidence"],
                        "is_rhythmic": bpm_info["is_rhythmic"],
                    })
                except Exception as analysis_exc:
                    results.append({
                        "file_path": full_path,
                        "filename": f,
                        "error": str(analysis_exc),
                        "error_type": "analysis",
                    })

    return results


def resolve_unique_path(target_path: str, reserved_paths: Set[str]) -> str:
    """Generate a deterministic non-colliding destination path (e.g. name_001.wav)."""
    norm_target = os.path.normpath(target_path)
    if not os.path.exists(norm_target) and norm_target not in reserved_paths:
        reserved_paths.add(norm_target)
        return norm_target

    base_dir, filename = os.path.split(norm_target)
    stem, ext = os.path.splitext(filename)

    counter = 1
    while True:
        candidate = os.path.normpath(os.path.join(base_dir, f"{stem}_{counter:03d}{ext}"))
        if not os.path.exists(candidate) and candidate not in reserved_paths:
            reserved_paths.add(candidate)
            return candidate
        counter += 1


def organize_samples(
    scan_results: List[Dict[str, Any]],
    output_dir: str,
    mode: str = "copy",  # "copy", "move", "dry-run"
    source_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sort analyzed samples into organized destination folders safely.

    Guarantees:
    - Rejects operation if source and destination are the same directory.
    - Never overwrites existing destination files via atomic exclusive file creation ('xb').
    - Guarantees data integrity via SHA-256 verification and clears partial files on failure.
    - For 'move', verifies destination integrity before removing source, then verifies source non-existence.
    - Captures and categorizes read, analysis, and write errors per file.
    """
    valid_modes = ("copy", "move", "dry-run")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: {valid_modes}")

    real_output_dir = os.path.realpath(output_dir)

    if source_dir is not None:
        real_src_dir = os.path.realpath(source_dir)
        if real_src_dir == real_output_dir:
            raise ValueError(f"Source and destination directories cannot be identical: '{output_dir}'")

    summary: Dict[str, Any] = {
        "total": len(scan_results),
        "processed": 0,
        "planned": 0,
        "errors": 0,
        "operations": [],
    }

    reserved_paths: Set[str] = set()

    for item in scan_results:
        src_path = item.get("file_path")
        if not src_path or "error" in item:
            summary["errors"] += 1
            summary["operations"].append({
                "source": src_path or "unknown",
                "destination": None,
                "category": "Unknown",
                "mode": mode,
                "status": "error",
                "error_type": item.get("error_type", "scan"),
                "error_msg": item.get("error", "Scan error"),
            })
            continue

        src_path = os.path.normpath(src_path)
        real_src = os.path.realpath(src_path)

        # Check directory collision for this sample
        if os.path.realpath(os.path.dirname(src_path)) == real_output_dir:
            summary["errors"] += 1
            summary["operations"].append({
                "source": src_path,
                "destination": None,
                "category": "Unknown",
                "mode": mode,
                "status": "error",
                "error_type": "safety",
                "error_msg": f"Source directory matches destination directory: '{real_output_dir}'",
            })
            continue

        category = item.get("category", "Unknown")

        has_key = item.get("key") and item.get("key") != "Unknown" and item.get("key_confidence", 0) >= 0.35
        has_bpm = item.get("bpm") is not None and item.get("is_rhythmic", False)

        key_part = f"{item['root']}_{item['scale']}" if has_key else None
        bpm_part = f"{int(round(item['bpm']))}BPM" if has_bpm else None

        if "Loops" in category:
            if bpm_part and key_part:
                subfolder = os.path.join(output_dir, category, f"{bpm_part}_{key_part}")
            elif bpm_part:
                subfolder = os.path.join(output_dir, category, bpm_part)
            elif key_part:
                subfolder = os.path.join(output_dir, category, key_part)
            else:
                subfolder = os.path.join(output_dir, category)
        else:
            subfolder = os.path.join(output_dir, category)

        orig_filename = item["filename"]
        prefix_parts = []
        if key_part:
            prefix_parts.append(item["root"])
        if bpm_part:
            prefix_parts.append(bpm_part)

        if prefix_parts and not orig_filename.startswith(tuple(prefix_parts)):
            dest_filename = f"{'_'.join(prefix_parts)}_{orig_filename}"
        else:
            dest_filename = orig_filename

        desired_target = os.path.join(subfolder, dest_filename)

        if mode == "dry-run":
            final_dest = resolve_unique_path(desired_target, reserved_paths)
            op: Dict[str, Any] = {
                "source": src_path,
                "destination": final_dest,
                "category": category,
                "mode": mode,
                "status": "planned",
            }
            summary["planned"] += 1
            summary["operations"].append(op)
            continue

        # Physical execution with atomic exclusive file creation & SHA-256 verification
        op = {
            "source": src_path,
            "destination": None,
            "category": category,
            "mode": mode,
            "status": "planned",
        }

        try:
            if os.path.realpath(desired_target) == real_src:
                raise ValueError("Source and destination paths are identical")

            final_dest, file_hash = copy_file_exclusive(src_path, desired_target, reserved_paths)
            op["destination"] = final_dest

            if mode == "copy":
                # Verify source file was not modified
                if not os.path.exists(src_path) or compute_sha256(src_path) != file_hash:
                    raise RuntimeError(f"Integrity check failed: source file '{src_path}' was modified during copy")
                op["status"] = "copied"

            elif mode == "move":
                # Safely delete source after destination was atomically written and verified
                os.remove(src_path)
                if os.path.exists(src_path):
                    raise RuntimeError(f"Failed to remove source file '{src_path}' after move")
                if not os.path.exists(final_dest) or compute_sha256(final_dest) != file_hash:
                    raise RuntimeError(f"Failed to verify integrity of moved destination '{final_dest}'")
                op["status"] = "moved"

            summary["processed"] += 1
            summary["planned"] += 1

        except Exception as exc:
            op["status"] = "error"
            op["error_type"] = "write"
            op["error_msg"] = str(exc)
            summary["errors"] += 1

        summary["operations"].append(op)

    return summary
