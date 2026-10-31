"""Sample organization engine: recursive scanning, classification, and safe file sorting."""
import os
import shutil
from typing import Dict, Any, List, Optional, Set
from .audio_io import load_audio
from .metrics import calculate_metrics
from .key_detector import estimate_key
from .bpm_detector import estimate_bpm

SUPPORTED_EXTENSIONS = [".wav", ".aif", ".aiff", ".flac"]


def classify_sample(
    metrics: Dict[str, Any],
    bpm_info: Dict[str, Any],
    key_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Heuristic classification into audio sample categories.

    Categories:
    - Kicks, Snares_Claps, Hats_Cymbals, Percussion (one-shots)
    - Drum_Loops, Melodic_Loops, FX_Textures (loops & long samples)
    - Unknown (insufficient evidence or silence)
    """
    duration = metrics.get("duration_sec", 0.0)
    crest_factor = metrics.get("crest_factor_db", 0.0)
    zcr = metrics.get("zcr", 0.0)
    peak_db = metrics.get("peak_db", -120.0)
    low_ratio = metrics.get("low_energy_ratio", 0.0)
    high_ratio = metrics.get("high_energy_ratio", 0.0)

    if peak_db < -55.0:
        return "Unknown"

    # Short percussive hits (one-shots)
    if duration < 1.2:
        if high_ratio > 0.35 or zcr > 0.15:
            return "Hats_Cymbals"
        if low_ratio > 0.35 and crest_factor > 11.0:
            return "Kicks"
        if crest_factor > 10.0:
            return "Snares_Claps"
        if crest_factor > 6.0:
            return "Percussion"
        return "Unknown"

    # Longer audio loops or stems (duration >= 1.2s)
    has_key = (
        key_info is not None
        and key_info.get("key") != "Unknown"
        and key_info.get("confidence", 0.0) >= 0.30
    )
    is_rhythmic = bpm_info.get("is_rhythmic", False)

    if has_key:
        return "Melodic_Loops"
    if is_rhythmic:
        return "Drum_Loops"
    if crest_factor > 4.0:
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
    """Scan a folder recursively and analyze all audio files."""
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
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if os.path.realpath(os.path.join(root, d)) not in excluded_canonical
        ]

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts_lower:
                full_path = os.path.join(root, f)
                try:
                    analysis = analyze_file(full_path)
                    results.append(analysis)
                except Exception as e:
                    results.append({"file_path": full_path, "error": str(e)})

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
) -> Dict[str, Any]:
    """
    Sort analyzed samples into organized destination folders without data loss.

    Parameters
    ----------
    scan_results : List[dict]
        Output list from scan_directory or analyze_file.
    output_dir : str
        Target folder for organized samples.
    mode : str, default 'copy'
        Operation mode: 'copy', 'move', or 'dry-run'.

    Returns
    -------
    dict with keys:
        total, processed, planned, errors, operations
    """
    valid_modes = ("copy", "move", "dry-run")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: {valid_modes}")

    summary: Dict[str, Any] = {
        "total": len(scan_results),
        "processed": 0,
        "planned": 0,
        "errors": 0,
        "operations": [],
    }

    reserved_paths: Set[str] = set()

    for item in scan_results:
        if "error" in item or "file_path" not in item:
            summary["errors"] += 1
            continue

        src_path = os.path.normpath(item["file_path"])
        category = item.get("category", "Unknown")

        # Extract musical metadata safely
        has_key = item.get("key") and item.get("key") != "Unknown" and item.get("key_confidence", 0) >= 0.35
        has_bpm = item.get("bpm") is not None and item.get("is_rhythmic", False)

        key_part = f"{item['root']}_{item['scale']}" if has_key else None
        bpm_part = f"{int(round(item['bpm']))}BPM" if has_bpm else None

        # Determine subfolder structure
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

        # Build clean non-polluting filename prefix
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
        final_dest = resolve_unique_path(desired_target, reserved_paths)

        op: Dict[str, Any] = {
            "source": src_path,
            "destination": final_dest,
            "category": category,
            "mode": mode,
            "status": "planned",
        }

        if mode == "dry-run":
            # Completely non-destructive: no mkdir, no file operations
            summary["planned"] += 1
            summary["operations"].append(op)
            continue

        # Physical copy or move
        try:
            if src_path == final_dest:
                raise ValueError("Source and destination paths are identical")

            os.makedirs(os.path.dirname(final_dest), exist_ok=True)

            if mode == "copy":
                shutil.copy2(src_path, final_dest)
                op["status"] = "copied"
            elif mode == "move":
                shutil.move(src_path, final_dest)
                op["status"] = "moved"

            summary["processed"] += 1
            summary["planned"] += 1
        except Exception as exc:
            op["status"] = "error"
            op["error_msg"] = str(exc)
            summary["errors"] += 1

        summary["operations"].append(op)

    return summary
