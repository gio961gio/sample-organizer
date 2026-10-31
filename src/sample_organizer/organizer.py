"""Sample organization engine: recursive scanning, classification, and file sorting."""
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .audio_io import load_audio
from .metrics import calculate_metrics
from .key_detector import estimate_key
from .bpm_detector import estimate_bpm


def classify_sample(metrics: Dict[str, Any], bpm_info: Dict[str, Any]) -> str:
    """Heuristic classification into audio sample categories."""
    duration = metrics["duration_sec"]
    crest_factor = metrics["crest_factor_db"]
    zcr = metrics["zcr"]

    # Short percussive hits
    if duration < 1.2:
        if zcr > 0.15:
            return "Hats_Cymbals"
        elif crest_factor > 14.0 and duration < 0.4:
            return "Kicks"
        elif crest_factor > 11.0:
            return "Snares_Claps"
        else:
            return "Percussion"

    # Longer audio loops or stems
    if bpm_info.get("is_rhythmic", False) and duration >= 1.5:
        return "Drum_Loops"

    if duration >= 1.5:
        return "Melodic_Loops"

    return "FX_Textures"


def analyze_file(file_path: str) -> Dict[str, Any]:
    """Run full acoustic analysis on a single audio sample."""
    signal, sr = load_audio(file_path)
    metrics = calculate_metrics(signal, sr)
    key_info = estimate_key(signal, sr)
    bpm_info = estimate_bpm(signal, sr)
    category = classify_sample(metrics, bpm_info)

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


def scan_directory(directory_path: str, supported_exts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Scan a folder recursively and analyze all audio files."""
    if supported_exts is None:
        supported_exts = [".wav", ".aif", ".aiff", ".flac"]

    results = []
    for root, _, files in os.walk(directory_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                full_path = os.path.join(root, f)
                try:
                    analysis = analyze_file(full_path)
                    results.append(analysis)
                except Exception as e:
                    results.append({"file_path": full_path, "error": str(e)})

    return results


def organize_samples(
    scan_results: List[Dict[str, Any]],
    output_dir: str,
    mode: str = "copy",  # "copy", "move", "dry-run"
) -> Dict[str, Any]:
    """Sort analyzed samples into organized destination folders."""
    os.makedirs(output_dir, exist_ok=True)
    summary = {"total": len(scan_results), "processed": 0, "errors": 0, "operations": []}

    for item in scan_results:
        if "error" in item:
            summary["errors"] += 1
            continue

        category = item["category"]
        key_str = item["key"].replace(" ", "_")
        bpm_str = f"{int(item['bpm'])}BPM"

        # Organize into category folders with harmonic sub-structure
        if "Loops" in category:
            target_subfolder = os.path.join(output_dir, category, f"{bpm_str}_{key_str}")
        else:
            target_subfolder = os.path.join(output_dir, category)

        os.makedirs(target_subfolder, exist_ok=True)
        dest_filename = f"{item['root']}_{bpm_str}_{item['filename']}" if item["key_confidence"] > 0.4 else item["filename"]
        dest_path = os.path.join(target_subfolder, dest_filename)

        op = {
            "source": item["file_path"],
            "destination": dest_path,
            "category": category,
            "mode": mode,
        }

        if mode == "copy":
            shutil.copy2(item["file_path"], dest_path)
        elif mode == "move":
            shutil.move(item["file_path"], dest_path)

        summary["operations"].append(op)
        summary["processed"] += 1

    return summary
