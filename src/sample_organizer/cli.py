"""Command-line interface for Sample Organizer."""
import os
import sys
import argparse
from typing import List
from .organizer import analyze_file, scan_directory, organize_samples


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Format tabular data cleanly for CLI display."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    row_lines = [" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)) for row in rows]

    return "\n".join([header_line, sep_line] + row_lines)


def handle_analyze(args: argparse.Namespace):
    """Analyze a single audio sample and display detailed acoustic metrics."""
    if not os.path.exists(args.file):
        print(f"[Error] File not found: '{args.file}'")
        sys.exit(1)

    print(f"\nAnalyzing: {os.path.basename(args.file)}...")
    try:
        res = analyze_file(args.file)
    except Exception as exc:
        print(f"[Error] Analysis failed: {exc}")
        sys.exit(1)

    bpm_display = (
        f"{res['bpm']} BPM (conf: {res['bpm_confidence']:.2f})"
        if res["bpm"] is not None
        else "None (non-rhythmic)"
    )
    key_display = (
        f"{res['key']} (conf: {res['key_confidence']:.2f})"
        if res["key"] != "Unknown"
        else "Unknown (non-tonal)"
    )

    print("\n--- Audio Overview ---")
    print(f"  File:            {res['filename']}")
    print(f"  Category:        {res['category']}")
    print(f"  Sample Rate:     {res['sample_rate']} Hz")
    print(f"  Duration:        {res['metrics']['duration_sec']} s")

    print("\n--- Musical Analysis ---")
    print(f"  Key:             {key_display}")
    print(f"  BPM:             {bpm_display}")
    print(f"  Rhythmic:        {'Yes' if res['is_rhythmic'] else 'No'}")

    print("\n--- Dynamics & Signal ---")
    print(f"  Peak:            {res['metrics']['peak_db']} dBFS")
    print(f"  RMS:             {res['metrics']['rms_db']} dBFS")
    print(f"  Crest Factor:    {res['metrics']['crest_factor_db']} dB")
    print(f"  ZCR:             {res['metrics']['zcr']}")
    print(f"  Centroid:        {res['metrics'].get('spectral_centroid', 0.0)} Hz")
    print(f"  Low/High Energy: {res['metrics'].get('low_energy_ratio', 0.0)} / {res['metrics'].get('high_energy_ratio', 0.0)}\n")


def handle_scan(args: argparse.Namespace):
    """Scan a folder and display summary table."""
    if not os.path.isdir(args.directory):
        print(f"[Error] Directory not found: '{args.directory}'")
        sys.exit(1)

    print(f"\nScanning directory: '{args.directory}'...")
    try:
        results = scan_directory(args.directory)
    except Exception as exc:
        print(f"[Error] Scan failed: {exc}")
        sys.exit(1)

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"Scanned {len(results)} files ({len(valid)} analyzed, {len(errors)} errors).\n")

    if not valid:
        print("No valid audio samples found.")
        return

    headers = ["Filename", "Category", "Key", "BPM", "Duration", "Peak (dB)"]
    rows = []
    for r in valid[:40]:
        bpm_str = f"{int(round(r['bpm']))}" if r.get("bpm") is not None else "-"
        key_str = r.get("key", "Unknown")
        rows.append([
            r["filename"][:25],
            r["category"],
            key_str,
            bpm_str,
            f"{r['metrics']['duration_sec']}s",
            f"{r['metrics']['peak_db']}",
        ])

    print(format_table(rows, headers))
    if len(valid) > 40:
        print(f"\n... and {len(valid) - 40} more samples.")

    if errors:
        print(f"\n[Warnings] {len(errors)} file(s) could not be read:")
        for err in errors[:5]:
            print(f"  - {os.path.basename(err.get('file_path', 'unknown'))}: {err.get('error', 'unknown error')}")


def handle_organize(args: argparse.Namespace):
    """Sort samples into target directory."""
    if not os.path.isdir(args.source):
        print(f"[Error] Source directory not found: '{args.source}'")
        sys.exit(1)

    print(f"\nScanning '{args.source}' for organization...")
    # Exclude output dir if it resides inside source
    results = scan_directory(args.source, exclude_dirs=[args.destination])
    valid_count = sum(1 for r in results if "error" not in r)

    print(f"Found {valid_count} samples to organize (Mode: {args.mode}). Destination: '{args.destination}'")

    summary = organize_samples(results, args.destination, mode=args.mode)

    if args.mode == "dry-run":
        print(f"\n[Dry Run] Planned operations: {summary['planned']} (No files written to disk)")
        for op in summary["operations"][:10]:
            print(f"  [Plan] {os.path.basename(op['source'])} -> {op['destination']}")
        if len(summary["operations"]) > 10:
            print(f"  ... and {len(summary['operations']) - 10} more planned operations.")
    else:
        print(f"\n[Completed] Processed: {summary['processed']} files, Errors: {summary['errors']}")
        if summary["errors"] > 0:
            for op in summary["operations"]:
                if op.get("status") == "error":
                    print(f"  - Error on '{os.path.basename(op['source'])}': {op.get('error_msg')}")


def main():
    parser = argparse.ArgumentParser(description="Sample Organizer: Audio sample analysis & library organization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze acoustic features of an audio file")
    p_analyze.add_argument("file", help="Path to audio file (WAV, AIFF, FLAC)")
    p_analyze.set_defaults(func=handle_analyze)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan a directory and display audio sample manifest")
    p_scan.add_argument("directory", help="Path to sample library folder")
    p_scan.set_defaults(func=handle_scan)

    # organize
    p_organize = subparsers.add_parser("organize", help="Sort and organize audio samples into structured folders")
    p_organize.add_argument("source", help="Source folder containing unorganized samples")
    p_organize.add_argument("destination", help="Target folder for organized library")
    p_organize.add_argument("--mode", choices=["copy", "move", "dry-run"], default="copy", help="File operation mode")
    p_organize.set_defaults(func=handle_organize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
