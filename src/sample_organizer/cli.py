"""Command-line interface for Sample Organizer."""
import os
import sys
import json
import argparse
from typing import List, Optional
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
        print(f"[Error] File not found: {args.file}")
        sys.exit(1)

    print(f"\nAnalyzing: {os.path.basename(args.file)}...")
    res = analyze_file(args.file)

    print("\n--- Audio Overview ---")
    print(f"  File:       {res['filename']}")
    print(f"  Category:   {res['category']}")
    print(f"  Sample Rate:{res['sample_rate']} Hz")
    print(f"  Duration:   {res['metrics']['duration_sec']} s")

    print("\n--- Musical Analysis ---")
    print(f"  Key:        {res['key']} (conf: {res['key_confidence']:.2f})")
    print(f"  BPM:        {res['bpm']} BPM (conf: {res['bpm_confidence']:.2f})")
    print(f"  Rhythmic:   {'Yes' if res['is_rhythmic'] else 'No'}")

    print("\n--- Dynamics & Signal ---")
    print(f"  Peak:       {res['metrics']['peak_db']} dBFS")
    print(f"  RMS:        {res['metrics']['rms_db']} dBFS")
    print(f"  Crest:      {res['metrics']['crest_factor_db']} dB")
    print(f"  ZCR:        {res['metrics']['zcr']}\n")


def handle_scan(args: argparse.Namespace):
    """Scan a folder and display summary table."""
    if not os.path.isdir(args.directory):
        print(f"[Error] Directory not found: {args.directory}")
        sys.exit(1)

    print(f"\nScanning directory: {args.directory}...")
    results = scan_directory(args.directory)

    valid = [r for r in results if "error" not in r]
    print(f"Found {len(valid)} audio samples.\n")

    headers = ["Filename", "Category", "Key", "BPM", "Duration", "Peak (dB)"]
    rows = []
    for r in valid[:30]:  # show up to first 30 in table
        rows.append([
            r["filename"][:25],
            r["category"],
            r["key"],
            f"{int(r['bpm'])}",
            f"{r['metrics']['duration_sec']}s",
            f"{r['metrics']['peak_db']}",
        ])

    print(format_table(rows, headers))
    if len(valid) > 30:
        print(f"\n... and {len(valid) - 30} more samples.")


def handle_organize(args: argparse.Namespace):
    """Sort samples into target directory."""
    print(f"\nScanning '{args.source}' for organization...")
    results = scan_directory(args.source)
    print(f"Sorting {len(results)} samples into '{args.destination}' (Mode: {args.mode})...")

    summary = organize_samples(results, args.destination, mode=args.mode)
    print(f"\nCompleted! Processed: {summary['processed']} files, Errors: {summary['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Sample Organizer: Audio sample analysis & library organization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze acoustic features of a single audio file")
    p_analyze.add_argument("file", help="Path to audio file (.wav)")
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
