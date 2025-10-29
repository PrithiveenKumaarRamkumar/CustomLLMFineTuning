"""
Basic statistics generation for processed code outputs.

Produces aggregate stats per language and overall: file counts, total bytes,
average size, and a small size histogram.
"""

from pathlib import Path
import argparse
import json
from datetime import datetime


def collect_stats(input_dir: Path) -> dict:
    report = {
        "input_dir": str(input_dir),
        "start_time": datetime.utcnow().isoformat() + "Z",
        "total_files": 0,
        "total_bytes": 0,
        "languages": {},
        "size_histogram": {
            "<=1KB": 0,
            "1-10KB": 0,
            "10-100KB": 0,
            ">100KB": 0,
        },
    }

    if not input_dir.exists():
        report["error"] = f"Input directory not found: {input_dir}"
        report["end_time"] = datetime.utcnow().isoformat() + "Z"
        return report

    for lang_dir in (d for d in input_dir.iterdir() if d.is_dir()):
        lang = lang_dir.name
        files = [f for f in lang_dir.glob("**/*") if f.is_file()]
        lang_bytes = 0

        for fp in files:
            try:
                size = fp.stat().st_size
            except Exception:
                size = 0

            report["total_files"] += 1
            report["total_bytes"] += size
            lang_bytes += size

            if size <= 1024:
                report["size_histogram"]["<=1KB"] += 1
            elif size <= 10 * 1024:
                report["size_histogram"]["1-10KB"] += 1
            elif size <= 100 * 1024:
                report["size_histogram"]["10-100KB"] += 1
            else:
                report["size_histogram"][">100KB"] += 1

        report["languages"][lang] = {
            "files": len(files),
            "bytes": lang_bytes,
            "avg_size_bytes": (lang_bytes / len(files)) if files else 0,
        }

    report["end_time"] = datetime.utcnow().isoformat() + "Z"
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate basic statistics for processed code")
    parser.add_argument("--input", type=Path, default=Path("data/processed/code"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_statistics.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = collect_stats(args.input)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Data statistics written to {args.output}")


if __name__ == "__main__":
    main()
