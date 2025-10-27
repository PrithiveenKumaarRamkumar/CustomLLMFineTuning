"""
Dataset organizer and filter for user-uploaded code.

- Classifies files by extension into language subfolders
- Applies basic size filtering
- Computes SHA-256 checksums
- Writes a simple manifest of moved files
- Optionally reads configuration from configs/data_config.yaml
- Optionally emits Prometheus-style metrics to monitoring/metrics

Supported languages:
    - python: .py
    - java: .java
    - cpp: .cpp, .cc, .cxx
    - javascript: .js, .jsx

Usage:
    python scripts/dataset_filter.py --input data/raw --output data/staged/raw_organized --max-size-mb 10
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from typing import Dict, Set, Optional

import yaml

EXT_TO_LANG = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".js": "javascript",
    ".jsx": "javascript",
}


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_config(config_path: Optional[Path]) -> Dict:
    """Load data acquisition/ingestion config if present."""
    default = {
        "filters": {
            "file_size": {
                "min_bytes": 1,
                "max_bytes": 10 * 1024 * 1024,
            }
        },
        "languages": ["Python", "Java", "C++", "JavaScript"],
        "metrics_output": "monitoring/metrics",
    }
    if not config_path or not config_path.exists():
        return default
    try:
        with config_path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge shallowly
        for k, v in user_cfg.items():
            default[k] = v
        return default
    except Exception:
        return default


def _emit_metrics(metrics_dir: Path, moved: int, skipped: int):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prom = metrics_dir / "dataset_filter.prom"
    lines = [
        f"dataset_filter_files_moved_total {moved}",
        f"dataset_filter_files_skipped_total {skipped}",
        f"dataset_filter_last_run_timestamp {int(datetime.utcnow().timestamp())}",
    ]
    prom.write_text("\n".join(lines) + "\n", encoding="utf-8")


def organize_and_filter(input_dir: Path, output_dir: Path, max_size_mb: int, config_path: Optional[Path] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(config_path)
    # Determine allowed languages set
    allowed_langs: Set[str] = set(l.lower() for l in (cfg.get("languages") or []))
    # If languages missing, allow all supported
    if not allowed_langs:
        allowed_langs = {"python", "java", "c++", "javascript"}

    # Size from CLI takes precedence; else config
    max_bytes = max_size_mb * 1024 * 1024 if max_size_mb else int(
        (cfg.get("filters", {}).get("file_size", {}).get("max_bytes") or (10 * 1024 * 1024))
    )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_time": datetime.utcnow().isoformat() + "Z",
        "moved": [],
        "skipped": [],
        "total_moved": 0,
        "total_skipped": 0,
    }

    # Gather candidate files (any files under input_dir)
    candidates = [p for p in input_dir.rglob("*") if p.is_file()]

    for src in candidates:
        ext = src.suffix.lower()
        lang = EXT_TO_LANG.get(ext)
        if not lang:
            manifest["skipped"].append({"path": str(src), "reason": "unsupported_extension"})
            continue

        # language allowlist from config (handles C++ naming)
        normalized_lang = "c++" if lang == "cpp" else lang
        if allowed_langs and normalized_lang.lower() not in allowed_langs:
            manifest["skipped"].append({"path": str(src), "reason": "language_not_allowed"})
            continue

        try:
            size = src.stat().st_size
        except Exception:
            size = 0

        if size == 0:
            manifest["skipped"].append({"path": str(src), "reason": "empty"})
            continue
        if size > max_bytes:
            manifest["skipped"].append({"path": str(src), "reason": "too_large"})
            continue

        # Prepare destination path (preserve filename only)
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        dst = lang_dir / src.name

        # Handle name collisions by appending a numeric suffix
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            i = 1
            while True:
                alt = lang_dir / f"{stem}_{i}{suffix}"
                if not alt.exists():
                    dst = alt
                    break
                i += 1

        # Copy and checksum
        shutil.copy2(src, dst)
        checksum = sha256_of_file(dst)
        (dst.with_suffix(dst.suffix + ".sha256")).write_text(checksum + "\n", encoding="utf-8")

        manifest["moved"].append({
            "src": str(src),
            "dst": str(dst),
            "lang": lang,
            "bytes": size,
            "sha256": checksum,
        })
        manifest["total_moved"] += 1

    manifest["total_skipped"] = len(manifest["skipped"])
    manifest["end_time"] = datetime.utcnow().isoformat() + "Z"

    # Write manifest inside output_dir
    (output_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Emit metrics if configured
    metrics_dir = Path(cfg.get("metrics_output", "monitoring/metrics"))
    try:
        _emit_metrics(metrics_dir, manifest["total_moved"], manifest["total_skipped"])
    except Exception:
        pass
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Organize and filter user-uploaded dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/staged/raw_organized"))
    parser.add_argument("--max-size-mb", type=int, default=10)
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()

    report = organize_and_filter(args.input, args.output, args.max_size_mb, config_path=args.config)
    print(json.dumps({k: report[k] for k in ["total_moved", "total_skipped", "output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
