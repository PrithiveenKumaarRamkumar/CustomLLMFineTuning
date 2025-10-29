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
import sys
from datetime import datetime
from typing import Dict, Set, Optional

import yaml

# Import monitoring
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring" / "scripts"))
    from monitoring import monitor
    MONITORING_ENABLED = True
except Exception:
    MONITORING_ENABLED = False

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


def _safe_filename(base_name: str, extension: str, max_len: int = 180) -> str:
    """Build a safe filename that avoids Windows path length issues.
    
    - Truncates long basenames and appends a short hash to preserve uniqueness
    - Ensures consistent extension handling
    """
    # Remove extension if already present
    ext = extension.lstrip('.')
    if base_name.lower().endswith(f".{ext.lower()}"):
        candidate = base_name
    else:
        candidate = f"{base_name}.{ext}"
    
    if len(candidate) <= max_len:
        return candidate
    
    # Truncate and add hash suffix for uniqueness
    digest = hashlib.sha1(candidate.encode('utf-8')).hexdigest()[:10]
    keep = max_len - (len(digest) + 3)
    head = candidate[:keep].rstrip('. ')
    if head.lower().endswith(f".{ext.lower()}"):
        head = head[:-(len(ext) + 1)]
    return f"{head}__{digest}.{ext}"


def _save_checksum(filepath: Path, checksum: str) -> None:
    """Attempt to save a sidecar checksum; fall back to a manifest file if path is too long."""
    try:
        checksum_file = filepath.with_suffix(filepath.suffix + ".sha256")
        checksum_file.write_text(checksum + "\n", encoding="utf-8")
        return
    except OSError as e:
        print(f"Warning: Failed to write checksum sidecar for {filepath.name}: {e}. Falling back to manifest.")
        # Fallback: write to a manifest file in the same directory
        manifest = filepath.parent / "_checksums.sha256"
        try:
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(f"{checksum}  {filepath.name}\n")
        except Exception as e2:
            print(f"Error: Also failed to write checksum manifest: {e2}")


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


# Apply monitoring decorator if available
if MONITORING_ENABLED:
    @monitor.track_stage_time("organize")
    def organize_and_filter(input_dir: Path, output_dir: Path, max_size_mb: int, config_path: Optional[Path] = None) -> dict:
        return _organize_and_filter_impl(input_dir, output_dir, max_size_mb, config_path)
else:
    def organize_and_filter(input_dir: Path, output_dir: Path, max_size_mb: int, config_path: Optional[Path] = None) -> dict:
        return _organize_and_filter_impl(input_dir, output_dir, max_size_mb, config_path)


def _organize_and_filter_impl(input_dir: Path, output_dir: Path, max_size_mb: int, config_path: Optional[Path] = None) -> dict:
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

        # Prepare destination path with safe filename
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Use safe filename to avoid Windows path length issues
        safe_name = _safe_filename(src.stem, src.suffix, max_len=180)
        dst = lang_dir / safe_name

        # Handle name collisions by appending a numeric suffix
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            i = 1
            while True:
                alt_name = _safe_filename(f"{src.stem}_{i}", src.suffix, max_len=180)
                alt = lang_dir / alt_name
                if not alt.exists():
                    dst = alt
                    break
                i += 1

        # Copy and checksum
        shutil.copy2(src, dst)
        checksum = sha256_of_file(dst)
        _save_checksum(dst, checksum)

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

    # Set pipeline status to running
    if MONITORING_ENABLED:
        monitor.set_pipeline_status("dataset_filter", 1)
    
    try:
        report = organize_and_filter(args.input, args.output, args.max_size_mb, config_path=args.config)
        
        # Record metrics
        if MONITORING_ENABLED:
            total_bytes = sum(item.get("bytes", 0) for item in report.get("moved", []))
            monitor.record_data_volume("dataset_filter", total_bytes)
            monitor.set_pipeline_status("dataset_filter", 0)
        
        print(json.dumps({k: report[k] for k in ["total_moved", "total_skipped", "output_dir"]}, indent=2))
    except Exception as e:
        if MONITORING_ENABLED:
            monitor.record_error("dataset_filter", type(e).__name__)
            monitor.set_pipeline_status("dataset_filter", -1)
        raise


if __name__ == "__main__":
    main()
