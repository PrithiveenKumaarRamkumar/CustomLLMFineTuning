"""
Unified Software Heritage (The Stack v2 content) downloader for multiple languages.

- Reads configuration from configs/data_config.yaml when available
- Supports languages: python, java, cpp, javascript
- Loads filtered metadata JSON per language
- Downloads blobs from S3 (softwareheritage/content/<blob_id>)
- Writes to output directory per language with checksum sidecars
- Respects existing files if checksum matches

Usage:
  python scripts/batch_swh_download.py --languages python,java,cpp,javascript
  python scripts/batch_swh_download.py --languages python

Config fallbacks (if config is missing):
- Metadata path: data/filtered_metadata_<lang>.json
- Output path:   data/code_files/<lang>
"""

from __future__ import annotations

import os
import json
import argparse
import gzip
import io
import hashlib
from pathlib import Path
from typing import Dict, List

import boto3
from smart_open import open as smart_open

try:
    import yaml
except Exception:
    yaml = None

from logger_config import setup_logger, get_log_filename


LANG_ALIAS = {
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "c++": "cpp",
    "javascript": "javascript",
    "js": "javascript",
}


def _clean_filename(repo_name: str, path: str) -> str:
    repo = repo_name.replace("/", "_")
    file_path = path.replace("/", "_").replace(" ", "_")
    return f"{repo}_{file_path}"


def _safe_filename(repo_name: str, path: str, extension: str, max_len: int = 180) -> str:
    """Build a safe filename that avoids Windows path length issues and duplicate extensions.

    - Deduplicates extension if it's already present at the end of path
    - Truncates long basenames and appends a short hash to preserve uniqueness
    """
    import hashlib as _hashlib

    base = _clean_filename(repo_name, path)
    # Remove an extra extension if already present
    ext = extension.lstrip('.')
    if base.lower().endswith(f".{ext.lower()}"):
        candidate = f"{base}"
    else:
        candidate = f"{base}.{ext}"

    if len(candidate) <= max_len:
        return candidate

    # Truncate and add hash suffix for uniqueness
    digest = _hashlib.sha1(candidate.encode('utf-8')).hexdigest()[:10]
    # Keep as much of the head as fits
    keep = max_len - (len(digest) + 3)  # for __ and no extra dot
    head = candidate[:keep].rstrip('. ')
    # Ensure we still end with the desired extension
    if head.lower().endswith(f".{ext.lower()}"):
        head = head[: -(len(ext) + 1)]  # strip ext, will re-add below
    return f"{head}__{digest}.{ext}"


def _calculate_file_hash(filepath: Path, algo: str = "sha256") -> str:
    hasher = hashlib.sha256() if algo.lower() == "sha256" else hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_existing_file(filepath: Path) -> bool:
    if not filepath.exists():
        return False
    checksum_file = filepath.with_suffix(filepath.suffix + ".sha256")
    if not checksum_file.exists():
        return False
    stored = checksum_file.read_text(encoding="utf-8").strip()
    current = _calculate_file_hash(filepath)
    return stored == current


def _save_checksum(filepath: Path, checksum: str, logger=None) -> None:
    """Attempt to save a sidecar checksum; fall back to a manifest file if path is too long.

    On Windows, very long paths for the sidecar (original + '.sha256') can exceed limits.
    If writing the sidecar fails, append to a directory-level manifest file instead.
    """
    try:
        checksum_file = filepath.with_suffix(filepath.suffix + ".sha256")
        checksum_file.write_text(checksum, encoding="utf-8")
        return
    except OSError as e:
        if logger:
            logger.warning(f"Failed to write checksum sidecar for {filepath.name}: {e}. Falling back to manifest.")
        # Fallback: write to a manifest file in the same directory
        manifest = filepath.parent / "_checksums.sha256"
        try:
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(f"{checksum}  {filepath.name}\n")
        except Exception as e2:
            if logger:
                logger.error(f"Also failed to write checksum manifest: {e2}")


def _load_config(config_path: Path) -> Dict:
    # Provide safe defaults if YAML or file missing
    default = {
        "languages": ["Python", "Java", "C++", "JavaScript"],
        "output_paths": {"base_dir": "data/raw"},
        "metadata_paths": {},
    }
    if not yaml or not config_path.exists():
        return default
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # Shallow merge
        for k, v in default.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return default


def _metadata_path_for(lang_norm: str, cfg: Dict) -> Path:
    # Prefer explicit metadata_paths from config if present
    meta_cfg = cfg.get("metadata_paths") or {}
    mapping = {
        "python": meta_cfg.get("python", "data/raw/filtered_metadata_python.json"),
        "java": meta_cfg.get("java", "data/raw/filtered_metadata_java.json"),
        "cpp": meta_cfg.get("cpp", "data/raw/filtered_metadata_cpp.json"),
        "javascript": meta_cfg.get("javascript", "data/raw/filtered_metadata_javascript.json"),
    }
    # Legacy fallback used by older scripts
    legacy = f"data/raw/metadata/filtered_metadata_{lang_norm}.json"
    p = Path(mapping.get(lang_norm, legacy))
    return p if p.exists() else Path(legacy)


def _output_dir_for(lang_norm: str, cfg: Dict) -> Path:
    base = cfg.get("output_paths", {}).get("base_dir", "data/code_files")
    return Path(base) / lang_norm


def download_for_language(lang: str, cfg: Dict, logger) -> Dict:
    lang_norm = LANG_ALIAS.get(lang.lower(), None)
    if not lang_norm:
        raise ValueError(f"Unsupported language: {lang}")

    meta_path = _metadata_path_for(lang_norm, cfg)
    out_dir = _output_dir_for(lang_norm, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Language: {lang_norm}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Output:   {out_dir}")

    # Load metadata
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {meta_path}")
        return {"language": lang_norm, "total": 0, "downloaded": 0, "skipped": 0, "errors": 1}
    except Exception as e:
        logger.error(f"Failed to read metadata {meta_path}: {e}")
        return {"language": lang_norm, "total": 0, "downloaded": 0, "skipped": 0, "errors": 1}

    # AWS session
    try:
        session = boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        s3 = session.client("s3")
    except KeyError as e:
        logger.error(f"Missing AWS credential: {e}")
        logger.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment.")
        return {"language": lang_norm, "total": len(metadata), "downloaded": 0, "skipped": 0, "errors": len(metadata)}

    downloaded = 0
    skipped = 0
    errors = 0

    for idx, entry in enumerate(metadata, 1):
        blob_id = entry.get("blob_id")
        if not blob_id:
            errors += 1
            continue
        src_encoding = entry.get("src_encoding", "UTF-8")
        repo_name = entry.get("repo_name", "unknownrepo")
        path = entry.get("path", f"{blob_id}.txt")
        extension = entry.get("extension", "txt")

        filename = _safe_filename(repo_name, path, extension)
        out_file = out_dir / filename

        if _verify_existing_file(out_file):
            skipped += 1
            continue

        s3_url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with smart_open(s3_url, "rb", transport_params={"client": s3}) as fin:
                with gzip.GzipFile(fileobj=io.BytesIO(fin.read())) as gz:
                    content = gz.read().decode(src_encoding, errors="replace")
            out_file.write_text(content, encoding="utf-8")
            checksum = _calculate_file_hash(out_file, "sha256")
            _save_checksum(out_file, checksum, logger)
            downloaded += 1
        except Exception as e:
            logger.error(f"[{idx}/{len(metadata)}] Error downloading {filename}: {e}")
            errors += 1

    return {
        "language": lang_norm,
        "total": len(metadata),
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Download code artifacts for selected languages")
    parser.add_argument("--languages", type=str, help="Comma-separated languages (e.g., python,java,cpp,javascript)")
    parser.add_argument("--config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.languages:
        langs = [LANG_ALIAS.get(x.strip().lower(), x.strip().lower()) for x in args.languages.split(",")]
    else:
        # Use config languages
        langs_cfg = cfg.get("languages") or []
        # Map display names to normalized keys
        name_map = {"Python": "python", "Java": "java", "C++": "cpp", "JavaScript": "javascript"}
        langs = [name_map.get(x, x.lower()) for x in langs_cfg]

    # Logger
    log_file = get_log_filename("data_acquisition_all")
    logger = setup_logger(__name__, log_file)

    logger.info("=" * 70)
    logger.info("Starting Unified Data Acquisition")
    logger.info("=" * 70)

    summaries: List[Dict] = []
    for lang in langs:
        logger.info("-" * 40)
        logger.info(f"Processing language: {lang}")
        result = download_for_language(lang, cfg, logger)
        logger.info(f"Result: {result}")
        summaries.append(result)

    logger.info("=" * 70)
    logger.info("Summary:")
    for r in summaries:
        logger.info(f"{r['language']}: downloaded={r['downloaded']} skipped={r['skipped']} errors={r['errors']} total={r['total']}")
    logger.info("=" * 70)

    print("Unified acquisition complete. See logs for details:", log_file)


if __name__ == "__main__":
    main()
