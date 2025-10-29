"""
Basic schema/quality validation for processed code outputs.

Validations:
- Input directory exists and has at least one language folder
- Files are non-empty and below a max size
- Simple PII signatures are not present (emails, ghp_ tokens)

Outputs a JSON report with summary and per-file findings.
Exit code: 0 on pass, 1 on fail.
"""

from pathlib import Path
import argparse
import json
import re
import sys
from datetime import datetime

PII_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PII_GITHUB_PAT = re.compile(r"ghp_[A-Za-z0-9_]{20,}")


def validate_directory(input_dir: Path, max_file_size_mb: int = 10) -> dict:
    report = {
        "input_dir": str(input_dir),
        "start_time": datetime.utcnow().isoformat() + "Z",
        "total_files": 0,
        "languages": {},
        "pii_hits": 0,
        "empty_files": 0,
        "oversize_files": 0,
        "status": "pass",
        "errors": []
    }

    if not input_dir.exists():
        report["status"] = "fail"
        report["errors"].append(f"Input directory not found: {input_dir}")
        return report

    lang_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not lang_dirs:
        report["status"] = "fail"
        report["errors"].append("No language subfolders found")
        return report

    max_bytes = max_file_size_mb * 1024 * 1024

    for lang_dir in lang_dirs:
        lang = lang_dir.name
        files = list(f for f in lang_dir.glob("**/*") if f.is_file())
        lang_stats = {"files": 0, "pii_hits": 0, "empty": 0, "oversize": 0}

        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""

            size = len(content.encode("utf-8"))
            report["total_files"] += 1
            lang_stats["files"] += 1

            if size == 0:
                report["empty_files"] += 1
                lang_stats["empty"] += 1

            if size > max_bytes:
                report["oversize_files"] += 1
                lang_stats["oversize"] += 1

            # Simple PII signatures (ignore safe example emails)
            pii = 0
            email_match = PII_EMAIL.search(content)
            if email_match:
                email_str = email_match.group(0).lower()
                # Allow common safe example/demo emails
                safe_domains = ("example.com", "example.org", "test.com", "sample.com", "foo.com", "bar.com", "domain.com")
                if not any(email_str.endswith("@" + d) or email_str.endswith(d) for d in safe_domains):
                    pii += 1
            if PII_GITHUB_PAT.search(content):
                pii += 1
            if pii:
                report["pii_hits"] += pii
                lang_stats["pii_hits"] += pii

        report["languages"][lang] = lang_stats

    # Determine status
    if report["pii_hits"] > 0 or report["empty_files"] > 0 or report["oversize_files"] > 0:
        report["status"] = "fail"

    report["end_time"] = datetime.utcnow().isoformat() + "Z"
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate processed code directory")
    parser.add_argument("--input", type=Path, default=Path("data/processed/code"))
    parser.add_argument("--output", type=Path, default=Path("reports/schema_validation_report.json"))
    parser.add_argument("--max-file-size-mb", type=int, default=10)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = validate_directory(args.input, max_file_size_mb=args.max_file_size_mb)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Schema validation report written to {args.output}")

    # Exit code reflects pass/fail
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
