import tempfile
from pathlib import Path
import shutil
import sys

# Ensure scripts/ is importable
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from dataset_filter import organize_and_filter


def test_organize_and_filter_basic(tmp_path: Path):
    # Arrange: create raw files of multiple types
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.py").write_text("print('hi')\n", encoding="utf-8")
    (raw / "b.java").write_text("class B{}\n", encoding="utf-8")
    (raw / "c.txt").write_text("ignore me\n", encoding="utf-8")

    out = tmp_path / "staged"

    # Act
    report = organize_and_filter(raw, out, max_size_mb=10, config_path=None)

    # Assert
    assert report["total_moved"] == 2
    assert report["total_skipped"] >= 1

    py_file = out / "python" / "a.py"
    java_file = out / "java" / "b.java"

    assert py_file.exists()
    assert java_file.exists()

    # Check checksum sidecar exists
    assert (py_file.with_suffix(py_file.suffix + ".sha256")).exists()
    assert (java_file.with_suffix(java_file.suffix + ".sha256")).exists()
