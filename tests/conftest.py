import pytest
import os
import tempfile
from typing import Dict, Any

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Global test configuration fixture.
    """
    return {
        "test_data_dir": os.path.join(os.path.dirname(__file__), "test_data"),
        "thresholds": {
            "file_size_max": 10_000_000,  # 10MB
            "complexity_threshold": 15,
            "doc_ratio_min": 0.05,
            "duplicate_threshold": 0.20,
        }
    }

@pytest.fixture(scope="function")
def temp_workspace():
    """
    Provides a temporary workspace for file operations.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="session")
def sample_code_data():
    """
    Provides sample code data for testing.
    """
    return {
        "python_sample": """
def hello_world():
    print("Hello, World!")
        """,
        "java_sample": """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
        """
    }

@pytest.fixture(scope="session")
def aws_credentials():
    """
    Mock AWS credentials for testing.
    """
    return {
        "aws_access_key_id": "test_access_key",
        "aws_secret_access_key": "test_secret_key",
        "region": "us-east-1"
    }