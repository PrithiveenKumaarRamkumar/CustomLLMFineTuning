import pytest
import os
import tempfile
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Global test configuration fixture.
    """
    logger.info("Creating test config fixture")
    config = {
        "test_data_dir": os.path.join(os.path.dirname(__file__), "test_data"),
        "thresholds": {
            "file_size_max": 10_000_000,  # 10MB
            "complexity_max": 15,
            "pii_patterns": ["password", "api_key", "token"],
            "doc_ratio_min": 0.05,
            "duplicate_threshold": 0.20,
        }
    }
    os.makedirs(config["test_data_dir"], exist_ok=True)
    return config

@pytest.fixture(scope="function")
def aws_credentials() -> Dict[str, str]:
    """
    Mock AWS credentials fixture.
    """
    logger.info("Creating AWS credentials fixture")
    creds = {
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret"
    }
    logger.info("Created AWS credentials")
    return creds

@pytest.fixture(scope="function")
def temp_workspace(test_config):
    """
    Creates a temporary workspace for tests.
    """
    logger.info("Creating temporary workspace")
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temp directory: {temp_dir}")
        yield temp_dir
        logger.info("Cleaning up temp directory")

@pytest.fixture(scope="session")
def sample_code_data() -> Dict[str, str]:
    """
    Sample code snippets for testing.
    """
    logger.info("Creating sample code data fixture")
    return {
        "python_sample": """def hello_world():
    print("Hello, World!")
    # API_KEY = "abc123xyz789"  # Testing PII detection
""",
        "java_sample": """public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        String password = "secret123";  // Testing PII detection
    }
}"""
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