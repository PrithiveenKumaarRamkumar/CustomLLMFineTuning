import pytest
import os
import tempfile
import json
import hashlib
import boto3
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Global test configuration fixture with comprehensive testing parameters.
    """
    return {
        "test_data_dir": os.path.join(os.path.dirname(__file__), "test_data"),
        "thresholds": {
            "file_size_max": 10_000_000,  # 10MB
            "file_size_min": 100,         # 100 bytes
            "complexity_threshold": 15,
            "doc_ratio_min": 0.05,
            "duplicate_threshold": 0.20,
            "pii_false_positive_rate": 0.03,
            "processing_time_multiplier": 3.0,
            "memory_usage_percent": 85,
        },
        "anomaly_detection": {
            "statistical_threshold": 3.0,  # 3 sigma
            "drift_threshold": 0.1,
            "quality_threshold": 0.95,
        },
        "bias_analysis": {
            "language_distribution": {
                "python": {"min": 0.40, "max": 0.50},
                "java": {"min": 0.25, "max": 0.35},
                "cpp": {"min": 0.15, "max": 0.25},
                "javascript": {"min": 0.05, "max": 0.15},
            },
            "license_diversity_min": 5,
            "high_star_threshold": 0.70,
        }
    }

@pytest.fixture(scope="function")
def temp_workspace():
    """
    Provides a temporary workspace for file operations with cleanup.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create standard directory structure
        data_dirs = ['raw', 'staged', 'processed', 'logs', 'metrics']
        for dir_name in data_dirs:
            os.makedirs(os.path.join(tmpdir, dir_name), exist_ok=True)
        yield tmpdir

@pytest.fixture(scope="session")
def sample_code_data():
    """
    Comprehensive sample code data for testing across languages.
    """
    return {
        "python_sample": """
def calculate_fibonacci(n: int) -> int:
    '''Calculate fibonacci number using dynamic programming.'''
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# Example usage
if __name__ == "__main__":
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")
        """,
        "java_sample": """
/**
 * Simple calculator class for basic arithmetic operations.
 * @author Test Developer
 */
public class Calculator {
    
    /**
     * Adds two integers and returns the result.
     * @param a First integer
     * @param b Second integer
     * @return Sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println("5 + 3 = " + calc.add(5, 3));
    }
}
        """,
        "cpp_sample": """
#include <iostream>
#include <vector>
#include <algorithm>

/**
 * Binary search implementation
 * @param arr Sorted array to search in
 * @param target Value to find
 * @return Index of target or -1 if not found
 */
int binary_search(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}

int main() {
    std::vector<int> nums = {1, 3, 5, 7, 9, 11};
    int index = binary_search(nums, 7);
    std::cout << "Found at index: " << index << std::endl;
    return 0;
}
        """,
        "javascript_sample": """
/**
 * Utility functions for array manipulation
 * @module ArrayUtils
 */

/**
 * Removes duplicates from an array
 * @param {Array} arr Input array
 * @returns {Array} Array with unique elements
 */
function removeDuplicates(arr) {
    return [...new Set(arr)];
}

/**
 * Groups array elements by a key function
 * @param {Array} arr Input array
 * @param {Function} keyFn Function to generate grouping key
 * @returns {Object} Object with grouped elements
 */
function groupBy(arr, keyFn) {
    return arr.reduce((groups, item) => {
        const key = keyFn(item);
        groups[key] = groups[key] || [];
        groups[key].push(item);
        return groups;
    }, {});
}

// Example usage
const numbers = [1, 2, 2, 3, 4, 4, 5];
console.log('Unique numbers:', removeDuplicates(numbers));
        """,
        "malformed_sample": """
def broken_function(
    # Missing closing parenthesis and colon
    print("This is broken code"
    return None  # Unreachable
        """,
        "pii_sample": """
# Sample code with PII patterns for testing
def process_user_data():
    email = "user@example.com"
    api_key = "ghp_1234567890abcdef1234567890abcdef12345678"
    phone = "+1-555-123-4567"
    ip_address = "192.168.1.100"
    ssn = "123-45-6789"
    credit_card = "4111-1111-1111-1111"
    return {"processed": True}
        """
    }

@pytest.fixture(scope="session")
def aws_credentials():
    """
    AWS credential management with environment detection and mock fallback.
    """
    # Check for real AWS credentials
    real_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    real_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if real_access_key and real_secret_key:
        # Validate credentials work
        try:
            # Test basic AWS connectivity
            session = boto3.Session(
                aws_access_key_id=real_access_key,
                aws_secret_access_key=real_secret_key,
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            sts = session.client('sts')
            sts.get_caller_identity()  # This will fail if credentials are invalid
            
            return {
                "aws_access_key_id": real_access_key,
                "aws_secret_access_key": real_secret_key,
                "region": os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                "is_mock": False
            }
        except Exception:
            # Fall back to mock if validation fails
            pass
    
    # Return mock credentials for testing
    return {
        "aws_access_key_id": "test_access_key",
        "aws_secret_access_key": "test_secret_key",
        "region": "us-east-1",
        "is_mock": True
    }

@pytest.fixture(scope="function")
def mock_s3_client():
    """
    Mock S3 client for testing without AWS dependency.
    """
    with patch('boto3.client') as mock_client:
        s3_mock = MagicMock()
        
        # Mock successful download
        s3_mock.download_file.return_value = None
        s3_mock.head_object.return_value = {'ContentLength': 1024}
        s3_mock.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'test/file1.py', 'Size': 1024},
                {'Key': 'test/file2.py', 'Size': 2048}
            ]
        }
        
        mock_client.return_value = s3_mock
        yield s3_mock

@pytest.fixture(scope="function")
def sample_metadata():
    """
    Sample metadata for testing data acquisition.
    """
    return {
        "repository_metadata": [
            {
                "repo_name": "test/repo1",
                "stars": 150,
                "license": "MIT",
                "language": "Python",
                "files": [
                    {
                        "path": "src/main.py",
                        "size": 2048,
                        "sha256": "abc123def456",
                        "content_url": "https://example.com/file1"
                    }
                ]
            },
            {
                "repo_name": "test/repo2", 
                "stars": 50,
                "license": "Apache-2.0",
                "language": "Java",
                "files": [
                    {
                        "path": "src/Main.java",
                        "size": 1536,
                        "sha256": "def456ghi789",
                        "content_url": "https://example.com/file2"
                    }
                ]
            }
        ]
    }

@pytest.fixture(scope="function")
def test_files(temp_workspace):
    """
    Create test files for preprocessing and filtering tests.
    """
    test_files = {}
    sample_data = {
        "valid_python.py": """
def hello_world():
    \"\"\"A simple hello world function.\"\"\"
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
        """,
        "large_file.py": "# " + "x" * 10_000_000,  # 10MB+ file
        "small_file.py": "# Small file",  # Very small file
        "duplicate1.py": "print('duplicate content')",
        "duplicate2.py": "print('duplicate content')",  # Exact duplicate
        "pii_file.py": """
# File with PII
user_email = "john.doe@company.com"
api_key = "ghp_abcdefghijklmnopqrstuvwxyz123456789"
        """,
        "malformed.py": """
def broken_function(
    print("Missing syntax")
        """,
    }
    
    for filename, content in sample_data.items():
        filepath = os.path.join(temp_workspace, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        test_files[filename] = filepath
    
    return test_files

@pytest.fixture(scope="function") 
def anomaly_test_data():
    """
    Generate test data with known anomalies for detection testing.
    """
    np.random.seed(42)  # For reproducible tests
    
    # Normal distribution data
    normal_data = np.random.normal(100, 15, 1000)
    
    # Add anomalies
    anomalies = [200, 300, -50, 500]  # Clear outliers
    data_with_anomalies = np.concatenate([normal_data, anomalies])
    
    return {
        "normal_data": normal_data,
        "data_with_anomalies": data_with_anomalies,
        "expected_anomalies": len(anomalies),
        "file_sizes": [100, 200, 500, 1000, 10_000_000, 50],  # One oversized, one undersized
        "processing_times": [1.0, 1.2, 0.8, 15.0, 1.1],  # One spike
        "duplicate_rates": [0.05, 0.10, 0.15, 0.25, 0.08],  # One above threshold
    }

@pytest.fixture(scope="function")
def bias_test_dataset():
    """
    Generate biased dataset for bias detection testing.
    """
    # Create imbalanced language distribution
    languages = ['python'] * 70 + ['java'] * 20 + ['cpp'] * 8 + ['javascript'] * 2
    
    # Create popularity bias (high star repos dominating)
    stars = [1000] * 80 + [50] * 15 + [10] * 5
    
    # Create license imbalance
    licenses = ['MIT'] * 60 + ['Apache-2.0'] * 25 + ['GPL-3.0'] * 10 + ['BSD-2-Clause'] * 5
    
    return pd.DataFrame({
        'language': languages,
        'stars': stars,
        'license': licenses,
        'file_size': np.random.lognormal(8, 2, len(languages)),
        'complexity': np.random.exponential(5, len(languages))
    })

@pytest.fixture(autouse=True)
def setup_test_logging():
    """
    Setup test-specific logging configuration.
    """
    import logging
    
    # Suppress noisy loggers during tests
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Configure test logger
    logger = logging.getLogger('test_pipeline')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    yield logger