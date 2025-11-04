import pytest
import numpy as np
from scripts.anomaly_detection import PipelineAnomalyDetector
from scripts.logger_config import setup_logger

logger = setup_logger(__name__)

class TestAnomalyDetection:
    @pytest.fixture(autouse=True)
    def setup(self, test_config):
        self.anomaly_detector = PipelineAnomalyDetector(
            thresholds=test_config["thresholds"]
        )
    
    def test_file_size_anomalies(self):
        """Test file size anomaly detection"""
        test_sizes = [1000, 5000, 15_000_000, 2000, 3000]
        anomalies = self.anomaly_detector.detect_file_size_anomalies(test_sizes)
        assert len(anomalies) == 1
        assert anomalies[0] == 2  # Index of the oversized file
    
    def test_complexity_anomalies(self):
        """Test code complexity anomaly detection"""
        test_complexities = {
            "file1.py": 5,
            "file2.py": 20,
            "file3.py": 8
        }
        anomalies = self.anomaly_detector.detect_complexity_anomalies(
            test_complexities
        )
        assert len(anomalies) == 1
        assert "file2.py" in anomalies
    
    def test_duplicate_detection(self, sample_code_data):
        """Test duplicate code detection"""
        test_files = {
            "file1.py": sample_code_data["python_sample"],
            "file2.py": sample_code_data["python_sample"],
            "file3.py": sample_code_data["java_sample"]
        }
        duplicates = self.anomaly_detector.detect_duplicates(test_files)
        assert len(duplicates) == 1
        assert ("file1.py", "file2.py") in duplicates
    
    def test_statistical_anomalies(self):
        """Test statistical anomaly detection"""
        data = np.random.normal(0, 1, 1000)
        outliers = np.array([10, -10, 15, -15])
        combined = np.concatenate([data, outliers])
        
        anomalies = self.anomaly_detector.detect_statistical_anomalies(
            combined, threshold=3
        )
        assert len(anomalies) == len(outliers)
    
    def test_pii_pattern_detection(self):
        """Test PII pattern detection"""
        test_content = """
        Here's my email: test@example.com
        API Key: sk_test_12345
        Phone: 123-456-7890
        """
        pii_results = self.anomaly_detector.detect_pii_patterns(test_content)
        assert len(pii_results["detected_patterns"]) == 3
        assert pii_results["risk_level"] == "high"
    
    def test_language_distribution_bias(self):
        """Test programming language distribution bias detection"""
        language_dist = {
            "Python": 450,
            "Java": 280,
            "C++": 180,
            "JavaScript": 90
        }
        bias_results = self.anomaly_detector.analyze_language_bias(language_dist)
        assert bias_results["has_significant_bias"] is False
        assert len(bias_results["recommendations"]) == 0