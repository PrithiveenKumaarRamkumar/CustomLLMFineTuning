import pytest
import time
import os
import logging
from typing import Dict, Any
from scripts.data_acquisition import DataAcquisition
from scripts.preprocessing import Preprocessor
from scripts.anomaly_detection import AnomalyDetector
from scripts.preprocessing import PIIRemover

# Set up logging
logging.basicConfig(level=logging.INFO)

class TestPipelineIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, aws_credentials, temp_workspace, test_config):
        """Set up test environment before each test."""
        self.test_thresholds = {
            "file_size_max": 10_000_000,
            "complexity_max": 15,
            "pii_patterns": ["password", "api_key", "token"]
        }
        
        self.data_acquisition = DataAcquisition(
            aws_credentials=aws_credentials,
            output_dir=temp_workspace
        )
        self.preprocessor = Preprocessor()
        self.pii_remover = PIIRemover()
        self.anomaly_detector = AnomalyDetector(
            thresholds=self.test_thresholds
        )
    
    def test_end_to_end_pipeline(self, sample_code_data):
        """Test complete data pipeline integration"""
        # Step 1: Data Acquisition
        download_result = self.data_acquisition.download_dataset(
            bucket="test-bucket",
            prefix="test-prefix"
        )
        assert "downloaded_files" in download_result
        assert len(download_result["downloaded_files"]) > 0

        # Step 2: Initial Processing
        processed_files = []
        for file in download_result["downloaded_files"]:
            processed = {
                "content": self.pii_remover.remove_pii(file["content"]),
                "file_path": file.get("path", "test.py"),
                "metadata": {
                    "language": "python",
                    "size": len(file["content"])
                }
            }
            processed_files.append(processed)
        assert len(processed_files) > 0
        
        # Step 3: Anomaly Detection
        anomalies = self.anomaly_detector.analyze_dataset(processed_files)
        assert "file_size_anomalies" in anomalies
        assert "complexity_anomalies" in anomalies
        assert "pii_detections" in anomalies
        
        # Step 4: Bias Analysis
        bias_metrics = self.anomaly_detector.analyze_bias(processed_files)
        assert bias_metrics["language_distribution"] is not None
        assert bias_metrics["complexity_distribution"] is not None
    
    def test_error_propagation(self):
        """Test error handling and propagation through pipeline"""
        # Trigger an error in data acquisition
        with pytest.raises(Exception, match="Invalid bucket"):
            self.data_acquisition.download_dataset(
                bucket="nonexistent-bucket",
                prefix="invalid-prefix"
            )
    
    def test_data_consistency(self, sample_code_data):
        """Test data consistency through pipeline stages"""
        # Process sample data
        processed_data = {
            "content": sample_code_data["python_sample"],
            "file_path": "test.py",
            "metadata": {
                "language": "python",
                "size": len(sample_code_data["python_sample"])
            }
        }

        # Apply preprocessing
        processed_content = processed_data["content"]
        processed_content = self.pii_remover.remove_pii(processed_content)
        processed_data["content"] = processed_content

        # Run anomaly detection
        anomalies = self.anomaly_detector.analyze_single_file(processed_data)
        assert isinstance(anomalies, dict)
        assert "file_size" in anomalies
        assert "complexity" in anomalies
        assert "pii_detected" in anomalies
        
    def test_performance_metrics(self):
        """Test pipeline performance metrics"""
        start_time = time.time()
        
        # Process a sample file
        file_data = {
            "content": "def test():\n    print('hello')",
            "file_path": "test.py",
            "size": 28
        }
        
        # Measure preprocessing time
        preprocess_start = time.time()
        processed = self.pii_remover.remove_pii(file_data["content"])
        file_data["content"] = processed
        preprocess_time = time.time() - preprocess_start
        
        # Measure anomaly detection time
        anomaly_start = time.time()
        self.anomaly_detector.analyze_single_file(file_data)
        anomaly_time = time.time() - anomaly_start
        
        total_time = time.time() - start_time
        
        # Basic performance assertions
        assert preprocess_time < 1.0, "Preprocessing took too long"
        assert anomaly_time < 1.0, "Anomaly detection took too long"
        assert total_time < 2.0, "Total pipeline execution took too long"
        self.anomaly_detector.analyze_single_file(processed)
        anomaly_time = time.time() - anomaly_start
        
        total_time = time.time() - start_time
        
        # Basic performance assertions
        assert preprocess_time < 1.0, "Preprocessing took too long"
        assert anomaly_time < 1.0, "Anomaly detection took too long"
        assert total_time < 2.0, "Total pipeline execution took too long"