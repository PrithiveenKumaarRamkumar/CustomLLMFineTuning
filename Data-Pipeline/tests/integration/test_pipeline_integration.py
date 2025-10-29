import pytest
from scripts.data_acquisition import DataAcquisition
from scripts.preprocessing import Preprocessor
from scripts.anomaly_detection import AnomalyDetector
from scripts.logger_config import setup_logger

logger = setup_logger(__name__)

class TestPipelineIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, aws_credentials, temp_workspace, test_config):
        self.data_acquisition = DataAcquisition(
            aws_credentials=aws_credentials,
            output_dir=temp_workspace
        )
        self.preprocessor = Preprocessor()
        self.anomaly_detector = AnomalyDetector(
            thresholds=test_config["thresholds"]
        )
    
    def test_end_to_end_pipeline(self, sample_code_data):
        """Test complete data pipeline integration"""
        # Step 1: Data Acquisition
        download_result = self.data_acquisition.download_dataset(
            bucket="test-bucket",
            prefix="test-prefix"
        )
        assert download_result["status"] == "success"
        
        # Step 2: Initial Processing
        processed_files = self.preprocessor.process_files(
            download_result["downloaded_files"]
        )
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
        with pytest.raises(Exception):
            # Trigger an error in data acquisition
            self.data_acquisition.download_dataset(
                bucket="nonexistent-bucket",
                prefix="invalid-prefix"
            )
    
    def test_data_consistency(self, sample_code_data):
        """Test data consistency through pipeline stages"""
        # Process sample data
        processed_data = self.preprocessor.process_content(
            sample_code_data["python_sample"]
        )
        
        # Verify processed data maintains required properties
        assert "content" in processed_data
        assert "metadata" in processed_data
        assert processed_data["metadata"]["language"] == "python"
        
        # Run anomaly detection
        anomalies = self.anomaly_detector.analyze_single_file(processed_data)
        assert isinstance(anomalies, dict)
        assert all(key in anomalies for key in [
            "file_size",
            "complexity",
            "pii_detected"
        ])
    
    def test_performance_metrics(self):
        """Test pipeline performance metrics"""
        # TODO: Implement performance testing
        # This should include:
        # - Processing time measurements
        # - Memory usage tracking
        # - Resource utilization stats
        pass