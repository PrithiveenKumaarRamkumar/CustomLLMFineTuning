import pytest
from scripts.data_acquisition import DataAcquisition
from scripts.logger_config import setup_logger

logger = setup_logger(__name__)

class TestDataAcquisition:
    @pytest.fixture(autouse=True)
    def setup(self, aws_credentials, temp_workspace):
        self.data_acquisition = DataAcquisition(
            aws_credentials=aws_credentials,
            output_dir=temp_workspace
        )
    
    def test_aws_credentials_validation(self):
        """Test AWS credentials validation"""
        # Test with valid credentials
        assert self.data_acquisition.validate_credentials() is True
        
        # Test with invalid credentials
        with pytest.raises(ValueError):
            DataAcquisition(aws_credentials=None, output_dir="test")
    
    def test_download_dataset(self, temp_workspace):
        """Test dataset download functionality"""
        result = self.data_acquisition.download_dataset(
            bucket="test-bucket",
            prefix="test-prefix"
        )
        assert result["status"] == "success"
        assert len(result["downloaded_files"]) > 0
    
    def test_metadata_validation(self, sample_code_data):
        """Test metadata validation for downloaded files"""
        validation_result = self.data_acquisition.validate_metadata({
            "file_size": 1024,
            "language": "python",
            "content": sample_code_data["python_sample"]
        })
        assert validation_result["is_valid"] is True
        assert "errors" not in validation_result
    
    def test_file_size_validation(self):
        """Test file size validation"""
        # Test normal file size
        assert self.data_acquisition.validate_file_size(1024) is True
        
        # Test oversized file
        assert self.data_acquisition.validate_file_size(20_000_000) is False
    
    def test_sha256_verification(self, sample_code_data):
        """Test SHA256 hash verification"""
        content = sample_code_data["python_sample"]
        hash_result = self.data_acquisition.verify_file_hash(
            content=content,
            expected_hash="ae2b1fca515949e5d54fb22b8ed95575"
        )
        assert hash_result["verified"] is True
    
    def test_error_handling(self):
        """Test error handling during data acquisition"""
        with pytest.raises(Exception):
            self.data_acquisition.download_dataset(
                bucket="nonexistent-bucket",
                prefix="invalid-prefix"
            )