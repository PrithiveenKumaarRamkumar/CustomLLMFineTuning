import pytest
import os
import json
import yaml
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from data_acquisition import DataAcquisition, HuggingFaceDatasetFilter
    from batch_swh_download import SWHDownloader, FileDownloadManager  
    from dataset_filter import DatasetOrganizer, LanguageClassifier
except ImportError as e:
    # Create mock classes if modules don't exist yet
    class DataAcquisition:
        def __init__(self, config): pass
        def fetch_metadata(self): return []
    class HuggingFaceDatasetFilter:
        def __init__(self, config): pass
        def apply_filters(self, data): return data
    class SWHDownloader:
        def __init__(self, config): pass
        def download_file(self, url, path): return True
    class FileDownloadManager:
        def __init__(self, config): pass
        def batch_download(self, files): return []
    class DatasetOrganizer:
        def __init__(self, config): pass
        def organize_files(self, files): return {}
    class LanguageClassifier:
        def __init__(self): pass
        def classify_file(self, filepath): return "python"

@pytest.mark.unit
class TestDataAcquisition:
    """Comprehensive test suite for data acquisition module - 16 tests"""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        requirements_path = os.path.join(os.path.dirname(__file__), '..', '..', 'requirements.txt')
        assert os.path.exists(requirements_path), "requirements.txt not found"
    
    def test_config_file_exists(self):
        """Test that data_config.yaml exists"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data_config.yaml')
        assert os.path.exists(config_path), "data_config.yaml not found"
    
    def test_config_file_loads(self):
        """Test that config file is valid YAML and loads correctly"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data_config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None, "Config file is empty"
        assert "languages" in config, "Config missing 'languages' key"
        assert "filters" in config, "Config missing 'filters' key"
        assert "output_paths" in config, "Config missing 'output_paths' key"
    
    def test_languages_configured(self):
        """Test that required languages are configured"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data_config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        required_languages = ['python', 'java', 'javascript', 'cpp']
        configured_languages = config.get('languages', [])
        
        for lang in required_languages:
            assert lang in configured_languages, f"Required language '{lang}' not configured"
    
    def test_filter_configuration_valid(self):
        """Test that filter configuration has valid values"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data_config.yaml')
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        filters = config.get('filters', {})
        
        # Test star filters
        if 'stars' in filters:
            assert 'min' in filters['stars'], "Star filter missing min value"
            assert filters['stars']['min'] >= 0, "Star min value should be non-negative"
        
        # Test file size filters  
        if 'file_size' in filters:
            assert 'min_bytes' in filters['file_size'], "File size filter missing min_bytes"
            assert 'max_bytes' in filters['file_size'], "File size filter missing max_bytes"
            assert filters['file_size']['min_bytes'] > 0, "Min file size should be positive"
            assert filters['file_size']['max_bytes'] > filters['file_size']['min_bytes'], \
                "Max file size should be greater than min"
    
    @patch('boto3.client')
    def test_data_acquisition_initialization(self, mock_boto):
        """Test DataAcquisition class initialization"""
        config = {
            'aws': {'region': 'us-east-1'},
            'datasets': {'huggingface_name': 'bigcode/the-stack-v2'},
            'filters': {'stars': {'min': 10}}
        }
        
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        acquirer = DataAcquisition(config)
        assert acquirer.config == config
        
    def test_huggingface_filter_initialization(self):
        """Test HuggingFaceDatasetFilter initialization"""
        config = {
            'filters': {
                'stars': {'min': 10, 'max': 1000},
                'languages': ['python', 'java'],
                'licenses': ['mit', 'apache-2.0']
            }
        }
        
        filter_obj = HuggingFaceDatasetFilter(config)
        assert filter_obj.config == config
    
    def test_metadata_filtering_stars(self, sample_metadata):
        """Test metadata filtering by star count"""
        config = {'filters': {'stars': {'min': 100}}}
        filter_obj = HuggingFaceDatasetFilter(config)
        
        # Test with sample metadata
        test_repos = [
            {'repo_name': 'test/high_stars', 'stars': 150},
            {'repo_name': 'test/low_stars', 'stars': 50}
        ]
        
        filtered = filter_obj.apply_filters(test_repos)
        
        # Should only keep high star repo
        assert len(filtered) == 1
        assert filtered[0]['stars'] >= 100
    
    def test_metadata_filtering_language(self, sample_metadata):
        """Test metadata filtering by programming language"""
        config = {'filters': {'languages': ['python', 'java']}}
        filter_obj = HuggingFaceDatasetFilter(config)
        
        test_repos = [
            {'repo_name': 'test/python_repo', 'language': 'python'},
            {'repo_name': 'test/java_repo', 'language': 'java'},
            {'repo_name': 'test/cpp_repo', 'language': 'cpp'}
        ]
        
        filtered = filter_obj.apply_filters(test_repos)
        
        # Should only keep python and java repos
        assert len(filtered) == 2
        languages = [repo['language'] for repo in filtered]
        assert 'python' in languages
        assert 'java' in languages
        assert 'cpp' not in languages
    
    @patch('requests.get')
    def test_sha256_validation(self, mock_get, temp_workspace):
        """Test SHA256 checksum validation during download"""
        # Create test file content
        test_content = b"print('Hello, World!')"
        expected_sha256 = hashlib.sha256(test_content).hexdigest()
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = test_content
        mock_get.return_value = mock_response
        
        downloader = SWHDownloader({'checksums_enabled': True})
        
        # Test download with correct checksum
        output_path = os.path.join(temp_workspace, 'test_file.py')
        result = downloader.download_file(
            'https://example.com/file.py', 
            output_path,
            expected_sha256=expected_sha256
        )
        
        assert result == True
        assert os.path.exists(output_path)
        
        # Verify file content
        with open(output_path, 'rb') as f:
            assert f.read() == test_content
    
    def test_sha256_validation_failure(self, temp_workspace):
        """Test SHA256 validation failure handling"""
        with patch('requests.get') as mock_get:
            # Create mismatched content
            actual_content = b"print('Hello, World!')"
            wrong_sha256 = hashlib.sha256(b"different content").hexdigest()
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = actual_content
            mock_get.return_value = mock_response
            
            downloader = SWHDownloader({'checksums_enabled': True})
            
            output_path = os.path.join(temp_workspace, 'test_file.py')
            result = downloader.download_file(
                'https://example.com/file.py',
                output_path, 
                expected_sha256=wrong_sha256
            )
            
            # Should fail due to checksum mismatch
            assert result == False
            assert not os.path.exists(output_path)
    
    @patch('boto3.client')
    def test_aws_s3_integration(self, mock_boto):
        """Test AWS S3 integration for file downloads"""
        # Mock S3 client and operations
        mock_s3 = MagicMock()
        mock_s3.download_file.return_value = None
        mock_s3.head_object.return_value = {'ContentLength': 1024}
        mock_boto.return_value = mock_s3
        
        config = {
            'aws': {'region': 'us-east-1', 'bucket': 'test-bucket'},
            's3_enabled': True
        }
        
        downloader = SWHDownloader(config)
        
        # Test S3 download
        result = downloader.download_from_s3('test-key', '/tmp/test-file')
        
        # Verify S3 client was called
        mock_boto.assert_called_with('s3', region_name='us-east-1')
        mock_s3.download_file.assert_called_once()
    
    def test_file_size_filtering(self, temp_workspace, test_files):
        """Test filtering files by size constraints"""
        organizer = DatasetOrganizer({
            'filters': {
                'file_size': {
                    'min_bytes': 100,
                    'max_bytes': 10000
                }
            }
        })
        
        # Create files of different sizes
        small_file = os.path.join(temp_workspace, 'small.py')
        with open(small_file, 'w') as f:
            f.write('# Small file')  # < 100 bytes
        
        normal_file = os.path.join(temp_workspace, 'normal.py')
        with open(normal_file, 'w') as f:
            f.write('# Normal file\n' + 'print("hello")\n' * 20)  # ~300 bytes
            
        large_file = os.path.join(temp_workspace, 'large.py')
        with open(large_file, 'w') as f:
            f.write('# Large file\n' + 'x = 1\n' * 1000)  # > 10KB
        
        files = [small_file, normal_file, large_file]
        filtered_files = organizer.filter_by_size(files)
        
        # Should only keep normal file
        assert len(filtered_files) == 1
        assert normal_file in filtered_files
        assert small_file not in filtered_files
        assert large_file not in filtered_files
    
    def test_language_classification(self, temp_workspace):
        """Test programming language classification by file extension"""
        classifier = LanguageClassifier()
        
        # Create test files with different extensions
        test_cases = [
            ('test.py', 'python'),
            ('Test.java', 'java'),
            ('main.cpp', 'cpp'),
            ('app.js', 'javascript'),
            ('program.c', 'c'),
            ('main.go', 'go'),
            ('lib.rs', 'rust')
        ]
        
        for filename, expected_lang in test_cases:
            filepath = os.path.join(temp_workspace, filename)
            with open(filepath, 'w') as f:
                f.write('// Test file')
            
            detected_lang = classifier.classify_file(filepath)
            assert detected_lang == expected_lang, f"Failed to classify {filename} as {expected_lang}"
    
    def test_duplicate_url_handling(self):
        """Test handling of duplicate download URLs"""
        download_manager = FileDownloadManager({'skip_duplicates': True})
        
        # Test with duplicate URLs
        files_to_download = [
            {'url': 'https://example.com/file1.py', 'path': '/tmp/file1.py'},
            {'url': 'https://example.com/file2.py', 'path': '/tmp/file2.py'},
            {'url': 'https://example.com/file1.py', 'path': '/tmp/file1_duplicate.py'},  # Duplicate
        ]
        
        unique_urls = download_manager.remove_duplicate_urls(files_to_download)
        
        # Should remove duplicate
        assert len(unique_urls) == 2
        urls = [f['url'] for f in unique_urls]
        assert urls.count('https://example.com/file1.py') == 1
    
    def test_download_error_handling(self, temp_workspace):
        """Test error handling during file downloads"""
        with patch('requests.get') as mock_get:
            # Mock HTTP error
            mock_get.side_effect = Exception("Network error")
            
            downloader = SWHDownloader({'retry_attempts': 2})
            
            output_path = os.path.join(temp_workspace, 'test_file.py')
            result = downloader.download_file('https://example.com/broken', output_path)
            
            # Should handle error gracefully
            assert result == False
            assert not os.path.exists(output_path)
            
            # Should have attempted retries
            assert mock_get.call_count == 2
        with open("configs/data_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        languages = config['languages']
        required_languages = ["Python", "Java", "C++", "JavaScript"]
        
        for lang in required_languages:
            assert lang in languages, f"{lang} not in configured languages"
    
    def test_output_directories_exist(self):
        """Test that output directories exist for all languages"""
        languages = ["python", "java", "cpp", "javascript"]
        
        for lang in languages:
            path = f"data/code_files/{lang}"
            assert os.path.exists(path), f"Output directory for {lang} not found: {path}"
    
    def test_metadata_files_exist(self):
        """Test that filtered metadata files exist for all languages"""
        languages = ["python", "java", "cpp", "javascript"]
        
        for lang in languages:
            path = f"data/filtered_metadata_{lang}.json"
            assert os.path.exists(path), f"Metadata file for {lang} not found: {path}"
    
    def test_metadata_files_valid_json(self):
        """Test that metadata files contain valid JSON"""
        languages = ["python", "java", "cpp", "javascript"]
        
        for lang in languages:
            path = f"data/filtered_metadata_{lang}.json"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                assert isinstance(data, list), f"Metadata for {lang} is not a list"
                assert len(data) > 0, f"Metadata for {lang} is empty"
    
    def test_metadata_entries_have_required_fields(self):
        """Test that metadata entries have required fields"""
        path = "data/filtered_metadata_python.json"
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if len(data) > 0:
                first_entry = data[0]
                required_fields = ["blob_id", "repo_name", "path", "language", "src_encoding"]
                
                for field in required_fields:
                    assert field in first_entry, f"Missing required field: {field}"
    
    def test_downloaded_files_exist(self):
        """Test that at least some code files were downloaded"""
        languages = ["python", "java", "cpp", "javascript"]
        
        for lang in languages:
            path = f"data/code_files/{lang}"
            if os.path.exists(path):
                files = list(Path(path).glob("*"))
                assert len(files) > 0, f"No code files found for {lang}"
    
    def test_downloaded_files_not_empty(self):
        """Test that downloaded files are not empty"""
        path = "data/code_files/python"
        
        if os.path.exists(path):
            files = list(Path(path).glob("*"))
            if len(files) > 0:
                # Check first file
                first_file = files[0]
                size = os.path.getsize(first_file)
                assert size > 0, f"Downloaded file is empty: {first_file}"
    
    def test_filter_criteria_in_config(self):
        """Test that filter criteria are properly configured"""
        with open("configs/data_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        filters = config['filters']
        
        assert 'stars' in filters, "Missing stars filter"
        assert 'file_size' in filters, "Missing file_size filter"
        assert 'licenses' in filters, "Missing licenses filter"
        
        assert filters['stars']['min'] >= 0, "Invalid min stars value"
        assert filters['file_size']['min_bytes'] > 0, "Invalid min file size"
        assert filters['file_size']['max_bytes'] > filters['file_size']['min_bytes'], \
            "Max file size must be greater than min"


class TestRequirements:
    """Test suite for requirements and dependencies"""
    
    def test_required_packages_in_requirements(self):
        """Test that critical packages are in requirements.txt"""
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        required_packages = ["datasets", "boto3", "smart-open", "pyyaml", "pytest"]
        
        for package in required_packages:
            assert package.lower() in content.lower(), \
                f"Required package '{package}' not found in requirements.txt"
