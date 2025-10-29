import pytest
import os
import json
import yaml
from pathlib import Path


class TestDataAcquisition:
    """Test suite for data acquisition module"""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        assert os.path.exists("requirements.txt"), "requirements.txt not found"
    
    def test_config_file_exists(self):
        """Test that data_config.yaml exists"""
        assert os.path.exists("configs/data_config.yaml"), "data_config.yaml not found"
    
    def test_config_file_loads(self):
        """Test that config file is valid YAML and loads correctly"""
        with open("configs/data_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert config is not None, "Config file is empty"
        assert "languages" in config, "Config missing 'languages' key"
        assert "filters" in config, "Config missing 'filters' key"
        assert "output_paths" in config, "Config missing 'output_paths' key"
    
    def test_languages_configured(self):
        """Test that required languages are configured"""
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
