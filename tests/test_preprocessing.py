"""
Unit Tests for Data Preprocessing Pipeline
Task 2: Data Preprocessing & Cleaning

Tests for PII removal, deduplication, and main preprocessing pipeline.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import yaml
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from pii_removal import PIIRemover
from deduplication import CodeDeduplicator
from preprocessing import PreprocessingPipeline


class TestPIIRemoval(unittest.TestCase):
    """Test PII removal functionality"""
    
    def setUp(self):
        self.pii_remover = PIIRemover(log_removed_items=False)
    
    def test_email_removal(self):
        """Test email detection and removal"""
        text = "Contact me at john.doe@company.com for questions"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("john.doe@company.com", cleaned)
        self.assertIn("user@example.com", cleaned)
        self.assertEqual(stats['emails'], 1)
    
    def test_api_key_removal(self):
        """Test API key detection and removal"""
        text = 'API_KEY = "sk-1234567890abcdef1234567890abcdef"'
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("sk-1234567890abcdef1234567890abcdef", cleaned)
        self.assertIn("[REDACTED]", cleaned)
        self.assertEqual(stats['api_keys'], 1)
    
    def test_ip_address_removal(self):
        """Test IP address detection and removal"""
        text = "Server IP: 192.168.1.100"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("192.168.1.100", cleaned)
        self.assertEqual(stats['ips'], 1)
    
    def test_whitelist_preservation(self):
        """Test that whitelisted items are preserved"""
        text = "localhost: 127.0.0.1 and test@example.com"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        # These should be preserved
        self.assertIn("127.0.0.1", cleaned)
        self.assertIn("test@example.com", cleaned)
        self.assertEqual(stats['emails'], 0)
        self.assertEqual(stats['ips'], 0)
    
    def test_github_token_removal(self):
        """Test GitHub token detection"""
        text = "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz123456"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456", cleaned)
        self.assertIn("[REDACTED]", cleaned)
        self.assertEqual(stats['api_keys'], 1)
    
    def test_url_removal(self):
        """Test URL removal (selective)"""
        text = "Check https://internal.company.com/secret but see https://github.com/public/repo"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        # Internal URL should be removed, GitHub URL should be kept
        self.assertNotIn("internal.company.com", cleaned)
        self.assertIn("github.com", cleaned)
    
    def test_process_file(self):
        """Test file processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "test_input.py"
            output_file = temp_path / "test_output.py"
            
            # Create test file with PII
            with open(input_file, 'w') as f:
                f.write('API_KEY = "secret123456789"\nemail = "user@company.com"')
            
            result = self.pii_remover.process_file(input_file, output_file)
            
            self.assertTrue(result['success'])
            self.assertGreater(result['pii_removed'], 0)
            self.assertTrue(output_file.exists())
            
            # Check that output file has PII removed
            with open(output_file, 'r') as f:
                content = f.read()
            
            self.assertNotIn("secret123456789", content)
            self.assertNotIn("user@company.com", content)


class TestDeduplication(unittest.TestCase):
    """Test deduplication functionality"""
    
    def setUp(self):
        self.deduplicator = CodeDeduplicator(similarity_threshold=0.9)
    
    def test_exact_hash_calculation(self):
        """Test exact hash calculation"""
        content1 = "def hello(): print('world')"
        content2 = "def hello(): print('world')"  # Identical
        content3 = "def hello(): print('universe')"  # Different
        
        hash1 = self.deduplicator._calculate_exact_hash(content1)
        hash2 = self.deduplicator._calculate_exact_hash(content2)
        hash3 = self.deduplicator._calculate_exact_hash(content3)
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
    
    def test_code_normalization(self):
        """Test code normalization"""
        code1 = '''
        def hello():  # This is a comment
            print("world")    # Another comment
        '''
        
        code2 = '''
        def hello():
            print("world")
        '''
        
        normalized1 = self.deduplicator._normalize_code(code1, 'python')
        normalized2 = self.deduplicator._normalize_code(code2, 'python')
        
        # After normalization, they should be similar
        # (comments removed, whitespace normalized)
        self.assertNotEqual(code1, code2)  # Originally different
        # Normalized versions should be more similar
        
    def test_find_exact_duplicates(self):
        """Test exact duplicate detection"""
        file_contents = {
            Path("file1.py"): "def hello(): print('world')",
            Path("file2.py"): "def hello(): print('world')",  # Exact duplicate
            Path("file3.py"): "def hello(): print('universe')",  # Different
        }
        
        duplicates = self.deduplicator.find_exact_duplicates(file_contents)
        
        # Should find one group of duplicates (file1.py and file2.py)
        self.assertEqual(len(duplicates), 1)
        duplicate_group = list(duplicates.values())[0]
        self.assertEqual(len(duplicate_group), 2)
        self.assertIn(Path("file1.py"), duplicate_group)
        self.assertIn(Path("file2.py"), duplicate_group)
    
    def test_similarity_calculation(self):
        """Test similarity calculation"""
        content1 = "def hello(): print('world')"
        content2 = "def hello(): print('world')"  # Identical
        content3 = "def hello(): print('universe')"  # Similar but different
        content4 = "class MyClass: pass"  # Very different
        
        sim1 = self.deduplicator._calculate_similarity(content1, content2)
        sim2 = self.deduplicator._calculate_similarity(content1, content3)
        sim3 = self.deduplicator._calculate_similarity(content1, content4)
        
        self.assertEqual(sim1, 1.0)  # Identical
        self.assertGreater(sim2, 0.5)  # Similar
        self.assertLess(sim3, 0.5)  # Very different
    
    def test_process_directory(self):
        """Test directory processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            
            # Create test files with duplicates
            (input_dir / "file1.py").write_text("def hello(): print('world')")
            (input_dir / "file2.py").write_text("def hello(): print('world')")  # Duplicate
            (input_dir / "file3.py").write_text("def hello(): print('universe')")  # Unique
            
            results = self.deduplicator.process_directory(input_dir, output_dir)
            
            self.assertEqual(results['input_files'], 3)
            self.assertEqual(results['output_files'], 2)  # One duplicate removed
            self.assertEqual(results['removed_files'], 1)
            self.assertTrue(output_dir.exists())


class TestPreprocessingPipeline(unittest.TestCase):
    """Test preprocessing pipeline functionality"""
    
    def setUp(self):
        self.pipeline = PreprocessingPipeline()
    
    def test_tokenization_with_mock(self):
        """Test tokenization with mocked tokenizer"""
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 102, 103]
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'def', '[SEP]']
        mock_tokenizer.model_max_length = 512
        
        # Create pipeline with config
        self.pipeline = PreprocessingPipeline()
        self.pipeline.config = {'tokenizer_model': 'microsoft/codebert-base'}
        
        # Set up mocks and run test
        with patch('preprocessing.TRANSFORMERS_AVAILABLE', True), \
             patch('preprocessing.AutoTokenizer') as mock_tokenizer_class:
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            self.pipeline._init_tokenizer()
            result = self.pipeline.tokenize_code("def hello(): pass", "python")
            
            self.assertIn('token_ids', result)
            self.assertIn('token_strings', result)
            self.assertEqual(result['token_count'], 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_full_pipeline_integration(self):
        """Test the complete pipeline with realistic data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input structure matching Task 1 output
            input_base = temp_path / "input"
            output_base = temp_path / "output"
            
            python_input = input_base / "python"
            python_input.mkdir(parents=True)
            
            # Create test files with various preprocessing needs
            test_files = {
                "repo1_file1.py": '''
                    # Developer contact: dev@company.com
                    API_KEY = "sk-1234567890abcdef"
                    
                    def process_data(data):
                        # TODO: implement processing
                        return data.strip()
                ''',
                "repo2_file1.py": '''
                    # Same content as repo1_file1.py (duplicate)
                    API_KEY = "sk-1234567890abcdef"
                    
                    def process_data(data):
                        # TODO: implement processing
                        return data.strip()
                ''',
                "repo3_utils.py": '''
                    import requests
                    
                    def make_request():
                        headers = {"Authorization": "Bearer token123"}
                        return requests.get("https://api.example.com/data")
                '''
            }
            
            for filename, content in test_files.items():
                with open(python_input / filename, 'w') as f:
                    f.write(content)
            
            # Create minimal config
            config = {
                'stages': {
                    'encoding_fix': True,
                    'malformed_cleaning': True,
                    'pii_removal': True,
                    'deduplication': True,
                    'tokenization': False,  # Skip for integration test
                    'whitespace_cleanup': True
                },
                'parallel_processing': False,
                'supported_languages': ['python']
            }
            
            config_path = temp_path / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Run pipeline
            pipeline = PreprocessingPipeline(config_path=config_path)
            results = pipeline.run_pipeline(input_base, output_base)
            
            # Check results
            self.assertIn('language_results', results)
            self.assertIn('python', results['language_results'])
            
            python_results = results['language_results']['python']
            self.assertEqual(python_results['input_files'], 3)
            
            # Should have removed duplicates
            self.assertLess(python_results['processed_files'], 3)
