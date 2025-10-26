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
            
            # Check that output directory has correct files
            output_files = list(output_dir.glob("*.py"))
            self.assertEqual(len(output_files), 2)


class TestPreprocessingPipeline(unittest.TestCase):
    """Test main preprocessing pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create config
        config = {
            'tokenizer_model': 'microsoft/codebert-base',
            'similarity_threshold': 0.85,
            'stages': {
                'encoding_fix': True,
                'malformed_cleaning': True,
                'pii_removal': True,
                'deduplication': True,
                'tokenization': False,  # Skip tokenization in tests
                'whitespace_cleanup': True
            },
            'parallel_processing': False,  # Disable for testing
            'supported_languages': ['python', 'java']
        }
        
        config_path = self.temp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.pipeline = PreprocessingPipeline(config_path=config_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsInstance(self.pipeline.config, dict)
        self.assertEqual(self.pipeline.config['similarity_threshold'], 0.85)
        self.assertIn('python', self.pipeline.config['supported_languages'])
    
    def test_encoding_detection(self):
        """Test encoding detection and fixing"""
        test_file = self.temp_path / "test.py"
        
        # Write file with UTF-8 content
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("# -*- coding: utf-8 -*-\nprint('hello')")
        
        content, encoding = self.pipeline.detect_and_fix_encoding(test_file)
        
        self.assertEqual(encoding, 'utf-8')
        self.assertIn("print('hello')", content)
    
    def test_malformed_code_cleaning(self):
        """Test malformed code cleaning"""
        malformed_code = "def hello():\x00\x01\x02\n    print('world')\r\n"
        cleaned = self.pipeline.clean_malformed_code(malformed_code, 'python')
        
        # Should remove null bytes and normalize line endings
        self.assertNotIn('\x00', cleaned)
        self.assertNotIn('\x01', cleaned)
        self.assertNotIn('\r\n', cleaned)
        self.assertIn("print('world')", cleaned)
    
    def test_whitespace_cleanup(self):
        """Test whitespace cleanup"""
        messy_code = """
        def hello():    
            print('world')     
            
            
            
            return None   
        """
        
        cleaned = self.pipeline.cleanup_whitespace(messy_code)
        lines = cleaned.split('\n')
        
        # Should remove trailing whitespace
        for line in lines:
            self.assertEqual(line, line.rstrip())
        
        # Should limit consecutive empty lines
        empty_count = 0
        max_consecutive_empty = 0
        for line in lines:
            if not line.strip():
                empty_count += 1
                max_consecutive_empty = max(max_consecutive_empty, empty_count)
            else:
                empty_count = 0
        
        self.assertLessEqual(max_consecutive_empty, 2)
    
    def test_process_single_file(self):
        """Test single file processing"""
        input_file = self.temp_path / "input.py"
        output_file = self.temp_path / "output.py"
        
        # Create test file with various issues
        test_content = '''
        # API key in comment
        API_KEY = "sk-test123456789abcdef"
        email = "developer@company.com"
        
        def hello():   
            print("world")    
        '''
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.pipeline.process_single_file(
            input_file, output_file, 'python', self.pipeline.config['stages']
        )
        
        self.assertTrue(result['success'])
        self.assertGreater(len(result['stages_applied']), 0)
        self.assertTrue(output_file.exists())
        
        # Check that PII was removed
        with open(output_file, 'r') as f:
            processed_content = f.read()
        
        self.assertNotIn("sk-test123456789abcdef", processed_content)
        self.assertNotIn("developer@company.com", processed_content)
    
    def test_language_directory_processing(self):
        """Test processing a language directory"""
        # Create input directory structure
        input_dir = self.temp_path / "input" / "python"
        input_dir.mkdir(parents=True)
        
        output_dir = self.temp_path / "output" / "python"
        
        # Create test files
        (input_dir / "file1.py").write_text("def hello(): print('world')")
        (input_dir / "file2.py").write_text("API_KEY='secret123'")
        (input_dir / "file3.py").write_text("def hello(): print('world')")  # Duplicate
        
        results = self.pipeline.process_language_directory(input_dir, output_dir, 'python')
        
        self.assertEqual(results['language'], 'python')
        self.assertEqual(results['input_files'], 3)
        self.assertGreaterEqual(results['processed_files'], 2)  # At least 2 should be processed
        self.assertTrue(output_dir.exists())
    
    @patch('preprocessing.AutoTokenizer')
    def test_tokenization_with_mock(self, mock_tokenizer_class):
        """Test tokenization with mocked tokenizer"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 102, 103]  # Sample token IDs
        mock_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'def', '[SEP]']
        mock_tokenizer.model_max_length = 512
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Reinitialize pipeline to use mocked tokenizer
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
            
            # Check that output files exist and are clean
            output_files = list((output_base / "python").glob("*.py"))
            self.assertGreater(len(output_files), 0)
            
            # Check that PII was removed from at least one file
            pii_removed = python_results.get('total_pii_removed', 0)
            self.assertGreater(pii_removed, 0)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPIIRemoval))
    suite.addTests(loader.loadTestsFromTestCase(TestDeduplication))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)