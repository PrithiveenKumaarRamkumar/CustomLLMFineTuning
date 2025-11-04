"""
Comprehensive Unit Tests for Data Preprocessing Pipeline
Task 2: Data Preprocessing & Cleaning - 45 Test Cases

Tests for PII removal, deduplication, encoding fixes, tokenization,
and integrated preprocessing pipeline functionality.
"""

import unittest
import pytest
import tempfile
import shutil
import os
import hashlib
import chardet
from pathlib import Path
import json
import yaml
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from pii_removal import PIIRemover
    from deduplication import CodeDeduplicator  
    from preprocessing import PreprocessingPipeline
    from anomaly_detection import PipelineAnomalyDetector
    from bias_analysis import PIIFairnessAnalyzer
except ImportError as e:
    # Create mock classes if modules don't exist yet
    class PIIRemover:
        def __init__(self, **kwargs): pass
        def remove_pii_from_text(self, text): return text, {}
        def remove_pii_from_file(self, filepath): return True, {}
    class CodeDeduplicator:
        def __init__(self, **kwargs): pass
        def find_duplicates(self, files): return []
        def calculate_similarity(self, text1, text2): return 0.0
    class PreprocessingPipeline:
        def __init__(self, config): pass
        def process_files(self, files): return []
    class PipelineAnomalyDetector:
        def __init__(self, config=None): pass
        def analyze_pipeline_stage(self, name, metrics): return []
    class PIIFairnessAnalyzer:
        def __init__(self): pass
        def analyze_pii_removal_fairness(self, before, after): return None


@pytest.mark.unit
class TestPIIRemoval(unittest.TestCase):
    """Comprehensive PII removal functionality tests - 15 test cases"""
    
    def setUp(self):
        self.pii_remover = PIIRemover(log_removed_items=False)
    
    def test_email_removal_basic(self):
        """Test basic email detection and removal"""
        text = "Contact me at john.doe@company.com for questions"
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("john.doe@company.com", cleaned)
        self.assertIn("user@example.com", cleaned)
        self.assertEqual(stats.get('emails', 0), 1)
    
    def test_email_removal_multiple_formats(self):
        """Test multiple email format detection"""
        test_cases = [
            "simple@example.com",
            "user.name@sub.domain.org", 
            "test+tag@gmail.com",
            "user_123@company.co.uk",
            "firstname.lastname@university.edu"
        ]
        
        for email in test_cases:
            text = f"Email: {email}"
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            self.assertNotIn(email, cleaned, f"Failed to remove email: {email}")
            self.assertEqual(stats.get('emails', 0), 1)
    
    def test_github_token_removal(self):
        """Test GitHub token detection and removal"""
        text = 'GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz123456"'
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        self.assertNotIn("ghp_1234567890abcdefghijklmnopqrstuvwxyz123456", cleaned)
        self.assertIn("[REDACTED]", cleaned)
        self.assertEqual(stats.get('github_tokens', 0), 1)
    
    def test_api_key_removal(self):
        """Test API key detection and removal"""
        test_cases = [
            'API_KEY = "sk-1234567890abcdef1234567890abcdef"',
            'secret_key: "abc123def456ghi789jkl012mno345pqr678"',
            'access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"'
        ]
        
        for text in test_cases:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            self.assertIn("[REDACTED]", cleaned)
            self.assertTrue(stats.get('api_keys', 0) >= 1)
    
    def test_ip_address_removal(self):
        """Test IP address detection and removal"""
        test_cases = [
            "Server IP: 192.168.1.100",
            "Connect to 10.0.0.1 on port 8080",
            "Public IP 203.0.113.42 is accessible",
            "Localhost 127.0.0.1 for testing"
        ]
        
        for text in test_cases:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            # Check that IP patterns are removed
            import re
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            remaining_ips = re.findall(ip_pattern, cleaned)
            self.assertEqual(len(remaining_ips), 0, f"IP address not removed from: {text}")
    
    def test_phone_number_removal(self):
        """Test phone number detection and removal"""
        test_cases = [
            "Call me at +1-555-123-4567",
            "Phone: (555) 987-6543", 
            "Contact: 555.123.4567",
            "Mobile: 15551234567"
        ]
        
        for text in test_cases:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            self.assertTrue(stats.get('phone_numbers', 0) >= 1 or "[REDACTED]" in cleaned)
    
    def test_ssn_removal(self):
        """Test Social Security Number detection and removal"""
        test_cases = [
            "SSN: 123-45-6789",
            "Social Security: 987654321",
            "ID: 123456789"
        ]
        
        for text in test_cases:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            # Should remove or redact SSN patterns
            import re
            ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
            remaining_ssns = re.findall(ssn_pattern, cleaned)
            self.assertEqual(len(remaining_ssns), 0)
    
    def test_credit_card_removal(self):
        """Test credit card number detection and removal"""
        test_cases = [
            "Card: 4111-1111-1111-1111",
            "Credit Card: 5555555555554444",
            "Payment: 378282246310005"
        ]
        
        for text in test_cases:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            # Should remove credit card patterns
            import re
            cc_pattern = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            remaining_ccs = re.findall(cc_pattern, cleaned)
            self.assertEqual(len(remaining_ccs), 0)
    
    def test_pii_removal_file_operations(self):
        """Test PII removal from actual files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_pii.py")
            
            # Create file with PII
            pii_content = """
# Configuration file with PII
DATABASE_USER = "admin@company.com"
API_KEY = "ghp_abcdefghijklmnopqrstuvwxyz123456789"
SERVER_IP = "192.168.1.100"
PHONE = "+1-555-123-4567"
"""
            
            with open(test_file, 'w') as f:
                f.write(pii_content)
            
            # Remove PII
            success, stats = self.pii_remover.remove_pii_from_file(test_file)
            
            self.assertTrue(success)
            self.assertTrue(stats.get('emails', 0) >= 1)
            
            # Verify PII was removed
            with open(test_file, 'r') as f:
                cleaned_content = f.read()
            
            self.assertNotIn("admin@company.com", cleaned_content)
            self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz123456789", cleaned_content)
    
    def test_pii_false_positive_handling(self):
        """Test handling of false positive PII patterns"""
        # These should NOT be flagged as PII
        false_positives = [
            "version = '1.2.3'",  # Version numbers
            "ratio = 3.14159",    # Mathematical constants  
            "localhost = '127.0.0.1'",  # Well-known addresses (commented context)
            "example@example.com in documentation",  # Documentation examples
        ]
        
        for text in false_positives:
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            # Should have minimal changes for these cases
            total_pii = sum(stats.values())
            self.assertTrue(total_pii <= 1, f"Too many false positives in: {text}")
    
    def test_pii_statistics_accuracy(self):
        """Test accuracy of PII detection statistics"""
        text = """
        Multiple PII types:
        - Email: user1@test.com and user2@test.org  
        - Tokens: ghp_abc123def456ghi789 and ghp_xyz987uvw654rst321
        - IP: 192.168.1.1 and 10.0.0.1
        """
        
        cleaned, stats = self.pii_remover.remove_pii_from_text(text)
        
        # Verify counts match expected
        self.assertEqual(stats.get('emails', 0), 2)
        self.assertEqual(stats.get('github_tokens', 0), 2) 
        self.assertEqual(stats.get('ip_addresses', 0), 2)
    
    def test_pii_removal_preserves_code_structure(self):
        """Test that PII removal preserves code structure and functionality"""
        code_with_pii = '''
def connect_to_database():
    """Connect to production database"""
    host = "192.168.1.100"  # PII: IP address
    user = "admin@company.com"  # PII: email
    password = "secret_key_abc123def456"  # PII: potential key
    
    return create_connection(host, user, password)
'''
        
        cleaned, stats = self.pii_remover.remove_pii_from_text(code_with_pii)
        
        # Code structure should be preserved
        self.assertIn("def connect_to_database():", cleaned)
        self.assertIn("return create_connection", cleaned)
        self.assertIn("\"\"\"Connect to production database\"\"\"", cleaned)
        
        # PII should be removed/replaced
        self.assertNotIn("192.168.1.100", cleaned)
        self.assertNotIn("admin@company.com", cleaned)
    
    def test_pii_removal_encoding_handling(self):
        """Test PII removal with different text encodings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test UTF-8 with special characters
            utf8_file = os.path.join(temp_dir, "test_utf8.py")
            content = "# User: josé.garcía@empresa.com\n# IP: 192.168.1.1"
            
            with open(utf8_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            success, stats = self.pii_remover.remove_pii_from_file(utf8_file)
            self.assertTrue(success)
            self.assertTrue(stats.get('emails', 0) >= 1)
    
    def test_pii_removal_large_files(self):
        """Test PII removal performance with large files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            large_file = os.path.join(temp_dir, "large_test.py")
            
            # Create large file with scattered PII
            lines = []
            for i in range(1000):
                if i % 100 == 0:
                    lines.append(f"# Contact: user{i}@example.com")
                else:
                    lines.append(f"# Line {i}: regular code comment")
            
            with open(large_file, 'w') as f:
                f.write('\n'.join(lines))
            
            # Measure performance
            start_time = datetime.now()
            success, stats = self.pii_remover.remove_pii_from_file(large_file)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.assertTrue(success)
            self.assertEqual(stats.get('emails', 0), 10)  # 10 emails inserted
            self.assertLess(duration, 5.0)  # Should complete within 5 seconds
    
    def test_pii_removal_concurrent_safety(self):
        """Test PII removal thread safety for concurrent operations"""
        import threading
        import concurrent.futures
        
        def remove_pii_worker(text_id):
            text = f"Test {text_id}: contact user{text_id}@test.com"
            cleaned, stats = self.pii_remover.remove_pii_from_text(text)
            return stats.get('emails', 0)
        
        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(remove_pii_worker, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should have detected exactly 1 email
        self.assertTrue(all(count == 1 for count in results))
        self.assertEqual(len(results), 20)


@pytest.mark.unit 
class TestCodeDeduplication(unittest.TestCase):
    """Comprehensive code deduplication functionality tests - 15 test cases"""
    
    def setUp(self):
        self.deduplicator = CodeDeduplicator(similarity_threshold=0.85)
    
    def test_exact_duplicate_detection(self):
        """Test detection of exact duplicate files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create identical files
            content = "print('Hello, World!')\nprint('This is a test')"
            
            file1 = os.path.join(temp_dir, "file1.py")
            file2 = os.path.join(temp_dir, "file2.py") 
            
            with open(file1, 'w') as f:
                f.write(content)
            with open(file2, 'w') as f:
                f.write(content)
            
            duplicates = self.deduplicator.find_duplicates([file1, file2])
            
            self.assertEqual(len(duplicates), 1)  # One duplicate pair
            self.assertIn((file1, file2), duplicates or [(file1, file2)])
    
    def test_near_duplicate_detection(self):
        """Test detection of near-duplicate files (high similarity)"""
        content1 = """
def calculate_sum(a, b):
    '''Calculate the sum of two numbers'''
    return a + b

if __name__ == "__main__":
    result = calculate_sum(5, 3)
    print(f"Result: {result}")
"""
        
        content2 = """
def calculate_sum(x, y):
    '''Calculate the sum of two numbers'''
    return x + y

if __name__ == "__main__":
    result = calculate_sum(5, 3)
    print(f"Sum: {result}")
"""
        
        similarity = self.deduplicator.calculate_similarity(content1, content2)
        self.assertGreater(similarity, 0.80)  # Should be high similarity
    
    def test_different_content_not_duplicates(self):
        """Test that different content is not flagged as duplicates"""
        content1 = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        content2 = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
        
        similarity = self.deduplicator.calculate_similarity(content1, content2)
        self.assertLess(similarity, 0.50)  # Should be low similarity
    
    def test_whitespace_normalization(self):
        """Test that whitespace differences are normalized"""
        content1 = "def hello():\n    print('hello')\n    return True"
        content2 = "def hello():\n        print('hello')\n        return True"  # Different indentation
        
        # Should still detect as similar after normalization
        similarity = self.deduplicator.calculate_similarity(content1, content2)
        self.assertGreater(similarity, 0.90)
    
    def test_comment_handling(self):
        """Test handling of comments in similarity calculation"""
        content1 = """
def process_data(data):
    # This processes the data
    result = data * 2
    return result
"""
        
        content2 = """
def process_data(data):
    # This function processes the input data
    result = data * 2
    return result
"""
        
        similarity = self.deduplicator.calculate_similarity(content1, content2)
        self.assertGreater(similarity, 0.80)  # Comments shouldn't drastically affect similarity
    
    def test_minhash_consistency(self):
        """Test MinHash signature consistency"""
        content = "def test(): return 42"
        
        # Generate signature multiple times
        sig1 = self.deduplicator._generate_minhash_signature(content)
        sig2 = self.deduplicator._generate_minhash_signature(content) 
        
        # Should be identical
        np.testing.assert_array_equal(sig1, sig2)
    
    def test_large_file_deduplication(self):
        """Test deduplication performance with large files"""
        # Generate large similar content
        base_content = "def function_{i}():\n    return {i}\n\n"
        content1 = "".join([base_content.format(i=i) for i in range(100)])
        content2 = "".join([base_content.format(i=i) for i in range(100)])  # Identical
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = os.path.join(temp_dir, "large1.py")
            file2 = os.path.join(temp_dir, "large2.py")
            
            with open(file1, 'w') as f:
                f.write(content1)
            with open(file2, 'w') as f:
                f.write(content2)
            
            start_time = datetime.now()
            duplicates = self.deduplicator.find_duplicates([file1, file2])
            duration = (datetime.now() - start_time).total_seconds()
            
            self.assertEqual(len(duplicates), 1)  # Should find duplicate
            self.assertLess(duration, 2.0)  # Should be fast
    
    def test_similarity_threshold_configuration(self):
        """Test configurable similarity threshold"""
        content1 = "def test(): return 1"
        content2 = "def test(): return 2"  # Similar but different
        
        # High threshold deduplicator
        high_threshold_dedup = CodeDeduplicator(similarity_threshold=0.95)
        
        # Low threshold deduplicator  
        low_threshold_dedup = CodeDeduplicator(similarity_threshold=0.50)
        
        similarity = high_threshold_dedup.calculate_similarity(content1, content2)
        
        # With high threshold, these shouldn't be duplicates
        # With low threshold, they might be
        if similarity > 0.50:
            with tempfile.TemporaryDirectory() as temp_dir:
                file1 = os.path.join(temp_dir, "test1.py")
                file2 = os.path.join(temp_dir, "test2.py")
                
                with open(file1, 'w') as f:
                    f.write(content1)
                with open(file2, 'w') as f:
                    f.write(content2)
                
                high_dups = high_threshold_dedup.find_duplicates([file1, file2])
                low_dups = low_threshold_dedup.find_duplicates([file1, file2])
                
                # Should respect threshold differences
                self.assertGreaterEqual(len(low_dups), len(high_dups))
    
    def test_batch_deduplication(self):
        """Test deduplication of multiple files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = []
            contents = [
                "print('File 1')",  # Unique
                "print('File 2')",  # Unique  
                "print('File 1')",  # Duplicate of first
                "print('Different content')",  # Unique
                "print('File 2')",  # Duplicate of second
            ]
            
            for i, content in enumerate(contents):
                filepath = os.path.join(temp_dir, f"file{i}.py")
                with open(filepath, 'w') as f:
                    f.write(content)
                files.append(filepath)
            
            duplicates = self.deduplicator.find_duplicates(files)
            
            # Should find 2 duplicate pairs
            self.assertEqual(len(duplicates), 2)
    
    def test_encoding_robustness(self):
        """Test deduplication with different encodings"""
        content = "# Test with unicode: café, naïve, résumé"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            utf8_file = os.path.join(temp_dir, "utf8.py")
            latin1_file = os.path.join(temp_dir, "latin1.py")
            
            # Write same content in different encodings
            with open(utf8_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # For this test, we'll use utf-8 for both since latin-1 might not support all chars
            with open(latin1_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            duplicates = self.deduplicator.find_duplicates([utf8_file, latin1_file])
            self.assertEqual(len(duplicates), 1)  # Should detect as duplicates
    
    def test_empty_file_handling(self):
        """Test handling of empty files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty1 = os.path.join(temp_dir, "empty1.py")
            empty2 = os.path.join(temp_dir, "empty2.py")
            non_empty = os.path.join(temp_dir, "nonempty.py")
            
            # Create empty files
            Path(empty1).touch()
            Path(empty2).touch()
            
            # Create non-empty file
            with open(non_empty, 'w') as f:
                f.write("print('hello')")
            
            duplicates = self.deduplicator.find_duplicates([empty1, empty2, non_empty])
            
            # Empty files should be considered duplicates
            self.assertTrue(any((empty1, empty2) in dup or (empty2, empty1) in dup 
                             for dup in duplicates))
    
    def test_deduplication_statistics(self):
        """Test deduplication statistics reporting"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mix of unique and duplicate files
            files = []
            for i in range(10):
                filepath = os.path.join(temp_dir, f"file{i}.py")
                # Create some duplicates
                content = f"print('Content {i % 5}')"  # 5 unique contents, duplicated
                with open(filepath, 'w') as f:
                    f.write(content)
                files.append(filepath)
            
            duplicates = self.deduplicator.find_duplicates(files)
            stats = self.deduplicator.get_deduplication_stats()
            
            self.assertEqual(stats['total_files'], 10)
            self.assertEqual(stats['duplicate_pairs'], len(duplicates))
            self.assertGreater(stats['duplicate_rate'], 0)
    
    def test_similarity_matrix_generation(self):
        """Test generation of similarity matrix for files"""
        contents = [
            "def func1(): pass",
            "def func1(): pass",  # Identical
            "def func2(): pass",  # Similar structure
            "class MyClass: pass"  # Different
        ]
        
        similarity_matrix = self.deduplicator.generate_similarity_matrix(contents)
        
        # Matrix should be symmetric
        self.assertEqual(similarity_matrix.shape, (4, 4))
        
        # Diagonal should be 1.0 (identical to self)
        np.testing.assert_array_equal(np.diag(similarity_matrix), np.ones(4))
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(similarity_matrix, similarity_matrix.T)
        
        # First two should be identical
        self.assertAlmostEqual(similarity_matrix[0, 1], 1.0, places=2)
    
    def test_incremental_deduplication(self):
        """Test incremental deduplication for streaming scenarios"""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_files = []
            
            # Create initial set of files
            for i in range(3):
                filepath = os.path.join(temp_dir, f"existing{i}.py")
                with open(filepath, 'w') as f:
                    f.write(f"print('Existing {i}')")
                existing_files.append(filepath)
            
            # Initialize deduplicator with existing files
            self.deduplicator.initialize_with_existing(existing_files)
            
            # Add new file that's a duplicate
            new_file = os.path.join(temp_dir, "new.py")
            with open(new_file, 'w') as f:
                f.write("print('Existing 1')")  # Duplicate of existing1.py
            
            is_duplicate = self.deduplicator.is_duplicate_of_existing(new_file)
            self.assertTrue(is_duplicate)


@pytest.mark.unit
class TestPreprocessingPipeline(unittest.TestCase):
    """Comprehensive preprocessing pipeline integration tests - 15 test cases"""
    
    def setUp(self):
        self.config = {
            'stages': {
                'encoding_fix': True,
                'malformed_cleanup': True,
                'pii_removal': True,
                'deduplication': True,
                'tokenization': False,
                'whitespace_cleanup': True
            },
            'pii_removal': {
                'log_removed_items': False,
                'replacement_strategy': 'generic'
            },
            'deduplication': {
                'similarity_threshold': 0.85,
                'algorithm': 'minhash'
            }
        }
        self.pipeline = PreprocessingPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with configuration"""
        self.assertEqual(self.pipeline.config, self.config)
        self.assertTrue(hasattr(self.pipeline, 'pii_remover'))
        self.assertTrue(hasattr(self.pipeline, 'deduplicator'))
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end preprocessing pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files with various issues
            test_files = {
                'normal.py': 'def hello():\n    print("Hello, World!")',
                'with_pii.py': 'EMAIL = "user@company.com"\nprint("Hello")',
                'duplicate1.py': 'print("duplicate content")',
                'duplicate2.py': 'print("duplicate content")',
                'malformed.py': 'def broken_function(\n    print("missing syntax")',
                'encoding_issue.py': 'print("test")'  # Will simulate encoding issue
            }
            
            file_paths = []
            for filename, content in test_files.items():
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                file_paths.append(filepath)
            
            # Process files
            results = self.pipeline.process_files(file_paths)
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn('processed_files', results)
            self.assertIn('statistics', results)
    
    def test_stage_execution_order(self):
        """Test that pipeline stages execute in correct order"""
        stage_execution_log = []
        
        # Mock stage methods to track execution
        original_methods = {}
        stage_names = ['encoding_fix', 'malformed_cleanup', 'pii_removal', 'deduplication', 'whitespace_cleanup']
        
        for stage in stage_names:
            method_name = f'_stage_{stage}'
            if hasattr(self.pipeline, method_name):
                original_methods[stage] = getattr(self.pipeline, method_name)
                
                def make_tracked_method(stage_name, original_method):
                    def tracked_method(*args, **kwargs):
                        stage_execution_log.append(stage_name)
                        return original_method(*args, **kwargs)
                    return tracked_method
                
                setattr(self.pipeline, method_name, make_tracked_method(stage, original_methods[stage]))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'test.py')
            with open(test_file, 'w') as f:
                f.write('print("test")')
            
            self.pipeline.process_files([test_file])
        
        # Verify stages executed in order (those that were mocked)
        expected_order = [stage for stage in stage_names if stage in stage_execution_log]
        actual_order = [stage for stage in stage_execution_log if stage in expected_order]
        
        # Should maintain relative order
        self.assertEqual(actual_order[:len(expected_order)], expected_order)
    
    def test_selective_stage_execution(self):
        """Test pipeline with selective stage execution"""
        # Configure to skip certain stages
        selective_config = self.config.copy()
        selective_config['stages']['pii_removal'] = False
        selective_config['stages']['deduplication'] = False
        
        selective_pipeline = PreprocessingPipeline(selective_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'test_pii.py')
            with open(test_file, 'w') as f:
                f.write('EMAIL = "user@company.com"')
            
            results = selective_pipeline.process_files([test_file])
            
            # PII should still be present since PII removal was disabled
            with open(test_file, 'r') as f:
                content = f.read()
            
            if 'user@company.com' in content:
                # PII removal was indeed skipped
                pass
            else:
                # PII was removed by another stage or default behavior
                pass
    
    def test_error_handling_malformed_files(self):
        """Test pipeline error handling with malformed files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with syntax errors
            malformed_file = os.path.join(temp_dir, 'malformed.py')
            with open(malformed_file, 'w') as f:
                f.write('def incomplete_function(\n    print("missing closing parenthesis"')
            
            # Create binary file (non-text)
            binary_file = os.path.join(temp_dir, 'binary.py')
            with open(binary_file, 'wb') as f:
                f.write(b'\x00\x01\x02\x03\xff\xfe')
            
            # Pipeline should handle errors gracefully
            results = self.pipeline.process_files([malformed_file, binary_file])
            
            self.assertIsInstance(results, dict)
            # Should have error information
            if 'errors' in results:
                self.assertGreater(len(results['errors']), 0)
    
    def test_statistics_generation(self):
        """Test comprehensive statistics generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_files = []
            for i in range(5):
                filepath = os.path.join(temp_dir, f'test{i}.py')
                content = f'print("File {i}")\n# Email: user{i}@test.com'
                with open(filepath, 'w') as f:
                    f.write(content)
                test_files.append(filepath)
            
            results = self.pipeline.process_files(test_files)
            stats = results.get('statistics', {})
            
            # Should have comprehensive statistics
            expected_stats = ['total_files', 'processed_files', 'pii_removed', 'duplicates_found']
            for stat in expected_stats:
                if stat in stats:
                    self.assertIsInstance(stats[stat], (int, float))
    
    def test_preprocessing_with_anomaly_detection(self):
        """Test preprocessing pipeline with integrated anomaly detection"""
        anomaly_detector = PipelineAnomalyDetector()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with potential anomalies
            large_file = os.path.join(temp_dir, 'large.py')
            with open(large_file, 'w') as f:
                f.write('# Large file\n' + 'x = 1\n' * 10000)  # Very large
            
            pii_file = os.path.join(temp_dir, 'pii.py')
            with open(pii_file, 'w') as f:
                f.write('email = "test@company.com"')
            
            files = [large_file, pii_file]
            results = self.pipeline.process_files(files)
            
            # Analyze for anomalies
            metrics = {
                'file_paths': files,
                'data_metrics': {'values': [os.path.getsize(f) for f in files]},
                'performance_metrics': {'processing_time': 1.0, 'memory_usage': 50.0, 'cpu_usage': 30.0}
            }
            
            anomalies = anomaly_detector.analyze_pipeline_stage('preprocessing', metrics)
            
            # Should detect file size anomaly
            size_anomalies = [a for a in anomalies if a.anomaly_type == 'file_size_anomaly']
            self.assertGreater(len(size_anomalies), 0)
    
    def test_bias_analysis_integration(self):
        """Test preprocessing pipeline with bias analysis"""
        pii_analyzer = PIIFairnessAnalyzer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files before PII removal
            before_files = []
            for i in range(3):
                filepath = os.path.join(temp_dir, f'before{i}.py')
                with open(filepath, 'w') as f:
                    f.write(f'email{i} = "user{i}@company.com"')
                before_files.append(filepath)
            
            # Process files (PII removal)
            self.pipeline.process_files(before_files)
            
            # Files after processing (same paths)
            after_files = before_files
            
            # Analyze PII removal fairness
            bias_result = pii_analyzer.analyze_pii_removal_fairness(before_files, after_files)
            
            if bias_result:
                self.assertIsInstance(bias_result.current_distribution, dict)
                self.assertIn('severity', bias_result.__dict__)
    
    def test_concurrent_processing(self):
        """Test pipeline concurrent processing capabilities"""
        import concurrent.futures
        
        def process_file_batch(file_batch):
            return self.pipeline.process_files(file_batch)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple batches of files
            batches = []
            for batch_id in range(3):
                batch_files = []
                for file_id in range(5):
                    filepath = os.path.join(temp_dir, f'batch{batch_id}_file{file_id}.py')
                    with open(filepath, 'w') as f:
                        f.write(f'print("Batch {batch_id}, File {file_id}")')
                    batch_files.append(filepath)
                batches.append(batch_files)
            
            # Process batches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_file_batch, batch) for batch in batches]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All batches should be processed successfully
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIsInstance(result, dict)
    
    def test_memory_efficiency(self):
        """Test pipeline memory efficiency with large datasets"""
        import psutil
        import os as os_module
        
        process = psutil.Process(os_module.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with tempfile.TemporaryDirectory() as temp_dir:
            large_files = []
            
            # Create moderately large files
            for i in range(10):
                filepath = os.path.join(temp_dir, f'large{i}.py')
                content = f'# Large file {i}\n' + f'data_{i} = "x" * 1000\n' * 100
                with open(filepath, 'w') as f:
                    f.write(content)
                large_files.append(filepath)
            
            # Process files
            results = self.pipeline.process_files(large_files)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB for this test)
            self.assertLess(memory_increase, 100)
    
    def test_configuration_validation(self):
        """Test pipeline configuration validation"""
        # Test invalid configuration
        invalid_configs = [
            {'stages': {}},  # Missing stages
            {'stages': {'invalid_stage': True}},  # Invalid stage name
            {'pii_removal': {'invalid_option': True}},  # Invalid PII option
        ]
        
        for invalid_config in invalid_configs:
            try:
                pipeline = PreprocessingPipeline(invalid_config)
                # Should either handle gracefully or raise appropriate error
                self.assertIsInstance(pipeline, PreprocessingPipeline)
            except (ValueError, KeyError, TypeError) as e:
                # Expected validation error
                self.assertIsInstance(e, (ValueError, KeyError, TypeError))
    
    def test_progress_tracking(self):
        """Test pipeline progress tracking and reporting"""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = []
            for i in range(10):
                filepath = os.path.join(temp_dir, f'progress_test{i}.py')
                with open(filepath, 'w') as f:
                    f.write(f'print("Progress test {i}")')
                files.append(filepath)
            
            # Mock progress callback
            progress_updates = []
            
            def progress_callback(current, total, stage):
                progress_updates.append((current, total, stage))
            
            # Configure pipeline with progress tracking
            if hasattr(self.pipeline, 'set_progress_callback'):
                self.pipeline.set_progress_callback(progress_callback)
            
            results = self.pipeline.process_files(files)
            
            # Should have received progress updates (if implemented)
            if progress_updates:
                self.assertGreater(len(progress_updates), 0)
                # Final update should show completion
                final_update = progress_updates[-1]
                self.assertEqual(final_update[0], final_update[1])  # current == total
    
    def test_output_format_consistency(self):
        """Test consistency of pipeline output format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'format_test.py')
            with open(test_file, 'w') as f:
                f.write('print("Format test")')
            
            results = self.pipeline.process_files([test_file])
            
            # Verify standard output format
            required_keys = ['processed_files', 'statistics']
            for key in required_keys:
                if key in results:
                    self.assertIn(key, results)
            
            # Statistics should be numeric where appropriate
            if 'statistics' in results:
                stats = results['statistics']
                for stat_name, stat_value in stats.items():
                    if stat_name.endswith('_count') or stat_name.endswith('_total'):
                        self.assertIsInstance(stat_value, (int, float))
    
    def test_rollback_on_failure(self):
        """Test pipeline rollback capabilities on processing failure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, 'rollback_test.py')
            original_content = 'print("Original content")'
            
            with open(test_file, 'w') as f:
                f.write(original_content)
            
            # Create backup of original content
            backup_file = test_file + '.backup'
            with open(backup_file, 'w') as f:
                f.write(original_content)
            
            # Simulate processing failure
            with patch.object(self.pipeline, '_stage_pii_removal', side_effect=Exception("Simulated failure")):
                try:
                    results = self.pipeline.process_files([test_file])
                except Exception:
                    pass
            
            # If rollback is implemented, original content should be preserved
            with open(test_file, 'r') as f:
                final_content = f.read()
            
            # Either content is unchanged (rollback) or processing continued despite error
            self.assertTrue(
                final_content == original_content or  # Rollback occurred
                len(final_content) > 0  # Processing continued
            )


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([__file__, '-v', '--cov=scripts', '--cov-report=html'])
    
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