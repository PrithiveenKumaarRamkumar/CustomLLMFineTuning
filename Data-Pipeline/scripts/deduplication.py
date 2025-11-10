"""
Deduplication Module
Task 2: Data Preprocessing & Cleaning

This module performs exact and near-duplicate detection on code files
using multiple hashing strategies and similarity metrics.
"""

import hashlib
import logging
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import json
from collections import defaultdict
import re
from difflib import SequenceMatcher
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class CodeDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85, chunk_size: int = 1000):
        """
        Initialize code deduplicator.
        
        Args:
            similarity_threshold: Threshold for near-duplicate detection (0.0-1.0)
            chunk_size: Number of files to process in each batch
        """
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Storage for different hash types
        self.exact_hashes = {}  # SHA-256 -> file_path
        self.normalized_hashes = {}  # Normalized content hash -> file_path
        self.structural_hashes = {}  # AST/structure hash -> file_path
        self.minihashes = {}  # MinHash for near-duplicate detection
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'unique_files': 0,
            'bytes_saved': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
    
    def find_duplicates(self, file_contents) -> List[Tuple[Path, Path]]:
        """
        Find exact duplicates using SHA-256 hashing.
        
        Args:
            file_contents: Either Dict[Path, str] mapping paths to content, 
                          or List[str] of file paths to read and compare
            
        Returns:
            List of tuples representing duplicate file pairs
        """
        hash_to_files = {}
        duplicates = []
        
        # Handle both dict and list formats
        if isinstance(file_contents, list):
            # Read files from paths
            files_dict = {}
            for file_path in file_contents:
                path_obj = Path(file_path)
                try:
                    with open(path_obj, 'r', encoding='utf-8') as f:
                        files_dict[path_obj] = f.read()
                except Exception as e:
                    self.logger.warning(f"Could not read file {file_path}: {e}")
                    continue
            file_contents = files_dict
        
        # Update stats
        total_files = len(file_contents)
        self.stats['files_processed'] = total_files
        
        for file_path, content in file_contents.items():
            # Handle empty files explicitly
            if not content.strip():
                content_hash = "empty_file_hash"  # Special hash for empty files
            else:
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if content_hash in hash_to_files:
                # Found a duplicate
                original = hash_to_files[content_hash]
                # Convert Path objects to strings for compatibility
                orig_str = str(original) if isinstance(original, Path) else original
                file_str = str(file_path) if isinstance(file_path, Path) else file_path
                duplicates.append((orig_str, file_str))
            else:
                hash_to_files[content_hash] = file_path
        
        # Update stats with results
        self.stats['duplicate_pairs'] = len(duplicates)
        self.stats['exact_duplicates'] = len(duplicates)
        self.stats['unique_files'] = total_files - len(duplicates)
        
        return duplicates
    
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two code files using token-based comparison.
        
        Args:
            content1: Content of first file
            content2: Content of second file
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        import re
        
        def extract_meaningful_tokens(content):
            # Extract identifiers, strings, and numbers (ignore keywords and structure)
            token_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|"[^"]*"|\'[^\']*\'|\d+\.?\d*'
            tokens = re.findall(token_pattern, content)
            
            # Filter out common Python keywords and very common variable names
            common_tokens = {
                'def', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 
                'from', 'class', 'try', 'except', 'finally', 'with', 'as', 'and', 
                'or', 'not', 'in', 'is', 'True', 'False', 'None', 'pass', 'break', 
                'continue', 'lambda', 'yield', 'global', 'nonlocal',
                # Common variable names that don't indicate similarity
                'n', 'i', 'x', 'y', 'z', 'self', 'cls', 'args', 'kwargs',
                # Common numbers that appear in both
                '1', '2', '0'
            }
            
            meaningful_tokens = [token for token in tokens if token not in common_tokens]
            return meaningful_tokens
        
        tokens1 = extract_meaningful_tokens(content1)
        tokens2 = extract_meaningful_tokens(content2)
        
        # If both are empty after token extraction
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        # Calculate similarity based on meaningful tokens
        tokens1_str = ' '.join(tokens1)
        tokens2_str = ' '.join(tokens2)
        
        return SequenceMatcher(None, tokens1_str, tokens2_str).ratio()
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Private method alias for calculate_similarity."""
        return self.calculate_similarity(content1, content2)
    
    def initialize_with_existing(self, existing_files):
        """
        Initialize deduplicator with existing file signatures for incremental deduplication.
        
        Args:
            existing_files: Either Dict[Path, str] or List[str] of file paths
        """
        # Handle both dict and list formats
        if isinstance(existing_files, list):
            files_dict = {}
            for file_path in existing_files:
                path_obj = Path(file_path)
                try:
                    with open(path_obj, 'r', encoding='utf-8') as f:
                        files_dict[path_obj] = f.read()
                except Exception as e:
                    self.logger.warning(f"Could not read file {file_path}: {e}")
                    continue
            existing_files = files_dict
            
        with self.lock:
            for file_path, content in existing_files.items():
                # Store exact hash
                content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                self.exact_hashes[content_hash] = file_path
                
                # Store normalized hash
                normalized = self._normalize_code(content)
                norm_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
                self.normalized_hashes[norm_hash] = file_path
    
    def generate_similarity_matrix(self, file_contents):
        """
        Generate a similarity matrix for all file pairs.
        
        Args:
            file_contents: Either Dict[Path, str], List[str] of file paths, or List[str] of contents
            
        Returns:
            Numpy array of similarity scores
        """
        import numpy as np
        
        # Handle different input formats
        if isinstance(file_contents, dict):
            # Dict of path -> content
            contents_list = list(file_contents.values())
        elif isinstance(file_contents, list):
            if len(file_contents) > 0 and isinstance(file_contents[0], str):
                # Check if it's file paths or content
                if len(file_contents[0]) < 500 and ('/' in file_contents[0] or '\\' in file_contents[0] or '.' in file_contents[0]):
                    # Likely file paths - try to read them
                    contents_list = []
                    for file_path in file_contents:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                contents_list.append(f.read())
                        except Exception as e:
                            self.logger.warning(f"Could not read file {file_path}: {e}")
                            contents_list.append("")  # Add empty content for failed reads
                else:
                    # Assume it's content strings
                    contents_list = file_contents
            else:
                contents_list = list(file_contents)
        else:
            raise ValueError("file_contents must be dict or list")
            
        n = len(contents_list)
        similarity_matrix = np.zeros((n, n))
        
        # Fill the matrix
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Identity
                elif i < j:
                    # Calculate similarity for upper triangle
                    similarity = self.calculate_similarity(contents_list[i], contents_list[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Make it symmetric
        
        return similarity_matrix
    
    def _generate_minhash_signature(self, content: str, num_hashes: int = 128) -> List[int]:
        """
        Generate MinHash signature for near-duplicate detection.
        
        Args:
            content: File content
            num_hashes: Number of hash functions to use
            
        Returns:
            MinHash signature as list of integers
        """
        # Simple MinHash implementation using built-in hash
        shingles = self._generate_shingles(content)
        signature = []
        
        for i in range(num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = hash((shingle, i)) % (2**32)
                if hash_val < min_hash:
                    min_hash = hash_val
            signature.append(min_hash)
        
        return signature
    
    def _generate_shingles(self, content: str, k: int = 3) -> Set[str]:
        """Generate k-shingles from content."""
        content = self._normalize_code(content)
        shingles = set()
        for i in range(len(content) - k + 1):
            shingles.add(content[i:i+k])
        return shingles
    
    def _calculate_exact_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of exact content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _normalize_code(self, content: str, language: str = 'unknown') -> str:
        """
        Normalize code by removing comments, extra whitespace, and formatting.
        
        Args:
            content: Raw code content
            language: Programming language for language-specific normalization
            
        Returns:
            Normalized code string
        """
        normalized = content
        
        # Remove single-line comments based on language
        comment_patterns = {
            'python': [r'#.*$'],
            'java': [r'//.*$'],
            'cpp': [r'//.*$'],
            'javascript': [r'//.*$'],
        }
        
        # Remove multi-line comments
        multiline_patterns = {
            'java': [r'/\*[\s\S]*?\*/'],
            'cpp': [r'/\*[\s\S]*?\*/'],
            'javascript': [r'/\*[\s\S]*?\*/'],
            'python': [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"]
        }
        
        # Apply single-line comment removal
        if language.lower() in comment_patterns:
            for pattern in comment_patterns[language.lower()]:
                normalized = re.sub(pattern, '', normalized, flags=re.MULTILINE)
        
        # Apply multi-line comment removal
        if language.lower() in multiline_patterns:
            for pattern in multiline_patterns[language.lower()]:
                normalized = re.sub(pattern, '', normalized, flags=re.DOTALL)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s*([{}();,])\s*', r'\1', normalized)
        
        # Remove empty lines
        lines = [line.strip() for line in normalized.split('\n') if line.strip()]
        normalized = '\n'.join(lines)
        
        # Convert to lowercase for case-insensitive comparison
        normalized = normalized.lower().strip()
        
        return normalized
    
    def _calculate_normalized_hash(self, content: str, language: str = 'unknown') -> str:
        """Calculate hash of normalized content"""
        normalized = self._normalize_code(content, language)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _extract_structural_features(self, content: str, language: str = 'unknown') -> str:
        """
        Extract structural features from code (simplified AST-like representation).
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Structural signature string
        """
        # This is a simplified approach. In a production system, 
        # you would use proper AST parsers for each language.
        
        features = []
        
        # Count different types of tokens/structures
        patterns = {
            'functions': [r'\bdef\s+\w+', r'\bfunction\s+\w+', r'\w+\s*\(.*\)\s*{'],
            'classes': [r'\bclass\s+\w+', r'\bclass\s+\w+\s*{'],
            'loops': [r'\bfor\s*\(', r'\bwhile\s*\(', r'\bfor\s+\w+\s+in\b'],
            'conditions': [r'\bif\s*\(', r'\belse\b', r'\belif\b'],
            'imports': [r'\bimport\s+', r'\bfrom\s+\w+\s+import', r'#include\s*<'],
            'variables': [r'\b[a-zA-Z_]\w*\s*=\s*'],
        }
        
        for feature_type, pattern_list in patterns.items():
            count = 0
            for pattern in pattern_list:
                count += len(re.findall(pattern, content, re.IGNORECASE))
            features.append(f"{feature_type}:{count}")
        
        # Add rough line count
        line_count = len([line for line in content.split('\n') if line.strip()])
        features.append(f"lines:{line_count}")
        
        # Create structural signature
        structure = '|'.join(features)
        return hashlib.md5(structure.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate code-aware similarity"""
        # Remove comments and normalize whitespace for better code comparison
        normalized1 = self._normalize_code(content1, 'unknown')
        normalized2 = self._normalize_code(content2, 'unknown')
        
        return SequenceMatcher(None, normalized1, normalized2).ratio()
    
    def _get_language_from_path(self, file_path: Path) -> str:
        """Extract programming language from file path"""
        suffix_mapping = {
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
            '.js': 'javascript', '.jsx': 'javascript'
        }
        return suffix_mapping.get(file_path.suffix.lower(), 'unknown')
    
    def _process_file_hashes(self, file_path: Path, content: str) -> Dict:
        """
        Calculate all hash types for a file.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            Dictionary containing all calculated hashes
        """
        language = self._get_language_from_path(file_path)
        
        return {
            'file_path': str(file_path),
            'language': language,
            'size': len(content),
            'exact_hash': self._calculate_exact_hash(content),
            'normalized_hash': self._calculate_normalized_hash(content, language),
            'structural_hash': self._extract_structural_features(content, language)
        }
    
    def find_exact_duplicates(self, file_contents: Dict[Path, str]) -> Dict[str, List[Path]]:
        """
        Find files with identical content.
        
        Args:
            file_contents: Dictionary mapping file paths to their content
            
        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        hash_to_files = defaultdict(list)
        
        for file_path, content in file_contents.items():
            exact_hash = self._calculate_exact_hash(content)
            hash_to_files[exact_hash].append(file_path)
        
        # Return only groups with duplicates
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        
        with self.lock:
            self.stats['exact_duplicates'] += sum(len(files) - 1 for files in duplicates.values())
        
        return duplicates
    
    def find_normalized_duplicates(self, file_contents: Dict[Path, str]) -> Dict[str, List[Path]]:
        """
        Find files with identical normalized content (ignoring formatting/comments).
        
        Args:
            file_contents: Dictionary mapping file paths to their content
            
        Returns:
            Dictionary mapping normalized hash to list of duplicate file paths
        """
        hash_to_files = defaultdict(list)
        
        for file_path, content in file_contents.items():
            language = self._get_language_from_path(file_path)
            normalized_hash = self._calculate_normalized_hash(content, language)
            hash_to_files[normalized_hash].append(file_path)
        
        # Return only groups with duplicates
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        
        return duplicates
    
    def find_near_duplicates(self, file_contents: Dict[Path, str], max_comparisons: int = 10000) -> List[Tuple[Path, Path, float]]:
        """
        Find near-duplicate files using similarity comparison.
        
        Args:
            file_contents: Dictionary mapping file paths to their content
            max_comparisons: Maximum number of pairwise comparisons to perform
            
        Returns:
            List of tuples (file1, file2, similarity_score)
        """
        near_duplicates = []
        file_list = list(file_contents.items())
        
        # Limit comparisons for performance
        if len(file_list) > max_comparisons:
            self.logger.warning(f"Too many files for full comparison. Limiting to {max_comparisons} comparisons.")
        
        comparisons_made = 0
        
        for i in range(len(file_list)):
            if comparisons_made >= max_comparisons:
                break
                
            for j in range(i + 1, len(file_list)):
                if comparisons_made >= max_comparisons:
                    break
                    
                file1_path, content1 = file_list[i]
                file2_path, content2 = file_list[j]
                
                # Skip if files are too different in size (optimization)
                size_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
                if size_ratio < 0.5:
                    comparisons_made += 1
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(content1, content2)
                
                if similarity >= self.similarity_threshold:
                    near_duplicates.append((file1_path, file2_path, similarity))
                
                comparisons_made += 1
        
        with self.lock:
            self.stats['near_duplicates'] += len(near_duplicates)
        
        return near_duplicates
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         keep_strategy: str = 'first') -> Dict:
        """
        Process a directory to remove duplicates.
        
        Args:
            input_dir: Input directory containing files to deduplicate
            output_dir: Output directory for unique files
            keep_strategy: Strategy for which duplicate to keep ('first', 'shortest_name', 'largest')
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting deduplication of {input_dir}")
        
        # Read all files
        file_contents = {}
        file_sizes = {}
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and not file_path.name.endswith('.sha256'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    file_contents[file_path] = content
                    file_sizes[file_path] = len(content.encode('utf-8'))
                    
                    with self.lock:
                        self.stats['files_processed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error reading {file_path}: {e}")
        
        self.logger.info(f"Read {len(file_contents)} files")
        
        # Find different types of duplicates
        exact_dups = self.find_exact_duplicates(file_contents)
        normalized_dups = self.find_normalized_duplicates(file_contents)
        near_dups = self.find_near_duplicates(file_contents)
        
        # Determine which files to keep
        files_to_keep = set(file_contents.keys())
        removed_files = set()
        
        # Remove exact duplicates
        for hash_val, duplicate_files in exact_dups.items():
            files_to_remove = self._select_files_to_remove(duplicate_files, keep_strategy, file_sizes)
            removed_files.update(files_to_remove)
            files_to_keep -= set(files_to_remove)
        
        # Remove normalized duplicates (from remaining files)
        remaining_normalized = {}
        for hash_val, duplicate_files in normalized_dups.items():
            remaining_files = [f for f in duplicate_files if f in files_to_keep]
            if len(remaining_files) > 1:
                remaining_normalized[hash_val] = remaining_files
        
        for hash_val, duplicate_files in remaining_normalized.items():
            files_to_remove = self._select_files_to_remove(duplicate_files, keep_strategy, file_sizes)
            removed_files.update(files_to_remove)
            files_to_keep -= set(files_to_remove)
        
        # Copy unique files to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        copied_files = 0
        
        for file_path in files_to_keep:
            try:
                # Maintain directory structure
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as src:
                    content = src.read()
                    
                with open(output_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
                
                copied_files += 1
                
            except Exception as e:
                self.logger.error(f"Error copying {file_path}: {e}")
        
        # Calculate statistics
        bytes_saved = sum(file_sizes[f] for f in removed_files)
        
        with self.lock:
            self.stats['unique_files'] = len(files_to_keep)
            self.stats['bytes_saved'] = bytes_saved
        
        results = {
            'input_files': len(file_contents),
            'output_files': len(files_to_keep),
            'removed_files': len(removed_files),
            'exact_duplicates_found': len(exact_dups),
            'normalized_duplicates_found': len(normalized_dups),
            'near_duplicates_found': len(near_dups),
            'bytes_saved': bytes_saved,
            'duplicate_groups': {
                'exact': exact_dups,
                'normalized': remaining_normalized,
                'near': near_dups
            }
        }
        
        self.logger.info(f"Deduplication complete. {len(files_to_keep)} unique files kept, "
                        f"{len(removed_files)} duplicates removed, "
                        f"{bytes_saved / (1024*1024):.1f}MB saved")
        
        return results
    
    def _select_files_to_remove(self, duplicate_files: List[Path], 
                              keep_strategy: str, file_sizes: Dict[Path, int]) -> List[Path]:
        """
        Select which files to remove from a group of duplicates.
        
        Args:
            duplicate_files: List of duplicate file paths
            keep_strategy: Strategy for selection
            file_sizes: Dictionary mapping file paths to sizes
            
        Returns:
            List of files to remove
        """
        if len(duplicate_files) <= 1:
            return []
        
        files_to_remove = []
        
        if keep_strategy == 'first':
            # Keep the first file, remove the rest
            files_to_remove = duplicate_files[1:]
            
        elif keep_strategy == 'shortest_name':
            # Keep the file with the shortest name
            shortest = min(duplicate_files, key=lambda f: len(f.name))
            files_to_remove = [f for f in duplicate_files if f != shortest]
            
        elif keep_strategy == 'largest':
            # Keep the largest file (in case there are minor differences)
            largest = max(duplicate_files, key=lambda f: file_sizes.get(f, 0))
            files_to_remove = [f for f in duplicate_files if f != largest]
            
        else:
            # Default: keep first
            files_to_remove = duplicate_files[1:]
        
        return files_to_remove
    
    def save_duplicate_report(self, results: Dict, output_path: Path):
        """
        Save detailed duplicate detection report.
        
        Args:
            results: Results from process_directory
            output_path: Path to save the report
        """
        report = {
            'summary': {
                'input_files': results['input_files'],
                'output_files': results['output_files'],
                'removed_files': results['removed_files'],
                'bytes_saved': results['bytes_saved']
            },
            'duplicate_groups': {}
        }
        
        # Convert Path objects to strings for JSON serialization
        for group_type, groups in results['duplicate_groups'].items():
            if group_type == 'near':
                # Near duplicates have different structure
                report['duplicate_groups'][group_type] = [
                    {
                        'file1': str(item[0]),
                        'file2': str(item[1]),
                        'similarity': item[2]
                    } for item in groups
                ]
            else:
                report['duplicate_groups'][group_type] = {
                    hash_val: [str(f) for f in files]
                    for hash_val, files in groups.items()
                }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Duplicate report saved to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    def get_deduplication_stats(self) -> Dict:
        """
        Get deduplication statistics including files processed, duplicates found, etc.
        
        Returns:
            Dictionary containing deduplication statistics
        """
        files_processed = self.stats.get('files_processed', 0)
        exact_duplicates = self.stats.get('exact_duplicates', 0)
        near_duplicates = self.stats.get('near_duplicates', 0)
        duplicate_pairs = self.stats.get('duplicate_pairs', 0)
        
        # Calculate duplicate rate
        duplicate_rate = duplicate_pairs / files_processed if files_processed > 0 else 0.0
        
        return {
            'files_processed': files_processed,
            'total_files': files_processed,  # Alias for backward compatibility
            'exact_duplicates': exact_duplicates,
            'near_duplicates': near_duplicates,
            'unique_files': self.stats.get('unique_files', 0),
            'bytes_saved': self.stats.get('bytes_saved', 0),
            'duplicate_pairs': duplicate_pairs,
            'duplicate_rate': duplicate_rate,
            'similarity_threshold': self.similarity_threshold,
            'total_duplicates': exact_duplicates + near_duplicates
        }
    
    def is_duplicate_of_existing(self, file_path) -> bool:
        """
        Check if a file is a duplicate of any existing file in the known set.
        
        Args:
            file_path: Path to file to check (str or Path)
            
        Returns:
            True if file is a duplicate of existing files, False otherwise
        """
        try:
            path_obj = Path(file_path)
            
            # Read file content
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate exact hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Check if exact hash exists
            if content_hash in self.exact_hashes:
                return True
            
            # Check normalized hash
            normalized = self._normalize_code(content)
            norm_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            
            if norm_hash in self.normalized_hashes:
                return True
                
            # Check similarity with existing files (expensive operation)
            for existing_path, existing_content in self.exact_hashes.items():
                if hasattr(existing_content, 'read'):
                    # If it's a file object, read it
                    try:
                        with open(existing_content, 'r', encoding='utf-8') as f:
                            existing_text = f.read()
                    except:
                        continue
                else:
                    existing_text = str(existing_content)
                
                similarity = self.calculate_similarity(content, existing_text)
                if similarity >= self.similarity_threshold:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking duplicate status for {file_path}: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    deduplicator = CodeDeduplicator(similarity_threshold=0.85)
    
    if len(sys.argv) >= 3:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
        
        results = deduplicator.process_directory(input_dir, output_dir, keep_strategy='first')
        
        # Save report
        report_path = output_dir / 'deduplication_report.json'
        deduplicator.save_duplicate_report(results, report_path)
        
        print(f"Deduplication complete!")
        print(f"Input files: {results['input_files']}")
        print(f"Output files: {results['output_files']}")
        print(f"Removed files: {results['removed_files']}")
        print(f"Space saved: {results['bytes_saved'] / (1024*1024):.1f}MB")
        
    else:
        print("Usage: python deduplication.py <input_directory> <output_directory>")