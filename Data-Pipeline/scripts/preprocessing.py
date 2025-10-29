"""
Main Preprocessing Pipeline
Task 2: Data Preprocessing & Cleaning

This module orchestrates the complete preprocessing pipeline including:
- PII removal
- Deduplication
- Code tokenization
- File cleaning and encoding fixes
- Modular, configurable preprocessing stages
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False
import chardet

# Import our custom modules
from pii_removal import PIIRemover
from deduplication import CodeDeduplicator
from logger_config import setup_logger, get_log_filename


class PreprocessingPipeline:
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            config_path: Path to configuration file (defaults to configs/preprocessing_config.yaml)
        """
        self.config_path = config_path or Path("configs/preprocessing_config.yaml")
        self.config = self._load_config()
        
        # Set up logging
        log_file = get_log_filename("data_preprocessing")
        self.logger = setup_logger(__name__, log_file)
        
        # Initialize components
        self.pii_remover = PIIRemover(log_removed_items=self.config.get('log_pii_removal', True))
        self.deduplicator = CodeDeduplicator(
            similarity_threshold=self.config.get('similarity_threshold', 0.85)
        )
        
        # Initialize tokenizer
        self.tokenizer = None
        self._init_tokenizer()
        
        # Statistics
        self.stats = {
            'total_files_processed': 0,
            'total_files_output': 0,
            'malformed_files_fixed': 0,
            'encoding_issues_fixed': 0,
            'pii_items_removed': 0,
            'duplicates_removed': 0,
            'bytes_processed': 0,
            'bytes_saved': 0,
            'processing_time': 0,
            'tokenization_stats': {}
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'tokenizer_model': 'microsoft/codebert-base',
            'similarity_threshold': 0.85,
            'log_pii_removal': True,
            'max_file_size_mb': 10,
            'supported_languages': ['python', 'java', 'cpp', 'javascript'],
            'stages': {
                'encoding_fix': True,
                'malformed_cleaning': True,
                'pii_removal': True,
                'deduplication': True,
                'tokenization': True,
                'whitespace_cleanup': True
            },
            'parallel_processing': True,
            'max_workers': min(8, multiprocessing.cpu_count()),
            'chunk_size': 100,
            'metrics_output': 'monitoring/metrics'
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                return default_config
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                return default_config
        else:
            # Create default config file
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            print(f"Created default config at {self.config_path}")
            return default_config
    
    
    def _emit_metrics(self, run_success: bool = True):
        """Emit Prometheus-style metrics to configured directory."""
        try:
            metrics_dir = Path(self.config.get('metrics_output', 'monitoring/metrics'))
            metrics_dir.mkdir(parents=True, exist_ok=True)
            prom_file = metrics_dir / 'preprocessing.prom'
            lines = [
                f"preprocessing_files_processed_total {self.stats.get('total_files_processed', 0)}",
                f"preprocessing_files_output_total {self.stats.get('total_files_output', 0)}",
                f"preprocessing_pii_items_removed_total {self.stats.get('pii_items_removed', 0)}",
                f"preprocessing_duplicates_removed_total {self.stats.get('duplicates_removed', 0)}",
                f"preprocessing_bytes_processed_total {self.stats.get('bytes_processed', 0)}",
                f"preprocessing_bytes_saved_total {self.stats.get('bytes_saved', 0)}",
                f"preprocessing_processing_seconds {self.stats.get('processing_time', 0)}",
                f"preprocessing_last_run_success {1 if run_success else 0}",
            ]
            prom_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        except Exception:
            # metrics emission should never break the pipeline
            pass

    def _init_tokenizer(self):
        """Initialize the CodeBERT tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.info("Transformers not available, skipping tokenizer")
            self.tokenizer = None
            return
            
        try:
            tokenizer_model = self.config.get('tokenizer_model', 'microsoft/codebert-base')
            self.logger.info(f"Loading tokenizer: {tokenizer_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

    
    def detect_and_fix_encoding(self, file_path: Path) -> Tuple[str, str]:
        """
        Detect and fix file encoding issues.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (content, detected_encoding)
        """
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content, 'utf-8'
        except UnicodeDecodeError:
            # Detect encoding
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                
                detected = chardet.detect(raw_data)
                encoding = detected.get('encoding', 'latin-1')
                confidence = detected.get('confidence', 0)
                
                if confidence < 0.7:
                    encoding = 'latin-1'  # Fallback
                
                content = raw_data.decode(encoding, errors='replace')
                
                self.logger.debug(f"Fixed encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
                self.stats['encoding_issues_fixed'] += 1
                
                return content, encoding
                
            except Exception as e:
                self.logger.error(f"Failed to detect encoding for {file_path}: {e}")
                # Last resort: read as binary and decode with errors='replace'
                with open(file_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='replace')
                return content, 'utf-8-with-errors'
    
    def clean_malformed_code(self, content: str, language: str) -> str:
        """
        Clean malformed code files by removing/fixing common issues.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            Cleaned content
        """
        original_length = len(content)
        cleaned = content
        
        # Remove null bytes
        cleaned = cleaned.replace('\x00', '')
        
        # Remove excessive repeated characters (potential corruption)
        import re
        cleaned = re.sub(r'(.)\1{50,}', r'\1\1\1', cleaned)  # Replace 50+ repeated chars with 3
        
        # Remove binary data markers
        cleaned = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
        
        # Language-specific cleaning
        if language == 'python':
            # Fix common Python issues
            cleaned = re.sub(r'\r\n', '\n', cleaned)  # Normalize line endings
            cleaned = re.sub(r'^\ufeff', '', cleaned)  # Remove BOM
        
        elif language in ['java', 'cpp']:
            # Fix common C-style language issues
            cleaned = re.sub(r'\r\n', '\n', cleaned)
            cleaned = re.sub(r'^\ufeff', '', cleaned)
        
        elif language == 'javascript':
            # Fix common JS issues
            cleaned = re.sub(r'\r\n', '\n', cleaned)
            cleaned = re.sub(r'^\ufeff', '', cleaned)
        
        # Remove files that are mostly binary or gibberish
        if len(cleaned) < original_length * 0.5:
            self.logger.warning(f"File appears heavily corrupted (lost {original_length - len(cleaned)} chars)")
        
        if len(cleaned) != original_length:
            self.stats['malformed_files_fixed'] += 1
        
        return cleaned
    
    def cleanup_whitespace(self, content: str) -> str:
        """
        Clean up whitespace while preserving code structure.
        
        Args:
            content: File content
            
        Returns:
            Cleaned content
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace but preserve leading (indentation)
            cleaned_line = line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Remove excessive empty lines (more than 2 consecutive)
        result_lines = []
        empty_count = 0
        
        for line in cleaned_lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def tokenize_code(self, content: str, language: str) -> Dict:
        """
        Tokenize code using CodeBERT tokenizer.
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Dictionary with tokenization results
        """
        if not self.tokenizer:
            return {'tokens': [], 'token_count': 0, 'error': 'Tokenizer not available'}
        
        try:
            # Add language identifier for better tokenization
            if language in ['python', 'java', 'cpp', 'javascript']:
                prefixed_content = f"<{language}>\n{content}"
            else:
                prefixed_content = content
            
            # Tokenize
            tokens = self.tokenizer.encode(
                prefixed_content,
                add_special_tokens=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
            
            # Get token strings for analysis
            token_strings = self.tokenizer.convert_ids_to_tokens(tokens)
            
            return {
                'token_ids': tokens,
                'token_strings': token_strings,
                'token_count': len(tokens),
                'truncated': len(prefixed_content) > self.tokenizer.model_max_length,
                'language': language
            }
            
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            return {'tokens': [], 'token_count': 0, 'error': str(e)}
    
    def process_single_file(self, input_path: Path, output_path: Path, 
                          language: str, stages: Dict[str, bool]) -> Dict:
        """
        Process a single file through the preprocessing pipeline.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            language: Programming language
            stages: Dictionary of enabled stages
            
        Returns:
            Processing results dictionary
        """
        start_time = datetime.now()
        result = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'language': language,
            'success': True,
            'stages_applied': [],
            'original_size': 0,
            'final_size': 0,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Check file size
            file_size_mb = input_path.stat().st_size / (1024 * 1024)
            max_size = self.config.get('max_file_size_mb', 10)
            
            if file_size_mb > max_size:
                self.logger.warning(f"Skipping {input_path.name}: file too large ({file_size_mb:.1f}MB > {max_size}MB)")
                result['success'] = False
                result['errors'].append(f"File too large: {file_size_mb:.1f}MB")
                return result
            
            # Stage 1: Encoding fix
            if stages.get('encoding_fix', True):
                content, encoding = self.detect_and_fix_encoding(input_path)
                result['stages_applied'].append('encoding_fix')
                result['detected_encoding'] = encoding
            else:
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                result['detected_encoding'] = 'utf-8'
            
            result['original_size'] = len(content)
            
            # Stage 2: Malformed code cleaning
            if stages.get('malformed_cleaning', True):
                content = self.clean_malformed_code(content, language)
                result['stages_applied'].append('malformed_cleaning')
            
            # Stage 3: PII removal
            if stages.get('pii_removal', True):
                content, pii_stats = self.pii_remover.remove_pii_from_text(content)
                result['stages_applied'].append('pii_removal')
                result['pii_removed'] = sum(pii_stats.values())
                self.stats['pii_items_removed'] += result['pii_removed']
            
            # Stage 4: Whitespace cleanup
            if stages.get('whitespace_cleanup', True):
                content = self.cleanup_whitespace(content)
                result['stages_applied'].append('whitespace_cleanup')
            
            # Stage 5: Tokenization (for analysis, not replacement)
            if stages.get('tokenization', True):
                tokenization_result = self.tokenize_code(content, language)
                result['stages_applied'].append('tokenization')
                result['tokenization'] = tokenization_result
                
                # Update tokenization stats
                if language not in self.stats['tokenization_stats']:
                    self.stats['tokenization_stats'][language] = {
                        'total_tokens': 0,
                        'files_tokenized': 0,
                        'avg_tokens_per_file': 0
                    }
                
                stats = self.stats['tokenization_stats'][language]
                stats['total_tokens'] += tokenization_result.get('token_count', 0)
                stats['files_tokenized'] += 1
                stats['avg_tokens_per_file'] = stats['total_tokens'] / stats['files_tokenized']
            
            # Save processed file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result['final_size'] = len(content)
            result['size_reduction'] = result['original_size'] - result['final_size']
            
            # Update statistics
            self.stats['total_files_processed'] += 1
            self.stats['bytes_processed'] += result['original_size']
            
            self.logger.debug(f"Processed {input_path.name}: {len(result['stages_applied'])} stages, "
                            f"{result['original_size']} -> {result['final_size']} bytes")
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            result['success'] = False
            result['errors'].append(str(e))
        
        finally:
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def process_language_directory(self, input_dir: Path, output_dir: Path, 
                                 language: str) -> Dict:
        """
        Process all files for a specific language.
        
        Args:
            input_dir: Input directory (e.g., data/raw/code/python)
            output_dir: Output directory (e.g., data/processed/code/python)
            language: Programming language
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing {language} files from {input_dir}")
        
        # Get all code files (excluding .sha256 files)
        code_files = [f for f in input_dir.iterdir() 
                     if f.is_file() and not f.name.endswith('.sha256')]
        
        self.logger.info(f"Found {len(code_files)} {language} files to process")
        
        if not code_files:
            return {'language': language, 'processed_files': 0, 'results': []}
        
        # Process files
        stages = self.config.get('stages', {})
        results = []
        
        if self.config.get('parallel_processing', True):
            # Parallel processing
            max_workers = self.config.get('max_workers', 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for code_file in code_files:
                    output_file = output_dir / code_file.name
                    future = executor.submit(
                        self.process_single_file, 
                        code_file, output_file, language, stages
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            self.stats['total_files_output'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Future failed: {e}")
        else:
            # Sequential processing
            for code_file in code_files:
                output_file = output_dir / code_file.name
                result = self.process_single_file(code_file, output_file, language, stages)
                results.append(result)
                
                if result['success']:
                    self.stats['total_files_output'] += 1
        
        # Deduplication stage (if enabled)
        if stages.get('deduplication', True):
            self.logger.info(f"Starting deduplication for {language}")
            
            # Create temporary directory for deduplication
            temp_dir = output_dir.parent / f"temp_dedup_{language}"
            
            try:
                dedup_results = self.deduplicator.process_directory(
                    output_dir, temp_dir, keep_strategy='first'
                )
                
                # Replace original with deduplicated files
                if temp_dir.exists():
                    shutil.rmtree(output_dir)
                    temp_dir.rename(output_dir)
                
                # Update statistics
                self.stats['duplicates_removed'] += dedup_results['removed_files']
                self.stats['bytes_saved'] += dedup_results['bytes_saved']
                
                # Save deduplication report
                report_path = output_dir.parent / f"deduplication_report_{language}.json"
                self.deduplicator.save_duplicate_report(dedup_results, report_path)
                
                self.logger.info(f"Deduplication complete for {language}: "
                               f"{dedup_results['removed_files']} duplicates removed")
                
            except Exception as e:
                self.logger.error(f"Deduplication failed for {language}: {e}")
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        
        successful_results = [r for r in results if r['success']]
        
        language_stats = {
            'language': language,
            'input_files': len(code_files),
            'processed_files': len(successful_results),
            'failed_files': len(results) - len(successful_results),
            'total_original_size': sum(r.get('original_size', 0) for r in successful_results),
            'total_final_size': sum(r.get('final_size', 0) for r in successful_results),
            'total_pii_removed': sum(r.get('pii_removed', 0) for r in successful_results),
            'results': results
        }
        
        self.logger.info(f"Completed {language}: {len(successful_results)}/{len(code_files)} files processed successfully")
        
        return language_stats
    
    def run_pipeline(self, input_base_dir: Path, output_base_dir: Path) -> Dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            input_base_dir: Base input directory (e.g., data/raw/code)
            output_base_dir: Base output directory (e.g., data/processed/code)
            
        Returns:
            Complete processing results
        """
        start_time = datetime.now()
        
        self.logger.info("=" * 70)
        self.logger.info("Starting Data Preprocessing Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Input directory: {input_base_dir}")
        self.logger.info(f"Output directory: {output_base_dir}")
        
        # Create output directory
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each language
        supported_languages = self.config.get('supported_languages', ['python', 'java', 'cpp', 'javascript'])
        language_results = {}
        
        for language in supported_languages:
            input_lang_dir = input_base_dir / language
            output_lang_dir = output_base_dir / language
            
            if input_lang_dir.exists() and input_lang_dir.is_dir():
                try:
                    lang_results = self.process_language_directory(
                        input_lang_dir, output_lang_dir, language
                    )
                    language_results[language] = lang_results
                except Exception as e:
                    self.logger.error(f"Failed to process {language}: {e}")
                    language_results[language] = {
                        'language': language,
                        'error': str(e),
                        'processed_files': 0
                    }
            else:
                self.logger.warning(f"No input directory found for {language}: {input_lang_dir}")
        
        # Calculate final statistics
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Save final results
        final_results = {
            'pipeline_config': self.config,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'processing_time': self.stats['processing_time'],
            'global_statistics': self.stats,
            'language_results': language_results
        }
        
        # Save results to file
        results_path = output_base_dir.parent / 'preprocessing_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info("=" * 70)
        self.logger.info("Preprocessing Pipeline Complete")
        self.logger.info(f"Total files processed: {self.stats['total_files_processed']}")
        self.logger.info(f"Total files output: {self.stats['total_files_output']}")
        self.logger.info(f"PII items removed: {self.stats['pii_items_removed']}")
        self.logger.info(f"Duplicates removed: {self.stats['duplicates_removed']}")
        self.logger.info(f"Processing time: {self.stats['processing_time']:.1f} seconds")
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info("=" * 70)
        
        return final_results

def main():
    """Main function for running the preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preprocessing Pipeline')
    parser.add_argument('--input', type=Path, default=Path('data/raw/code'),
                       help='Input directory containing raw code files')
    parser.add_argument('--output', type=Path, default=Path('data/processed/code'),
                       help='Output directory for processed files')
    parser.add_argument('--config', type=Path,
                       help='Path to configuration file')
    parser.add_argument('--language', type=str, choices=['python', 'java', 'cpp', 'javascript'],
                       help='Process only specific language')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without actually processing')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=args.config)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be processed")
        print(f"Input directory: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Configuration: {pipeline.config}")
        return
    
    # Run pipeline
    if args.language:
        # Process single language
        input_lang_dir = args.input / args.language
        output_lang_dir = args.output / args.language
        
        if not input_lang_dir.exists():
            print(f"Error: Input directory for {args.language} not found: {input_lang_dir}")
            return
        
        results = pipeline.process_language_directory(
            input_lang_dir, output_lang_dir, args.language
        )
        print(f"Processed {results['processed_files']}/{results['input_files']} {args.language} files")
    else:
        # Process all languages
        results = pipeline.run_pipeline(args.input, args.output)
        print("Pipeline complete! Check the log file for detailed results.")


if __name__ == "__main__":
    main()