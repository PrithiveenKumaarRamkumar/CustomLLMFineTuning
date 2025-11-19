# pipeline/02_tokenizer.py
# ================================================================================
# Module 2: Tokenize split datasets from folders
# ================================================================================

from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from datasets import Dataset
import json
import logging
import re

logger = logging.getLogger(__name__)


class DatasetTokenizer:
    """Tokenize code datasets from split folders"""
    
    def __init__(
        self,
        tokenizer_path: str,
        splits_dir: str,
        output_dir: str,
        max_length: int = 2048,
        instruction_format: str = "starcoder2"
    ):
        """
        Initialize tokenizer
        
        Args:
            tokenizer_path: Path to tokenizer (model directory)
            splits_dir: Root directory containing train/val/test folders
            output_dir: Output directory for tokenized datasets
            max_length: Maximum sequence length
            instruction_format: Instruction format ('starcoder2', 'alpaca', 'plain')
        """
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.instruction_format = instruction_format
        
        # Verify splits directory exists
        if not self.splits_dir.exists():
            raise ValueError(f"Splits directory does not exist: {self.splits_dir}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DatasetTokenizer initialized")
        logger.info(f"Splits directory: {splits_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max length: {max_length}")
    
    def find_split_directories(self) -> Tuple[Path, Path, Path]:
        """
        Find train, val, and test directories
        
        Returns:
            Tuple of (train_dir, val_dir, test_dir)
        """
        train_dir = self.splits_dir / "train"
        val_dir = self.splits_dir / "val"
        test_dir = self.splits_dir / "test"
        
        # Check if directories exist
        if not train_dir.exists():
            logger.warning(f"Train directory not found: {train_dir}")
        if not val_dir.exists():
            logger.warning(f"Val directory not found: {val_dir}")
        if not test_dir.exists():
            logger.warning(f"Test directory not found: {test_dir}")
        
        return train_dir, val_dir, test_dir
    
    def load_code_files_from_directory(self, directory: Path) -> List[Dict]:
        """
        Load all code files from a directory
        
        Args:
            directory: Directory containing code files
            
        Returns:
            List of dictionaries with 'code' and 'file_path' keys
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        # Support multiple programming language extensions
        extensions = ['.py', '.java', '.cpp', '.js', '.ts', '.go', '.rs', '.c', '.h', '.jsx', '.tsx']
        
        code_samples = []
        
        for ext in extensions:
            files = list(directory.glob(f'*{ext}'))
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Skip empty files
                    if not code.strip():
                        logger.debug(f"Skipping empty file: {file_path}")
                        continue
                    
                    # Skip very short files (likely not useful)
                    if len(code.strip()) < 20:
                        logger.debug(f"Skipping very short file: {file_path}")
                        continue
                    
                    code_samples.append({
                        'code': code,
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'language': ext[1:]  # Remove the dot
                    })
                
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(code_samples)} code files from {directory}")
        return code_samples
    
    def extract_docstring_or_intent(self, code: str) -> str:
        """
        Extract docstring or infer intent from code
        
        Args:
            code: Source code string
            
        Returns:
            Extracted intent or description
        """
        # Try to extract Python docstring
        docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
        match = re.search(docstring_pattern, code, re.DOTALL)
        if match:
            docstring = (match.group(1) or match.group(2)).strip()
            if len(docstring) > 10:  # Only use if substantial
                return docstring
        
        # Try to extract comments from first few lines
        lines = code.split('\n')
        comments = []
        for line in lines[:15]:  # Check first 15 lines
            line = line.strip()
            if line.startswith('#'):
                comment = line.lstrip('#').strip()
                if len(comment) > 5:
                    comments.append(comment)
            elif line.startswith('//'):
                comment = line.lstrip('/').strip()
                if len(comment) > 5:
                    comments.append(comment)
        
        if comments:
            intent = ' '.join(comments[:3])  # Use first 3 comments
            return intent
        
        # Fallback: generic instruction
        return "Implement the following code"
    
    def format_into_instruction(self, code_sample: Dict) -> str:
        """
        Format code into instruction-following format
        
        Args:
            code_sample: Dictionary with 'code' and metadata
            
        Returns:
            Formatted instruction string
        """
        code = code_sample['code']
        file_name = code_sample.get('file_name', 'code.py')
        
        # Extract intent/instruction
        instruction = self.extract_docstring_or_intent(code)
        
        # Format based on specified format
        if self.instruction_format == "starcoder2":
            formatted = f"""<|im_start|>system
You are an expert programming assistant that writes clear, efficient, and well-documented code.<|im_end|>
<|im_start|>user
{instruction}

File: {file_name}<|im_end|>
<|im_start|>assistant
{code}<|im_end|>"""
        
        elif self.instruction_format == "alpaca":
            formatted = f"""### Instruction:
{instruction}

### Input:
File: {file_name}

### Response:
{code}"""
        
        else:  # plain
            formatted = f"# {instruction}\n# File: {file_name}\n\n{code}"
        
        return formatted
    
    def create_dataset_from_samples(self, code_samples: List[Dict]) -> Dataset:
        """
        Create HuggingFace Dataset from code samples
        
        Args:
            code_samples: List of code sample dictionaries
            
        Returns:
            HuggingFace Dataset
        """
        if not code_samples:
            logger.warning("No code samples provided, creating empty dataset")
            return Dataset.from_dict({'text': []})
        
        # Format into instructions
        formatted_texts = []
        for sample in code_samples:
            try:
                formatted = self.format_into_instruction(sample)
                formatted_texts.append(formatted)
            except Exception as e:
                logger.warning(f"Error formatting sample {sample.get('file_name', 'unknown')}: {e}")
                continue
        
        # Create dataset
        dataset = Dataset.from_dict({'text': formatted_texts})
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset, split_name: str) -> Dataset:
        """
        Tokenize dataset
        
        Args:
            dataset: HuggingFace Dataset
            split_name: Name of split (for logging)
            
        Returns:
            Tokenized dataset
        """
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for {split_name}, skipping tokenization")
            return dataset
        
        def tokenize_function(examples):
            """Tokenization function"""
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors=None
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        logger.info(f"Tokenizing {split_name} dataset...")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}"
        )
        
        logger.info(f"✓ Tokenized {len(tokenized_dataset)} samples for {split_name}")
        return tokenized_dataset
    
    def save_tokenized_dataset(self, dataset: Dataset, split_name: str):
        """
        Save tokenized dataset to disk
        
        Args:
            dataset: Tokenized dataset
            split_name: Name of split ('train', 'val', 'test')
        """
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for {split_name}, not saving")
            return
        
        output_path = self.output_dir / f"{split_name}_tokenized"
        dataset.save_to_disk(str(output_path))
        
        logger.info(f"✓ Saved {split_name} dataset to: {output_path}")
        logger.info(f"  - Samples: {len(dataset)}")
        logger.info(f"  - Features: {list(dataset.features.keys())}")
    
    def run(self) -> Dict:
        """
        Execute tokenization pipeline for all splits
        
        Returns:
            Dictionary with tokenization results
        """
        logger.info("="*80)
        logger.info("STEP 2: TOKENIZATION")
        logger.info("="*80)
        
        # Find split directories
        train_dir, val_dir, test_dir = self.find_split_directories()
        
        results = {}
        
        # Process each split
        for split_name, directory in [
            ('train', train_dir),
            ('val', val_dir),
            ('test', test_dir)
        ]:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {split_name.upper()} split from: {directory}")
            logger.info(f"{'='*80}")
            
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                results[split_name] = {
                    'status': 'skipped',
                    'reason': 'directory_not_found',
                    'num_samples': 0
                }
                continue
            
            try:
                # Load code files
                code_samples = self.load_code_files_from_directory(directory)
                
                if not code_samples:
                    logger.warning(f"No code files found in {directory}")
                    results[split_name] = {
                        'status': 'skipped',
                        'reason': 'no_files_found',
                        'num_samples': 0
                    }
                    continue
                
                # Create dataset
                dataset = self.create_dataset_from_samples(code_samples)
                
                # Tokenize
                tokenized_dataset = self.tokenize_dataset(dataset, split_name)
                
                # Save
                self.save_tokenized_dataset(tokenized_dataset, split_name)
                
                # Store results
                results[split_name] = {
                    'status': 'success',
                    'num_samples': len(tokenized_dataset),
                    'num_files': len(code_samples),
                    'output_path': str(self.output_dir / f"{split_name}_tokenized")
                }
            
            except Exception as e:
                logger.error(f"Error processing {split_name} split: {e}", exc_info=True)
                results[split_name] = {
                    'status': 'failed',
                    'reason': str(e),
                    'num_samples': 0
                }
        
        # Save tokenization metadata
        self._save_metadata(results)
        
        # Print summary
        self._print_summary(results)
        
        logger.info("="*80)
        logger.info("✓ STEP 2 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _save_metadata(self, results: Dict):
        """Save tokenization metadata"""
        metadata = {
            'tokenizer_name': str(self.tokenizer.name_or_path),
            'max_length': self.max_length,
            'instruction_format': self.instruction_format,
            'vocab_size': self.tokenizer.vocab_size,
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'splits': results
        }
        
        metadata_path = self.output_dir / "tokenization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✓ Metadata saved to: {metadata_path}")
    
    def _print_summary(self, results: Dict):
        """Print tokenization summary"""
        logger.info("\n" + "="*80)
        logger.info("TOKENIZATION SUMMARY")
        logger.info("="*80)
        
        total_samples = 0
        total_files = 0
        
        for split_name, result in results.items():
            status = result.get('status', 'unknown')
            num_samples = result.get('num_samples', 0)
            num_files = result.get('num_files', 0)
            
            total_samples += num_samples
            total_files += num_files
            
            if status == 'success':
                logger.info(f"✓ {split_name.upper()}: {num_files} files → {num_samples} samples")
            else:
                reason = result.get('reason', 'unknown')
                logger.info(f"✗ {split_name.upper()}: {status} ({reason})")
        
        logger.info(f"\nTOTAL: {total_files} files → {total_samples} samples")
        logger.info("="*80)


def run_tokenization(config: dict) -> Dict:
    """
    Convenience function to run tokenization
    
    Args:
        config: Configuration dictionary with keys:
            - tokenizer_path: Path to tokenizer
            - splits_dir: Root directory with train/val/test folders
            - output_dir: Output directory for tokenized datasets
            - max_length: Maximum sequence length (default: 2048)
            - instruction_format: Format (default: 'starcoder2')
    
    Returns:
        Dictionary with tokenization results
    """
    tokenizer = DatasetTokenizer(
        tokenizer_path=config['tokenizer_path'],
        splits_dir=config['splits_dir'],
        output_dir=config['output_dir'],
        max_length=config.get('max_length', 2048),
        instruction_format=config.get('instruction_format', 'starcoder2')
    )
    
    return tokenizer.run()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenize split datasets")
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='./models/starcoder2-3b',
        help='Path to tokenizer (model directory)'
    )
    parser.add_argument(
        '--splits-dir',
        type=str,
        default='./data/splits',
        help='Directory containing train/val/test folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/tokenized',
        help='Output directory for tokenized datasets'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--instruction-format',
        type=str,
        choices=['starcoder2', 'alpaca', 'plain'],
        default='starcoder2',
        help='Instruction format'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tokenization
    config = {
        'tokenizer_path': args.tokenizer_path,
        'splits_dir': args.splits_dir,
        'output_dir': args.output_dir,
        'max_length': args.max_length,
        'instruction_format': args.instruction_format
    }
    
    results = run_tokenization(config)
    
    # Print results
    print("\n" + "="*80)
    print("TOKENIZATION RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)