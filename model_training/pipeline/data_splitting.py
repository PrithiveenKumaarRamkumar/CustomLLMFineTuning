# pipeline/01_data_splitter.py
# ================================================================================
# Module 1: Split preprocessed data into train/val/test
# ================================================================================

import os
import shutil
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import json
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split preprocessed code files into train/validation/test sets"""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize data splitter
        
        Args:
            source_dir: Directory containing preprocessed code files
            output_dir: Output directory for splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        logger.info("DataSplitter initialized")
        logger.info(f"Source: {source_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Split ratios: {train_ratio}/{val_ratio}/{test_ratio}")
    
    def collect_files(self) -> List[Path]:
        """
        Collect all preprocessed code files
        
        Returns:
            List of file paths
        """
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.source_dir}")
        
        # Support multiple file types
        extensions = ['.py', '.java', '.cpp', '.js', '.ts', '.go', '.rs', '.c', '.h']
        
        files = []
        for ext in extensions:
            found = list(self.source_dir.rglob(f'*{ext}'))
            files.extend(found)
        
        # Filter out special files
        files = [
            f for f in files 
            if f.name != "__init__.py" 
            and "__pycache__" not in str(f)
            and ".git" not in str(f)
        ]
        
        logger.info(f"Collected {len(files)} files")
        return files
    
    def split_files(self, files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split files into train/val/test sets
        
        Args:
            files: List of file paths
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        if len(files) < 3:
            raise ValueError(f"Need at least 3 files to split, got {len(files)}")
        
        # First split: separate test set
        train_val_files, test_files = train_test_split(
            files,
            test_size=self.test_ratio,
            random_state=self.random_seed
        )
        
        # Second split: separate validation from training
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed
        )
        
        logger.info(f"Split complete:")
        logger.info(f"  Train: {len(train_files)} files ({len(train_files)/len(files)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_files)} files ({len(val_files)/len(files)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_files)} files ({len(test_files)/len(files)*100:.1f}%)")
        
        return train_files, val_files, test_files
    
    def copy_files(
        self,
        train_files: List[Path],
        val_files: List[Path],
        test_files: List[Path]
    ):
        """
        Copy files to train/val/test directories
        
        Args:
            train_files: Training files
            val_files: Validation files
            test_files: Test files
        """
        # Create output directories
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        test_dir = self.output_dir / "test"
        
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        def copy_file_list(files, dest_dir):
            for file_path in files:
                dest_path = dest_dir / file_path.name
                
                # Handle duplicate filenames
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1
                
                shutil.copy2(file_path, dest_path)
        
        logger.info("Copying files...")
        copy_file_list(train_files, train_dir)
        logger.info(f"  ✓ Copied {len(train_files)} files to {train_dir}")
        
        copy_file_list(val_files, val_dir)
        logger.info(f"  ✓ Copied {len(val_files)} files to {val_dir}")
        
        copy_file_list(test_files, test_dir)
        logger.info(f"  ✓ Copied {len(test_files)} files to {test_dir}")
    
    def save_metadata(
        self,
        train_files: List[Path],
        val_files: List[Path],
        test_files: List[Path]
    ):
        """
        Save split metadata
        
        Args:
            train_files: Training files
            val_files: Validation files
            test_files: Test files
        """
        metadata = {
            'source_directory': str(self.source_dir),
            'output_directory': str(self.output_dir),
            'total_files': len(train_files) + len(val_files) + len(test_files),
            'train_files': len(train_files),
            'val_files': len(val_files),
            'test_files': len(test_files),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'file_paths': {
                'train': [str(f) for f in train_files],
                'val': [str(f) for f in val_files],
                'test': [str(f) for f in test_files]
            }
        }
        
        metadata_path = self.output_dir / "split_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def run(self) -> dict:
        """
        Execute data splitting pipeline
        
        Returns:
            Dictionary with split information
        """
        logger.info("="*80)
        logger.info("STEP 1: DATA SPLITTING")
        logger.info("="*80)
        
        # Collect files
        files = self.collect_files()
        
        if len(files) == 0:
            raise ValueError("No files found to split")
        
        # Split files
        train_files, val_files, test_files = self.split_files(files)
        
        # Copy files
        self.copy_files(train_files, val_files, test_files)
        
        # Save metadata
        self.save_metadata(train_files, val_files, test_files)
        
        result = {
            'train_dir': str(self.output_dir / "train"),
            'val_dir': str(self.output_dir / "val"),
            'test_dir': str(self.output_dir / "test"),
            'train_files': len(train_files),
            'val_files': len(val_files),
            'test_files': len(test_files),
            'total_files': len(files)
        }
        
        logger.info("="*80)
        logger.info("✓ STEP 1 COMPLETE")
        logger.info("="*80)
        
        return result


def run_data_splitting(config: dict) -> dict:
    """
    Convenience function to run data splitting
    
    Args:
        config: Configuration dictionary with keys:
            - source_dir: Source directory
            - output_dir: Output directory
            - train_ratio: Training ratio (default: 0.8)
            - val_ratio: Validation ratio (default: 0.1)
            - test_ratio: Test ratio (default: 0.1)
            - random_seed: Random seed (default: 42)
    
    Returns:
        Dictionary with split information
    """
    splitter = DataSplitter(
        source_dir=config['source_dir'],
        output_dir=config['output_dir'],
        train_ratio=config.get('train_ratio', 0.8),
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1),
        random_seed=config.get('random_seed', 42)
    )
    
    return splitter.run()


if __name__ == "__main__":
    # Example usage
    config = {
        'source_dir': './data/preprocessed',
        'output_dir': './data/splits',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'random_seed': 42
    }
    
    result = run_data_splitting(config)
    print(json.dumps(result, indent=2))