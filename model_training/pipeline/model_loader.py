# pipeline/03_model_loader.py
# ================================================================================
# Module 3: Load base model
# ================================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load base model with quantization"""
    
    def __init__(
        self,
        model_path: str,
        use_4bit: bool = True,
        use_8bit: bool = False,
        device: str = "auto",
        trust_remote_code: bool = True
    ):
        """
        Initialize model loader
        
        Args:
            model_path: Path to base model
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            device: Device to load model on
            trust_remote_code: Trust remote code
        """
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.device = device
        self.trust_remote_code = trust_remote_code
        
        logger.info("ModelLoader initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"4-bit: {use_4bit}, 8-bit: {use_8bit}")
    
    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration"""
        if self.use_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.use_8bit:
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.info("No quantization")
            return None
    
    def get_dtype(self) -> torch.dtype:
        """Get model dtype"""
        if self.use_4bit or self.use_8bit:
            return torch.float16
        return torch.float32
    
    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info("="*80)
        logger.info("STEP 3: MODEL LOADING")
        logger.info("="*80)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Load model
        logger.info("Loading model...")
        quantization_config = self.get_quantization_config()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.get_dtype(),
            low_cpu_mem_usage=True
        )
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        logger.info(f"Model dtype: {model.dtype}")
        
        logger.info("="*80)
        logger.info("✓ STEP 3 COMPLETE")
        logger.info("="*80)
        
        return model, tokenizer


def run_model_loading(config: dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to load model
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(
        model_path=config['model_path'],
        use_4bit=config.get('use_4bit', True),
        use_8bit=config.get('use_8bit', False),
        device=config.get('device', 'auto'),
        trust_remote_code=config.get('trust_remote_code', True)
    )
    
    return loader.load()


if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': './models/starcoder2-3b',
        'use_4bit': True,
        'use_8bit': False,
        'device': 'auto'
    }
    
    model, tokenizer = run_model_loading(config)
    print(f"Model loaded: {type(model).__name__}")
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")