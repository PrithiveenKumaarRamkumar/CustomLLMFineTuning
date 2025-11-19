# pipeline/06_model_saver.py
# ================================================================================
# Module 6: Save fine-tuned model
# ================================================================================

from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)


class ModelSaver:
    """Save fine-tuned model and adapters"""
    
    def __init__(
        self,
        output_dir: str,
        save_full_model: bool = True,
        save_adapters_only: bool = True
    ):
        """
        Initialize model saver
        
        Args:
            output_dir: Output directory
            save_full_model: Save full merged model
            save_adapters_only: Save LoRA adapters separately
        """
        self.output_dir = Path(output_dir)
        self.save_full_model = save_full_model
        self.save_adapters_only = save_adapters_only
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelSaver initialized")
        logger.info(f"Output directory: {output_dir}")
    
    def save(self, trainer, tokenizer, training_results: dict) -> dict:
        """
        Save model and tokenizer
        
        Args:
            trainer: Trainer object
            tokenizer: Tokenizer
            training_results: Training results dictionary
        
        Returns:
            Dictionary with save information
        """
        logger.info("="*80)
        logger.info("STEP 6: SAVING MODEL")
        logger.info("="*80)
        
        results = {}
        
        # Save full model
        if self.save_full_model:
            full_model_dir = self.output_dir / "full_model"
            full_model_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving full model to: {full_model_dir}")
            trainer.save_model(str(full_model_dir))
            tokenizer.save_pretrained(str(full_model_dir))
            
            results['full_model_path'] = str(full_model_dir)
            logger.info(f"✓ Full model saved")
        
        # Save LoRA adapters separately
        if self.save_adapters_only:
            adapters_dir = self.output_dir / "lora_adapters"
            adapters_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving LoRA adapters to: {adapters_dir}")
            trainer.model.save_pretrained(str(adapters_dir))
            
            results['adapters_path'] = str(adapters_dir)
            logger.info(f"✓ LoRA adapters saved")
        
        # Save model card
        model_card = self._create_model_card(training_results)
        model_card_path = self.output_dir / "MODEL_CARD.md"
        
        with open(model_card_path, 'w') as f:
            f.write(model_card)
        
        results['model_card_path'] = str(model_card_path)
        logger.info(f"✓ Model card saved to: {model_card_path}")
        
        # Save metadata
        metadata = {
            'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_results': training_results,
            'paths': results
        }
        
        metadata_path = self.output_dir / "save_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to: {metadata_path}")
        
        logger.info("="*80)
        logger.info("✓ STEP 6 COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _create_model_card(self, training_results: dict) -> str:
        """Create model card"""
        training_time = training_results.get('training_time', 0)
        final_loss = training_results.get('final_loss', 0)
        
        model_card = f"""# Fine-Tuned StarCoder2 Model

## Model Information
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Fine-tuning Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Training Time**: {training_time/60:.2f} minutes
- **Final Loss**: {final_loss:.4f}

## Usage

### Load Full Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.output_dir}/full_model")
tokenizer = AutoTokenizer.from_pretrained("{self.output_dir}/full_model")

# Generate code
prompt = "Write a Python function to calculate factorial"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Load LoRA Adapters
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
tokenizer = AutoTokenizer.from_pretrained("base_model_path")

# Load adapters
model = PeftModel.from_pretrained(base_model, "{self.output_dir}/lora_adapters")
```

## Training Details
- See `save_metadata.json` for complete training configuration
- Logs available in `logs/` directory
```
"""
        return model_card


def run_model_saving(trainer, tokenizer, training_results: dict, config: dict) -> dict:
    """
    Convenience function to save model
    
    Args:
        trainer: Trainer object
        tokenizer: Tokenizer
        training_results: Training results
        config: Save configuration
    
    Returns:
        Dictionary with save information
    """
    saver = ModelSaver(
        output_dir=config['output_dir'],
        save_full_model=config.get('save_full_model', True),
        save_adapters_only=config.get('save_adapters_only', True)
    )
    
    return saver.save(trainer, tokenizer, training_results)


if __name__ == "__main__":
    # Example usage
    save_config = {
        'output_dir': './output/final_model',
        'save_full_model': True,
        'save_adapters_only': True
    }
    
    # Assuming trainer and tokenizer are available
    # results = run_model_saving(trainer, tokenizer, training_results, save_config)
    print("Model saver configured")