# pipeline/08_triton_deployer.py
# ================================================================================
# Module 8: Deploy model to Triton Inference Server
# ================================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class TritonDeployer:
    """Deploy model to Triton Inference Server"""
    
    def __init__(
        self,
        model_path: str,
        triton_model_repository: str,
        model_name: str,
        model_version: str = "1"
    ):
        """
        Initialize Triton deployer
        
        Args:
            model_path: Path to fine-tuned model
            triton_model_repository: Triton model repository path
            model_name: Name for deployed model
            model_version: Model version
        """
        self.model_path = Path(model_path)
        self.triton_model_repository = Path(triton_model_repository)
        self.model_name = model_name
        self.model_version = model_version
        
        logger.info("TritonDeployer initialized")
        logger.info(f"Model: {model_path}")
        logger.info(f"Triton repository: {triton_model_repository}")
        logger.info(f"Model name: {model_name}")
    
    def create_model_directory(self):
        """Create Triton model directory structure"""
        model_dir = self.triton_model_repository / self.model_name
        version_dir = model_dir / self.model_version
        
        version_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created model directory: {model_dir}")
        
        return model_dir, version_dir
    
    def export_model_to_torchscript(self, version_dir: Path):
        """
        Export model to TorchScript format
        
        Args:
            version_dir: Version directory path
        """
        logger.info("Exporting model to TorchScript...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        
        # Create example inputs
        example_input_ids = torch.randint(0, 1000, (1, 128))
        example_attention_mask = torch.ones((1, 128), dtype=torch.long)
        
        try:
            # Try tracing
            logger.info("Attempting to trace model...")
            traced_model = torch.jit.trace(
                model,
                (example_input_ids, example_attention_mask)
            )
            
            model_path = version_dir / "model.pt"
            torch.jit.save(traced_model, str(model_path))
            logger.info(f"✓ Traced model saved to: {model_path}")
        
        except Exception as e:
            logger.warning(f"Tracing failed: {e}")
            logger.info("Attempting to script model...")
            
            # Try scripting
            scripted_model = torch.jit.script(model)
            model_path = version_dir / "model.pt"
            torch.jit.save(scripted_model, str(model_path))
            logger.info(f"✓ Scripted model saved to: {model_path}")
    
    def save_tokenizer(self, model_dir: Path):
        """
        Save tokenizer
        
        Args:
            model_dir: Model directory path
        """
        logger.info("Saving tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        tokenizer_dir = model_dir / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        
        tokenizer.save_pretrained(str(tokenizer_dir))
        logger.info(f"✓ Tokenizer saved to: {tokenizer_dir}")
    
    def generate_config_pbtxt(self, model_dir: Path, max_batch_size: int = 8):
        """
        Generate Triton config.pbtxt
        
        Args:
            model_dir: Model directory path
            max_batch_size: Maximum batch size
        """
        logger.info("Generating config.pbtxt...")
        
        config = f"""name: "{self.model_name}"
platform: "pytorch_libtorch"
max_batch_size: {max_batch_size}
default_model_filename: "model.pt"

input [
  {{
    name: "INPUT_IDS"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "ATTENTION_MASK"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]

output [
  {{
    name: "OUTPUT"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100000
}}
"""
        
        config_path = model_dir / "config.pbtxt"
        with open(config_path, 'w') as f:
            f.write(config)
        
        logger.info(f"✓ Config saved to: {config_path}")
    
    def deploy(self, max_batch_size: int = 8) -> dict:
        """
        Deploy model to Triton
        
        Args:
            max_batch_size: Maximum batch size
        
        Returns:
            Dictionary with deployment information
        """
        logger.info("="*80)
        logger.info("STEP 8: TRITON DEPLOYMENT")
        logger.info("="*80)
        
        # Create directory structure
        model_dir, version_dir = self.create_model_directory()
        
        # Export model to TorchScript
        self.export_model_to_torchscript(version_dir)
        
        # Save tokenizer
        self.save_tokenizer(model_dir)
        
        # Generate config
        self.generate_config_pbtxt(model_dir, max_batch_size)
        
        # Create deployment metadata
        metadata = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_repository': str(self.triton_model_repository),
            'model_directory': str(model_dir),
            'max_batch_size': max_batch_size
        }
        
        metadata_path = model_dir / "deployment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to: {metadata_path}")
        
        logger.info("\n" + "="*80)
        logger.info("DEPLOYMENT INSTRUCTIONS")
        logger.info("="*80)
        logger.info("To start Triton server, run:")
        logger.info(f"  tritonserver --model-repository={self.triton_model_repository}")
        logger.info("\nTo test the deployment:")
        logger.info(f"  curl -X POST http://localhost:8000/v2/models/{self.model_name}/infer")
        logger.info("="*80)
        logger.info("✓ STEP 8 COMPLETE")
        logger.info("="*80)
        
        return metadata


def run_triton_deployment(config: dict) -> dict:
    """
    Convenience function to deploy to Triton
    
    Args:
        config: Deployment configuration
    
    Returns:
        Dictionary with deployment information
    """
    deployer = TritonDeployer(
        model_path=config['model_path'],
        triton_model_repository=config['triton_model_repository'],
        model_name=config['model_name'],
        model_version=config.get('model_version', '1')
    )
    
    return deployer.deploy(max_batch_size=config.get('max_batch_size', 8))


if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': './output/final_model/full_model',
        'triton_model_repository': './triton_models',
        'model_name': 'starcoder2_finetuned',
        'model_version': '1',
        'max_batch_size': 8
    }
    
    result = run_triton_deployment(config)
    print(json.dumps(result, indent=2))