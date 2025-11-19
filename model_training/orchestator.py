# orchestrator.py
# ================================================================================
# Main pipeline orchestrator - ties all modules together
# ================================================================================

import logging
import json
from pathlib import Path
from typing import Dict
import sys

# Import all pipeline modules
from pipeline.data_splitting import run_data_splitting
from pipeline.tokenizer import run_tokenization
from pipeline.model_loader import run_model_loading
from pipeline.layer_freezer import run_layer_freezing
from pipeline.qlora_trainer import run_qlora_training, QLoRATrainer
from pipeline.model_saver import run_model_saving
from pipeline.evaluator import run_evaluation
from pipeline.triton_deployer import run_triton_deployment
from pipeline.fastapi_server import run_fastapi_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrate the entire fine-tuning pipeline"""
    
    def __init__(self, config_path: str):
        """
        Initialize orchestrator
        
        Args:
            config_path: Path to pipeline configuration JSON
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {}
        
        logger.info("="*80)
        logger.info("PIPELINE ORCHESTRATOR INITIALIZED")
        logger.info("="*80)
        logger.info(f"Config: {config_path}")
    
    def _load_config(self) -> Dict:
        """Load pipeline configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("Configuration loaded successfully")
        return config
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Data Splitting
            if self.config.get('run_data_splitting', True):
                logger.info("\n>>> Running Step 1: Data Splitting")
                self.results['data_splitting'] = run_data_splitting(
                    self.config['data_splitting']
                )
            
            # Step 2: Tokenization
            if self.config.get('run_tokenization', True):
                logger.info("\n>>> Running Step 2: Tokenization")
                self.results['tokenization'] = run_tokenization(
                    self.config['tokenization']
                )
            
            # Step 3: Model Loading
            if self.config.get('run_training', True):
                logger.info("\n>>> Running Step 3: Model Loading")
                model, tokenizer = run_model_loading(
                    self.config['model_loading']
                )
                
                # Step 4: Layer Freezing
                logger.info("\n>>> Running Step 4: Layer Freezing")
                self.results['layer_freezing'] = run_layer_freezing(
                    model,
                    self.config['layer_freezing']
                )
                
                # Step 5: QLoRA Training
                logger.info("\n>>> Running Step 5: QLoRA Training")
                trainer_config = {
                    'train_dataset_path': self.results['tokenization']['train']['output_path'],
                    'val_dataset_path': self.results['tokenization']['val']['output_path'],
                    'output_dir': self.config['training']['output_dir'],
                    'lora_config': self.config['training']['lora_config'],
                    'training_config': self.config['training']['training_config']
                }
                
                qlora_trainer = QLoRATrainer(
                    model=model,
                    tokenizer=tokenizer,
                    **trainer_config
                )
                self.results['training'] = qlora_trainer.train()
                
                # Step 6: Model Saving
                logger.info("\n>>> Running Step 6: Model Saving")
                self.results['model_saving'] = run_model_saving(
                    qlora_trainer.trainer,
                    tokenizer,
                    self.results['training'],
                    self.config['model_saving']
                )
            
            # Step 7: Evaluation
            if self.config.get('run_evaluation', True):
                logger.info("\n>>> Running Step 7: Evaluation")
                eval_config = self.config['evaluation'].copy()
                eval_config['model_path'] = self.results['model_saving']['full_model_path']
                eval_config['test_dataset_path'] = self.results['tokenization']['test']['output_path']
                
                self.results['evaluation'] = run_evaluation(eval_config)
            
            # Step 8: Triton Deployment
            if self.config.get('run_triton_deployment', False):
                logger.info("\n>>> Running Step 8: Triton Deployment")
                triton_config = self.config['triton_deployment'].copy()
                triton_config['model_path'] = self.results['model_saving']['full_model_path']
                
                self.results['triton_deployment'] = run_triton_deployment(triton_config)
            
            # Save final results
            self._save_results()
            
            logger.info("\n" + "="*80)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            self._save_results()
            raise
    
    def run_inference_server(self):
        """Run FastAPI inference server"""
        logger.info("\n>>> Running Step 9: FastAPI Server")
        
        server_config = self.config['fastapi_server'].copy()
        
        # Use saved model path if available
        if 'model_saving' in self.results:
            server_config['model_path'] = self.results['model_saving']['full_model_path']
        
        run_fastapi_server(server_config)
    
    def _save_results(self):
        """Save pipeline results"""
        results_path = Path(self.config.get('results_output_dir', './pipeline_results'))
        results_path.mkdir(parents=True, exist_ok=True)
        
        results_file = results_path / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Pipeline results saved to: {results_file}")
    
    def _print_summary(self):
        """Print pipeline summary"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*80)
        
        if 'data_splitting' in self.results:
            logger.info(f"✓ Data Split: {self.results['data_splitting']['total_files']} files")
        
        if 'tokenization' in self.results:
            total_samples = sum(
                split['num_samples']
                for split in self.results['tokenization'].values()
            )
            logger.info(f"✓ Tokenization: {total_samples} total samples")
        
        if 'layer_freezing' in self.results:
            freeze_pct = (
                self.results['layer_freezing']['frozen_params'] /
                self.results['layer_freezing']['total_params'] * 100
            )
            logger.info(f"✓ Layer Freezing: {freeze_pct:.2f}% frozen")
        
        if 'training' in self.results:
            train_time = self.results['training']['training_time']
            logger.info(f"✓ Training: {train_time/60:.2f} minutes")
        
        if 'evaluation' in self.results:
            codebleu = self.results['evaluation'].get('codebleu', 0)
            syntax_valid = self.results['evaluation'].get('syntax_validity', 0)
            logger.info(f"✓ Evaluation: CodeBLEU={codebleu:.4f}, Syntax={syntax_valid:.2%}")
        
        if 'model_saving' in self.results:
            model_path = self.results['model_saving']['full_model_path']
            logger.info(f"✓ Model saved to: {model_path}")
        
        logger.info("="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration JSON'
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Run inference server after pipeline'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(args.config)
    orchestrator.run_full_pipeline()
    
    # Optionally start server
    if args.serve:
        orchestrator.run_inference_server()


if __name__ == "__main__":
    main()