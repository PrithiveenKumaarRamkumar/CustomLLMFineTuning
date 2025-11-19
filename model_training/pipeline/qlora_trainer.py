# pipeline/05_qlora_trainer.py
# ================================================================================
# Module 5: QLoRA fine-tuning
# ================================================================================

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_from_disk
import mlflow
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class QLoRATrainer:
    """QLoRA-based model fine-tuning"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset_path: str,
        val_dataset_path: str,
        output_dir: str,
        lora_config: dict,
        training_config: dict
    ):
        """
        Initialize QLoRA trainer
        
        Args:
            model: Base model
            tokenizer: Tokenizer
            train_dataset_path: Path to tokenized training dataset
            val_dataset_path: Path to tokenized validation dataset
            output_dir: Output directory
            lora_config: LoRA configuration
            training_config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.output_dir = Path(output_dir)
        self.lora_config = lora_config
        self.training_config = training_config
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("QLoRATrainer initialized")
    
    def apply_qlora(self):
        """Apply QLoRA adapters to model"""
        logger.info("Preparing model for QLoRA training...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('alpha', 32),
            lora_dropout=self.lora_config.get('dropout', 0.1),
            target_modules=self.lora_config.get(
                'target_modules',
                ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense']
            ),
            bias=self.lora_config.get('bias', 'none'),
            inference_mode=False
        )
        
        logger.info("LoRA Configuration:")
        logger.info(f"  Rank (r): {lora_config.r}")
        logger.info(f"  Alpha: {lora_config.lora_alpha}")
        logger.info(f"  Dropout: {lora_config.lora_dropout}")
        logger.info(f"  Target modules: {lora_config.target_modules}")
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def load_datasets(self):
        """Load tokenized datasets"""
        logger.info("Loading datasets...")
        
        self.train_dataset = load_from_disk(self.train_dataset_path)
        self.val_dataset = load_from_disk(self.val_dataset_path)
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Val dataset: {len(self.val_dataset)} samples")
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments"""
        # Auto-detect precision support
        has_gpu = torch.cuda.is_available()
        supports_bf16 = False
        
        if has_gpu:
            capability = torch.cuda.get_device_capability()
            supports_bf16 = capability[0] >= 8
        
        use_bf16 = self.training_config.get('bf16', True) and supports_bf16
        use_fp16 = self.training_config.get('fp16', False) and has_gpu and not use_bf16
        
        if self.training_config.get('bf16', True) and not supports_bf16:
            logger.warning("bf16 not supported, using fp16")
        
        logger.info(f"Training precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_config.get('num_epochs', 3),
            per_device_train_batch_size=self.training_config.get('batch_size', 2),
            per_device_eval_batch_size=self.training_config.get('batch_size', 2),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 8),
            
            learning_rate=self.training_config.get('learning_rate', 2e-4),
            lr_scheduler_type=self.training_config.get('lr_scheduler', 'cosine'),
            warmup_steps=self.training_config.get('warmup_steps', 100),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            max_grad_norm=self.training_config.get('max_grad_norm', 1.0),
            
            optim=self.training_config.get('optimizer', 'paged_adamw_8bit'),
            
            fp16=use_fp16,
            bf16=use_bf16,
            
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=self.training_config.get('logging_steps', 10),
            logging_first_step=True,
            
            eval_strategy='steps',
            eval_steps=self.training_config.get('eval_steps', 500),
            
            save_strategy='steps',
            save_steps=self.training_config.get('save_steps', 500),
            save_total_limit=self.training_config.get('save_total_limit', 3),
            
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            
            report_to=['mlflow'] if self.training_config.get('use_mlflow', True) else [],
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            
            seed=42,
            data_seed=42,
        )
        
        return training_args
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        if not self.training_config.get('use_mlflow', True):
            return
        
        mlflow.set_tracking_uri(self.training_config.get('mlflow_uri', './mlruns'))
        mlflow.set_experiment(
            self.training_config.get('experiment_name', 'starcoder2-finetuning')
        )
        
        run_name = f"qlora-{time.strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Log parameters
        mlflow.log_params({
            'lora_r': self.lora_config.get('r', 16),
            'lora_alpha': self.lora_config.get('alpha', 32),
            'lora_dropout': self.lora_config.get('dropout', 0.1),
            'num_epochs': self.training_config.get('num_epochs', 3),
            'batch_size': self.training_config.get('batch_size', 2),
            'learning_rate': self.training_config.get('learning_rate', 2e-4),
            'train_samples': len(self.train_dataset),
            'val_samples': len(self.val_dataset)
        })
        
        logger.info(f"MLflow run started: {run_name}")
    
    def train(self) -> dict:
        """
        Execute training
        
        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info("STEP 5: QLORA TRAINING")
        logger.info("="*80)
        
        # Apply QLoRA
        self.apply_qlora()
        
        # Load datasets
        self.load_datasets()
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.training_config.get('num_epochs', 3)}")
        logger.info(f"Batch size: {self.training_config.get('batch_size', 2)}")
        logger.info(f"Gradient accumulation: {self.training_config.get('gradient_accumulation_steps', 8)}")
        
        start_time = time.time()
        
        try:
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            logger.info("\n" + "="*80)
            logger.info("TRAINING COMPLETE")
            logger.info("="*80)
            logger.info(f"Training time: {training_time:.2f}s ({training_time/60:.2f}m)")
            logger.info(f"Final loss: {train_result.training_loss:.4f}")
            
            # Log to MLflow
            if self.training_config.get('use_mlflow', True):
                mlflow.log_metric('training_time_seconds', training_time)
                mlflow.log_metric('final_train_loss', train_result.training_loss)
            
            results = {
                'training_time': training_time,
                'final_loss': train_result.training_loss,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            if self.training_config.get('use_mlflow', True):
                mlflow.log_param('training_status', 'failed')
                mlflow.end_run()
            raise
        
        # End MLflow run
        if self.training_config.get('use_mlflow', True):
            mlflow.end_run()
        
        logger.info("="*80)
        logger.info("✓ STEP 5 COMPLETE")
        logger.info("="*80)
        
        # Store trainer for later use
        self.trainer = trainer
        
        return results


def run_qlora_training(model, tokenizer, config: dict) -> dict:
    """
    Convenience function to run QLoRA training
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        config: Training configuration
    
    Returns:
        Dictionary with training results
    """
    trainer = QLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset_path=config['train_dataset_path'],
        val_dataset_path=config['val_dataset_path'],
        output_dir=config['output_dir'],
        lora_config=config['lora_config'],
        training_config=config['training_config']
    )
    
    return trainer.train()


if __name__ == "__main__":
    # Example usage
    from model_loader import run_model_loading
    from layer_freezer import run_layer_freezing
    
    # Load model
    model_config = {
        'model_path': './models/starcoder2-3b',
        'use_4bit': True
    }
    model, tokenizer = run_model_loading(model_config)
    
    # Freeze layers
    freeze_config = {
        'strategy': 'first_n',
        'n_layers': 15,
        'freeze_embeddings': True
    }
    run_layer_freezing(model, freeze_config)
    
    # Train with QLoRA
    train_config = {
        'train_dataset_path': './data/tokenized/train_tokenized',
        'val_dataset_path': './data/tokenized/val_tokenized',
        'output_dir': './output/qlora_training',
        'lora_config': {
            'r': 16,
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense']
        },
        'training_config': {
            'num_epochs': 3,
            'batch_size': 2,
            'gradient_accumulation_steps': 8,
            'learning_rate': 2e-4,
            'use_mlflow': True
        }
    }
    
    results = run_qlora_training(model, tokenizer, train_config)
    print(json.dumps(results, indent=2))