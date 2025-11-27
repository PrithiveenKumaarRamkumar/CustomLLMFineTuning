# src/training/train.py
import torch
import mlflow
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MLflowCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    try:
                        mlflow.log_metric(k, v, step=state.global_step)
                    except:
                        pass

def train_model(config):
    '''Train StarCoder2-3B with LoRA and MLflow tracking'''
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config.get('experiment_name', 'starcoder2-production'))
    
    USE_GPU = torch.cuda.is_available()
    
    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log config
        mlflow.log_params(config)
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        if USE_GPU:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                config['base_model'], quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config['base_model'], torch_dtype=torch.float32,
                low_cpu_mem_usage=True, trust_remote_code=True
            )
        
        # Apply LoRA
        for param in model.parameters():
            param.requires_grad = False
        
        lora_config = LoraConfig(
            r=config['lora_r'], lora_alpha=config['lora_alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config['lora_dropout'], bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        
        # Load dataset
        dataset = load_dataset("json", data_files=config['dataset_path'], split="train")
        
        def preprocess(examples):
            texts = examples["text"]
            result = tokenizer(texts, truncation=True, max_length=config['max_length'],
                              padding="max_length", return_tensors=None)
            result["labels"] = result["input_ids"].copy()
            return result
        
        tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
        
        # Training
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            num_train_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            fp16=USE_GPU, logging_steps=5, save_strategy="epoch",
            save_total_limit=2, optim="paged_adamw_8bit" if USE_GPU else "adamw_torch",
            warmup_steps=config['warmup_steps'], lr_scheduler_type="cosine",
            gradient_checkpointing=False, report_to="none",
            dataloader_pin_memory=USE_GPU, remove_unused_columns=False
        )
        
        trainer = Trainer(
            model=model, args=training_args, train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[MLflowCallback()]
        )
        
        result = trainer.train()
        
        # Save and log
        trainer.save_model(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])
        mlflow.log_artifacts(config['output_dir'], artifact_path="model")
        
        metrics = {
            'final_loss': result.training_loss,
            'train_runtime': result.metrics["train_runtime"],
        }
        mlflow.log_metrics(metrics)
        
        return mlflow.active_run().info.run_id, metrics
