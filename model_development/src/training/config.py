# src/training/config.py
def get_training_config():
    return {
        'base_model': 'bigcode/starcoder2-3b',
        'output_dir': '/opt/airflow/models/checkpoints/starcoder-finetuned',
        'dataset_path': '/opt/airflow/data/processed/code_dataset.json',
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'num_epochs': 3,
        'learning_rate': 2e-4,
        'max_length': 512,
        'warmup_steps': 50,
        'mlflow_uri': 'http://mlflow:5000',
        'experiment_name': 'starcoder2-production',
    }
