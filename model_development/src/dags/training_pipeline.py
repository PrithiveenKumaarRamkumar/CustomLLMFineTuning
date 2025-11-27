# dags/training_pipeline.py
"""
Airflow DAG for StarCoder2-3B Training Pipeline
- Data validation
- Model training with LoRA
- MLflow tracking
- Model evaluation
- Deployment trigger
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
import sys
sys.path.append('/opt/airflow/src')

from training.train import train_model
from evaluation.evaluate import evaluate_model
from utils.mlflow_utils import promote_to_production, compare_models

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'starcoder2_training_pipeline',
    default_args=default_args,
    description='Complete training pipeline for StarCoder2-3B',
    schedule_interval='@weekly',  # Run weekly
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'starcoder', 'training'],
)

# Task 1: Validate data
def validate_data(**context):
    """Validate training data quality"""
    import json
    from pathlib import Path
    
    dataset_path = "/opt/airflow/data/processed/code_dataset.json"
    
    print(f"Validating dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Validation checks
    assert len(data) > 0, "Dataset is empty"
    assert len(data) >= 10, f"Dataset too small: {len(data)} samples"
    
    # Check required fields
    required_fields = ['text', 'language']
    for i, item in enumerate(data[:5]):
        for field in required_fields:
            assert field in item, f"Missing {field} in sample {i}"
    
    # Check for empty samples
    empty_samples = sum(1 for item in data if not item.get('text', '').strip())
    assert empty_samples == 0, f"Found {empty_samples} empty samples"
    
    print(f"Validation passed: {len(data)} samples")
    
    # Push metadata to XCom
    context['task_instance'].xcom_push(key='dataset_size', value=len(data))
    context['task_instance'].xcom_push(key='dataset_path', value=dataset_path)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Task 2: Train model
def train_model_task(**context):
    """Train StarCoder2-3B with LoRA"""
    from training.train import train_model
    from training.config import get_training_config
    
    # Get config
    config = get_training_config()
    
    # Get dataset info from previous task
    dataset_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='dataset_path'
    )
    
    config['dataset_path'] = dataset_path
    
    print("Starting training...")
    print(f"Config: {config}")
    
    # Train model
    run_id, metrics = train_model(config)
    
    print(f"Training complete!")
    print(f"  Run ID: {run_id}")
    print(f"  Final Loss: {metrics.get('final_loss', 'N/A')}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    context['task_instance'].xcom_push(key='final_loss', value=metrics.get('final_loss'))
    
    return run_id

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# Task 3: Evaluate model
def evaluate_model_task(**context):
    """Evaluate trained model"""
    from evaluation.evaluate import evaluate_model
    from evaluation.config import get_eval_config
    
    # Get run ID from training
    run_id = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='run_id'
    )
    
    print(f"Evaluating model from run: {run_id}")
    
    # Get config
    config = get_eval_config()
    config['run_id'] = run_id
    
    # Evaluate
    results = evaluate_model(config)
    
    print(f"Evaluation complete!")
    print(f"  CodeBLEU: {results.get('codebleu', 'N/A')}")
    print(f"  BLEU: {results.get('bleu', 'N/A')}")
    print(f"  Syntax Validity: {results.get('syntax_validity', 'N/A')}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='codebleu', value=results.get('codebleu'))
    context['task_instance'].xcom_push(key='bleu', value=results.get('bleu'))
    context['task_instance'].xcom_push(key='syntax_validity', value=results.get('syntax_validity'))
    
    return results

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Task 4: Quality gate
def quality_gate(**context):
    """Check if model meets quality thresholds"""
    
    # Get metrics from evaluation
    codebleu = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='codebleu'
    )
    bleu = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='bleu'
    )
    syntax_validity = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='syntax_validity'
    )
    
    # Define thresholds
    THRESHOLDS = {
        'codebleu_min': 0.3,
        'bleu_min': 0.2,
        'syntax_validity_min': 0.7
    }
    
    print("Checking quality thresholds...")
    print(f"  CodeBLEU: {codebleu} (threshold: {THRESHOLDS['codebleu_min']})")
    print(f"  BLEU: {bleu} (threshold: {THRESHOLDS['bleu_min']})")
    print(f"  Syntax: {syntax_validity} (threshold: {THRESHOLDS['syntax_validity_min']})")
    
    # Check thresholds
    passed = True
    failures = []
    
    if codebleu is not None and codebleu < THRESHOLDS['codebleu_min']:
        passed = False
        failures.append(f"CodeBLEU too low: {codebleu}")
    
    if bleu is not None and bleu < THRESHOLDS['bleu_min']:
        passed = False
        failures.append(f"BLEU too low: {bleu}")
    
    if syntax_validity is not None and syntax_validity < THRESHOLDS['syntax_validity_min']:
        passed = False
        failures.append(f"Syntax validity too low: {syntax_validity}")
    
    if not passed:
        error_msg = "Quality gate failed:\n" + "\n".join(failures)
        print(f"{error_msg}")
        raise ValueError(error_msg)
    
    print("Quality gate passed!")
    context['task_instance'].xcom_push(key='quality_passed', value=True)

quality_gate_task = PythonOperator(
    task_id='quality_gate',
    python_callable=quality_gate,
    dag=dag,
)

# Task 5: Promote to production
def promote_model(**context):
    """Promote model to production in MLflow"""
    from utils.mlflow_utils import promote_to_production
    
    run_id = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='run_id'
    )
    
    print(f"Promoting model {run_id} to production...")
    
    promote_to_production(
        model_name="starcoder2-finetuned",
        run_id=run_id
    )
    
    print("Model promoted to production")

promote_task = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_model,
    dag=dag,
)

# Task 6: Deploy to FastAPI
deploy_task = BashOperator(
    task_id='deploy_to_fastapi',
    bash_command="""
    echo "Restarting FastAPI service with new model..."
    docker-compose restart inference-api
    echo "FastAPI service restarted"
    """,
    dag=dag,
)

# Task 7: Send success notification
def send_success_notification(**context):
    """Send success notification"""
    run_id = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='run_id'
    )
    codebleu = context['task_instance'].xcom_pull(
        task_ids='evaluate_model',
        key='codebleu'
    )
    
    message = f"""
    ✓ StarCoder2 Training Pipeline Completed Successfully!
    
    Run ID: {run_id}
    CodeBLEU: {codebleu:.4f if codebleu else 'N/A'}
    
    Model deployed to production.
    """
    
    print(message)
    # Add Slack/Email notification here
    
send_success = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    dag=dag,
)

# Task 8: Send failure notification
def send_failure_notification(**context):
    """Send failure notification"""
    print("Training pipeline failed!")
    # Add Slack/Email notification here

send_failure = PythonOperator(
    task_id='send_failure_notification',
    python_callable=send_failure_notification,
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# Define task dependencies
validate_data_task >> train_task >> evaluate_task >> quality_gate_task
quality_gate_task >> promote_task >> deploy_task >> send_success
quality_gate_task >> send_failure