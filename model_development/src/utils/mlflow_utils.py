import mlflow

def promote_to_production(model_name, run_id):
    '''Promote model to production'''
    client = mlflow.tracking.MlflowClient()
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    
    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Model version {mv.version} promoted to Production")

def compare_models(experiment_name):
    '''Compare all models in experiment'''
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.final_loss ASC"]
    )
    return runs