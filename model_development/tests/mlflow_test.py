import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("starcoder2-production")

# Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get best model
best_run = runs.sort_values('metrics.final_loss').iloc[0]
print(f"Best run: {best_run['run_id']}")