# StarCoder2-3B MLOps Pipeline

Production-ready ML pipeline for training, evaluating, and deploying StarCoder2-3B.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start services
docker-compose up -d

# 3. Initialize Airflow
docker-compose run airflow-webserver airflow db init
docker-compose run airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# 4. Access services
# - Airflow: http://localhost:8080 (admin/admin)
# - MLflow: http://localhost:5000
# - FastAPI: http://localhost:8000/docs
```

## Pipeline Architecture

```
Data -> Validation -> Training -> Evaluation -> Quality Gate -> Deploy
                       |
                    MLflow
```

## Triggering Pipelines

### Via Airflow UI
1. Go to http://localhost:8080
2. Find `starcoder2_training_pipeline`
3. Click "Trigger DAG"

### Via API
```bash
curl -X POST http://localhost:8080/api/v1/dags/starcoder2_training_pipeline/dagRuns \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{}'
```

### Via CLI
```bash
docker-compose run airflow-scheduler airflow dags trigger starcoder2_training_pipeline
```

## Monitoring

- **Airflow Logs**: Check DAG execution in Airflow UI
- **MLflow**: Track experiments at http://localhost:5000
- **API Logs**: `docker-compose logs -f inference-api`

## Configuration

Edit files in `config/`:
- `training_config.yaml`: Training hyperparameters
- `eval_config.yaml`: Evaluation settings
- `inference_config.yaml`: API configuration

## Troubleshooting

**Airflow not starting:**
```bash
docker-compose down
docker-compose up -d
```

**API failing:**
```bash
docker-compose logs inference-api
```
