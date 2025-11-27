#!/bin/bash

echo "Setting up StarCoder2 MLOps Pipeline..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize Airflow
export AIRFLOW_HOME=$(pwd)
airflow db init

# Create Airflow user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Create directories
mkdir -p data/processed data/eval models/base models/checkpoints mlruns logs

echo "✓ Setup complete!"
echo "Run: docker-compose up -d"