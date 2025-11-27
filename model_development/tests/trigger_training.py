import requests

# Trigger Airflow DAG
response = requests.post(
    "http://localhost:8080/api/v1/dags/starcoder2_training_pipeline/dagRuns",
    json={},
    auth=("admin", "admin")
)
print(response.json())