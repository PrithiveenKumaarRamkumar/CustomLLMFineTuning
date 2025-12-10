# LLM Fine-Tuning Platform

A production-ready multi-tenant platform for fine-tuning large language models using QLoRA with integrated data management, experiment tracking, and inference serving.

**Status**: Production-Ready | **Version**: v1.0+ 

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## Overview

The LLM Fine-Tuning Platform is a comprehensive solution for democratizing custom LLM development while maintaining enterprise-grade reliability, security, and observability. It handles the complete lifecycle:

- **Multi-Tenant Architecture**: Isolated per-user environments with JWT authentication
- **Data Management**: Upload, validate, preprocess, and version datasets
- **Parameter-Efficient Training**: QLoRA fine-tuning on StarCoder2-3B
- **Experiment Tracking**: Vertex AI integration for reproducible science
- **Model Serving**: High-performance FastAPI inference engine with adapter loading
- **Monitoring & Observability**: Prometheus, Grafana, and drift detection
- **Cloud Integration**: Google Cloud Storage and Vertex AI support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Frontend (Next.js/TypeScript)                  │
│                   Web Dashboard                             │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     AUTH     │  │     DATA     │  │   TRAINING   │
│              │  │              │  │              │
│ • JWT/RBAC   │  │ • Upload     │  │ • QLoRA      │
│ • User Mgmt  │  │ • Preprocess │  │ • Distribute │
│ • Token Mgmt │  │ • Validate   │  │ • Track      │
│              │  │ • Split      │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                    ┌────▼────┐
                    │ FastAPI │
                    │ Serving  │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌────────────┐ ┌──────────┐ ┌───────────┐
    │PostgreSQL  │ │  Redis   │ │    GCS    │
    │(Metadata)  │ │(Caching) │ │(Storage)  │
    └────────────┘ └──────────┘ └───────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌──────────┐ ┌───────────────┐
    │Prometheus│ │  Vertex AI    │
    │(Metrics) │ │(Tracking &    │
    └──────────┘ │ Training)     │
                 └───────────────┘
```

### Data Flow

```
User Upload → Validation → Preprocessing → Deduplication → PII Removal → 
Bias Detection → Dataset Splitting → GCS Upload → Training Job → 
Vertex AI Tracking → Model Registry → Inference Service → Monitoring
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for local dev)
- GCP account with billing (for cloud features)
- CUDA 11.8+ (optional, for GPU training)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd llm-finetuning-platform

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d

# Run migrations
alembic upgrade head

# Start API server
python run.py
```

**API Docs**: http://localhost:8000/docs

---

## Key Features

| Feature | Description |
|---------|-------------|
| **QLoRA Fine-Tuning** | 4-bit quantization + LoRA adapters for memory efficiency |
| **Multi-Tenant** | Per-user data isolation in PostgreSQL and GCS |
| **Dataset Pipeline** | Upload, validate, deduplicate, remove PII, split data |
| **Experiment Tracking** | Vertex AI integration with metrics, artifacts, and model cards |
| **Async API** | FastAPI with async/await for high concurrency |
| **GPU Inference** | Dynamic adapter loading with model caching |
| **Monitoring** | Prometheus metrics, Grafana dashboards, drift detection |
| **Cloud-Native** | Vertex AI integration, GCS storage, Docker/Kubernetes ready |
| **Email Alerts** | Training notifications (success/failure) |
| **Security** | JWT authentication, CORS, rate limiting, input validation |

---

## Project Structure

```
llm-finetuning-platform/
├── auth/                          # Authentication & Authorization
│   ├── models.py                 # SQLAlchemy ORM models (User, Dataset, Adapter)
│   ├── routes.py                 # FastAPI auth endpoints (/signup, /login, /refresh)
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── jwt_handler.py            # JWT token creation & verification
│   ├── dependencies.py           # Dependency injection (get_current_user)
│   └── database.py               # Database connection & initialization
│
├── data/                          # Dataset Management & Preprocessing
│   ├── pipeline.py               # Main orchestrator combining all steps
│   ├── file_handler.py           # File upload handling (JSON, CSV, Python)
│   ├── preprocessing.py          # Cleaning, deduplication, PII removal
│   ├── splitter.py               # Train/val/test splitting logic
│   ├── bias_detection.py         # Statistical bias analysis (Evidently)
│   ├── gcs_pipeline.py           # GCS integration
│   ├── routes.py                 # FastAPI data endpoints (/upload, /datasets)
│   ├── processed/                # Local processed cache
│   ├── raw/                      # Local raw cache
│   └── versioning/               # Version tracking
│
├── training/                      # Model Fine-Tuning
│   ├── train.py                  # Core QLoRA training script
│   ├── routes.py                 # FastAPI training endpoints (/train, /status)
│   ├── vertex_manager.py         # Vertex AI integration
│   ├── email_utils.py            # Training notifications
│   └── Dockerfile                # Training container
│
├── serving/                       # Inference Engine
│   ├── engine.py                 # Model + adapter loading/caching
│   ├── api/
│   │   ├── main.py              # FastAPI app setup, middleware, routes
│   │   ├── inference.py          # Prediction endpoints
│   │   └── __pycache__/
│   └── docker/                   # Serving container config
│
├── storage/                       # Cloud & Local Storage
│   ├── gcs_storage.py            # GCS client (upload/download/list)
│   └── tenant_storage.py         # Multi-tenant path management
│
├── monitoring/                    # Observability
│   ├── prometheus.yml            # Prometheus config
│   └── dashboards/               # Grafana JSON dashboards
│
├── orchestration/                 # Workflow Orchestration
│   ├── auth/, data/, storage/    # Orchestration tasks
│
├── alembic/                       # Database Migrations
│   ├── env.py                    # Migration environment
│   └── versions/                 # Migration files
│
├── configs/                       # Configuration Files
│   ├── database_config.yaml
│   ├── model_config.yaml
│   ├── data_config.yaml
│   ├── serving_config.yaml
│   └── monitoring_config.yaml
│
├── tests/                         # Unit & Integration Tests
│   ├── test_auth_api.py
│   ├── test_data_api.py
│   └── test_env.py
│
├── scripts/                       # Utility Scripts
│   ├── register_model.py
│   ├── test_inference_flow.py
│   └── test_training_flow.py
│
├── utils/                         # Shared Utilities
│   ├── email.py
│   └── email_templates.py
│
├── frontend/                      # Next.js Web UI
│   ├── src/app/
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
│
├── docker-compose.yml             # Local dev services
├── requirements.txt               # Python dependencies
├── alembic.ini                    # Migration config
├── run.py                         # Quick start script
├── SETUP_GUIDE.md                 # Detailed setup
└── docs/GCP_SETUP_GUIDE.md        # Cloud setup guide
```

---

## Core Components

### 1. Authentication Module (`auth/`)

**Purpose**: Multi-tenant user management with JWT tokens

**Endpoints**:
```
POST   /api/auth/signup          - User registration
POST   /api/auth/login           - User login (returns access + refresh tokens)
POST   /api/auth/refresh         - Refresh access token
GET    /api/auth/me              - Get current user profile
POST   /api/auth/change-password - Update password
```

**Features**: Password hashing (bcrypt), JWT signing, token validation, user isolation

---

### 2. Data Management Module (`data/`)

**Purpose**: Dataset upload, validation, preprocessing, and quality checks

**Processing Pipeline**:
```
Upload → Parse → Clean → Deduplicate → Detect PII → Analyze Bias → Split → GCS Upload
```

**Endpoints**:
```
POST   /api/data/upload          - Upload dataset
GET    /api/data/datasets        - List user's datasets
POST   /api/data/process         - Run preprocessing
GET    /api/data/status/{id}     - Check status
DELETE /api/data/{id}            - Delete dataset
```

**Features**: Multi-format support (JSON, CSV, Python), fuzzy deduplication, PII masking, Evidently bias detection

---

### 3. Training Module (`training/`)

**Purpose**: Fine-tune base models with QLoRA

**QLoRA Config**:
- Base Model: `bigcode/starcoder2-3b`
- Quantization: 4-bit NF4
- LoRA Rank: 16
- Learning Rate: 2e-4
- Batch Size: 4

**Endpoints**:
```
POST   /api/training/train       - Submit training job
GET    /api/training/jobs        - List jobs
GET    /api/training/status/{id} - Get job status
CANCEL /api/training/{id}        - Cancel job
```

**Features**: Vertex AI training, metric logging, email notifications, checkpoint management

---

### 4. Inference Engine (`serving/`)

**Purpose**: High-performance model inference with adapter loading

**Endpoints**:
```
POST   /api/inference/predict    - Single prediction
POST   /api/inference/batch      - Batch inference
GET    /api/inference/models     - List adapters
```

**Features**: Lazy model loading, adapter caching, GPU batching, async processing

---

### 5. Monitoring Stack (`monitoring/`)

**Purpose**: System observability and alerting

**Components**:
- Prometheus (metrics collection)
- Grafana (dashboards)
- Custom metrics (request rates, GPU usage, training loss)

**Dashboards**:
- API Performance
- GPU Utilization
- Training Progress
- Data Quality

---

## Getting Started

### For Data Scientists

```bash
# 1. Upload dataset
curl -X POST http://localhost:8000/api/data/upload \
  -H "Authorization: Bearer {token}" \
  -F "file=@dataset.json"

# 2. Check preprocessing status
curl http://localhost:8000/api/data/status/{dataset_id} \
  -H "Authorization: Bearer {token}"

# 3. Submit training job
curl -X POST http://localhost:8000/api/training/train \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "...", "epochs": 3, "learning_rate": 2e-4}'

# 4. Monitor training
curl http://localhost:8000/api/training/status/{job_id} \
  -H "Authorization: Bearer {token}"
```

### For MLOps Engineers

```bash
# Deploy monitoring stack
docker-compose up -d prometheus grafana

# View dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090

# Check training jobs
# Vertex AI: https://console.cloud.google.com/vertex-ai/training
```

### For Application Developers

```bash
# Authenticate
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "..."}'

# Get predictions
curl -X POST http://localhost:8000/api/inference/predict \
  -H "Authorization: Bearer {access_token}" \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "...", "prompt": "def hello"}'
```

---

## API Documentation

### Interactive Swagger UI

Visit: **http://localhost:8000/docs**

All endpoints fully documented with request/response schemas.

### Authentication

Include JWT token in all protected requests:
```
Authorization: Bearer {access_token}
```

### Response Format

```json
{
  "status": "success",
  "data": { ... },
  "error": null,
  "timestamp": "2024-12-10T10:30:00Z"
}
```

---

## Documentation

- [Setup Guide](SETUP_GUIDE.md) - Detailed installation & configuration
- [GCP Setup](docs/GCP_SETUP_GUIDE.md) - Cloud SQL & GCS configuration
- [API Reference](serving/api/main.py) - Endpoint documentation
- [Model Training](training/README.md) - Training guide
- [Data Pipeline](data/README.md) - Data processing guide
- [Monitoring](monitoring/README.md) - Observability setup

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI, Uvicorn, Pydantic |
| **Auth** | Python-Jose, Passlib/bcrypt |
| **Database** | PostgreSQL, SQLAlchemy, Alembic |
| **Cache** | Redis |
| **ML/DL** | PyTorch, Transformers, PEFT (LoRA) |
| **Quantization** | BitsAndBytes (4-bit) |
| **Cloud** | Google Cloud Storage, Vertex AI |
| **Experiment Tracking** | Vertex AI Experiments |
| **Monitoring** | Prometheus, Grafana |
| **Validation** | Evidently AI, Great Expectations |
| **Frontend** | Next.js, TypeScript, React |
| **Orchestration** | Apache Airflow, DVC |
| **Testing** | pytest, pytest-asyncio |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=auth,data,training --cov-report=html

# Integration tests
python scripts/test_training_flow.py
python scripts/test_inference_flow.py
```

---

## Deployment

### Docker Compose (Local)
```bash
docker-compose up -d
python run.py
```

### Docker Container
```bash
docker build -t llm-platform .
docker run -p 8000:8000 --env-file .env llm-platform
```

### Kubernetes
```bash
kubectl apply -f orchestration/k8s/
```

### Google Cloud Run
```bash
gcloud run deploy llm-platform --source . --region us-central1
```

---

## Security

- **Authentication**: JWT tokens with HS256 signing
- **Authorization**: Per-user data isolation
- **Passwords**: Bcrypt hashing with 12 rounds
- **Secrets**: Environment variables + Google Cloud Secret Manager
- **API**: CORS whitelisting, rate limiting, input validation
- **Storage**: GCS paths include user_id for segregation
- **Network**: HTTPS in production, firewall rules in GCP

---

## Monitoring Dashboard

Access Grafana dashboards at: **http://localhost:3000**

**Pre-configured Dashboards**:
- API Request Metrics
- GPU Utilization
- Training Job Progress
- Data Quality Metrics
- System Resources

---

## Contributing

```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8 isort

# 4. Make changes & test
pytest tests/ -v

# 5. Format code
black . && isort . && flake8 .

# 6. Push & create PR
git push origin feature/amazing-feature
```

---

## Roadmap

- [ ] Advanced hyperparameter optimization (Optuna)
- [ ] Multi-model ensemble serving
- [ ] Federated learning support
- [ ] ONNX export and optimization
- [ ] Real-time retraining pipelines
- [ ] Advanced model interpretability
- [ ] Web UI improvements
- [ ] Edge deployment support

---

**Last Updated**: December 2025 | **Version**: 1.0.0
