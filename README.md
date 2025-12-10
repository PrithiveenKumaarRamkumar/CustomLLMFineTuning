# Custom LLM Fine-Tuning Platform

An enterprise-grade, end-to-end platform for fine-tuning and serving large language models (LLMs) on domain-specific datasets. Built with LoRA/QLoRA parameter-efficient fine-tuning, distributed training infrastructure, production-grade deployment patterns, and comprehensive MLOps capabilities.

**Status**: Production-Ready | **Version**: v1.0+ | **License**: MIT

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Recent Work](#recent-work)
- [How It Works](#how-it-works)
- [Example Use Cases](#example-use-cases)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## 🎯 Overview

The Custom LLM Fine-Tuning Platform is a comprehensive solution designed to democratize LLM customization while maintaining production-grade reliability, scalability, and governance. It addresses the complete lifecycle of custom language model development:

- **Data Management**: Acquire, validate, and version large-scale code datasets
- **Model Training**: Memory-efficient fine-tuning with advanced parameter freezing strategies
- **Experiment Tracking**: Full MLflow integration for reproducible science
- **Model Registry**: Centralized model versioning and deployment management
- **Inference Engine**: High-performance serving with GPU batching and adaptive scheduling
- **Monitoring & Observability**: Real-time performance tracking, drift detection, and quality gates
- **Orchestration**: Automated workflows with Apache Airflow for production deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                             │
│                    (Web Dashboard + API Gateway)                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐
│  DATA PIPELINE  │  │  MODEL TRAINING  │  │  SERVING LAYER │
│                 │  │                  │  │                │
│ • Acquisition   │  │ • QLoRA Fine     │  │ • FastAPI      │
│ • Preprocessing │  │   Tuning         │  │ • GPU Batching │
│ • Validation    │  │ • Distributed    │  │ • LoRA Adapters│
│ • DVC Versioning│  │   Training       │  │ • Auth & Auth  │
│ • Monitoring    │  │ • Hyperparameter │  │ • Metrics      │
│                 │  │   Search         │  │                │
└────────┬────────┘  └────────┬─────────┘  └────────┬───────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────────┐                  ┌────────────────────┐
│  EXPERIMENT TRACKING │                  │  MONITORING STACK  │
│  & REGISTRY          │                  │                    │
│                      │                  │ • Prometheus       │
│ • MLflow Server      │                  │ • Grafana          │
│ • Model Cards        │                  │ • Alertmanager     │
│ • Version Control    │                  │ • Custom Dashboards│
│                      │                  │ • Drift Detection  │
└──────────────────────┘                  └────────────────────┘
        ▲                                           ▲
        │                                           │
        └───────────────────────┬───────────────────┘
                                │
                    ┌───────────┴──────────┐
                    │                      │
                    ▼                      ▼
            ┌──────────────────┐  ┌───────────────┐
            │  ORCHESTRATION   │  │  STORAGE      │
            │                  │  │               │
            │ • Airflow DAGs   │  │ • S3/Local FS │
            │ • Job Scheduler  │  │ • DVC Cache   │
            │ • CI/CD Pipeline │  │ • MLflow DB   │
            └──────────────────┘  └───────────────┘
```

### Data Flow Architecture

```
Raw Code Dataset
       │
       ▼
┌─────────────────────────────────────────┐
│        DATA ACQUISITION LAYER            │
│  (HuggingFace, Software Heritage, S3)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │  Multi-language  │
         │  Organization    │
         │  & Filtering     │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Preprocessing   │
         │  • Deduplication │
         │  • PII Removal   │
         │  • Normalization │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Quality Checks  │
         │  & Validation    │
         │  (Schema, Stats) │
         └────────┬─────────┘
                  │
                  ▼
        ┌───────────────────────┐
        │  DVC Versioning       │
        │  & Training Dataset   │
        └────────┬──────────────┘
                 │
                 ├─────────────────┬──────────────────┐
                 │                 │                  │
                 ▼                 ▼                  ▼
          Model A         Model B          Model C
       (StarCoder)       (LLaMA)          (Falcon)
             │              │                 │
             └──────────────┬─────────────────┘
                            │
                    ┌───────▼────────┐
                    │  QLoRA Fine    │
                    │  Tuning        │
                    │  (Distributed) │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  MLflow        │
                    │  Tracking &    │
                    │  Registry      │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Model Cards   │
                    │  & Evaluation  │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Containerize  │
                    │  & Deploy      │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  FastAPI       │
                    │  Inference     │
                    │  Service       │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Monitoring &  │
                    │  Observability │
                    │  Dashboards    │
                    └────────────────┘
```

### Component Interaction

| Component | Purpose | Technology Stack |
|-----------|---------|------------------|
| **Data Pipeline** | Acquire, validate, and version datasets | Python, DVC, Airflow, HuggingFace, S3 |
| **Training Module** | Fine-tune models with QLoRA | PyTorch, PEFT/LoRA, Transformers, HF Accelerate |
| **Model Registry** | Track experiments and versions | MLflow, Model Cards, DVC |
| **Serving Engine** | Deploy and inference | FastAPI, CUDA, TorchServe, Docker |
| **Monitoring Stack** | Observability and alerts | Prometheus, Grafana, AlertManager |
| **Orchestration** | Workflow automation | Apache Airflow, Kubernetes-ready |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU with CUDA 11.8+ (recommended for training)
- Docker & Docker Compose (for containerized deployment)
- Git LFS (for model checkpoints)

### Installation

```bash
# Clone repository
git clone https://github.com/pranudeepmetuku10/CustomLLMFineTuning.git
cd CustomLLMFineTuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install module-specific requirements
pip install -r Data-Pipeline/requirements.txt
pip install -r model_training/requirements.txt
pip install -r serving/requirements.txt
```

### Basic Usage

```bash
# 1. Set up Data Pipeline
cd Data-Pipeline
# Configure data_config.yaml
python scripts/data_acquisition.py
python scripts/preprocessing.py

# 2. Fine-tune a model
cd ../model_training
# Configure pipeline_config.json
python orchestrator.py

# 3. Deploy inference service
cd ../serving
python start_api.py
# Access at http://localhost:8000/docs
```

---

## ✨ Key Features

**Fine-Tuning** → Parameter-efficient training (LoRA/QLoRA) on user datasets with custom layer freezing strategies.

**Scalability** → Distributed training across multiple GPUs and nodes with automatic batch optimization.

**Experiment Tracking** → Complete MLflow integration with automatic model cards and metrics dashboards.

**Serving** → Production-ready FastAPI service with GPU batching, async processing, and Kubernetes support.

**Monitoring** → Real-time drift detection, performance dashboards, quality gates, and feedback loops.

**Data Versioning** → DVC integration for reproducible data pipelines and dataset tracking.

**Multi-Model Support** → Pre-configured pipelines for StarCoder2, LLaMA, Falcon, and custom base models.

**Security** → Bearer token authentication, API rate limiting, and input validation.

---

## 📁 Project Structure

```
CustomLLMFineTuning/
├── Data-Pipeline/              # Data acquisition, preprocessing, validation
│   ├── configs/               # YAML configuration files
│   ├── dags/                  # Apache Airflow orchestration
│   ├── scripts/               # Core data processing modules
│   ├── monitoring/            # Prometheus & Grafana setup
│   ├── tests/                 # Unit, integration, E2E tests
│   └── README.md              # Detailed data pipeline documentation
│
├── model_training/            # QLoRA fine-tuning and evaluation
│   ├── pipeline/              # Training pipeline modules
│   ├── data/                  # Training datasets
│   ├── models/                # Model checkpoints
│   ├── orchestrator.py        # Main training orchestrator
│   └── Readme.Md              # Training documentation
│
├── serving/                   # FastAPI inference service
│   ├── api/                   # API endpoints and models
│   ├── docker/                # Dockerization
│   ├── assets/                # Architecture diagrams
│   ├── start_api.py           # API server entry point
│   └── README.md              # Serving documentation
│
├── registry/                  # MLflow model registry
│   ├── mlflow_registry.py     # Registry client
│   ├── model_card_gen.py      # Auto model card generation
│   └── registry_client.py     # Registry utilities
│
├── orchestration/             # Workflow orchestration
│   ├── ci_cd.yaml             # CI/CD pipeline definition
│   ├── kuberflow_pipeline.py  # Kubernetes workflow
│   └── job_scheduler.py       # Job scheduling utilities
│
├── training/                  # Distributed training utilities
│   ├── distributed_train.sh   # Multi-node training script
│   ├── hyperparam_search.py   # Hyperparameter optimization
│   └── trainer_utils.py       # Training utilities
│
├── ui/                        # User interface
│   ├── backend/               # Backend API
│   └── frontend/              # Web dashboard
│
├── api/                       # REST API utilities
├── reports/                   # Generated reports
├── requirements.txt           # Main dependencies
└── README.md                  # This file
```

---

## Recent Work

**Data Pipeline Implementation** → For detailed information checkout [Data-Pipeline README](Data-Pipeline/README.md) 

The data pipeline includes:
- Multi-language code dataset acquisition from HuggingFace and Software Heritage
- Comprehensive preprocessing with deduplication (MinHash-based), PII removal, and quality validation
- Apache Airflow orchestration for production deployments
- Real-time monitoring with Prometheus & Grafana
- End-to-end testing with pytest and coverage reporting

---

## 📊 How It Works

```
Upload Dataset
      │
      ▼
   Preprocessing & Validation
      │
      ▼
   Fine-tune Base Model
   (StarCoder, LLaMA, Falcon)
      │
      ▼
   Track Experiments & Metrics
   (MLflow)
      │
      ▼
   Deploy as API Endpoint
   (FastAPI + Docker)
      │
      ▼
   Monitor Performance
      │
      ▼
   Retrain with New Data
   (Feedback Loop)
```

### End-to-End Workflow

1. **Data Ingestion**: Upload or select domain-specific code datasets
2. **Preprocessing**: Automatic cleaning, deduplication, and schema validation
3. **Experiment Setup**: Configure model, hyperparameters, and training strategy
4. **Fine-Tuning**: Train with QLoRA/LoRA with distributed capabilities
5. **Evaluation**: Automated metrics collection (CodeBLEU, perplexity, syntax validity)
6. **Versioning**: Automatic model card generation and registry management
7. **Deployment**: Container-ready deployment with one command
8. **Monitoring**: Real-time performance tracking and drift detection
9. **Iteration**: Feedback loop for continuous improvement

---

## 💼 Example Use Cases

**FinTech Copilots** → Trained on regulatory compliance codebases and financial algorithms for secure, compliant code generation.

**Healthcare Assistants** → Fine-tuned on medical knowledge bases and healthcare-specific code patterns for clinical decision support.

**Enterprise AI Copilots** → Customized for private code repositories, internal frameworks, and proprietary architectures.

**DevOps Automation** → Models specialized in infrastructure-as-code, deployment scripts, and system administration.

**Domain-Specific Code Generation** → Quantum computing, scientific computing, embedded systems, or specialized domains.

---

## 🔧 Core Components

### 1. Data Pipeline (`Data-Pipeline/`)

**Purpose**: Robust, scalable data acquisition and processing

**Key Capabilities**:
- Fetches code from The Stack v2 dataset (2.8B files)
- Multi-language support (Python, Java, C++, JavaScript)
- Automatic deduplication with MinHash (85% threshold)
- PII detection and removal
- Schema validation and quality gates
- DVC integration for version control
- Airflow orchestration for production

**Configuration**: `Data-Pipeline/configs/`

**Further Reading**: [Data Pipeline README](Data-Pipeline/README.md)

---

### 2. Model Training (`model_training/`)

**Purpose**: Memory-efficient fine-tuning of large language models

**Key Capabilities**:
- QLoRA 4-bit quantization for memory efficiency
- Custom layer freezing strategies
- Distributed training (multi-GPU, multi-node)
- Comprehensive evaluation metrics (CodeBLEU, syntax validity, perplexity)
- Hyperparameter search and optimization
- MLflow experiment tracking
- Production-ready model export

**Configuration**: `model_training/pipeline_config_template.json`

**Further Reading**: [Model Training README](model_training/Readme.Md)

---

### 3. Model Registry (`registry/`)

**Purpose**: Centralized model versioning and metadata management

**Key Features**:
- MLflow model registry integration
- Automatic model card generation
- Experiment comparison tools
- Version promotion workflows
- Model metadata tracking

**Usage**:
```python
from registry.mlflow_registry import ModelRegistry
registry = ModelRegistry()
registry.register_model("my-model", "models/checkpoint")
```

---

### 4. Serving Engine (`serving/`)

**Purpose**: Production-grade inference with high performance

**Key Capabilities**:
- FastAPI async framework
- GPU batching with adaptive scheduling
- LoRA adapter dynamic loading
- Bearer token authentication
- Prometheus metrics collection
- Health checks for Kubernetes
- Comprehensive error handling
- OpenAPI documentation

**API Endpoints**:
- `POST /predict` - Single inference
- `POST /predict/batch` - Batch inference
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Swagger UI

**Further Reading**: [Serving README](serving/README.md)

---

### 5. Monitoring Stack (`Data-Pipeline/monitoring/`)

**Purpose**: Observability and operational insights

**Components**:
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Visualization dashboards
- **AlertManager**: Alert routing and notifications
- **Custom Exporter**: Pipeline-specific metrics

**Dashboards Included**:
- Data Pipeline Metrics
- Model Performance
- Inference Latency
- Data Drift Detection
- System Resources

**Further Reading**: [Monitoring Guide](Data-Pipeline/monitoring/DASHBOARD_GUIDE.md)

---

### 6. Orchestration (`orchestration/`)

**Purpose**: Automated workflow management

**Components**:
- Apache Airflow DAGs for data pipeline
- CI/CD pipeline definition
- Kubernetes workflow support
- Job scheduling and retry logic

---

## 📚 Getting Started

### For Data Scientists

1. **Prepare your dataset**: Use the Data Pipeline to acquire and preprocess data
2. **Configure training**: Edit `model_training/pipeline_config_template.json`
3. **Run fine-tuning**: Execute `python orchestrator.py`
4. **Monitor experiments**: View MLflow dashboard at `http://localhost:5000`

### For MLOps Engineers

1. **Set up infrastructure**: Deploy Airflow, Prometheus, Grafana using provided Docker Compose
2. **Configure monitoring**: Customize dashboards in `Data-Pipeline/monitoring/dashboards/`
3. **Deploy service**: Use Docker Compose or Kubernetes manifests
4. **Monitor pipelines**: Access Grafana at configured URL

### For Application Developers

1. **Query the API**: Use `curl`, Python `requests`, or auto-generated OpenAPI clients
2. **Authenticate**: Include Bearer token in Authorization header
3. **Handle responses**: Parse structured JSON responses with error handling
4. **Monitor performance**: Track metrics and latency via dashboards

---

## 📖 Documentation

- [Data Pipeline Guide](Data-Pipeline/README.md)
- [Model Training Guide](model_training/Readme.Md)
- [Serving API Guide](serving/README.md)
- [Monitoring Dashboard Guide](Data-Pipeline/monitoring/DASHBOARD_GUIDE.md)
- [MLflow Model Registry Guide](registry/)
- [Deployment Guide](orchestration/)

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort

# Run tests
pytest tests/ -v --cov

# Format code
black .
isort .
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙋 Support

For questions, issues, or suggestions:

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check relevant README files in each component directory
- **Email**: Contact the maintainers

---

## 🔮 Roadmap

- [ ] Web UI for experiment management
- [ ] Advanced hyperparameter optimization (Optuna integration)
- [ ] Multi-model ensemble serving
- [ ] Federated learning support
- [ ] ONNX export and optimization
- [ ] Edge deployment (TensorRT, NCNN)
- [ ] Real-time retraining pipelines
- [ ] Advanced model interpretability features

---

**Last Updated**: December 2024 | **Maintained By**: CustomLLM Team
