# Data Pipeline - CustomLLMFineTuning

A production-ready data pipeline for acquiring, preprocessing, and managing code datasets for Large Language Model fine-tuning. This pipeline integrates data versioning with DVC, monitoring with Prometheus & Grafana, and comprehensive testing.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Code Structure](#code-structure)
- [Environment Setup](#environment-setup)
- [Running the Pipeline](#running-the-pipeline)
  - [Option 1: Airflow (Recommended)](#option-1-run-with-airflow-recommended-for-production)
  - [Option 2: DVC Directly](#option-2-run-dvc-pipeline-directly)
  - [Option 3: Individual Scripts](#option-3-run-individual-scripts)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This data pipeline processes code from **The Stack v2** dataset (via HuggingFace and Software Heritage) with the following stages:

1. **Data Acquisition**: Download code files from Software Heritage S3 bucket
2. **Organization & Filtering**: Organize by language and filter by size constraints
3. **Preprocessing**: Clean, deduplicate, and remove PII from code
4. **Validation**: Ensure data quality and schema compliance
5. **Statistics Generation**: Generate dataset statistics and reports

## ✨ Features

- **Multi-language Support**: Python, Java, C++, JavaScript
- **Data Versioning**: Full DVC integration for reproducibility
- **Orchestration**: Apache Airflow DAG for production workflows
- **Monitoring Stack**: Real-time metrics with Prometheus & Grafana dashboards
- **PII Removal**: Automated detection and removal of sensitive information
- **Deduplication**: MinHash-based similarity detection (85% threshold)
- **Quality Validation**: Schema validation and data quality checks
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Windows Path Handling**: Special handling for Windows MAX_PATH limitations
- **Checksum Verification**: SHA-256 checksums for data integrity
- **Automatic Retries**: Airflow retry logic for failed tasks
- **Metrics Integration**: Pipeline metrics pushed to Prometheus Pushgateway

## 📁 Code Structure

```
Data-Pipeline/
├── configs/                          # Configuration files
│   ├── data_config.yaml             # Data acquisition settings
│   ├── preprocessing_config.yaml    # Preprocessing parameters
│   ├── monitoring_config.yaml       # Monitoring configuration
│   └── serving_config.yaml          # Model serving settings
│
├── dags/                            # Airflow DAGs (orchestration)
│   ├── data_pipeline_dag.py         # Main pipeline DAG with DVC integration
│   ├── docker-compose.airflow.yml   # Airflow services (webserver, scheduler, postgres)
│   └── Dockerfile.airflow           # Custom Airflow image with dependencies
│
├── data/                            # Data storage (DVC-tracked)
│   ├── raw/                         # Raw downloaded code files
│   ├── staged/                      # Organized and filtered data
│   └── processed/                   # Preprocessed, clean data
│
├── logs/                            # Pipeline execution logs
│   └── .gitignore                   # Keeps folder, ignores files
│
├── monitoring/                      # Monitoring infrastructure
│   ├── docker-compose.monitoring.yml  # Docker services
│   ├── prometheus/                  # Prometheus configuration
│   ├── dashboards/                  # Grafana dashboards
│   ├── alertmanager/                # Alert configuration
│   ├── scripts/                     # Monitoring utilities
│   │   ├── monitoring.py            # PipelineMonitor class
│   │   ├── metrics_exporter.py      # Pushgateway integration
│   │   └── start_metrics_server.py  # Metrics HTTP server
│   ├── start_monitoring.ps1         # Quick-start script
│   ├── MONITORING_SETUP.md          # Monitoring documentation
│   └── DASHBOARD_GUIDE.md           # Dashboard usage guide
│
├── scripts/                         # Core pipeline scripts
│   ├── data_acquisition.py          # Fetch metadata from HuggingFace
│   ├── batch_swh_download.py        # Download from Software Heritage
│   ├── dataset_filter.py            # Organize & filter by language/size
│   ├── preprocessing.py             # Main preprocessing orchestrator
│   ├── pii_removal.py               # PII detection and removal
│   ├── deduplication.py             # MinHash deduplication
│   ├── schema_validation.py         # Data quality validation
│   ├── statistics_generation.py     # Dataset statistics
│   └── logger_config.py             # Centralized logging
│
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests
│   ├── integration/                 # Integration tests
│   ├── test_Acquisition.py          # Data acquisition tests
│   ├── test_preprocessing.py        # Preprocessing tests
│   ├── test_dataset_filter.py       # Filtering tests
│   ├── test_monitoring.py           # Monitoring tests
│   └── test_end_to_end.py          # Full pipeline tests
│
├── dvc.yaml                         # DVC pipeline definition
├── dvc.lock                         # DVC lockfile (version tracking)
└── README.md                        # This file
```

### Key Components

#### Scripts

- **`data_acquisition.py`**: Fetches metadata from HuggingFace's "bigcode/the-stack-v2" dataset
  - Requires HuggingFace token for gated dataset access
  - Filters by: stars, licenses, file size
  - Outputs filtered metadata JSONs per language

- **`batch_swh_download.py`**: Downloads actual code files from Software Heritage S3
  - Reads filtered metadata
  - Downloads from `s3://softwareheritage/content/<blob_id>`
  - Handles long Windows paths (MAX_PATH limitations)
  - Computes SHA-256 checksums with fallback to manifest files

- **`dataset_filter.py`**: Organizes and filters raw data
  - Classifies files by extension (.py, .java, .cpp, .js)
  - Filters by configurable size limits (default: 10 MB max)
  - Generates ingest manifest with metadata
  - Integrated with monitoring (tracks data volume, processing time)

- **`preprocessing.py`**: Main preprocessing orchestrator
  - Encoding detection and fixing
  - Malformed code cleaning
  - PII removal (emails, API keys, SSNs, etc.)
  - MinHash-based deduplication
  - Tokenization preparation
  - Whitespace normalization

- **`pii_removal.py`**: Detects and removes sensitive information
  - Email addresses
  - API keys and tokens
  - Social Security Numbers
  - Credit card numbers
  - Phone numbers
  - IP addresses

- **`deduplication.py`**: Removes near-duplicate code files
  - MinHash algorithm with LSH
  - Configurable similarity threshold (default: 85%)
  - Generates deduplication reports

- **`schema_validation.py`**: Validates processed data
  - Checks file structure compliance
  - Validates metadata completeness
  - Ensures data quality standards

- **`statistics_generation.py`**: Generates dataset statistics
  - File counts per language
  - Size distributions
  - Code complexity metrics
  - Token statistics

## 🛠️ Environment Setup

### Prerequisites

- **Python**: 3.12+ (tested with 3.12.5)
- **Git**: For version control
- **DVC**: For data versioning
- **Docker & Docker Compose**: For monitoring stack (optional but recommended)
- **AWS Credentials**: For downloading from Software Heritage S3 (Requester Pays)
- **HuggingFace Account**: For accessing The Stack v2 dataset

### 1. Clone the Repository

```bash
git clone https://github.com/Aparnashree11/CustomLLMFineTuning.git
cd CustomLLMFineTuning/Data-Pipeline
```

### 2. Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `datasets`, `huggingface_hub`, `transformers` - HuggingFace integration
- `boto3`, `smart-open` - AWS S3 access
- `dvc` - Data versioning
- `prometheus-client` - Metrics collection
- `pytest`, `pytest-cov` - Testing
- `pandas`, `numpy`, `scikit-learn` - Data processing

### 4. Configure Authentication

#### HuggingFace Token
The Stack v2 is a gated dataset requiring authentication:

1. Create account at https://huggingface.co
2. Request access to `bigcode/the-stack-v2`
3. Generate access token: Settings → Access Tokens → New token
4. Set environment variable:

```bash
# Windows
$env:HUGGINGFACE_TOKEN="hf_your_token_here"

# Linux/Mac
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

Or pass via CLI:
```bash
python scripts/data_acquisition.py --language python --hf-token hf_your_token_here
```

#### AWS Credentials (for Software Heritage downloads)
Software Heritage uses a "Requester Pays" S3 bucket:

1. Create AWS account
2. Create IAM user with S3 read permissions
3. Generate access keys
4. Configure credentials:

```bash
# Windows
$env:AWS_ACCESS_KEY_ID="your_key_id"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:AWS_DEFAULT_REGION="us-east-1"

# Linux/Mac
export AWS_ACCESS_KEY_ID="your_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

Or use `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_key_id
aws_secret_access_key = your_secret_key
region = us-east-1
```

### 5. Initialize DVC

```bash
# Initialize DVC (if not already done)
dvc init

# Pull DVC-tracked data (if available from remote)
dvc pull
```

## 🚀 Running the Pipeline

You can run the pipeline in three ways:

1. **Airflow DAG (Recommended for Production)**: Orchestrates DVC stages with scheduling, monitoring, and retries
2. **DVC Directly**: Manual execution of versioned pipeline stages
3. **Individual Scripts**: Run specific steps independently for development/debugging

> **Note**: The Airflow DAG automatically executes DVC stages (`dvc repro`), so running the DAG is sufficient for complete pipeline execution. You don't need to run DVC separately when using Airflow.

### Option 1: Run with Airflow (Recommended for Production)

Apache Airflow orchestrates the entire pipeline with DVC integration, providing scheduling, retries, and monitoring.

#### Airflow Setup

1. **Build Airflow Docker Image**:
```bash
cd dags
docker build -f Dockerfile.airflow -t airflow-custom:latest .
```

2. **Start Airflow Services**:
```bash
# Start Airflow (webserver, scheduler, postgres)
docker-compose -f docker-compose.airflow.yml up -d

# Check status
docker-compose -f docker-compose.airflow.yml ps
```

3. **Access Airflow UI**:
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

4. **Enable and Trigger DAG**:
- Navigate to DAGs page
- Find `data_pipeline_dag`
- Toggle ON to enable
- Click play button to trigger manually

#### Airflow DAG Structure

The DAG (`dags/data_pipeline_dag.py`) orchestrates all DVC stages:

```
organize → preprocessing → validation → statistics
```

**Features**:
- **DVC Integration**: Each task runs `dvc repro <stage>` (DVC handles dependency tracking)
- **Metrics Export**: Success/failure metrics pushed to Pushgateway
- **Retry Logic**: Automatic retries on failure (configurable)
- **Scheduling**: Set `schedule_interval` for automated runs (default: manual trigger)
- **Environment Variables**:
  - `AIRFLOW_REPO_DIR`: Path to repository root (default: `.`)
  - `PUSHGATEWAY_URL`: Metrics endpoint (default: `pushgateway:9091`)

#### Stop Airflow

```bash
cd dags
docker-compose -f docker-compose.airflow.yml down

# Remove volumes (data will be lost)
docker-compose -f docker-compose.airflow.yml down -v
```

### Option 2: Run DVC Pipeline Directly

DVC automatically tracks dependencies and only re-runs changed stages:

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro organize
dvc repro preprocessing
```

**DVC Pipeline Stages**:
1. `organize` - Filter and organize raw data by language
2. `preprocessing` - Clean, deduplicate, remove PII
3. `validation` - Schema validation
4. `statistics` - Generate dataset statistics

> **When to use DVC directly**: Development, testing specific stages, or when Airflow is not available.

### Option 3: Run Individual Scripts

For development, debugging, or running specific steps independently:

#### Step 1: Data Acquisition
```bash
# Fetch metadata for all languages
python scripts/data_acquisition.py --all

# Fetch metadata for specific language
python scripts/data_acquisition.py --language python --max-samples 1000

# Download actual code files
python scripts/batch_swh_download.py --languages python,java,cpp,javascript
```

#### Step 2: Organization & Filtering
```bash
python scripts/dataset_filter.py \
  --input data/raw \
  --output data/staged/raw_organized \
  --max-size-mb 10
```

#### Step 3: Preprocessing
```bash
python scripts/preprocessing.py \
  --input data/staged/raw_organized \
  --output data/processed/code \
  --config configs/preprocessing_config.yaml
```

#### Step 4: Validation
```bash
python scripts/schema_validation.py \
  --input data/processed/code \
  --output reports/schema_validation_report.json
```

#### Step 5: Statistics Generation
```bash
python scripts/statistics_generation.py \
  --input data/processed/code \
  --output reports/data_statistics.json
```

### Pipeline Outputs

After running the pipeline, you'll have:

```
data/
├── raw/                                    # Original downloaded files
├── staged/raw_organized/                   # Organized by language
│   ├── python/
│   ├── java/
│   ├── cpp/
│   ├── javascript/
│   └── ingest_manifest.json               # Tracking metadata
└── processed/code/                         # Clean, preprocessed code
    ├── python/
    ├── java/
    ├── cpp/
    ├── javascript/
    └── preprocessing_results.json         # Processing statistics

reports/
├── schema_validation_report.json          # Validation results
└── data_statistics.json                   # Dataset statistics

logs/
└── *.log                                   # Execution logs
```

## 📊 Data Versioning with DVC

### Why DVC?

DVC (Data Version Control) tracks:
- Large data files (code datasets)
- Pipeline stages and dependencies
- Experiment reproducibility
- Model artifacts

### DVC Pipeline Structure

The pipeline is defined in `dvc.yaml`:

```yaml
stages:
  organize:
    cmd: python scripts/dataset_filter.py ...
    deps:
      - scripts/dataset_filter.py
      - data/raw
    outs:
      - data/staged/raw_organized

  preprocessing:
    cmd: python scripts/preprocessing.py ...
    deps:
      - scripts/preprocessing.py
      - data/staged/raw_organized
    outs:
      - data/processed/code
      - data/processed/preprocessing_results.json
  
  # ... more stages
```

### Key DVC Commands

```bash
# Show pipeline status
dvc status

# Reproduce pipeline (run only changed stages)
dvc repro

# Visualize pipeline DAG
dvc dag

# Track new data files
dvc add data/new_dataset.csv

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull

# Show data metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

### Reproducibility Workflow

1. **Version Code**: Commit changes to Git
   ```bash
   git add scripts/preprocessing.py configs/preprocessing_config.yaml
   git commit -m "Updated preprocessing: increased similarity threshold"
   ```

2. **Version Data**: DVC tracks data automatically via `dvc.lock`
   ```bash
   dvc repro  # Re-runs pipeline
   git add dvc.lock  # Lock file has data checksums
   git commit -m "Updated data pipeline outputs"
   ```

3. **Push Everything**:
   ```bash
   git push origin data_pipeline
   dvc push  # Upload data to remote (if configured)
   ```

4. **Reproduce on Another Machine**:
   ```bash
   git clone <repo>
   cd Data-Pipeline
   dvc pull  # Download data
   dvc repro  # Verify reproducibility
   ```

### DVC Remote Storage (Optional)

Configure remote storage for data sharing:

```bash
# S3
dvc remote add -d myremote s3://mybucket/dvc-storage

# Google Cloud Storage
dvc remote add -d myremote gs://mybucket/dvc-storage

# Azure Blob Storage
dvc remote add -d myremote azure://mycontainer/dvc-storage

# Local/Network drive
dvc remote add -d myremote /mnt/shared/dvc-storage
```

## 📈 Monitoring

The pipeline includes a complete monitoring stack with Prometheus, Grafana, Pushgateway, and Alertmanager.

### Start Monitoring Stack

```bash
# Windows (PowerShell)
cd monitoring
.\start_monitoring.ps1

# Linux/Mac
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
python scripts/start_metrics_server.py
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Pushgateway**: http://localhost:9091
- **Metrics Endpoint**: http://localhost:8000/metrics

### Available Metrics

#### System Metrics
- `system_cpu_usage_percent` - CPU utilization
- `system_memory_usage_percent` - Memory usage
- `system_disk_usage_percent` - Disk usage

#### Pipeline Metrics
- `pipeline_status{pipeline_name}` - Status (1=running, 0=stopped, -1=failed)
- `pipeline_processing_seconds{stage}` - Processing duration histogram
- `data_volume_bytes_total{stage}` - Data volume processed (counter)
- `pipeline_errors_total{stage,error_type}` - Error counts by type

#### Grafana Dashboard Panels
1. **Pipeline Status** - Current status of each pipeline
2. **CPU Usage** - Real-time CPU percentage
3. **Memory Usage** - RAM consumption
4. **Disk Usage** - Storage utilization
5. **System Resources Over Time** - Historical trends
6. **Data Volume Processed** - Data throughput (5min rate)
7. **Pipeline Errors** - Error tracking by stage

### Monitoring Integration in Code

```python
from monitoring.scripts.monitoring import monitor

# Track processing time
@monitor.track_stage_time("my_stage")
def process_data():
    # Your code here
    pass

# Record data volume
monitor.record_data_volume("my_stage", bytes_processed=1024*1024*50)  # 50 MB

# Set pipeline status
monitor.set_pipeline_status("my_pipeline", status=1)  # 1=running

# Record errors
monitor.record_error("my_stage", "FileNotFoundError")
```
#### Monitoring Architecture
![Monitoring Architecture](assets/monitoringArchitecture.png)

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with multiple workers (parallel)
pytest tests/ -n auto
```

### Test Categories

- **Unit Tests**: `tests/unit/` - Individual function tests
- **Integration Tests**: `tests/integration/` - Component interaction tests
- **End-to-End Tests**: `test_end_to_end.py` - Full pipeline tests
- **Monitoring Tests**: `test_monitoring.py` - Metrics and monitoring tests

### Test Coverage

Current coverage targets:
- Scripts: >80% coverage
- Critical paths: 100% coverage
- Integration: Key workflows validated

View coverage report:
```bash
pytest --cov=scripts --cov-report=html
# Open htmlcov/index.html in browser
```

## ⚙️ Configuration

### Data Acquisition (`configs/data_config.yaml`)

```yaml
languages:
  - Python
  - Java
  - C++
  - JavaScript

filters:
  stars:
    min: 10
  file_size:
    min_bytes: 100
    max_bytes: 1048576  # 1 MB

output_paths:
  base_dir: "data/code_files"

download:
  batch_size: 100
  skip_existing: true
```

### Preprocessing (`configs/preprocessing_config.yaml`)

```yaml
tokenizer_model: microsoft/codebert-base
similarity_threshold: 0.85  # Deduplication threshold

stages:
  encoding_fix: true
  malformed_cleaning: true
  pii_removal: true
  deduplication: true
  tokenization: true
  whitespace_cleanup: true

parallel_processing: false
max_workers: 4
```

### Monitoring (`configs/monitoring_config.yaml`)

Configure Prometheus scrape intervals, Grafana datasources, and alert rules.

## 🐛 Troubleshooting

### Common Issues

#### 1. **"InvalidAccessKeyId" Error**
**Problem**: AWS credentials are invalid or not set.

**Solution**:
```bash
# Check credentials
python scripts/check_aws_creds.py

# Set environment variables
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
```

#### 2. **"Gated Dataset" Error (HuggingFace)**
**Problem**: No access token or haven't requested dataset access.

**Solution**:
1. Request access: https://huggingface.co/datasets/bigcode/the-stack-v2
2. Generate token: https://huggingface.co/settings/tokens
3. Set token: `$env:HUGGINGFACE_TOKEN="hf_..."`

#### 3. **Long Filename Errors (Windows)**
**Problem**: File paths exceed Windows MAX_PATH (260 characters).

**Solution**: Pipeline automatically handles this with:
- Filename truncation to 180 characters
- Hash suffix for uniqueness
- Checksum manifest fallback

If issues persist, enable long paths:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### 4. **Grafana Dashboard Empty**
**Problem**: No data source configured or metrics server not running.

**Solution**:
```bash
# Start metrics server
python monitoring/scripts/start_metrics_server.py

# Restart Grafana
cd monitoring
docker-compose -f docker-compose.monitoring.yml restart grafana
```

#### 5. **DVC Pipeline Fails**
**Problem**: Missing dependencies or corrupted cache.

**Solution**:
```bash
# Check status
dvc status

# Force re-run stage
dvc repro --force organize

# Clear cache and re-run
dvc gc --workspace
dvc repro
```

#### 6. **Import Errors**
**Problem**: Module not found or Python path issues.

**Solution**:
```bash
# Verify installation
pip install -r requirements.txt

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip install -e .
```

#### 7. **Airflow DAG Not Appearing**
**Problem**: DAG file not detected or has syntax errors.

**Solution**:
```bash
# Check DAG for errors
docker exec -it orchestration-airflow-scheduler-1 airflow dags list

# View DAG import errors
docker exec -it orchestration-airflow-scheduler-1 airflow dags list-import-errors

# Restart scheduler to refresh
docker-compose -f orchestration/docker-compose.airflow.yml restart airflow-scheduler
```

#### 8. **Airflow Task Fails with "command not found"**
**Problem**: DVC or Python not available in Airflow container.

**Solution**:
Ensure `Dockerfile.airflow` includes:
```dockerfile
RUN pip install dvc apache-airflow
```

Rebuild image:
```bash
cd dags
docker build -f Dockerfile.airflow -t airflow-custom:latest .
docker-compose -f docker-compose.airflow.yml up -d --force-recreate
```

### Logs and Debugging

- **Pipeline Logs**: `logs/*.log`
- **DVC Logs**: `.dvc/tmp/`
- **Docker Logs**: `docker-compose logs <service>`
- **Prometheus Metrics**: http://localhost:9090/targets
- **Airflow Logs**: 
  - Web UI: http://localhost:8080 → DAGs → Select task → View logs
  - Container: `docker logs orchestration-airflow-scheduler-1`
  - Task logs: `docker exec orchestration-airflow-scheduler-1 cat /opt/airflow/logs/...`

Enable debug logging:
```bash
# Set log level in config
# configs/data_config.yaml
logging:
  level: "DEBUG"

# Or via environment
$env:LOG_LEVEL="DEBUG"
```

## Team Contributions
| Team Member | Task | Components | Status |
|-------------|------|-----------|--------|
| **Siddiq Mohiuddin Mohammed** | Task 1 | Data Acquisition, S3 Integration, Metadata Filtering | ✅ Complete |
| **Uzma Fatima** | Task 2 | Preprocessing, PII Removal, Deduplication | ✅ Complete |
| **Aparna Shree** | Task 3 | Airflow DAG, Docker Deployment, Metrics Integration | ✅ Complete |
| **Pranudeep Metuku** | Task 4 | DVC Pipeline, Schema Validation, Statistics, Lineage Tracking | ✅ Complete |
| **Prithiveen** | Task 5 | Testing, Anomaly Detection, Bias Analysis | ✅ Complete |
| **Ketaki Salway** | Task 6 | Monitoring, Logging, Documentation | ✅ Complete |

---

**Last Updated**: October 28, 2025
