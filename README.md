#  Custom LLM Fine-Tuning - MLOps Pipeline

**Complete Data Processing Pipeline for LLM Fine-Tuning**  
*MLOps for Generative AI - Tasks 1 & 2 Complete*

##  Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Task 1: Data Acquisition](#task-1-data-acquisition)
* [Task 2: Data Preprocessing](#task-2-data-preprocessing)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Testing](#testing)
* [Team](#team)

##  Overview

This project implements a complete MLOps pipeline for custom LLM fine-tuning, featuring data acquisition from The Stack v2 dataset and comprehensive preprocessing including PII removal, deduplication, and tokenization.

**Pipeline Flow:**
```
The Stack v2 → Task 1 (Acquisition) → Task 2 (Preprocessing) → Task 3 (Airflow) → Task 4 (Versioning) → Task 5 (Testing) → Task 6 (Monitoring)
```

## ✨ Features

### Task 1: Data Acquisition  COMPLETE
*  Multi-language support: Python, Java, C++, JavaScript
*  Smart filtering by stars, licenses, file sizes
*  Resume capability with SHA-256 validation
*  AWS S3 integration with Software Heritage
*  Comprehensive logging and monitoring

### Task 2: Data Preprocessing  COMPLETE
*  PII detection and removal (emails, API keys, secrets, IPs)
*  Multi-level deduplication (exact, normalized, near-duplicate)
*  CodeBERT tokenization for ML readiness
*  Malformed code cleaning and encoding fixes
*  Modular, configurable processing pipeline
*  Detailed processing statistics and reports

##  Project Structure

```
CustomLLMFineTuning/
├── configs/
│   ├── data_config.yaml              # Task 1 configuration
│   └── preprocessing_config.yaml     # Task 2 configuration
├── data/
│   ├── raw/
│   │   ├── metadata/                 # Filtered JSON files
│   │   └── code/                     # Downloaded code by language
│   └── processed/
│       └── code/                     # Cleaned, deduplicated code
├── scripts/
│   ├── logger_config.py              # Centralized logging
│   ├── batch_swh_download_*.py       # Task 1: Data acquisition
│   ├── pii_removal.py               # Task 2: PII detection/removal
│   ├── deduplication.py             # Task 2: Duplicate detection
│   └── preprocessing.py             # Task 2: Main pipeline
├── tests/
│   ├── test_acquisition.py          # Task 1 tests
│   └── test_preprocessing.py        # Task 2 tests
├── logs/                            # Auto-generated logs
├── requirements.txt                 # All dependencies
└── README.md                        # This file
```

##  Task 1: Data Acquisition


### What it does:
1. Filters The Stack v2 dataset by programming languages
2. Applies repository-level filtering (stars, licenses, sizes)
3. Downloads code files from Software Heritage S3
4. Validates integrity with SHA-256 checksums
5. Implements resume capability for interrupted downloads

### Key Files:
- `scripts/batch_swh_download_*.py` - Download scripts for each language
- `configs/data_config.yaml` - Filter configuration
- `data/raw/metadata/filtered_metadata_*.json` - Pre-filtered metadata

##  Task 2: Data Preprocessing

**Status:**  Production Ready

### What it does:
1. **PII Removal:** Detects and removes emails, API keys, secrets, IP addresses, database URLs
2. **Deduplication:** Finds exact, normalized, and near-duplicate files
3. **Code Tokenization:** Uses CodeBERT tokenizer for ML readiness
4. **File Cleaning:** Handles encoding issues, malformed code, whitespace
5. **Pipeline Orchestration:** Modular, configurable processing stages

### Processing Results (Test Data):
```
 Total files processed: 5
 PII items removed: 9
 Duplicates removed: 1  
 Processing time: 0.1 seconds
 All languages supported: Python, Java, C++, JavaScript
```

### Key Files:
- `scripts/preprocessing.py` - Main preprocessing pipeline
- `scripts/pii_removal.py` - PII detection and removal
- `scripts/deduplication.py` - Duplicate detection algorithms
- `configs/preprocessing_config.yaml` - Processing configuration

##  Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Aparnashree11/CustomLLMFineTuning.git
cd CustomLLMFineTuning
```

### 2. Create Virtual Environment
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# For Task 1 + Task 2
pip install -r requirements.txt

# Minimal install (Task 2 only)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers chardet pyyaml
```

### 4. Set AWS Credentials (Task 1 only)
**Windows:**
```powershell
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
```

**Mac/Linux:**
```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
```

##  Usage

### Task 1: Data Acquisition
```bash
# Download code files (requires AWS credentials)
python scripts/batch_swh_download_python.py
python scripts/batch_swh_download_java.py
python scripts/batch_swh_download_cpp.py
python scripts/batch_swh_download_javascript.py
```

### Task 2: Data Preprocessing
```bash
# Process all languages
python scripts/preprocessing.py

# Process specific language
python scripts/preprocessing.py --language python

# Custom directories
python scripts/preprocessing.py --input data/raw/code --output data/processed/code
```

### Verify Results
```bash
# Check processed files
ls data/processed/code/python/
ls data/processed/code/java/

# View processing statistics
cat data/processed/preprocessing_results.json
```

##  Testing

### Run All Tests
```bash
# Task 1 tests
python tests/test_acquisition.py

# Task 2 tests  
python tests/test_preprocessing.py

# Full test suite
pytest tests/ -v
```

### Test Coverage
- Data acquisition validation
-  PII removal patterns
-  Deduplication algorithms
- File processing pipeline
-  Error handling and edge cases

##  Pipeline Statistics

**Task 1 Capabilities:**
- Languages: 4 (Python, Java, C++, JavaScript)
- Files per language: 100 (configurable)
- Integrity validation: SHA-256 checksums
- Resume capability: Skip existing files

**Task 2 Processing:**
- PII patterns detected: 15+ types
- Deduplication methods: 3 (exact, normalized, similarity-based)
- Processing speed: ~100-200 files/minute
- Languages supported: All Task 1 languages

## Integration with Remaining Tasks

**Ready for Task 3 (Airflow Orchestration):**
- Modular components can be called separately in Airflow DAGs
- Output directory structure: `data/processed/code/{language}/`
- Processing statistics available for monitoring
- Error handling and retry logic built-in

**Prepared for Task 4 (Versioning):**
- Clean, deduplicated datasets ready for DVC tracking
- Processing metadata for lineage tracking
- Schema validation friendly data structure

##  Configuration

### Task 1 Config (`configs/data_config.yaml`)
```yaml
languages: [Python, Java, C++, JavaScript]
min_stars: 10
max_file_size_mb: 5
licenses: [MIT, Apache-2.0, BSD]
```

### Task 2 Config (`configs/preprocessing_config.yaml`)
```yaml
stages:
  pii_removal: true
  deduplication: true
  tokenization: true
similarity_threshold: 0.85
max_file_size_mb: 10
```

## 👥 Team
**Project Group 25:**
* Aparna Shree Govindarajan (Task 3: Airflow)
* Ketaki Salway (Task 1: Acquisition)
* Pranudeep Metuku (Task 4: Versioning)
* Prithiveen Ramkumar (Task 5: Testing & Bias)
* Siddiq Mohiuddin Mohammed (Task 1: Acquisition)
* Uzma Fatima (Task 2: Preprocessing)

##  Next Steps

1. **Task 3:** Airflow DAG orchestration (In Progress - Aparna)
2. **Task 4:** DVC versioning and schema management (Pending - Pranudeep)
3. **Task 5:** Testing, anomaly detection, bias analysis (Pending - Prithiveen)
4. **Task 6:** Logging, monitoring, documentation (Pending - Siddiq & Ketaki)

---

**Last Updated:** October 26, 2025  
**Status:** Tasks 1 & 2 Complete ✅ 
**Next:** Task 3 Integration Ready 