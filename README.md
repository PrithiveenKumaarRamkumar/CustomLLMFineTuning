<!-- # Custom LLM Fine-Tuning Platform

An end-to-end platform for fine-tuning and serving large language models (LLMs) on domain-specific datasets. Built with LoRA/PEFT, distributed training, and production-grade deployment.

---

## Key Features

**Fine-Tuning** → Parameter-efficient training (LoRA/QLoRA) on user datasets.

**Scalability** → Distributed systems for fast training.

**Experiment Tracking** → MLflow integration with auto-generated model cards.

**Serving** → FastAPI + Docker/Kubernetes deployment with GPU batching.

**Monitoring** → Drift detection, performance dashboards, and feedback loops.

---

## How It Works

Upload dataset → preprocessing & validation.

Fine-tune base model (StarCoderBase, LLaMA, Falcon).

Track experiments and metrics.

Deploy model as an API endpoint.

Monitor → retrain with new data.

---

## Example Use Cases

FinTech copilots trained on regulatory codebases.

Healthcare assistants fine-tuned on medical knowledge.

Enterprise AI copilots for private code repositories.

--- -->


# 🚀 Custom LLM Fine-Tuning - Data Acquisition Pipeline

**Task 1: Data Acquisition & Initial Processing**  
*Part of the Custom LLM Fine-Tuning Deployment Platform - MLOps for Generative AI*

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Testing](#testing)
- [Logging & Monitoring](#logging--monitoring)
- [Troubleshooting](#troubleshooting)
- [Team](#team)


## 🎯 Overview

This module acquires and organizes code data from **The Stack v2 dataset**. It downloads code in Python, Java, C++, and JavaScript, applies filtering by repository stars and licenses, validates file integrity with SHA-256 checksums, and provides automatic resume capability.

### What This Pipeline Does:

1. Filters The Stack v2 by programming languages
2. Applies repository-level filtering (stars, licenses, sizes)
3. Downloads from Software Heritage S3
4. Validates integrity with checksums
5. Organizes by language
6. Logs all operations

---

## ✨ Features

- ✅ Multi-language: Python, Java, C++, JavaScript
- ✅ Smart filtering: Stars, licenses, file sizes
- ✅ Resume capability: Skip downloaded files
- ✅ SHA-256 checksum validation
- ✅ Comprehensive logging
- ✅ YAML configuration
- ✅ Unit tested
- ✅ AWS S3 integration


## 📁 Project Structure

CustomLLMFineTuning/
├── configs/
│ └── data_config.yaml
├── data/
│ └── raw/
│ ├── metadata/ # Filtered JSON files
│ └── code/ # Downloaded code by language
├── scripts/
│ ├── logger_config.py
│ ├── batch_swh_download_python.py
│ ├── batch_swh_download_java.py
│ ├── batch_swh_download_cpp.py
│ └── batch_swh_download_javascript.py
├── tests/
│ └── test_acquisition.py
├── logs/ # Auto-generated
├── requirements.txt
└── README.md


## 🚀 Installation & Setup

### 1. Clone Repository


git clone https://github.com/your-team/CustomLLMFineTuning.git
cd CustomLLMFineTuning


### 2. Create Virtual Environment

**Windows:**


python -m venv venv
.\venv\Scripts\Activate.ps1


**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate


### 3. Install Dependencies
pip install -r requirements.txt


### 4. Set AWS Credentials

**Windows:**
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"

**Mac/Linux:**
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"

### 5. Verify Setup
pytest tests/ -v

## 🎮 Usage

Run each language download:

python scripts/batch_swh_download_python.py
python scripts/batch_swh_download_java.py
python scripts/batch_swh_download_cpp.py
python scripts/batch_swh_download_javascript.py

**Resume interrupted downloads:** Just rerun the same script - it will skip existing files.

---

## 🔍 Troubleshooting

### "Access Denied" Error
**Solution:** Reset AWS credentials
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"

### "Module not found"
**Solution:** Install dependencies
pip install -r requirements.txt


### Files not being skipped
**Solution:** Ensure `.sha256` checksum files exist alongside code files

---

## 👥 Team

**Task 1 Team:**
- Siddiq Mohiuddin Mohammed
- Ketaki Salway

**Project Group 25:**
- Aparna Shree Govindarajan
- Ketaki Salway
- Pranudeep Metuku
- Prithiveen Ramkumar
- Siddiq Mohiuddin Mohammed
- Uzma Fatima

---

**Last Updated:** October 21, 2025
