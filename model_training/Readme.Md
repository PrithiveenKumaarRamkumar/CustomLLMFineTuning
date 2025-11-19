# StarCoder2 Fine-Tuning Platform

Complete end-to-end pipeline for fine-tuning StarCoder2 models with QLoRA, custom layer freezing, and production deployment.

## 🎯 Features

- ✅ **Modular Pipeline**: 9 independent modules for each task
- ✅ **User-Friendly**: Interactive CLI wizard for configuration
- ✅ **QLoRA Training**: Memory-efficient 4-bit quantized training
- ✅ **Custom Layer Freezing**: Freeze specific layers, embeddings, or norms
- ✅ **Comprehensive Evaluation**: CodeBLEU, syntax validity, exact match, perplexity
- ✅ **Production Ready**: Triton Inference Server + FastAPI endpoints
- ✅ **MLflow Tracking**: Complete experiment tracking and monitoring

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Modules](#pipeline-modules)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU RAM (for 3B model with 4-bit quantization)

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Use Template Configuration
```bash
# Copy template
cp pipeline_config_template.json my_config.json

# Edit configuration
nano my_config.json

# Run pipeline
python orchestrator.py --config my_config.json
```

### Option 3: Manual Step-by-Step
```bash
# Step 1: Prepare your preprocessed code files in a directory
# ./data/preprocessed/

# Step 2: Split data
python -m data_splitter

# Step 3: Tokenize
python -m tokenizer

# Step 4-6: Train
python -m qlora_trainer

# Step 7: Evaluate
python -m evaluator

# Step 8: Deploy
python -m triton_deployer

# Step 9: Serve
python -m fastapi_server
```

## 📦 Pipeline Modules

Each module is independent and can be run separately:

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `data_splitter.py` | Split data into train/val/test | Preprocessed code files | Split directories |
| `tokenizer.py` | Tokenize datasets | Split code files | Tokenized datasets |
| `model_loader.py` | Load base model | Model path | Model + tokenizer |
| `layer_freezer.py` | Freeze layers | Model + strategy | Frozen model |
| `qlora_trainer.py` | QLoRA fine-tuning | Datasets + model | Trained model |
| `model_saver.py` | Save model | Trained model | Saved checkpoints |
| `evaluator.py` | Evaluate model | Test dataset | Metrics report |
| `triton_deployer.py` | Deploy to Triton | Saved model | Triton-ready model |
| `fastapi_server.py` | Inference API | Saved model | Running API server |

## 📖 Usage Examples

### Example 1: Basic Fine-Tuning
```bash
# Configure:
# - Source directory: ./data/my_code
# - Model: starcoder2-3b
# - Strategy: Freeze first 15 layers
# - Epochs: 3
# - Batch size: 2
```

### Example 2: Custom Layer Freezing
```python
# Create custom config
config = {
    "layer_freezing": {
        "strategy": "specific",
        "layer_indices": [0, 1, 2, 5, 10, 15],
        "freeze_embeddings": True,
        "freeze_layer_norms": False
    }
}
```

### Example 3: Evaluation Only
```bash
# Evaluate existing model
python orchestrator.py --config eval_only_config.json
```

`eval_only_config.json`:
```json
{
  "run_data_splitting": false,
  "run_tokenization": false,
  "run_training": false,
  "run_evaluation": true,
  "evaluation": {
    "model_path": "./output/final_model/full_model",
    "test_dataset_path": "./data/tokenized/test_tokenized",
    "output_dir": "./eval_results"
  }
}
```

### Example 4: Deploy and Serve
```bash
# Deploy to Triton
python -c "
from pipeline.08_triton_deployer import run_triton_deployment
result = run_triton_deployment({
    'model_path': './output/final_model/full_model',
    'triton_model_repository': './triton_models',
    'model_name': 'my_model'
})
"

# Start Triton server
tritonserver --model-repository=./triton_models

# Or use FastAPI
python -c "
from pipeline.09_fastapi_server import run_fastapi_server
run_fastapi_server({
    'model_path': './output/final_model/full_model',
    'host': '0.0.0.0',
    'port': 8000
})
"
```

## ⚙️ Configuration

### Layer Freezing Strategies

**First N Layers**:
```json
{
  "strategy": "first_n",
  "n_layers": 15
}
```

**Last N Layers**:
```json
{
  "strategy": "last_n",
  "n_layers": 10,
  "total_layers": 30
}
```

**Specific Layers**:
```json
{
  "strategy": "specific",
  "layer_indices": [0, 1, 2, 5, 10, 15, 20]
}
```

**No Freezing**:
```json
{
  "strategy": "none"
}
```

### LoRA Configuration
```json
{
  "lora_config": {
    "r": 16,              // Rank: 8, 16, 32, 64
    "alpha": 32,          // Alpha: typically 2*r
    "dropout": 0.1,       // Dropout: 0.0-0.3
    "target_modules": [   // Which layers to adapt
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "dense"
    ]
  }
}
```

### Training Configuration
```json
{
  "training_config": {
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "optimizer": "paged_adamw_8bit",
    "lr_scheduler": "cosine",
    "use_mlflow": true
  }
}
```

## 📊 Evaluation Metrics

The evaluator computes:

- **CodeBLEU**: Code-specific BLEU score (0-1)
- **Syntax Validity**: % of syntactically correct code
- **Exact Match**: % of perfect predictions
- **Perplexity**: Model confidence score
- **Generation Length**: Average tokens generated

Example output:
```json
{
  "codebleu": 0.7234,
  "syntax_validity": 0.8900,
  "exact_match": 0.1500,
  "perplexity": 12.45,
  "num_test_samples": 100
}
```

## Deployment

### Triton Inference Server
```bash
# Deploy
python orchestrator.py --config deploy_config.json

# Start Triton
tritonserver --model-repository=./triton_models

# Test
curl -X POST http://localhost:8000/v2/models/my_model/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "INPUT_IDS", "data": [...]}]}'
```

### FastAPI Server
```bash
# Start server
python orchestrator.py --config config.json --serve

# Or directly
python -m pipeline.09_fastapi_server

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate factorial",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'

# API Documentation
open http://localhost:8000/docs
```

## 🔧 Troubleshooting

### Issue: CUDA out of memory

**Solution**:
```json
{
  "training_config": {
    "batch_size": 1,
    "gradient_accumulation_steps": 16
  }
}
```

### Issue: Model loading fails

**Solution**: Check transformers version
```bash
pip install transformers>=4.38.0
# Restart kernel!
```

### Issue: Slow training

**Solution**: Enable 4-bit quantization
```json
{
  "model_loading": {
    "use_4bit": true
  }
}
```

### Issue: Low CodeBLEU scores

**Solution**: 
- Increase training epochs
- Use larger LoRA rank
- Freeze fewer layers
- Increase dataset size

## 📁 Project Structure
```
starcoder2-finetuning/
├── pipeline/
│   ├── data_splitter.py
│   ├── tokenizer.py
│   ├── model_loader.py
│   ├── layer_freezer.py
│   ├── qlora_trainer.py
│   ├── model_saver.py
│   ├── evaluator.py
│   ├── triton_deployer.py
│   └── fastapi_server.py
├── orchestrator.py              # Main pipeline runner
├── pipeline_config_template.json
├── requirements.txt
├── README.md
├── data/
│   ├── preprocessed/            # Your code files here
│   ├── splits/                  # After step 1
│   └── tokenized/               # After step 2
├── models/
│   └── starcoder2-3b/          # Downloaded model
├── output/
│   ├── qlora_training/         # Training checkpoints
│   ├── final_model/            # Saved model
│   └── evaluation_results/     # Metrics
├── triton_models/              # Triton deployment
└── mlruns/                     # MLflow tracking
```

## 🎓 Best Practices

1. **Start Small**: Test with 100 samples first
2. **Monitor Training**: Use MLflow UI (`mlflow ui`)
3. **Save Checkpoints**: Keep save_total_limit=3
4. **Evaluate Often**: Run evaluation after each experiment
5. **Version Control**: Save configs in git
6. **GPU Memory**: Use 4-bit quantization for 3B+ models
7. **Batch Size**: Start with 1-2, increase with gradient accumulation