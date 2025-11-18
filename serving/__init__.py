"""
CustomLLM Inference Serving Module

This module provides production-ready LLM inference capabilities with:
- FastAPI-based REST API
- MLflow model management
- GPU batching for efficient inference
- Prometheus metrics integration
- LoRA adapter support
- Comprehensive error handling and monitoring

Main components:
- api: FastAPI application and routes
- inference_engine: Core inference logic with GPU batching
- adapter_loader: Model and LoRA adapter management

Usage:
    python -m serving.api.main

Environment Variables:
    API_KEY: Bearer token for API authentication
    MODEL_NAME: MLflow model name to load
    MLFLOW_TRACKING_URI: MLflow tracking server URI
"""

__version__ = "1.0.0"
__author__ = "CustomLLM Team"

from .inference_engine import InferenceEngine, ModelManager, GPUBatchingQueue
from .adapter_loader import AdapterLoader, MultiAdapterManager

__all__ = [
    "InferenceEngine",
    "ModelManager", 
    "GPUBatchingQueue",
    "AdapterLoader",
    "MultiAdapterManager"
]