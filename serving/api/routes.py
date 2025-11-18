import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import PlainTextResponse
import yaml
import time

from .schemas import (
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    ModelInfo,
    MetricsResponse,
    ErrorResponse,
    BatchInferenceRequest,
    BatchInferenceResponse
)
from ..inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None

# Security scheme
security = HTTPBearer()

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data-Pipeline", "configs", "serving_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Router instance
router = APIRouter()


def get_api_key() -> str:
    """Get API key from environment variables."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable not set")
    return api_key


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify Bearer token authentication."""
    try:
        expected_token = get_api_key()
        if credentials.credentials != expected_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


async def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance."""
    global inference_engine
    if inference_engine is None or not inference_engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )
    return inference_engine


@router.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate text prediction",
    description="Main inference endpoint for text generation using the loaded LLM model"
)
async def predict(
    request: InferenceRequest,
    request_obj: Request,
    token: str = Depends(verify_token),
    engine: InferenceEngine = Depends(get_inference_engine)
) -> InferenceResponse:
    """Generate text prediction based on input prompt."""
    
    # Add request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"Processing inference request {request_id} for prompt length: {len(request.prompt)}")
    
    try:
        # Prepare request data
        request_data = {
            "prompt": request.prompt,
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop_tokens": request.stop_tokens
        }
        
        # Process inference request
        start_time = time.time()
        result = await engine.generate(request_data)
        processing_time = time.time() - start_time
        
        # Update processing time in result
        result["processing_time"] = processing_time
        
        logger.info(f"Completed inference request {request_id} in {processing_time:.2f}s")
        
        return InferenceResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request validation failed: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Runtime error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/predict/batch",
    response_model=BatchInferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch text prediction",
    description="Process multiple prompts in a single batch request"
)
async def predict_batch(
    request: BatchInferenceRequest,
    token: str = Depends(verify_token),
    engine: InferenceEngine = Depends(get_inference_engine)
) -> BatchInferenceResponse:
    """Process batch inference requests."""
    
    request_id = str(uuid.uuid4())
    logger.info(f"Processing batch inference request {request_id} with {len(request.prompts)} prompts")
    
    try:
        start_time = time.time()
        results = []
        
        # Process each prompt individually (could be optimized for true batching)
        for prompt in request.prompts:
            request_data = {
                "prompt": prompt,
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop_tokens": request.stop_tokens
            }
            
            result = await engine.generate(request_data)
            results.append(InferenceResponse(**result))
        
        total_processing_time = time.time() - start_time
        
        logger.info(f"Completed batch inference request {request_id} in {total_processing_time:.2f}s")
        
        return BatchInferenceResponse(
            results=results,
            batch_size=len(request.prompts),
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch inference request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference failed"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Comprehensive health check for Kubernetes readiness and liveness probes"
)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint."""
    
    try:
        global inference_engine
        
        if inference_engine is None:
            # Engine not initialized
            return HealthResponse(
                status="unhealthy",
                api_status="running",
                model_status="not_loaded",
                gpu_status="unknown",
                uptime_seconds=0
            )
        
        health_status = inference_engine.get_health_status()
        
        # Determine HTTP status based on health
        if health_status["status"] != "healthy":
            # Return 503 for unhealthy status to fail K8s probes
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is unhealthy"
            )
        
        return HealthResponse(**health_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Prometheus metrics endpoint",
    description="Expose Prometheus-compatible metrics for monitoring"
)
async def get_metrics() -> str:
    """Expose Prometheus-compatible metrics."""
    
    try:
        global inference_engine
        
        if inference_engine is None:
            # Return basic metrics if engine not initialized
            return """# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total 0

# HELP api_request_duration_seconds Average request duration in seconds
# TYPE api_request_duration_seconds gauge
api_request_duration_seconds 0

# HELP gpu_memory_usage_bytes Current GPU memory usage in bytes
# TYPE gpu_memory_usage_bytes gauge
gpu_memory_usage_bytes 0

# HELP tokens_per_second Average tokens generated per second
# TYPE tokens_per_second gauge
tokens_per_second 0
"""
        
        metrics = inference_engine.get_metrics()
        
        # Format as Prometheus metrics
        prometheus_metrics = f"""# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total {metrics.get('api_requests_total', 0)}

# HELP api_request_duration_seconds Average request duration in seconds
# TYPE api_request_duration_seconds gauge
api_request_duration_seconds {metrics.get('api_request_duration_seconds', 0):.6f}

# HELP gpu_memory_usage_bytes Current GPU memory usage in bytes
# TYPE gpu_memory_usage_bytes gauge
gpu_memory_usage_bytes {metrics.get('gpu_memory_usage_bytes', 0)}

# HELP tokens_per_second Average tokens generated per second
# TYPE tokens_per_second gauge
tokens_per_second {metrics.get('tokens_per_second', 0):.2f}

# HELP model_loaded Indicates if the model is loaded (1) or not (0)
# TYPE model_loaded gauge
model_loaded {1 if inference_engine.is_ready else 0}
"""
        
        return prometheus_metrics
        
    except Exception as e:
        logger.error(f"Metrics collection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection failed"
        )


@router.get(
    "/info",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Model information endpoint",
    description="Get metadata about the currently loaded model from MLflow"
)
async def get_model_info(
    engine: InferenceEngine = Depends(get_inference_engine)
) -> ModelInfo:
    """Get information about the currently loaded model."""
    
    try:
        model_info_data = engine.get_model_info()
        
        if not model_info_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model information not available"
            )
        
        return ModelInfo(**model_info_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


# Additional utility endpoints

@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Basic status endpoint",
    description="Simple status check without detailed health information"
)
async def get_status() -> Dict[str, Any]:
    """Basic status endpoint."""
    return {
        "service": "CustomLLM Inference API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow(),
        "status": "running"
    }


# Initialize inference engine function
async def initialize_inference_engine():
    """Initialize the global inference engine."""
    global inference_engine
    
    try:
        # Load environment variables for MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        model_name = os.getenv("MODEL_NAME", config["mlflow"]["model_name"])
        
        # Update config with environment variables
        updated_config = config.copy()
        updated_config["mlflow"]["tracking_uri"] = mlflow_uri
        updated_config["mlflow"]["model_name"] = model_name
        
        # Create and initialize inference engine
        inference_engine = InferenceEngine(updated_config)
        success = await inference_engine.initialize()
        
        if not success:
            logger.error("Failed to initialize inference engine")
            inference_engine = None
            return False
        
        logger.info("Inference engine initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing inference engine: {str(e)}")
        inference_engine = None
        return False


# Shutdown inference engine function
async def shutdown_inference_engine():
    """Shutdown the global inference engine."""
    global inference_engine
    
    if inference_engine:
        await inference_engine.shutdown()
        inference_engine = None
        logger.info("Inference engine shutdown complete")