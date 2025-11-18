from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator


class InferenceRequest(BaseModel):
    """Pydantic model for inference request validation."""
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=8192,  # Character limit for prompt
        description="Input prompt for text generation"
    )
    max_length: Optional[int] = Field(
        default=100,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for text generation"
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    stop_tokens: Optional[List[str]] = Field(
        default=None,
        max_items=10,
        description="List of tokens to stop generation at"
    )
    
    @validator('prompt')
    def validate_prompt_tokens(cls, v):
        """Estimate token count and validate against limits."""
        # Rough estimation: 1 token ≈ 4 characters for most models
        estimated_tokens = len(v) // 4
        if estimated_tokens > 2048:
            raise ValueError("Prompt exceeds maximum token limit of 2048")
        return v
    
    @validator('stop_tokens')
    def validate_stop_tokens(cls, v):
        """Validate stop tokens format."""
        if v is not None:
            for token in v:
                if not isinstance(token, str) or len(token.strip()) == 0:
                    raise ValueError("Stop tokens must be non-empty strings")
        return v


class InferenceResponse(BaseModel):
    """Pydantic model for inference response."""
    
    generated_text: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original input prompt")
    tokens_generated: int = Field(..., ge=0, description="Number of tokens generated")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    """Pydantic model for health check response."""
    
    status: str = Field(..., description="Overall health status")
    api_status: str = Field(..., description="FastAPI application status")
    model_status: str = Field(..., description="Model loading status")
    gpu_status: str = Field(..., description="GPU availability status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    uptime_seconds: float = Field(..., ge=0, description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Pydantic model for model metadata."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    creation_timestamp: datetime = Field(..., description="Model creation timestamp")
    mlflow_run_id: Optional[str] = Field(None, description="MLflow run ID")
    model_uri: Optional[str] = Field(None, description="MLflow model URI")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")


class MetricsResponse(BaseModel):
    """Pydantic model for metrics response."""
    
    api_requests_total: int = Field(..., ge=0, description="Total API requests")
    api_request_duration_seconds: float = Field(..., ge=0, description="Average request duration")
    gpu_memory_usage_bytes: int = Field(..., ge=0, description="Current GPU memory usage")
    tokens_per_second: float = Field(..., ge=0, description="Average tokens generated per second")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model for all API errors."""
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class BatchInferenceRequest(BaseModel):
    """Pydantic model for batch inference requests."""
    
    prompts: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of prompts for batch processing"
    )
    max_length: Optional[int] = Field(default=100, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stop_tokens: Optional[List[str]] = Field(default=None, max_items=10)
    
    @validator('prompts')
    def validate_prompts(cls, v):
        """Validate each prompt in the batch."""
        for prompt in v:
            if len(prompt.strip()) == 0:
                raise ValueError("Prompts cannot be empty")
            # Rough token estimation
            if len(prompt) // 4 > 2048:
                raise ValueError(f"Prompt exceeds token limit: {prompt[:50]}...")
        return v


class BatchInferenceResponse(BaseModel):
    """Pydantic model for batch inference response."""
    
    results: List[InferenceResponse] = Field(..., description="List of inference results")
    batch_size: int = Field(..., ge=1, description="Number of prompts processed")
    total_processing_time: float = Field(..., ge=0, description="Total batch processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch response timestamp")