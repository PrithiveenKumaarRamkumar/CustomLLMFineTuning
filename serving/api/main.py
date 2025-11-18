import os
import sys
import logging
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any
import yaml
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import ValidationError
import time
import traceback

from .routes import router, initialize_inference_engine, shutdown_inference_engine
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data-Pipeline", "configs", "serving_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    # Startup
    logger.info("Starting up CustomLLM Inference API...")
    
    # Validate required environment variables
    required_env_vars = ["API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        # Don't exit here, let the API start but mark as unhealthy
    
    # Initialize inference engine
    try:
        success = await initialize_inference_engine()
        if success:
            logger.info("Inference engine initialized successfully")
        else:
            logger.warning("Inference engine initialization failed, API will start in degraded mode")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {str(e)}")
    
    logger.info("CustomLLM Inference API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CustomLLM Inference API...")
    await shutdown_inference_engine()
    logger.info("CustomLLM Inference API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=config["api"]["title"],
    description=config["api"]["description"],
    version=config["api"]["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url} - "
        f"Status: {response.status_code} - Time: {process_time:.4f}s"
    )
    
    return response


# Exception handlers

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors (400 Bad Request)."""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = ErrorResponse(
        error="validation_error",
        message="Request validation failed",
        details={
            "validation_errors": error_details,
            "invalid_fields": len(error_details)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.dict()
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic model validation errors."""
    logger.warning(f"Pydantic validation error for {request.url}: {exc.errors()}")
    
    error_response = ErrorResponse(
        error="model_validation_error",
        message="Data model validation failed",
        details={"validation_errors": exc.errors()}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception for {request.url}: {exc.status_code} - {exc.detail}")
    
    # Map specific status codes to error types
    error_type_mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "unprocessable_entity",
        429: "rate_limit_exceeded",
        500: "internal_server_error",
        503: "service_unavailable"
    }
    
    error_response = ErrorResponse(
        error=error_type_mapping.get(exc.status_code, "http_error"),
        message=exc.detail,
        details={"status_code": exc.status_code}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error for {request.url}: {type(exc).__name__}: {str(exc)}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    error_response = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        details={
            "exception_type": type(exc).__name__,
            "debug_info": str(exc) if app.debug else "Contact support for assistance"
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


# Custom exception for payload size errors
class PayloadTooLargeError(HTTPException):
    """Exception for when request payload exceeds limits."""
    
    def __init__(self, message: str = "Request payload too large"):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=message
        )


@app.exception_handler(PayloadTooLargeError)
async def payload_too_large_handler(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
    """Handle payload size limit errors (413)."""
    logger.warning(f"Payload too large for {request.url}: {exc.detail}")
    
    error_response = ErrorResponse(
        error="payload_too_large",
        message=exc.detail,
        details={
            "max_tokens": config["model"]["max_tokens"],
            "suggestion": "Reduce prompt length or split into multiple requests"
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content=error_response.dict()
    )


# Add request size limit check
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request size to prevent large payloads."""
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB limit
    
    content_length = request.headers.get('content-length')
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            error_response = ErrorResponse(
                error="payload_too_large",
                message=f"Request size ({content_length} bytes) exceeds limit ({MAX_REQUEST_SIZE} bytes)",
                details={"max_size_bytes": MAX_REQUEST_SIZE}
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=error_response.dict()
            )
    
    return await call_next(request)


# Include router
app.include_router(router)


# Root endpoint
@app.get("/", summary="Root endpoint", description="Welcome message and basic API information")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "Welcome to CustomLLM Inference API",
        "version": config["api"]["version"],
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "model_info": "/info"
    }


# Development server
if __name__ == "__main__":
    # Get configuration from environment or config file
    host = os.getenv("API_HOST", config["api"]["host"])
    port = int(os.getenv("API_PORT", config["api"]["port"]))
    workers = int(os.getenv("API_WORKERS", config["api"]["workers"]))
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False,  # Set to True for development
        access_log=True
    )