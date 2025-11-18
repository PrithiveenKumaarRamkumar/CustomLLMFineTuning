#!/usr/bin/env python3
"""
Startup script for CustomLLM Inference API

This script provides a convenient way to start the inference API with
proper configuration validation and error handling.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.api.main import app
import uvicorn


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("api.log")
        ]
    )


def validate_environment():
    """Validate required environment variables."""
    required_vars = ["API_KEY"]
    optional_vars = {
        "MODEL_NAME": "custom-llm",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "API_WORKERS": "1"
    }
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("  API_KEY: Bearer token for API authentication")
        print("\nOptional environment variables:")
        for var, default in optional_vars.items():
            print(f"  {var}: {os.getenv(var, default)} (current: {os.getenv(var, 'not set')})")
        return False
    
    print("✅ Environment validation passed")
    
    # Set defaults for optional variables
    for var, default in optional_vars.items():
        if not os.getenv(var):
            os.environ[var] = default
            print(f"📝 Set {var}={default}")
    
    return True


def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Start CustomLLM Inference API")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Environment validation completed successfully")
        sys.exit(0)
    
    # Get configuration
    host = args.host or os.getenv("API_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("API_PORT", "8000"))
    workers = args.workers or int(os.getenv("API_WORKERS", "1"))
    
    print(f"""
🚀 Starting CustomLLM Inference API
   Host: {host}
   Port: {port}
   Workers: {workers}
   Model: {os.getenv('MODEL_NAME', 'custom-llm')}
   MLflow: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}
   Log Level: {args.log_level}
   Reload: {args.reload}
   
🌐 API will be available at:
   - Swagger UI: http://{host}:{port}/docs
   - ReDoc: http://{host}:{port}/redoc
   - Health Check: http://{host}:{port}/health
   - Metrics: http://{host}:{port}/metrics
   """)
    
    try:
        # Start the server
        uvicorn.run(
            "serving.api.main:app",
            host=host,
            port=port,
            workers=1 if args.reload else workers,  # Workers must be 1 with reload
            log_level=args.log_level.lower(),
            reload=args.reload,
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()