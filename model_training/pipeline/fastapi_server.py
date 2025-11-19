# pipeline/09_fastapi_server.py
# ================================================================================
# Module 9: FastAPI inference server
# ================================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(50, ge=0, description="Top-k sampling")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences")


class GenerateResponse(BaseModel):
    generated_text: List[str]
    prompt: str
    generation_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str


class FastAPIServer:
    """FastAPI inference server"""
    
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """
        Initialize FastAPI server
        
        Args:
            model_path: Path to model
            host: Host address
            port: Port number
        """
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create FastAPI app
        self.app = FastAPI(
            title="StarCoder2 Inference API",
            description="Fine-tuned StarCoder2 code generation API",
            version="1.0.0"
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("FastAPIServer initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
    
    def load_model(self):
        """Load model and tokenizer"""
        logger.info("Loading model and tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        logger.info("✓ Model and tokenizer loaded")
    
    def _register_routes(self):
        """Register FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Load model on startup"""
            self.load_model()
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy" if self.model is not None else "unhealthy",
                model_loaded=self.model is not None,
                model_name=self.model_path.name,
                device=self.device
            )
        
        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate_code(request: GenerateRequest):
            """Generate code endpoint"""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    request.prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Generate
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        num_return_sequences=request.num_return_sequences,
                        do_sample=True if request.temperature > 0 else False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                generation_time = (time.time() - start_time) * 1000
                
                # Decode
                generated_texts = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                
                # Remove prompt from outputs
                prompt_length = len(request.prompt)
                generated_texts = [
                    text[prompt_length:].strip() if text.startswith(request.prompt) else text
                    for text in generated_texts
                ]
                
                return GenerateResponse(
                    generated_text=generated_texts,
                    prompt=request.prompt,
                    generation_time_ms=generation_time,
                    model_name=self.model_path.name
                )
            
            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "StarCoder2 Inference API",
                "endpoints": {
                    "health": "/health",
                    "generate": "/generate",
                    "docs": "/docs"
                }
            }
    
    def run(self):
        """Run FastAPI server"""
        import uvicorn
        
        logger.info("="*80)
        logger.info("STEP 9: FASTAPI SERVER")
        logger.info("="*80)
        logger.info(f"Starting server on {self.host}:{self.port}")
        logger.info(f"API documentation: http://{self.host}:{self.port}/docs")
        logger.info("="*80)
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def run_fastapi_server(config: dict):
    """
    Convenience function to run FastAPI server
    
    Args:
        config: Server configuration
    """
    server = FastAPIServer(
        model_path=config['model_path'],
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000)
    )
    
    server.run()


if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': './output/final_model/full_model',
        'host': '0.0.0.0',
        'port': 8000
    }
    
    run_fastapi_server(config)