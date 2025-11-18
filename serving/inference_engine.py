import asyncio
import logging
import time
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    """Represents a single inference task in the queue."""
    task_id: str
    prompt: str
    max_length: int
    temperature: float
    top_p: float
    stop_tokens: Optional[List[str]]
    future: asyncio.Future
    timestamp: datetime


class ModelManager:
    """Manages model loading and metadata from MLflow."""
    
    def __init__(self, model_name: str, mlflow_uri: str, stage: str = "Production"):
        self.model_name = model_name
        self.mlflow_uri = mlflow_uri
        self.stage = stage
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        self.model = None
        self.tokenizer = None
        self.model_info = None
        self.device = None
        self._setup_device()
    
    def _setup_device(self):
        """Setup GPU/CPU device for inference."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for inference")
    
    async def load_model(self) -> bool:
        """Load model from MLflow registry."""
        try:
            # Get model version info from MLflow
            model_version = self.client.get_latest_versions(
                name=self.model_name,
                stages=[self.stage]
            )[0]
            
            model_uri = f"models:/{self.model_name}/{self.stage}"
            
            # Load model artifacts
            model_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Set padding token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Store model metadata
            self.model_info = {
                "name": self.model_name,
                "version": model_version.version,
                "creation_timestamp": datetime.fromtimestamp(model_version.creation_timestamp / 1000),
                "mlflow_run_id": model_version.run_id,
                "model_uri": model_uri,
                "parameters": {}
            }
            
            logger.info(f"Successfully loaded model: {self.model_name} v{model_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model metadata."""
        return self.model_info or {}
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None
    
    def get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0)
        return 0


class GPUBatchingQueue:
    """Asynchronous GPU batching queue for efficient inference."""
    
    def __init__(self, model_manager: ModelManager, batch_size: int = 4, timeout: float = 0.1):
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.running = False
        self.processor_task = None
        
        # Metrics
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.total_tokens_generated = 0
    
    async def start(self):
        """Start the batch processing loop."""
        self.running = True
        self.processor_task = asyncio.create_task(self._batch_processor())
        logger.info("GPU batching queue started")
    
    async def stop(self):
        """Stop the batch processing loop."""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        logger.info("GPU batching queue stopped")
    
    async def submit_task(self, task: InferenceTask) -> str:
        """Submit an inference task to the queue."""
        await self.queue.put(task)
        self.total_requests += 1
        return await task.future
    
    async def _batch_processor(self):
        """Main batch processing loop."""
        while self.running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Error in batch processor: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[InferenceTask]:
        """Collect tasks into a batch."""
        batch = []
        start_time = time.time()
        
        # Wait for at least one task
        try:
            task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            batch.append(task)
        except asyncio.TimeoutError:
            return batch
        
        # Collect additional tasks up to batch size or timeout
        while len(batch) < self.batch_size and (time.time() - start_time) < self.timeout:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                batch.append(task)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[InferenceTask]):
        """Process a batch of inference tasks."""
        if not self.model_manager.is_model_loaded():
            for task in batch:
                task.future.set_exception(RuntimeError("Model not loaded"))
            return
        
        start_time = time.time()
        
        try:
            # Prepare batch inputs
            prompts = [task.prompt for task in batch]
            
            # Tokenize inputs
            inputs = self.model_manager.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if self.model_manager.device.type == "cuda":
                inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            # Generate responses
            with torch.no_grad():
                # Use the first task's parameters for the entire batch
                # (In production, you might want to group by parameters)
                first_task = batch[0]
                
                generation_config = GenerationConfig(
                    max_new_tokens=first_task.max_length,
                    temperature=first_task.temperature,
                    top_p=first_task.top_p,
                    do_sample=True,
                    pad_token_id=self.model_manager.tokenizer.eos_token_id
                )
                
                outputs = self.model_manager.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode outputs
            generated_texts = []
            for i, output in enumerate(outputs):
                # Skip input tokens
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                
                generated_text = self.model_manager.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                # Apply stop tokens if specified
                if batch[i].stop_tokens:
                    for stop_token in batch[i].stop_tokens:
                        if stop_token in generated_text:
                            generated_text = generated_text.split(stop_token)[0]
                
                generated_texts.append(generated_text)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Set results for each task
            for i, (task, generated_text) in enumerate(zip(batch, generated_texts)):
                tokens_generated = len(generated_text.split())  # Rough estimate
                self.total_tokens_generated += tokens_generated
                
                result = {
                    "generated_text": generated_text,
                    "prompt": task.prompt,
                    "tokens_generated": tokens_generated,
                    "processing_time": processing_time / len(batch),
                    "model_name": self.model_manager.model_info["name"],
                    "model_version": self.model_manager.model_info["version"],
                    "timestamp": datetime.utcnow()
                }
                
                task.future.set_result(result)
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            for task in batch:
                task.future.set_exception(e)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current queue metrics."""
        avg_duration = self.total_processing_time / max(self.total_requests, 1)
        avg_tokens_per_second = self.total_tokens_generated / max(self.total_processing_time, 0.001)
        
        return {
            "api_requests_total": self.total_requests,
            "api_request_duration_seconds": avg_duration,
            "gpu_memory_usage_bytes": self.model_manager.get_gpu_memory_usage(),
            "tokens_per_second": avg_tokens_per_second
        }


class InferenceEngine:
    """Main inference engine that orchestrates model loading and request processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            model_name=config["mlflow"]["model_name"],
            mlflow_uri=config["mlflow"]["tracking_uri"],
            stage=config["mlflow"]["model_stage"]
        )
        
        # Initialize GPU batching queue
        self.gpu_queue = GPUBatchingQueue(
            model_manager=self.model_manager,
            batch_size=config["model"]["batch_size"],
            timeout=config["model"].get("timeout", 60)
        )
        
        self.is_ready = False
    
    async def initialize(self) -> bool:
        """Initialize the inference engine."""
        try:
            # Load model
            if not await self.model_manager.load_model():
                return False
            
            # Start GPU batching queue
            await self.gpu_queue.start()
            
            self.is_ready = True
            logger.info("Inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {str(e)}")
            return False
    
    async def shutdown(self):
        """Shutdown the inference engine."""
        await self.gpu_queue.stop()
        self.is_ready = False
        logger.info("Inference engine shutdown complete")
    
    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text based on the request."""
        if not self.is_ready:
            raise RuntimeError("Inference engine not ready")
        
        # Create inference task
        task = InferenceTask(
            task_id=f"task_{int(time.time() * 1000000)}",
            prompt=request_data["prompt"],
            max_length=request_data.get("max_length", 100),
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 0.95),
            stop_tokens=request_data.get("stop_tokens"),
            future=asyncio.Future(),
            timestamp=datetime.utcnow()
        )
        
        # Submit task and wait for result
        return await self.gpu_queue.submit_task(task)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        gpu_available = torch.cuda.is_available()
        model_loaded = self.model_manager.is_model_loaded()
        
        overall_status = "healthy" if (self.is_ready and model_loaded and gpu_available) else "unhealthy"
        
        return {
            "status": overall_status,
            "api_status": "running",
            "model_status": "loaded" if model_loaded else "not_loaded",
            "gpu_status": "available" if gpu_available else "unavailable",
            "timestamp": datetime.utcnow(),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        return self.model_manager.get_model_info()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = self.gpu_queue.get_metrics()
        metrics["timestamp"] = datetime.utcnow()
        return metrics