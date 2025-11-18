import os
import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class AdapterLoader:
    """Manages loading and switching between different model adapters."""
    
    def __init__(self, base_model_path: str, mlflow_uri: str):
        self.base_model_path = base_model_path
        self.mlflow_uri = mlflow_uri
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        
        # Model components
        self.base_model = None
        self.tokenizer = None
        self.current_adapter = None
        self.loaded_adapters = {}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model metadata
        self.model_metadata = {}
    
    async def load_base_model(self, model_name: str, model_stage: str = "Production") -> bool:
        """Load the base model from MLflow."""
        try:
            logger.info(f"Loading base model: {model_name}")
            
            # Get model version from MLflow
            model_versions = self.client.get_latest_versions(
                name=model_name,
                stages=[model_stage]
            )
            
            if not model_versions:
                logger.error(f"No model found with name {model_name} in stage {model_stage}")
                return False
            
            model_version = model_versions[0]
            model_uri = f"models:/{model_name}/{model_stage}"
            
            # Download model artifacts
            local_model_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # Store metadata
            self.model_metadata = {
                "name": model_name,
                "version": model_version.version,
                "creation_timestamp": model_version.creation_timestamp,
                "run_id": model_version.run_id,
                "model_uri": model_uri,
                "base_model_path": local_model_path
            }
            
            logger.info(f"Successfully loaded base model: {model_name} v{model_version.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            return False
    
    async def load_adapter(
        self,
        adapter_name: str,
        adapter_path: Optional[str] = None,
        mlflow_model_name: Optional[str] = None,
        mlflow_stage: str = "Production"
    ) -> bool:
        """Load a LoRA adapter."""
        try:
            if adapter_name in self.loaded_adapters:
                logger.info(f"Adapter {adapter_name} already loaded")
                return True
            
            # Determine adapter path
            if mlflow_model_name:
                # Load from MLflow
                model_uri = f"models:/{mlflow_model_name}/{mlflow_stage}"
                adapter_path = mlflow.artifacts.download_artifacts(model_uri)
            elif not adapter_path:
                raise ValueError("Either adapter_path or mlflow_model_name must be provided")
            
            logger.info(f"Loading adapter: {adapter_name} from {adapter_path}")
            
            # Load PEFT config to validate
            peft_config = PeftConfig.from_pretrained(adapter_path)
            
            # Create adapter model
            if self.current_adapter:
                # Unload current adapter first
                self.base_model = self.base_model.unload()
            
            adapter_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Store adapter information
            self.loaded_adapters[adapter_name] = {
                "model": adapter_model,
                "path": adapter_path,
                "config": peft_config,
                "mlflow_model_name": mlflow_model_name
            }
            
            self.current_adapter = adapter_name
            
            logger.info(f"Successfully loaded adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {str(e)}")
            return False
    
    def switch_adapter(self, adapter_name: str) -> bool:
        """Switch to a different loaded adapter."""
        try:
            if adapter_name not in self.loaded_adapters:
                logger.error(f"Adapter {adapter_name} not loaded")
                return False
            
            if self.current_adapter == adapter_name:
                logger.info(f"Adapter {adapter_name} already active")
                return True
            
            # Switch adapter
            self.current_adapter = adapter_name
            logger.info(f"Switched to adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to adapter {adapter_name}: {str(e)}")
            return False
    
    def unload_adapter(self, adapter_name: str) -> bool:
        """Unload a specific adapter."""
        try:
            if adapter_name not in self.loaded_adapters:
                logger.warning(f"Adapter {adapter_name} not loaded")
                return True
            
            # If this is the current adapter, switch to base model
            if self.current_adapter == adapter_name:
                if hasattr(self.loaded_adapters[adapter_name]["model"], "unload"):
                    self.loaded_adapters[adapter_name]["model"].unload()
                self.current_adapter = None
            
            # Remove from loaded adapters
            del self.loaded_adapters[adapter_name]
            
            logger.info(f"Unloaded adapter: {adapter_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_name}: {str(e)}")
            return False
    
    def get_current_model(self):
        """Get the currently active model (base or adapter)."""
        if self.current_adapter and self.current_adapter in self.loaded_adapters:
            return self.loaded_adapters[self.current_adapter]["model"]
        return self.base_model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = self.model_metadata.copy()
        
        # Add adapter information
        if self.current_adapter:
            adapter_info = self.loaded_adapters[self.current_adapter]
            info.update({
                "active_adapter": self.current_adapter,
                "adapter_path": adapter_info["path"],
                "adapter_config": adapter_info["config"].to_dict() if adapter_info["config"] else {},
                "mlflow_adapter_model": adapter_info.get("mlflow_model_name")
            })
        
        # Add available adapters
        info["loaded_adapters"] = list(self.loaded_adapters.keys())
        
        # Add device information
        info["device"] = str(self.device)
        info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["gpu_memory_cached"] = torch.cuda.memory_reserved(0)
        
        return info
    
    def list_available_adapters(self) -> List[str]:
        """List all loaded adapters."""
        return list(self.loaded_adapters.keys())
    
    def is_adapter_loaded(self, adapter_name: str) -> bool:
        """Check if a specific adapter is loaded."""
        return adapter_name in self.loaded_adapters
    
    def get_adapter_config(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific adapter."""
        if adapter_name in self.loaded_adapters:
            config = self.loaded_adapters[adapter_name]["config"]
            return config.to_dict() if config else None
        return None
    
    async def reload_model(self, model_name: str, model_stage: str = "Production") -> bool:
        """Reload the base model (useful for model updates)."""
        try:
            # Unload all adapters first
            for adapter_name in list(self.loaded_adapters.keys()):
                self.unload_adapter(adapter_name)
            
            # Clear base model
            if self.base_model:
                del self.base_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Reload base model
            return await self.load_base_model(model_name, model_stage)
            
        except Exception as e:
            logger.error(f"Failed to reload model: {str(e)}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_info = {
            "cpu_memory_percent": 0,
            "gpu_memory_allocated": 0,
            "gpu_memory_cached": 0,
            "gpu_memory_total": 0
        }
        
        # CPU memory (requires psutil)
        try:
            import psutil
            memory_info["cpu_memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_cached": torch.cuda.memory_reserved(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
            })
        
        return memory_info
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Unload all adapters
            for adapter_name in list(self.loaded_adapters.keys()):
                self.unload_adapter(adapter_name)
            
            # Clear base model
            if self.base_model:
                del self.base_model
            
            if self.tokenizer:
                del self.tokenizer
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("AdapterLoader cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


class MultiAdapterManager:
    """Manages multiple adapter loaders for different model types."""
    
    def __init__(self, mlflow_uri: str):
        self.mlflow_uri = mlflow_uri
        self.adapter_loaders = {}
        self.default_loader = None
    
    async def add_model(
        self,
        model_key: str,
        model_name: str,
        model_stage: str = "Production",
        set_as_default: bool = False
    ) -> bool:
        """Add a new model with its adapter loader."""
        try:
            loader = AdapterLoader(
                base_model_path="",  # Will be determined from MLflow
                mlflow_uri=self.mlflow_uri
            )
            
            success = await loader.load_base_model(model_name, model_stage)
            if success:
                self.adapter_loaders[model_key] = loader
                
                if set_as_default or not self.default_loader:
                    self.default_loader = model_key
                
                logger.info(f"Added model {model_key} ({model_name})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add model {model_key}: {str(e)}")
            return False
    
    def get_loader(self, model_key: Optional[str] = None) -> Optional[AdapterLoader]:
        """Get adapter loader for a specific model or default."""
        if model_key:
            return self.adapter_loaders.get(model_key)
        elif self.default_loader:
            return self.adapter_loaders.get(self.default_loader)
        return None
    
    def list_models(self) -> List[str]:
        """List all available model keys."""
        return list(self.adapter_loaders.keys())
    
    def remove_model(self, model_key: str) -> bool:
        """Remove a model and its adapter loader."""
        if model_key in self.adapter_loaders:
            self.adapter_loaders[model_key].cleanup()
            del self.adapter_loaders[model_key]
            
            if self.default_loader == model_key:
                self.default_loader = list(self.adapter_loaders.keys())[0] if self.adapter_loaders else None
            
            logger.info(f"Removed model {model_key}")
            return True
        
        return False
    
    def cleanup_all(self):
        """Cleanup all adapter loaders."""
        for loader in self.adapter_loaders.values():
            loader.cleanup()
        
        self.adapter_loaders.clear()
        self.default_loader = None
        logger.info("All adapter loaders cleaned up")