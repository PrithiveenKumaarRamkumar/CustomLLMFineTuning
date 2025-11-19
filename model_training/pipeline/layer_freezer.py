# pipeline/04_layer_freezer.py
# ================================================================================
# Module 4: Freeze model layers
# ================================================================================

import torch
from typing import Dict, List, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)


class LayerFreezer:
    """Freeze model layers based on user configuration"""
    
    def __init__(self, model):
        """
        Initialize layer freezer
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.stats = {
            'total_params': 0,
            'frozen_params': 0,
            'trainable_params': 0,
            'frozen_layers': []
        }
        
        logger.info("LayerFreezer initialized")
    
    def get_layer_info(self) -> Dict:
        """Get information about model layers"""
        layer_info = {}
        
        for name, param in self.model.named_parameters():
            # Extract layer number
            layer_num = self._extract_layer_number(name)
            
            if layer_num is not None:
                if layer_num not in layer_info:
                    layer_info[layer_num] = {
                        'params': [],
                        'total_params': 0
                    }
                
                layer_info[layer_num]['params'].append(name)
                layer_info[layer_num]['total_params'] += param.numel()
        
        return layer_info
    
    def _extract_layer_number(self, param_name: str) -> Optional[int]:
        """Extract layer number from parameter name"""
        patterns = [
            r'layers\.(\d+)\.',
            r'h\.(\d+)\.',
            r'layer\.(\d+)\.',
            r'model\.layers\.(\d+)\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        
        return None
    
    def freeze_by_strategy(self, strategy: Dict) -> Dict:
        """
        Freeze layers based on strategy
        
        Args:
            strategy: Freezing strategy configuration
                {
                    'strategy': 'first_n' | 'last_n' | 'specific' | 'embeddings' | 'layer_norms',
                    'n_layers': int (for first_n/last_n),
                    'layer_indices': List[int] (for specific),
                    'freeze_embeddings': bool,
                    'freeze_layer_norms': bool
                }
        
        Returns:
            Dictionary with freezing statistics
        """
        logger.info("="*80)
        logger.info("STEP 4: LAYER FREEZING")
        logger.info("="*80)
        
        # Count total parameters
        self.stats['total_params'] = sum(p.numel() for p in self.model.parameters())
        
        strategy_type = strategy.get('strategy', 'first_n')
        logger.info(f"Strategy: {strategy_type}")
        
        # Apply strategy
        if strategy_type == 'first_n':
            n_layers = strategy.get('n_layers', 15)
            logger.info(f"Freezing first {n_layers} layers")
            self._freeze_first_n_layers(n_layers)
        
        elif strategy_type == 'last_n':
            n_layers = strategy.get('n_layers', 10)
            total_layers = strategy.get('total_layers', 30)
            logger.info(f"Freezing last {n_layers} layers")
            self._freeze_last_n_layers(n_layers, total_layers)
        
        elif strategy_type == 'specific':
            layer_indices = strategy.get('layer_indices', [])
            logger.info(f"Freezing specific layers: {layer_indices}")
            self._freeze_specific_layers(layer_indices)
        
        elif strategy_type == 'none':
            logger.info("No layer freezing applied")
        
        else:
            logger.warning(f"Unknown strategy: {strategy_type}")
        
        # Freeze embeddings if requested
        if strategy.get('freeze_embeddings', False):
            logger.info("Freezing embeddings")
            self._freeze_embeddings()
        
        # Freeze layer norms if requested
        if strategy.get('freeze_layer_norms', False):
            logger.info("Freezing layer normalizations")
            self._freeze_layer_norms()
        
        # Calculate statistics
        self.stats['frozen_params'] = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        self.stats['trainable_params'] = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        freeze_percentage = (
            self.stats['frozen_params'] / self.stats['total_params'] * 100
            if self.stats['total_params'] > 0 else 0
        )
        
        logger.info("\n" + "="*80)
        logger.info("FREEZING STATISTICS")
        logger.info("="*80)
        logger.info(f"Total parameters:      {self.stats['total_params']:,}")
        logger.info(f"Frozen parameters:     {self.stats['frozen_params']:,}")
        logger.info(f"Trainable parameters:  {self.stats['trainable_params']:,}")
        logger.info(f"Freeze percentage:     {freeze_percentage:.2f}%")
        logger.info("="*80)
        logger.info("✓ STEP 4 COMPLETE")
        logger.info("="*80)
        
        return self.stats
    
    def _freeze_first_n_layers(self, n_layers: int):
        """Freeze first N layers"""
        frozen_layers = set()
        
        for name, param in self.model.named_parameters():
            layer_num = self._extract_layer_number(name)
            
            if layer_num is not None and layer_num < n_layers:
                param.requires_grad = False
                frozen_layers.add(layer_num)
        
        self.stats['frozen_layers'] = sorted(list(frozen_layers))
        logger.info(f"Frozen layers: {self.stats['frozen_layers']}")
    
    def _freeze_last_n_layers(self, n_layers: int, total_layers: int):
        """Freeze last N layers"""
        start_layer = total_layers - n_layers
        frozen_layers = set()
        
        for name, param in self.model.named_parameters():
            layer_num = self._extract_layer_number(name)
            
            if layer_num is not None and layer_num >= start_layer:
                param.requires_grad = False
                frozen_layers.add(layer_num)
        
        self.stats['frozen_layers'] = sorted(list(frozen_layers))
        logger.info(f"Frozen layers: {self.stats['frozen_layers']}")
    
    def _freeze_specific_layers(self, layer_indices: List[int]):
        """Freeze specific layers"""
        frozen_layers = set()
        
        for name, param in self.model.named_parameters():
            layer_num = self._extract_layer_number(name)
            
            if layer_num is not None and layer_num in layer_indices:
                param.requires_grad = False
                frozen_layers.add(layer_num)
        
        self.stats['frozen_layers'] = sorted(list(frozen_layers))
        logger.info(f"Frozen layers: {self.stats['frozen_layers']}")
    
    def _freeze_embeddings(self):
        """Freeze embedding layers"""
        embed_keywords = ['embed', 'wte', 'wpe', 'token_embedding', 'position_embedding']
        
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in embed_keywords):
                param.requires_grad = False
                logger.debug(f"Frozen embedding: {name}")
    
    def _freeze_layer_norms(self):
        """Freeze layer normalization layers"""
        norm_keywords = ['ln', 'norm', 'layernorm', 'layer_norm']
        
        for name, param in self.model.named_parameters():
            if any(keyword in name.lower() for keyword in norm_keywords):
                param.requires_grad = False
                logger.debug(f"Frozen norm: {name}")


def run_layer_freezing(model, config: dict) -> Dict:
    """
    Convenience function to freeze layers
    
    Args:
        model: PyTorch model
        config: Freezing configuration
    
    Returns:
        Dictionary with freezing statistics
    """
    freezer = LayerFreezer(model)
    return freezer.freeze_by_strategy(config)


if __name__ == "__main__":
    # Example usage
    from model_loader import run_model_loading
    
    # Load model
    model_config = {
        'model_path': './models/starcoder2-3b',
        'use_4bit': True
    }
    model, _ = run_model_loading(model_config)
    
    # Freeze layers
    freeze_config = {
        'strategy': 'first_n',
        'n_layers': 15,
        'freeze_embeddings': True,
        'freeze_layer_norms': True
    }
    
    stats = run_layer_freezing(model, freeze_config)
    print(json.dumps(stats, indent=2))