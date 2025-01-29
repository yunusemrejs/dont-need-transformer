from .common_imports import *
from .config import SiegelModelConfig
from .attention_layers import (
    ConformalSelfAttention,
    AdaptiveHybridAttention,
    SiegelAttention
)
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from enum import Enum

class AttentionType(Enum):
    CONFORMAL = "conformal"
    HYBRID = "hybrid"
    SIEGEL = "siegel"

class StandardizedAttention(nn.Module):
    """Standardized attention mechanisms for Siegel-Kahler architecture."""
    
    def __init__(
        self,
        config: SiegelModelConfig,
        attention_type: AttentionType
    ):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        if attention_type == AttentionType.CONFORMAL:
            self.attention = ConformalSelfAttention(
                dim=config.latent_dim,
                hyperbolic_scale=config.hyperbolic_scale,
                temperature=config.attention_temperature
            )
        elif attention_type == AttentionType.HYBRID:
            self.attention = AdaptiveHybridAttention(
                dim=config.latent_dim,
                nhead=config.nhead,
                dropout=config.dropout,
                hyperbolic_scale=config.hyperbolic_scale,
                temperature=config.attention_temperature,
                use_gradient_checkpointing=config.use_gradient_checkpointing
            )
        elif attention_type == AttentionType.SIEGEL:
            self.attention = SiegelAttention(config=config)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        return self.attention(x, mask)

    def update_parameters(self, scale: float, temp: float):
        """Update shared parameters."""
        if hasattr(self.attention, 'update_parameters'):
            self.attention.update_parameters(scale, temp)
