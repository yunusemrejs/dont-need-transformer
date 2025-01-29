from .common_imports import *
from .config import SiegelModelConfig
from .memory_utils import MemoryManager
from .attention import StandardizedAttention, AttentionType

class SiegelKahlerEncoder(nn.Module):
    """Enhanced Siegel-Kahler encoder with standardized architecture."""
    
    def __init__(self, config: SiegelModelConfig):
        super().__init__()
        self.config = config
        self.memory_manager = MemoryManager(config)
        
        # Initialize weights using Lorentz centroid
        self._init_weights()
        
        # Initialize attention mechanism
        self.attention = StandardizedAttention(
            config=config,
            attention_type=AttentionType.HYBRID if config.use_adaptive_hybrid 
                         else AttentionType.CONFORMAL
        )
        
        # Initialize normalizations
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.latent_norm = nn.LayerNorm(config.latent_dim)
        
        # Initialize manifold and metric
        self.manifold = geoopt.manifolds.SiegelDisk(k=config.latent_dim//2)
        self.metric = geoopt.Metric("siegel")
        
        # Gradient scaler for mixed precision
        self.grad_scaler = GradScaler(
            enabled=config.use_mixed_precision,
            init_scale=0.2,
            growth_interval=100
        )
        
        # Initialize grid patterns
        self.grid_patterns, self.modulation_weights = config.create_shared_patterns(
            device=self.W_real.device
        )
        
        # Register cleanup hooks
        self._register_cleanup_hooks()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize real and imaginary weights
        init_scale = 1.0 / np.sqrt(self.config.input_dim)
        w_real = self._lorentz_centroid_init(
            (self.config.input_dim, self.config.latent_dim),
            scale=init_scale
        )
        w_imag = self._lorentz_centroid_init(
            (self.config.input_dim, self.config.latent_dim),
            scale=init_scale
        )
        
        # Register as manifold parameters
        self.W_real = geoopt.ManifoldParameter(w_real, manifold=geoopt.Stiefel())
        self.W_imag = geoopt.ManifoldParameter(w_imag, manifold=geoopt.Stiefel())
        
        # Initialize biases
        self.b_real = nn.Parameter(torch.zeros(self.config.latent_dim))
        self.b_imag = nn.Parameter(torch.zeros(self.config.latent_dim))
    
    def _lorentz_centroid_init(self, shape: Tuple[int, ...], scale: float = 1.0) -> Tensor:
        """Initialize weights using Lorentz centroid method."""
        init_tensor = torch.randn(shape)
        norm = torch.sqrt((init_tensor ** 2).sum(dim=-1, keepdim=True) + 1e-6)
        return scale * init_tensor / norm
    
    def _register_cleanup_hooks(self):
        """Register resource cleanup hooks."""
        def cleanup(*args):
            self.memory_manager.clear_cache()
            torch.cuda.empty_cache()
        
        self.register_forward_hook(lambda *args: cleanup())
    
    @torch.jit.script_method
    def project_to_manifold(self, Z: Tensor) -> Tensor:
        """Project tensor to Siegel disk manifold."""
        return self.manifold.projx(Z)
    
    def compute_grid_activations(self, z: Tensor) -> Tensor:
        """Compute grid activations with memory optimization."""
        return self.memory_manager.efficient_attention(
            self._compute_grid_activations_impl,
            z,
            use_checkpoint=self.config.use_gradient_checkpointing
        )
    
    def _compute_grid_activations_impl(self, z: Tensor) -> Tensor:
        """Implementation of grid activation computation."""
        # Project with adaptive scaling
        grid_proj = oe.contract('bnd,gd->bng', z, self.grid_patterns)
        
        # Apply periodic activation with modulation
        periodic_act = torch.tanh(torch.cos(2 * np.pi * grid_proj))
        weighted_act = periodic_act * F.softplus(self.modulation_weights[None, None, :])
        
        # Competitive normalization
        return F.softmax(weighted_act / self.config.grid_temperature, dim=-1)
    
    @autocast()
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass with enhanced processing."""
        if x.numel() == 0:
            raise ValueError("Empty input tensor")
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Project to complex space
        h_real = F.gelu(x @ self.W_real + self.b_real)
        h_imag = F.gelu(x @ self.W_imag + self.b_imag)
        
        # Form complex tensor
        Z = torch.complex(h_real, h_imag)
        Z = self.project_to_manifold(Z)
        
        # Apply attention
        Z_attended, attention_metadata = self.attention(Z)
        
        # Compute grid activations
        grid_acts = self.compute_grid_activations(Z_attended)
        
        # Prepare metadata
        metadata = {
            'grid_activations': grid_acts,
            'memory_metrics': self.memory_manager.get_metrics(),
            **attention_metadata
        }
        
        return Z_attended, metadata
    
    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_cache(clear_metrics=True)
