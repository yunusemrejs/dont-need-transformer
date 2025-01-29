from dataclasses import dataclass
from typing import Optional
import torch
from .common_imports import *

@dataclass
class SiegelModelConfig:
    """Shared configuration for Siegel-Kahler encoder-decoder architecture."""
    
    latent_dim: int
    vocab_size: Optional[int] = None
    n_hexagonal_cells: int = 6
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    epsilon: float = 1e-6
    hyperbolic_scale: float = 1.0
    attention_temperature: float = 1.0
    nhead: int = 8
    dropout: float = 0.1
    
    # Grid pattern parameters
    grid_scale: float = 0.1
    edge_modulation: float = 0.3
    grid_temperature: float = 1.0
    
    # Memory optimization
    memory_efficient_attention: bool = True
    chunk_size: int = 32
    
    # Yeni eklenecek parametreler
    max_seq_len: int = 512
    eos_token_id: int = 2
    beam_size: int = 4
    length_penalty: float = 0.6
    
    # Eksik parametreler eklendi
    use_conformal_attention: bool = True
    use_adaptive_hybrid: bool = True
    input_dim: Optional[int] = None
    fft_mode: str = "ortho"
    cache_size: int = 1000
    
    def create_shared_patterns(self, device: torch.device):
        """Initialize shared grid patterns with modular group symmetries."""
        base_points = torch.tensor([
            [0.0, 0.0],
            [0.5, torch.sqrt(torch.tensor(3.0))/2],
            [1.0, 0.0]
        ], device=device)
        
        def modular_transform(z: torch.Tensor, a: int, b: int, c: int, d: int) -> torch.Tensor:
            """Apply modular transformation (az + b)/(cz + d)."""
            num = a * z + b
            den = c * z + d
            return num / (den + 1e-8)  # Numerical stability
            
        def generate_modular_group_patterns() -> torch.Tensor:
            """Generate patterns using modular group transformations."""
            patterns = []
            # Fundamental domain transformations
            for k in range(-2, 3):  # Translation range
                for l in range(-2, 3):  # Rotation range
                    for p in base_points:
                        z = torch.complex(p[0], p[1])
                        # Apply T^k transformation (parabolic)
                        t_transform = modular_transform(z, 1, k, 0, 1)
                        # Apply S transformation (elliptic)
                        s_transform = modular_transform(t_transform, 0, -1, 1, 0)
                        # Apply T^l transformation
                        st_transform = modular_transform(s_transform, 1, l, 0, 1)
                        # Store real and imaginary parts
                        patterns.append(torch.stack([st_transform.real, st_transform.imag]))
            
            return torch.stack(patterns)
            
        def apply_hexagonal_symmetries(patterns: torch.Tensor) -> torch.Tensor:
            """Apply hexagonal lattice symmetries."""
            # Rotation matrices for 60° rotations
            theta = torch.tensor([np.pi/3], device=device)
            rot_60 = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]
            ], device=device)
            
            # Apply rotational symmetries
            rotated_patterns = []
            for pattern in patterns:
                for i in range(6):  # 6-fold symmetry
                    rot_matrix = torch.matrix_power(rot_60, i)
                    rotated = torch.matmul(pattern, rot_matrix)
                    rotated_patterns.append(rotated)
                    
            return torch.stack(rotated_patterns)
            
        def optimize_pattern_distribution(patterns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Optimize pattern distribution for better coverage."""
            # Remove duplicates within small numerical threshold
            dists = torch.cdist(patterns, patterns)
            mask = (dists > 1e-6).all(dim=1)
            unique_patterns = patterns[mask]
            
            # Compute optimal weights using maximum entropy principle
            n_patterns = unique_patterns.size(0)
            weights = torch.ones(n_patterns, device=device) / np.sqrt(n_patterns)
            
            # Normalize patterns
            patterns_normalized = F.normalize(unique_patterns, dim=-1)
            
            return patterns_normalized, weights
            
        # Generate basic patterns using modular group
        patterns = generate_modular_group_patterns()
        
        # Apply hexagonal symmetries
        patterns_with_symmetry = apply_hexagonal_symmetries(patterns)
        
        # Optimize pattern distribution
        final_patterns, modulation_weights = optimize_pattern_distribution(patterns_with_symmetry)
        
        # Scale patterns to unit disk
        final_patterns = final_patterns * self.grid_scale
        
        return final_patterns, modulation_weights
    
    # Doğrulama metodları eklendi
    def validate(self):
        """Validate configuration parameters."""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.latent_dim % 2 == 0, "latent_dim must be even for Siegel disk"
        if self.vocab_size is not None:
            assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_hexagonal_cells > 0, "n_hexagonal_cells must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
