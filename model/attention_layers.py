from .common_imports import *
from .config import SiegelModelConfig

class ConformalSelfAttention(nn.Module):
    """Conformal Self-Attention using hyperbolic geometry."""
    def __init__(self, dim: int, hyperbolic_scale: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(hyperbolic_scale))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.manifold = geoopt.manifolds.PoincareBall()

        # Projections
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict]:
        q = self.manifold.expmap0(self.proj_q(x))
        k = self.manifold.expmap0(self.proj_k(x))
        v = self.manifold.expmap0(self.proj_v(x))

        # Compute attention scores
        scores = self.manifold.dist(q.unsqueeze(-2), k.unsqueeze(-3))
        scores = -scores * self.scale / self.temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum('...ij,...jd->...id', attn, v)

        return attn_output, {"type": "conformal"}

class AdaptiveHybridAttention(nn.Module):
    """Hybrid attention combining conformal and Euclidean attention."""
    def __init__(
        self,
        dim: int,
        nhead: int = 8,
        dropout: float = 0.1,
        hyperbolic_scale: float = 1.0,
        temperature: float = 1.0,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.conformal_attn = ConformalSelfAttention(dim, hyperbolic_scale, temperature)
        self.euclidean_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.use_checkpoint = use_gradient_checkpointing

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict]:
        conf_out, _ = self.conformal_attn(x, mask)
        eucl_out, _ = self.euclidean_attn(x, x, x, key_padding_mask=mask)

        # Adaptive combination
        combined = self.alpha * conf_out + (1 - self.alpha) * eucl_out
        return combined, {"alpha": self.alpha.item()}

class SiegelAttention(nn.Module):
    """Attention mechanism specifically for Siegel disk manifold."""
    def __init__(self, config: SiegelModelConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.hybrid_attn = AdaptiveHybridAttention(
            dim=config.hidden_size,
            nhead=config.num_attention_heads,
            dropout=config.attention_dropout
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict]:
        return self.hybrid_attn(x, mask)
