from .common_imports import *
from .config import SiegelModelConfig
from .memory_utils import MemoryManager
from .attention import StandardizedAttention, AttentionType

class EnhancedSiegelKahlerDecoder(nn.Module):
    def __init__(self, config: SiegelModelConfig):
        super().__init__()
        # Replace individual parameters with config
        self.config = config
        
        # Use standardized attention
        self.attention = StandardizedAttention(
            config,
            AttentionType.HYBRID
        )
        
        # Use shared memory manager
        self.memory_manager = MemoryManager(config)
        
        # Temel parametreler
        self.latent_dim = config.latent_dim
        self.vocab_size = config.vocab_size
        self.use_mixed_precision = config.use_mixed_precision
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
        # Siegel manifold ve metrik tanımları
        self.manifold = geoopt.manifolds.SiegelDisk(k=config.latent_dim//2)
        self.metric = geoopt.Metric("siegel")
        
        # Hiperbolik token embedding
        self.token_embedding = geoopt.ManifoldParameter(
            self.manifold.random(config.vocab_size, config.latent_dim),
            manifold=self.manifold
        )

        # Hexagonal hücre tabanlı decoder katmanları
        self.decoder_layers = nn.ModuleList([
            SiegelDecoderLayer(
                latent_dim=config.latent_dim,
                hyperbolic_scale=config.hyperbolic_scale,
                attention_temperature=config.attention_temperature,
                nhead=config.nhead,
                dropout=config.dropout,
                use_gradient_checkpointing=config.use_gradient_checkpointing
            ) for _ in range(config.n_hexagonal_cells)
        ])

        # Hiperbolik çıktı projeksiyonu
        self.output_proj = HyperbolicOutputProjection(
            latent_dim=config.latent_dim,
            vocab_size=config.vocab_size,
            manifold=self.manifold
        )

        # Gradient ve bellek optimizasyonları
        self.grad_scaler = GradScaler(enabled=config.use_mixed_precision)
        self._register_cleanup_hooks()

        # Encoder ile senkronize grid patterns
        self._init_enhanced_grid_patterns()
        
        # Global hiperparametreler
        self.global_hyperbolic_scale = nn.Parameter(torch.tensor(config.hyperbolic_scale))
        self.global_attention_temp = nn.Parameter(torch.tensor(config.attention_temperature))
        
        # Encoder ile paylaşılan metrik
        self.shared_metric = geoopt.Metric("siegel")

    def _init_enhanced_grid_patterns(self):
        """Initialize grid patterns identical to encoder."""
        # Base points for hexagonal lattice
        base_points = torch.tensor([
            [0.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [1.0, 0.0]
        ], device=self.token_embedding.device)

        def modular_transform(z, a, b, c, d):
            return (a*z + b)/(c*z + d)

        transformed_points = []
        for p in base_points:
            z = torch.complex(p[0], p[1])
            for k in range(-2, 3):
                for l in range(-2, 3):
                    t_transform = modular_transform(z, 1, k, 0, 1)
                    st_transform = modular_transform(t_transform, 0, -1, 1, l)
                    transformed_points.append(
                        torch.stack([st_transform.real, st_transform.imag])
                    )

        patterns = torch.stack(transformed_points)
        patterns = F.normalize(patterns, dim=-1)

        # Register patterns and modulation
        self.register_buffer('grid_patterns', patterns)
        self.pattern_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.modulation_weights = nn.Parameter(
            torch.ones(patterns.size(0)) * np.sqrt(1.0 / patterns.size(0))
        )

    def compute_grid_activations(self, z: torch.Tensor) -> torch.Tensor:
        """Compute grid activations consistent with encoder."""
        # Project to grid patterns
        grid_proj = torch.einsum('bnd,gd->bng', z, self.grid_patterns)
        
        # Periodic activation with modulation
        periodic_act = torch.tanh(torch.cos(2 * np.pi * grid_proj * self.pattern_scale))
        weighted_act = periodic_act * F.softplus(self.modulation_weights[None, None, :])
        
        # Competitive normalization
        return F.softmax(weighted_act / self.global_attention_temp, dim=-1)

    def compute_siegel_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute Siegel distance using same method as encoder.
        
        Args:
            z1, z2: Points in Siegel disk
        Returns:
            torch.Tensor: Distance between points
        """
        # Compute necessary terms
        z1_star = z1.conj().transpose(-2, -1)
        z2_star = z2.conj().transpose(-2, -1)
        
        im_z1 = 0.5j * (z1 - z1_star)
        im_z2 = 0.5j * (z2 - z2_star)
        
        # Fast eigendecomposition
        eigenvals1, eigenvecs1 = self.optimized_eigendecomp(im_z1)
        im_z1_sqrt_inv = eigenvecs1 @ torch.diag_embed(torch.rsqrt(eigenvals1)) @ eigenvecs1.transpose(-2, -1).conj()
        
        # Compute distance terms
        diff = z1 - z2
        inner_term = im_z1_sqrt_inv @ diff @ torch.linalg.inv(im_z2) @ diff.transpose(-2, -1).conj() @ im_z1_sqrt_inv
        
        # Use same eigendecomposition as encoder
        eigenvals = torch.diagonal(inner_term, dim1=-2, dim2=-1).real
        return torch.sqrt(torch.abs(torch.sum(torch.log(eigenvals)**2)))

    def update_global_parameters(self, encoder_scale: float, encoder_temp: float):
        """Senkronize hiperparametreleri encoder ile."""
        self.global_hyperbolic_scale.data = torch.tensor(encoder_scale)
        self.global_attention_temp.data = torch.tensor(encoder_temp)
        
        # Update all attention layers
        for layer in self.decoder_layers:
            layer.update_scale_and_temperature(
                self.global_hyperbolic_scale.item(),
                self.global_attention_temp.item()
            )

    def optimized_eigendecomp(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized eigendecomposition matching encoder implementation."""
        return torch.linalg.eigh(matrix)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Poincaré disk üzerinde Möbius toplama."""
        xy = (x * y).sum(dim=-1, keepdim=True)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*xy + y2)*x + (1 - x2)*y
        den = 1 + 2*xy + x2*y2
        return num / den.clamp(min=1e-6)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hiperbolik uzayda forward pass.
        
        Args:
            tgt: Target sequence [batch_size, seq_len]
            memory: Encoder memory [batch_size, seq_len, latent_dim]
            tgt_mask: Target mask
            memory_mask: Memory mask
        """
        # Token'ları hiperbolik uzaya gömme
        x = self.manifold.expmap0(self.token_embedding(tgt))
        
        attention_weights = []
        for layer in self.decoder_layers:
            if self.use_gradient_checkpointing and self.training:
                x, attn = checkpoint(
                    layer,
                    x, memory, tgt_mask, memory_mask
                )
            else:
                x, attn = layer(x, memory, tgt_mask, memory_mask)
            attention_weights.append(attn)

        # Hiperbolik uzaydan vocabulary uzayına projeksiyon
        logits = self.output_proj(x)
        
        return logits, {"attention_weights": torch.stack(attention_weights)}

    def generate(
        self,
        memory: torch.Tensor,
        start_token: int,
        max_length: int = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        min_length: int = 10,
    ) -> torch.Tensor:
        """Hiperbolik uzayda autoregressive generation."""
        if use_beam_search:
            return self._beam_search_generate(
                memory, 
                start_token,
                max_length=max_length,
                min_length=min_length
            )
            
        max_length = max_length or self.max_seq_len
        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), start_token, 
                             device=memory.device)
        past_tokens = set()  # Tekrar kontrolü için
        
        for cur_len in range(max_length):
            # Token embedding
            token_emb = self.token_embedding(generated)
            
            with torch.no_grad():
                logits, _ = self.forward(token_emb, memory)
                logits = logits[:, -1, :] / temperature
                
                # Tekrar cezası uygula
                for prev_token in past_tokens:
                    logits[:, prev_token] /= repetition_penalty
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Minimum uzunluk kontrolü
                if cur_len < min_length:
                    logits[:, self.eos_token_id] = float('-inf')
                    
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Tekrar kontrolü için token'ı kaydet
                past_tokens.add(next_token.item())
                
                if next_token.item() == self.eos_token_id and cur_len >= min_length:
                    break
                    
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _beam_search_generate(
        self,
        memory: torch.Tensor,
        start_token: int,
        max_length: int = None,
        min_length: int = 10
    ) -> torch.Tensor:
        """
        Beam search implementasyonu
        
        Args:
            memory: Encoder'dan gelen bellek tensörü [batch_size, seq_len, dim]
            start_token: Başlangıç token ID'si
            max_length: Maksimum üretilecek sekans uzunluğu
            min_length: Minimum sekans uzunluğu
            
        Returns:
            torch.Tensor: Üretilen token dizisi [batch_size, seq_len]
        """
        max_length = max_length or self.max_seq_len
        batch_size = memory.size(0)
        device = memory.device

        # Her batch için beam tutma
        beams = [Beam(self.beam_size, device=device) for _ in range(batch_size)]
        
        # Başlangıç durumu
        input_ids = torch.full(
            (batch_size * self.beam_size, 1),
            start_token,
            dtype=torch.long,
            device=device
        )
        
        # Memory'i beam_size kadar çoğalt
        memory = memory.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        memory = memory.contiguous().view(batch_size * self.beam_size, -1, memory.size(-1))
        
        # Beam search
        for step in range(max_length):
            # Token embedding ve forward pass
            token_emb = self.token_embedding(input_ids)
            logits, _ = self.forward(token_emb, memory)
            
            # Son token'ın logits'leri
            next_token_logits = logits[:, -1, :]
            
            # Minimum uzunluk kontrolü
            if step < min_length:
                next_token_logits[:, self.eos_token_id] = float('-inf')
            
            # Her batch için ayrı işlem
            next_token_logits = next_token_logits.view(batch_size, self.beam_size, -1)
            
            # Her beam için next_tokens ve next_scores hesapla 
            for batch_idx, beam in enumerate(beams):
                beam.advance(next_token_logits[batch_idx])
                
                # Beam içindeki sekansları güncelle
                if step < max_length - 1:
                    input_ids_batch = input_ids.view(batch_size, self.beam_size, -1)
                    input_ids_batch[batch_idx] = beam.get_current_state()
                    input_ids = input_ids_batch.view(batch_size * self.beam_size, -1)
            
            # Tüm beam'ler tamamlandı mı kontrol et
            if all(beam.is_done for beam in beams):
                break
        
        # En iyi sekansları al
        output_ids = []
        for beam in beams:
            best_score, best_seq = beam.get_best_sequence()
            output_ids.append(best_seq)
            
        return torch.stack(output_ids)

class Beam:
    """Beam search yardımcı sınıfı"""
    
    def __init__(self, beam_size: int, device: torch.device):
        self.beam_size = beam_size
        self.device = device
        
        # Skor ve sekans tutucular
        self.scores = torch.zeros(beam_size, device=device)
        self.tokens = torch.full((beam_size, 1), 0, dtype=torch.long, device=device)
        self.is_done = False
        
        # Tamamlanan sekansları tut
        self.finished_sequences = []
        self.finished_scores = []
        
    def advance(self, log_probs: torch.Tensor):
        """Beam'i bir adım ilerlet"""
        vocab_size = log_probs.size(-1)
        
        if self.is_done:
            return
            
        # Mevcut skorlara log probları ekle
        scores = log_probs + self.scores.unsqueeze(1)
        
        # En iyi beam_size kadar skoru ve token'ı seç
        flat_scores = scores.view(-1)
        best_scores, best_idxs = flat_scores.topk(self.beam_size, dim=0)
        
        # Seçilen token'ları ve beam indekslerini hesapla
        beam_idxs = best_idxs // vocab_size
        token_idxs = best_idxs % vocab_size
        
        # Yeni sekansları oluştur
        new_tokens = torch.cat([
            self.tokens[beam_idxs],
            token_idxs.unsqueeze(1)
        ], dim=1)
        
        # Tamamlanan sekansları kontrol et ve kaydet
        for idx, (score, tokens) in enumerate(zip(best_scores, new_tokens)):
            if tokens[-1].item() == self.eos_token_id:
                self.finished_sequences.append(tokens)
                self.finished_scores.append(score)
                best_scores[idx] = float('-inf')
                
        # Beam'i güncelle
        self.scores = best_scores
        self.tokens = new_tokens
        
        # Tüm beam'ler tamamlandı mı kontrol et
        if len(self.finished_sequences) >= self.beam_size:
            self.is_done = True
            
    def get_current_state(self) -> torch.Tensor:
        """Mevcut beam durumunu döndür"""
        return self.tokens
        
    def get_best_sequence(self) -> Tuple[float, torch.Tensor]:
        """En iyi sekansı döndür"""
        if self.finished_sequences:
            # Tamamlanmış sekanslar varsa en iyisini seç
            best_idx = torch.tensor(self.finished_scores).argmax()
            return self.finished_scores[best_idx], self.finished_sequences[best_idx]
        else:
            # Tamamlanmamış en iyi sekansı döndür
            best_idx = self.scores.argmax()
            return self.scores[best_idx], self.tokens[best_idx]

class HyperbolicOutputProjection(nn.Module):
    """Hiperbolik uzaydan vocabulary uzayına projeksiyon."""
    
    def __init__(self, latent_dim: int, vocab_size: int, manifold: geoopt.manifolds.SiegelDisk):
        super().__init__()
        self.manifold = manifold
        self.vocab_proj = nn.Sequential(
            geoopt.ManifoldParameter(
                torch.empty(latent_dim, latent_dim * 2),
                manifold=geoopt.Stiefel()
            ),
            nn.GELU(),
            geoopt.ManifoldParameter(
                torch.empty(latent_dim * 2, vocab_size),
                manifold=geoopt.Stiefel()
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Optimize edilmiş parametre başlatma."""
        for param in self.parameters():
            if isinstance(param, geoopt.ManifoldParameter):
                param.data = param.manifold.random(param.size())
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced hiperbolik projeksiyon."""
        # Manifold-aware intermediate space
        x_lorentz = self.manifold.logmap0(x)
        
        # Apply grid pattern modulation
        grid_acts = self.parent_module.compute_grid_activations(x_lorentz)
        x_modulated = x_lorentz * grid_acts.mean(dim=1, keepdim=True)
        
        # Final projection with preserved geometry
        return self.vocab_proj(x_modulated)

class SiegelDecoderLayer(nn.Module):
    """Siegel disk üzerinde decoder katmanı."""
    def __init__(
        self,
        latent_dim: int,
        hyperbolic_scale: float = 1.0,
        attention_temperature: float = 1.0,
        nhead: int = 8,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Self-attention bloğu
        self.self_attn = AdaptiveHybridEncoder(
            dim=latent_dim,
            nhead=nhead,
            dropout=dropout,
            hyperbolic_scale=hyperbolic_scale,
            temperature=attention_temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # Cross-attention bloğu
        self.cross_attn = AdaptiveHybridEncoder(
            dim=latent_dim,
            nhead=nhead,
            dropout=dropout,
            hyperbolic_scale=hyperbolic_scale,
            temperature=attention_temperature,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        # Manifold-aware feed forward
        self.ffn = nn.Sequential(
            geoopt.manifolds.Stiefel().proj_manifold(
                nn.Linear(latent_dim, latent_dim * 4)
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            geoopt.manifolds.Stiefel().proj_manifold(
                nn.Linear(latent_dim * 4, latent_dim)
            )
        )

        # Normalizasyon katmanları
        self.norm_self_attn = geoopt.manifolds.SiegelDisk().proj_manifold(
            nn.LayerNorm(latent_dim)
        )
        self.norm_cross_attn = geoopt.manifolds.SiegelDisk().proj_manifold(
            nn.LayerNorm(latent_dim)
        )
        self.norm_ffn = geoopt.manifolds.SiegelDisk().proj_manifold(
            nn.LayerNorm(latent_dim)
        )

        # Hiperbolik manifold
        self.manifold = geoopt.manifolds.SiegelDisk(k=latent_dim//2)

    def _manifold_residual(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Siegel manifold üzerinde residual bağlantı."""
        return self.manifold.projx(
            self.manifold.logmap(x) + self.manifold.logmap(res)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with manifold-aware operations.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            memory: Encoder memory [batch_size, seq_len, dim]
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Output tensor
                - Attention weights
        """
        # Self-attention with gradient checkpointing
        def self_attn_block(x, mask):
            attn_out, meta = self.self_attn(x, mask)
            return self.norm_self_attn(
                self._manifold_residual(x, attn_out)
            ), meta['routing_weights']

        if self.use_gradient_checkpointing and self.training:
            x, self_attn_weights = checkpoint(self_attn_block, x, tgt_mask)
        else:
            x, self_attn_weights = self_attn_block(x, tgt_mask)

        # Cross-attention with memory
        def cross_attn_block(x, memory, mask):
            # Concat current state with memory for global context
            context, _ = self.cross_attn(
                torch.cat([x, memory], dim=1),
                mask=torch.cat([tgt_mask, memory_mask], dim=-1) if mask is not None else None
            )
            # Extract relevant part
            attn_out = context[:, :x.size(1), :]
            return self.norm_cross_attn(
                self._manifold_residual(x, attn_out)
            )

        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(cross_attn_block, x, memory, memory_mask)
        else:
            x = cross_attn_block(x, memory, memory_mask)

        # Feed-forward with manifold projection
        def ffn_block(x):
            ffn_out = self.ffn(self.manifold.logmap(x))
            return self.norm_ffn(
                self._manifold_residual(x, ffn_out)
            )

        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(ffn_block, x)
        else:
            x = ffn_block(x)

        return x, self_attn_weights

    def update_temperature(self, epoch: int, max_epochs: int):
        """Update attention temperature using cosine schedule."""
        for attn in [self.self_attn, self.cross_attn]:
            if hasattr(attn, 'update_temperature'):
                attn.update_temperature(epoch, max_epochs)

    def update_scale_and_temperature(self, scale: float, temp: float):
        """Update scale and temperature parameters."""
        self.self_attn.update_parameters(scale, temp)
        self.cross_attn.update_parameters(scale, temp)
