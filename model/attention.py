"""
Attention mechanisms for Conformer ASR model.

Includes:
- Rotary Positional Embeddings (RoPE)
- Multi-Query Attention (MQA)
- Flash Attention support (when available)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =====================================================
# Rotary Positional Embeddings (RoPE)
# =====================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Applies rotation to queries and keys based on position,
    enabling relative position awareness without explicit position embeddings.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int) -> None:
        """Update cached cos/sin for sequence length."""
        self.max_seq_len = max(seq_len, self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache as (1, 1, T, D) for broadcasting
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for sequence length."""
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    return (x * cos) + (rotate_half(x) * sin)


# =====================================================
# Flash Attention (when available)
# =====================================================

# Check for Flash Attention availability
FLASH_ATTENTION_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """
    Use PyTorch 2.0+ Flash Attention if available.
    
    Args:
        q: (B, H, T, D)
        k: (B, H, T, D)
        v: (B, H, T, D)
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        
    Returns:
        Attention output (B, H, T, D)
    """
    if FLASH_ATTENTION_AVAILABLE and q.is_cuda:
        # Convert boolean mask to float if needed
        attn_mask = None
        if mask is not None:
            # Flash attention expects additive mask
            attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
            
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=is_causal
        )
    else:
        # Fallback to standard attention
        return _standard_attention(q, k, v, mask, dropout_p)


def _standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
        
    return torch.matmul(attn_weights, v)


# =====================================================
# Multi-Head Attention with RoPE and MQA
# =====================================================

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with:
    - Rotary Position Embeddings (RoPE)
    - Multi-Query Attention (MQA) for efficiency
    - Flash Attention support (PyTorch 2.0+)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_mqa: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_mqa = use_mqa
        self.use_flash = use_flash and FLASH_ATTENTION_AVAILABLE
        
        # Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(self.d_head)
        
        # Query projection (always multi-head)
        self.linear_q = nn.Linear(d_model, d_model)
        
        # Key/Value projections
        if use_mqa:
            # Multi-Query: single head for K,V (shared across all query heads)
            self.linear_k = nn.Linear(d_model, self.d_head)
            self.linear_v = nn.Linear(d_model, self.d_head)
            self.n_kv_heads = 1
        else:
            # Standard multi-head
            self.linear_k = nn.Linear(d_model, d_model)
            self.linear_v = nn.Linear(d_model, d_model)
            self.n_kv_heads = n_heads
        
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Query input (B, T, D)
            x_k: Key input (B, T, D)
            x_v: Value input (B, T, D)
            mask: Attention mask (B, 1, 1, T) or (B, 1, T, T)
            
        Returns:
            output: (B, T, D)
            attn_weights: (B, H, T, T) or None if using flash attention
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Projections
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        if self.use_mqa:
            k = self.linear_k(x_k).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
            v = self.linear_v(x_v).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        else:
            k = self.linear_k(x_k).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
            v = self.linear_v(x_v).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 2. Apply RoPE to Q and K
        cos, sin = self.rotary_emb(q, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # 3. Expand K, V for MQA if needed
        if self.use_mqa and self.n_heads > 1:
            k = k.expand(-1, self.n_heads, -1, -1)
            v = v.expand(-1, self.n_heads, -1, -1)
        
        # 4. Attention
        if self.use_flash and self.training:
            # Flash Attention (no attention weights returned)
            context = flash_attention(q, k, v, mask, self.dropout_p)
            attn_weights = None
        else:
            # Standard attention with weights
            context = _standard_attention(q, k, v, mask, self.dropout_p if self.training else 0.0)
            attn_weights = None  # Skip storing for memory
        
        # 5. Output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.linear_out(context)
        
        return output, attn_weights


# =====================================================
# Legacy Compatibility
# =====================================================

class RelativePositionalEncoding(nn.Module):
    """Legacy class for backward compatibility."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> None:
        return None
