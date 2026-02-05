import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.attention import RelativeMultiHeadAttention

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super(SwiGLUFeedForward, self).__init__()
        # SwiGLU: (Swish(xW) * xV)W_out
        # We project to 2 * dim_feedforward to have gate and value
        self.linear1 = nn.Linear(d_model, 2 * dim_feedforward) 
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.linear1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class TransposeGroupNorm(nn.Module):
    """
    GroupNorm expects (N, C, L), but we have (N, L, C).
    This wrapper handles the permute.
    """
    def __init__(self, num_channels, num_groups=32):
        super(TransposeGroupNorm, self).__init__()
        if num_channels % num_groups != 0:
            # Fallback if d_model is not divisible by 32 (e.g. 100, 80)
            # Find closest divisor or default to 1 group (LayerNorm-ish but over channels)
            num_groups = 1
            for i in [32, 16, 8, 4, 2]:
                if num_channels % i == 0:
                    num_groups = i
                    break
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    
    def forward(self, x):
        # x: (N, L, C)
        x = x.transpose(1, 2) # (N, C, L)
        x = self.norm(x)
        return x.transpose(1, 2) # (N, L, C)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super(ConformerConvModule, self).__init__()

        # 1. Giriş Normalizasyonu (LayerNorm -> GroupNorm)
        self.norm = TransposeGroupNorm(d_model)

        # 2. Pointwise Conv + GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # 3. Depthwise Conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )

        # 4. Batch Norm + Aktivasyon (GELU -> Swish/SiLU)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()

        # 5. Çıkış Pointwise Conv
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, D_Model)
        x = self.norm(x)
        
        x = x.transpose(1, 2) # (B, C, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        
        return x.transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ConformerBlock, self).__init__()

        # --- 1. Feed Forward (Macaron Style - 1. Yarı) ---
        # Gelu -> SwiGLU implementation
        self.ff1 = SwiGLUFeedForward(d_model, d_model * 4, dropout)
        self.norm_ff1 = TransposeGroupNorm(d_model) # FF öncesi norm (Pre-norm yapısı)

        # --- 2. Multi-Head Self Attention ---
        self.attn = RelativeMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm_attn = TransposeGroupNorm(d_model)

        # --- 3. Convolution Module ---
        self.conv = ConformerConvModule(d_model)
        self.norm_conv = TransposeGroupNorm(d_model)

        # --- 4. Feed Forward (Macaron Style - 2. Yarı) ---
        self.ff2 = SwiGLUFeedForward(d_model, d_model * 4, dropout)
        self.norm_ff2 = TransposeGroupNorm(d_model)

        # --- 5. Final Normalizasyon ---
        self.final_norm = TransposeGroupNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-Norm mimarisi: x + f(norm(x))
        
        # 1. FF1
        # Note: Original code applied norm inside FF block? No, it had LayerNorm layer in sequential.
        # Here we apply norm before FF explicitly or inside? 
        # Standard Conformer: x = x + 0.5 * FF(Norm(x))
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        
        # 2. Attention
        attn_out, _ = self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x), mask=mask)
        x = x + attn_out
        
        # 3. Conv
        # Conv module has internal norm as per previous code, but standard is PreNorm outside.
        # My ConvModule has 'self.norm' at start. So passing x is fine.
        x = x + self.conv(x)
        
        # 4. FF2
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        
        return self.final_norm(x)

class TurkishASRModel(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        d_model=256,
        n_heads=4,
        n_blocks=6,
        n_classes=31,
        dropout=0.1
    ):
        super(TurkishASRModel, self).__init__()
        
        # --- 1. Subsampling (CNN) ---
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.SiLU(), # GELU -> SiLU
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.SiLU()  # GELU -> SiLU
        )
        
        flattened_dim = d_model * (n_mel_channels // 4)
        self.input_proj = nn.Linear(flattened_dim, d_model)
        
        # Positional Encoding is now handled inside Attention via RoPE
        # Leaving this empty or removing it.
        
        # --- 2. Conformer Blocks ---
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # --- 3. Classifier ---
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, input_lengths=None):
        """
        x: (Batch, Time, Mel_Channels)
        input_lengths: (Batch,) - Length of each sequence
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)
        
        # 1. Subsampling
        x = self.subsample(x) # (B, d_model, T/4, F/4)
        
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, t, -1) 
        x = self.input_proj(x)
        
        # Mask Creation
        mask = None
        if input_lengths is not None:
            # Subsampling divides length by 4
            subsampled_lengths = input_lengths // 4
            # Create (B, 1, 1, T) mask
            # mask[b, :, :, t] is 0 if t >= length[b]
            max_len = x.size(1)
            # Create range [0, 1, ..., max_len-1]
            seq_range = torch.arange(max_len, device=x.device).unsqueeze(0) # (1, T)
            # Expand lengths to (B, 1)
            lens = subsampled_lengths.unsqueeze(1) # (B, 1)
            # Mask: (B, T) -> True if valid
            mask = seq_range < lens
            # Reshape for Attention: (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # 2. Conformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)
            
        # 3. Classifier
        x = self.fc(x)
        
        return x
