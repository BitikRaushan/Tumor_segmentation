"""
Multi-Level Dual Aspect Self-Attention Guided Transformer (MDA-SGT)
Combines Token Transformer Self-Attention (TSA) and Global Spatial Attention (GSA).
Reconstructed from BSPC 2025 Paper (Figure 3 & Equations 3-12)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerSelfAttention(nn.Module):
    """TSA: Standard Vision Multi-Head Self-Attention over flattened tokens."""
    def __init__(self, in_channels: int, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feat = self.proj_in(x)  # [B, embed_dim, H, W]
        tokens = feat.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        # Learnable / broadcastable positional coordinates added
        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))

        out = tokens.transpose(1, 2).view(b, self.embed_dim, h, w)
        return self.proj_out(out)

class GlobalSpatialAttention(nn.Module):
    """GSA: Spatial boundary affinity matrix (IB) based on Equations 10 & 11."""
    def __init__(self, in_channels: int):
        super().__init__()
        reduced_ch = max(in_channels // 2, 32)
        self.query_conv = nn.Conv2d(in_channels, reduced_ch, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_ch, kernel_size=1)
        self.val_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        proj_q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, N, C']
        proj_k = self.key_conv(x).view(b, -1, h * w)                      # [B, C', N]
        
        # Spatial energy / intermediate boundary matrix (IB_ij)
        energy = torch.bmm(proj_q, proj_k)  # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        proj_v = self.val_conv(x).view(b, -1, h * w)  # [B, C, N]
        out = torch.bmm(proj_v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x

class MDASGTBottleneck(nn.Module):
    """Dual-branch fusion of TSA and GSA with learnable scaling factors lambda_1, lambda_2."""
    def __init__(self, channels: int, embed_dim: int = 768):
        super().__init__()
        self.tsa = TransformerSelfAttention(channels, embed_dim=embed_dim)
        self.gsa = GlobalSpatialAttention(channels)
        self.lambda_1 = nn.Parameter(torch.tensor(0.5))
        self.lambda_2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_tsa = self.tsa(x)
        f_gsa = self.gsa(x)
        return self.lambda_1 * f_tsa + self.lambda_2 * f_gsa