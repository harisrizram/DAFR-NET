"""
src/encoder/dual_branch.py
Dual-Branch Encoder: Structure Branch (Swin Transformer) + Frequency Branch (FFT)
Fused via Cross-Attention — core novelty #2 of DAFR-Net
"""

import torch
import torch.nn as nn
import torch.fft
import timm
from einops import rearrange


class StructureBranch(nn.Module):
    """
    Swin Transformer backbone extracts hierarchical structural features.
    Captures edges, outlines, and spatial patterns of the mural.
    """
    def __init__(self, backbone: str = "swin_base_patch4_window7_224",
                 pretrained: bool = True, out_channels: int = 256):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        # Project last feature map to out_channels
        in_ch = self.encoder.feature_info[-1]["num_chs"]
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        feat = features[-1]                   # take deepest feature map
        if feat.dim() == 3:                   # Swin returns (B, HW, C)
            B, HW, C = feat.shape
            H = W = int(HW ** 0.5)
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
        return self.proj(feat)


class FrequencyBranch(nn.Module):
    """
    FFT-based frequency feature extractor.
    Captures texture statistics and periodic patterns lost in degraded regions.
    Core of novelty #2 — no prior mural paper combines this with structure jointly.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 256,
                 num_layers: int = 4):
        super().__init__()
        self.freq_conv = nn.ModuleList()
        ch = in_channels * 2               # real + imaginary components
        for i in range(num_layers):
            out_ch = out_channels // (2 ** (num_layers - i - 1))
            self.freq_conv.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply 2D FFT and split into real/imag
        freq = torch.fft.fft2(x, norm="ortho")
        freq = torch.fft.fftshift(freq)
        real = freq.real
        imag = freq.imag
        feat = torch.cat([real, imag], dim=1)   # (B, 2C, H, W)

        for layer in self.freq_conv:
            feat = layer(feat)
        return feat


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between structure and frequency features.
    Structure features attend to frequency features and vice versa.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_s = nn.Linear(embed_dim // 2, embed_dim)
        self.proj_f = nn.Linear(embed_dim // 2, embed_dim)

        self.attn_s2f = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_f2s = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, struct_feat: torch.Tensor,
                freq_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = struct_feat.shape

        # Flatten spatial dims for attention
        s = rearrange(struct_feat, "b c h w -> b (h w) c")
        f = rearrange(freq_feat,   "b c h w -> b (h w) c")

        s = self.proj_s(s)
        f = self.proj_f(f)

        # Structure attends to frequency
        s2f, _ = self.attn_s2f(query=s, key=f, value=f)
        s2f = self.norm1(s + s2f)

        # Frequency attends to structure
        f2s, _ = self.attn_f2s(query=f, key=s, value=s)
        f2s = self.norm2(f + f2s)

        # Concatenate and project
        fused = torch.cat([s2f, f2s], dim=-1)        # (B, HW, 2*embed)
        fused = self.out_proj(fused)                  # (B, HW, embed)
        fused = rearrange(fused, "b (h w) c -> b c h w", h=H, w=W)
        return fused


class DualBranchEncoder(nn.Module):
    """
    Full dual-branch encoder: structure + frequency → cross-attention fusion.
    Output is the shared latent representation fed into the decoder.
    """
    def __init__(self, config: dict):
        super().__init__()
        struct_cfg = config["model"]["structure_branch"]
        freq_cfg   = config["model"]["frequency_branch"]
        fusion_cfg = config["model"]["fusion"]

        self.structure_branch = StructureBranch(
            backbone=struct_cfg["backbone"],
            pretrained=struct_cfg["pretrained"],
            out_channels=struct_cfg["out_channels"],
        )
        self.frequency_branch = FrequencyBranch(
            out_channels=freq_cfg["out_channels"],
            num_layers=freq_cfg["freq_layers"],
        )
        self.fusion = CrossAttentionFusion(
            embed_dim=fusion_cfg["embed_dim"],
            num_heads=fusion_cfg["num_heads"],
            dropout=fusion_cfg["dropout"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        struct_feat = self.structure_branch(x)
        freq_feat   = self.frequency_branch(x)

        # Align spatial resolution (freq branch preserves H×W, struct may downsample)
        if freq_feat.shape[-2:] != struct_feat.shape[-2:]:
            freq_feat = nn.functional.interpolate(
                freq_feat, size=struct_feat.shape[-2:],
                mode="bilinear", align_corners=False
            )

        fused = self.fusion(struct_feat, freq_feat)
        return fused
