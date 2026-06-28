"""
tests/test_encoder.py
Unit tests for StructureBranch, FrequencyBranch, CrossAttentionFusion,
and DualBranchEncoder.

Run:  pytest tests/test_encoder.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.encoder.dual_branch import (
    FrequencyBranch,
    CrossAttentionFusion,
    DualBranchEncoder,
)


# swin_tiny is the smallest Swin variant — avoids downloading large weights
SMALL_CFG = {
    "model": {
        "structure_branch": {
            "backbone": "swin_tiny_patch4_window7_224",
            "pretrained": False,
            "out_channels": 256,
        },
        "frequency_branch": {
            "fft_norm": "ortho",
            "out_channels": 256,
            "freq_layers": 4,
        },
        "fusion": {
            "type": "cross_attention",
            "num_heads": 4,
            "dropout": 0.0,
            "embed_dim": 512,
        },
    }
}


# ---------------------------------------------------------------------------
# FrequencyBranch
# ---------------------------------------------------------------------------

def test_freq_branch_preserves_spatial():
    branch = FrequencyBranch(in_channels=3, out_channels=256, num_layers=4)
    x      = torch.randn(2, 3, 64, 64)
    out    = branch(x)
    assert out.shape == (2, 256, 64, 64), f"Got {out.shape}"


def test_freq_branch_output_channels():
    branch = FrequencyBranch(in_channels=3, out_channels=128, num_layers=4)
    x      = torch.randn(1, 3, 32, 32)
    out    = branch(x)
    assert out.shape[1] == 128


def test_freq_branch_gradient_flows():
    branch = FrequencyBranch(in_channels=3, out_channels=64, num_layers=2)
    x      = torch.randn(1, 3, 32, 32, requires_grad=True)
    out    = branch(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# CrossAttentionFusion
# ---------------------------------------------------------------------------

def test_fusion_output_shape():
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=4, dropout=0.0)
    s      = torch.randn(2, 256, 8, 8)
    f      = torch.randn(2, 256, 8, 8)
    out    = fusion(s, f)
    assert out.shape == (2, 512, 8, 8), f"Got {out.shape}"


def test_fusion_different_content():
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=4, dropout=0.0)
    s = torch.randn(1, 256, 4, 4)
    f = torch.zeros(1, 256, 4, 4)
    out1 = fusion(s, f)
    out2 = fusion(s, s)
    assert not torch.allclose(out1, out2), "Fusion output should differ for different inputs"


# ---------------------------------------------------------------------------
# DualBranchEncoder (end-to-end)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    return DualBranchEncoder(SMALL_CFG)


def test_encoder_forward_shape(encoder):
    x   = torch.randn(1, 3, 224, 224)
    out = encoder(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 512          # embed_dim from fusion
    assert out.ndim == 4                 # (B, C, H, W)


def test_encoder_batch_consistency(encoder):
    x1  = torch.randn(1, 3, 224, 224)
    x2  = torch.cat([x1, x1], dim=0)
    o1  = encoder(x1)
    o2  = encoder(x2)
    # First sample in batch should match single-sample result
    assert torch.allclose(o1, o2[:1], atol=1e-5)


def test_encoder_gradient_flows(encoder):
    x   = torch.randn(1, 3, 224, 224)
    out = encoder(x)
    out.sum().backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in encoder.parameters() if p.requires_grad
    )
    assert has_grad, "No gradients reached encoder parameters"
