"""
tests/test_decoder.py
Unit tests for InpaintingHead and InpaintingLoss.

Run:  pytest tests/test_decoder.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.decoder.inpainting_head import InpaintingHead, InpaintingLoss


@pytest.fixture
def head():
    return InpaintingHead(in_channels=512, decoder_channels=[256, 128, 64, 32])


@pytest.fixture
def loss_fn():
    return InpaintingLoss(l1_weight=1.0, ssim_weight=0.2)


# ---------------------------------------------------------------------------
# InpaintingHead
# ---------------------------------------------------------------------------

def test_output_shape(head):
    latent = torch.randn(2, 512, 8, 8)
    out    = head(latent)
    # 4 UpBlocks each 2× → 8 → 128
    assert out.shape == (2, 3, 128, 128), f"Got {out.shape}"


def test_output_in_tanh_range(head):
    latent = torch.randn(2, 512, 8, 8)
    out    = head(latent)
    assert out.min() >= -1.0 - 1e-5, f"Min {out.min():.4f} < -1"
    assert out.max() <=  1.0 + 1e-5, f"Max {out.max():.4f} > +1"


def test_output_with_mask_composite(head):
    latent   = torch.randn(1, 512, 8, 8)
    original = torch.randn(1, 3, 128, 128).clamp(-1, 1)
    mask     = torch.ones(1, 1, 128, 128)   # fully masked

    out = head(latent, mask=mask, original=original)
    assert out.shape == (1, 3, 128, 128)


def test_no_mask_no_original(head):
    latent = torch.randn(1, 512, 4, 4)
    out    = head(latent)
    assert out.ndim == 4


def test_gradient_flows_through_head():
    h      = InpaintingHead(in_channels=64, decoder_channels=[32, 16])
    latent = torch.randn(1, 64, 4, 4)
    out    = h(latent)
    out.sum().backward()
    has_grad = any(p.grad is not None for p in h.parameters())
    assert has_grad


# ---------------------------------------------------------------------------
# InpaintingLoss
# ---------------------------------------------------------------------------

def test_loss_keys(loss_fn):
    pred   = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    losses = loss_fn(pred, target)
    assert {"total", "l1", "ssim"} <= losses.keys()


def test_loss_total_positive(loss_fn):
    pred   = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    assert loss_fn(pred, target)["total"].item() > 0


def test_loss_identical_inputs(loss_fn):
    x      = torch.rand(1, 3, 32, 32)
    losses = loss_fn(x, x)
    assert losses["l1"].item() < 1e-6


def test_loss_with_mask(loss_fn):
    pred   = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    mask   = torch.randint(0, 2, (2, 1, 64, 64)).float()
    losses = loss_fn(pred, target, mask=mask)
    assert losses["total"].item() >= 0


def test_loss_backward(loss_fn):
    pred   = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    losses = loss_fn(pred, target)
    losses["total"].backward()
    assert pred.grad is not None
