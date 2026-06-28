"""
tests/test_pipeline.py
End-to-end pipeline test: DualBranchEncoder → InpaintingHead → loss → backward.
No dataset required — runs on random tensors.

Run:  pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import pytest

from src.encoder.dual_branch import DualBranchEncoder
from src.decoder.inpainting_head import InpaintingHead, InpaintingLoss
from src.utils.metrics import evaluate_batch


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


@pytest.fixture(scope="module")
def pipeline():
    enc = DualBranchEncoder(SMALL_CFG)
    dec = InpaintingHead(in_channels=512, decoder_channels=[256, 128, 64, 32])
    return enc, dec


# ---------------------------------------------------------------------------

def test_pipeline_output_is_image(pipeline):
    enc, dec = pipeline
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        latent   = enc(x)
        restored = dec(latent)
    assert restored.ndim == 4
    assert restored.shape[1] == 3
    assert restored.min() >= -1.0 - 1e-5
    assert restored.max() <=  1.0 + 1e-5


def test_pipeline_encoder_output_channels(pipeline):
    enc, _ = pipeline
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = enc(x)
    assert out.shape[1] == 512          # embed_dim


def test_pipeline_loss_and_backward(pipeline):
    enc, dec = pipeline
    loss_fn  = InpaintingLoss()

    x      = torch.randn(2, 3, 224, 224)
    target = torch.randn(2, 3, 224, 224).clamp(-1, 1)

    latent   = enc(x)
    restored = dec(latent)

    # Resize to align with target if Swin downsampled more than 4×UpBlocks cover
    if restored.shape[-2:] != target.shape[-2:]:
        restored = F.interpolate(restored, size=target.shape[-2:],
                                 mode="bilinear", align_corners=False)

    losses = loss_fn(restored, target)
    losses["total"].backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in list(enc.parameters()) + list(dec.parameters())
        if p.requires_grad
    )
    assert has_grad, "No gradients reached model parameters"


def test_pipeline_evaluate_batch(pipeline):
    enc, dec = pipeline
    x      = torch.randn(1, 3, 224, 224)
    target = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        latent   = enc(x)
        restored = dec(latent)
        if restored.shape[-2:] != target.shape[-2:]:
            restored = F.interpolate(restored, size=target.shape[-2:],
                                     mode="bilinear", align_corners=False)
        metrics = evaluate_batch(restored, target)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["psnr"] > 0
