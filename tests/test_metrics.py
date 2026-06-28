"""
tests/test_metrics.py
Unit tests for PSNR / SSIM evaluation metrics.

Run:  pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from src.utils.metrics import compute_psnr, compute_ssim, evaluate_batch


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_identical():
    x    = torch.rand(1, 3, 64, 64)
    psnr = compute_psnr(x, x)
    assert psnr > 50.0, f"Identical images should have very high PSNR, got {psnr:.2f}"


def test_psnr_range():
    pred   = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    psnr   = compute_psnr(pred, target)
    assert 0 < psnr < 60.0


def test_psnr_decreases_with_noise():
    target    = torch.rand(1, 3, 64, 64)
    small_err = target + 0.01 * torch.randn_like(target)
    large_err = target + 0.1  * torch.randn_like(target)
    small_err = small_err.clamp(0, 1)
    large_err = large_err.clamp(0, 1)
    assert compute_psnr(small_err, target) > compute_psnr(large_err, target)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical():
    x    = torch.rand(1, 3, 64, 64)
    ssim = compute_ssim(x, x)
    assert abs(ssim - 1.0) < 0.01, f"Identical images SSIM should ≈ 1, got {ssim:.4f}"


def test_ssim_range():
    pred   = torch.rand(1, 3, 64, 64)
    target = torch.rand(1, 3, 64, 64)
    ssim   = compute_ssim(pred, target)
    assert 0.0 <= ssim <= 1.0, f"SSIM out of range: {ssim}"


def test_ssim_decreases_with_noise():
    target    = torch.rand(1, 3, 64, 64)
    small_err = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
    large_err = (target + 0.2  * torch.randn_like(target)).clamp(0, 1)
    assert compute_ssim(small_err, target) > compute_ssim(large_err, target)


# ---------------------------------------------------------------------------
# evaluate_batch (operates on [-1, 1] tensors)
# ---------------------------------------------------------------------------

def test_evaluate_batch_keys():
    pred   = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    result = evaluate_batch(pred, target)
    assert "psnr" in result
    assert "ssim" in result


def test_evaluate_batch_identical():
    x      = torch.randn(1, 3, 64, 64)  # [-1, 1] range
    result = evaluate_batch(x, x)
    assert result["psnr"] > 50.0
    assert abs(result["ssim"] - 1.0) < 0.01


def test_evaluate_batch_values_sane():
    pred   = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    result = evaluate_batch(pred, target)
    assert 0 < result["psnr"]
    assert 0 <= result["ssim"] <= 1.0
