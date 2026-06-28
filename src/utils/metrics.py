"""
src/utils/metrics.py
Evaluation metrics for DAFR-Net: PSNR, SSIM, LPIPS.
"""

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Inputs in [0,1]."""
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return psnr(pred, target).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM. Inputs in [0,1]."""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim(pred, target).item()


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Full evaluation of a batch. Returns dict with psnr, ssim.
    pred, target: (B, 3, H, W) in [-1, 1] — will be normalized to [0, 1].
    """
    pred_01   = (pred.clamp(-1, 1)   + 1) / 2
    target_01 = (target.clamp(-1, 1) + 1) / 2
    return {
        "psnr": compute_psnr(pred_01, target_01),
        "ssim": compute_ssim(pred_01, target_01),
    }
