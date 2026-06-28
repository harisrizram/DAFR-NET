"""
src/decoder/inpainting_head.py
UNet-style inpainting decoder — Phase 1 output head.
Takes fused encoder features → outputs restored image patch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class InpaintingHead(nn.Module):
    """
    Phase 1 decoder: takes fused latent (B, 512, H/8, W/8) → restored image (B, 3, H, W).
    Simple UNet upsampling — replaced by diffusion decoder in Phase 2.
    """
    def __init__(self, in_channels: int = 512,
                 decoder_channels: list = None):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        self.ups = nn.ModuleList()
        ch = in_channels
        for out_ch in decoder_channels:
            self.ups.append(UpBlock(ch, out_ch))
            ch = out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(ch, 3, kernel_size=1),
            nn.Tanh(),           # output in [-1, 1]; denorm to [0, 255] for display
        )

    def forward(self, latent: torch.Tensor,
                mask: torch.Tensor = None,
                original: torch.Tensor = None) -> torch.Tensor:
        x = latent
        for up in self.ups:
            x = up(x)

        restored = self.final_conv(x)

        # Composite: use original pixels outside mask, prediction inside
        if mask is not None and original is not None:
            mask_up = F.interpolate(mask, size=restored.shape[-2:], mode="nearest")
            restored = original * (1 - mask_up) + restored * mask_up

        return restored


class InpaintingLoss(nn.Module):
    """
    Combined loss for Phase 1 inpainting training.
    L1 + SSIM + optional VGG perceptual (added in Phase 2).
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.2):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> dict:
        if mask is not None:
            # Compute loss only inside the damaged region
            mask_up = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            pred_m   = pred   * mask_up
            target_m = target * mask_up
        else:
            pred_m, target_m = pred, target

        l1_loss = self.l1(pred_m, target_m)

        # Simple SSIM approximation via local means
        ssim_loss = 1 - self._ssim(pred_m, target_m)

        total = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        return {"total": total, "l1": l1_loss, "ssim": ssim_loss}

    @staticmethod
    def _ssim(x, y, window_size=11):
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
        sigma_x  = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
        sigma_y  = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        return ssim.mean()
