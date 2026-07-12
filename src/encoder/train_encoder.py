"""
src/encoder/train_encoder.py
Training entry point for DAFR-Net: DualBranchEncoder + InpaintingHead.

Usage:
    python src/encoder/train_encoder.py --config configs/encoder.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm

from src.encoder.dual_branch import DualBranchEncoder
from src.decoder.inpainting_head import InpaintingHead, InpaintingLoss
from src.utils.metrics import evaluate_batch


class MuralRestorationDataset(Dataset):
    """
    Paired dataset: (damaged_image, clean_target, binary_mask).

    Directory layout expected:
        image_dir/   — damaged input images
        mask_dir/    — binary masks (255 = damaged pixel)
        target_dir/  — clean reference images (optional; if absent, input = target)

    When target_dir is None the model trains in reconstruction mode —
    it learns to reproduce the undamaged regions, which is useful for
    self-supervised pre-training before fine-tuning on real pairs.
    """

    def __init__(self, image_dir: str, mask_dir: str, target_dir: str = None,
                 image_size: int = 256, augment: bool = True):
        self.image_dir  = Path(image_dir)
        self.mask_dir   = Path(mask_dir)
        self.target_dir = Path(target_dir) if target_dir else None

        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(
            p for p in self.image_dir.rglob("*") if p.suffix.lower() in exts
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        self.augment = augment

    # ------------------------------------------------------------------
    def _load_mask(self, img_path: Path) -> torch.Tensor:
        mask_path = self.mask_dir / img_path.with_suffix(".png").name
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", (256, 256), color=255)
        return self.mask_tf(mask)  # (1, H, W) in [0, 1]

    @staticmethod
    def _hflip(img, target, mask):
        img    = torch.flip(img,    dims=[-1])
        target = torch.flip(target, dims=[-1])
        mask   = torch.flip(mask,   dims=[-1])
        return img, target, mask

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img      = self.img_tf(Image.open(img_path).convert("RGB"))

        if self.target_dir is not None:
            tgt_path = self.target_dir / img_path.name
            target   = self.img_tf(Image.open(tgt_path).convert("RGB")) \
                       if tgt_path.exists() else img.clone()
        else:
            target = img.clone()

        mask = self._load_mask(img_path)

        if self.augment and torch.rand(1) > 0.5:
            img, target, mask = self._hflip(img, target, mask)

        return img, target, mask


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG16 feature-matching loss at relu2_2 and relu3_3.
    Encourages texture fidelity beyond per-pixel accuracy.
    Weights are frozen; runs at fp32 regardless of training precision.
    """

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        self.slice1 = nn.Sequential(*feats[:9]).eval()   # up to relu2_2
        self.slice2 = nn.Sequential(*feats[9:16]).eval() # up to relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1) / 2              # → [0, 1]
        return (x.float() - self.mean) / self.std  # ImageNet-normalize

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, t = self._prep(pred), self._prep(target)
        f1p, f1t = self.slice1(p), self.slice1(t)
        f2p, f2t = self.slice2(f1p), self.slice2(f1t)
        return F.l1_loss(f1p, f1t) + F.l1_loss(f2p, f2t)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DAFRNetModule(pl.LightningModule):
    """
    End-to-end training module: DualBranchEncoder → InpaintingHead.

    Forward produces a restored image at the same spatial resolution as
    the input (encoder output is upsampled bilinearly if decoder is shallower
    than required to reach input resolution — common with Swin backbones).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        fusion_dim = config["model"]["fusion"]["embed_dim"]

        self.encoder = DualBranchEncoder(config)
        self.decoder = InpaintingHead(
            in_channels=fusion_dim,
            decoder_channels=config["model"]["inpainting_head"]["decoder_channels"],
        )

        loss_cfg = config.get("loss", {})
        self.criterion      = InpaintingLoss(
            l1_weight=1.0,
            ssim_weight=loss_cfg.get("ssim_weight", 0.2),
        )
        perc_w = loss_cfg.get("perceptual_weight", 0.1)
        self.perceptual        = PerceptualLoss() if perc_w > 0 else None
        self.perceptual_weight = perc_w

        self._log_img_every = config.get("logging", {}).get(
            "log_images_every_n_epochs", 10
        )
        self._val_samples: list = []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                original: torch.Tensor = None) -> torch.Tensor:
        latent   = self.encoder(x)
        restored = self.decoder(latent)

        # Bilinear resize to input resolution when decoder is shallower
        if restored.shape[-2:] != x.shape[-2:]:
            restored = F.interpolate(
                restored, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        # Composite: keep original pixels outside mask, prediction inside
        if mask is not None and original is not None:
            mask_rs  = F.interpolate(mask, size=restored.shape[-2:], mode="nearest")
            restored = original * (1 - mask_rs) + restored * mask_rs

        return restored

    # ------------------------------------------------------------------
    def _loss(self, pred, target, mask):
        losses = self.criterion(pred, target, mask)
        if self.perceptual is not None:
            losses["perceptual"] = self.perceptual(pred, target)
            losses["total"] = losses["total"] + self.perceptual_weight * losses["perceptual"]
        return losses

    def training_step(self, batch, batch_idx):
        img, target, mask = batch
        pred   = self(img, mask=mask, original=img)
        losses = self._loss(pred, target, mask)
        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=(k == "total"),
                     on_step=True, on_epoch=True)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        img, target, mask = batch
        pred   = self(img, mask=mask, original=img)
        losses = self._loss(pred, target, mask)
        m      = evaluate_batch(pred, target)

        for k, v in losses.items():
            self.log(f"val/{k}", v, prog_bar=(k == "total"))
        self.log("val/psnr", m["psnr"], prog_bar=True)
        self.log("val/ssim", m["ssim"], prog_bar=True)

        if batch_idx == 0 and len(self._val_samples) == 0:
            self._val_samples = [
                img[:4].detach(), target[:4].detach(), pred[:4].detach()
            ]
        return losses["total"]

    def on_validation_epoch_end(self):
        if self._val_samples and self.current_epoch % self._log_img_every == 0:
            try:
                import wandb
                imgs, targets, preds = self._val_samples
                panels = []
                for i in range(min(4, imgs.shape[0])):
                    panels += [
                        wandb.Image(self._to_pil(imgs[i]),    caption=f"Damaged #{i}"),
                        wandb.Image(self._to_pil(targets[i]), caption=f"Target #{i}"),
                        wandb.Image(self._to_pil(preds[i]),   caption=f"Restored #{i}"),
                    ]
                self.logger.experiment.log(
                    {"val/restoration_samples": panels}, step=self.global_step
                )
            except Exception:
                pass
        self._val_samples = []

    @staticmethod
    def _to_pil(t: torch.Tensor):
        from PIL import Image as PILImage
        t = (t.clamp(-1, 1) + 1) / 2
        arr = (t * 255).byte().permute(1, 2, 0).cpu().numpy()
        return PILImage.fromarray(arr)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        tcfg = self.config["training"]
        opt  = torch.optim.AdamW(
            self.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
        )
        warmup = tcfg.get("warmup_epochs", 5)
        total  = tcfg["epochs"]

        def lr_lambda(epoch):
            if epoch < warmup:
                return epoch / max(1, warmup)
            frac = (epoch - warmup) / max(1, total - warmup)
            import math
            return 0.5 * (1 + math.cos(math.pi * frac))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

    # ------------------------------------------------------------------
    def export_weights(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), out / "encoder.pth")
        torch.save(self.decoder.state_dict(), out / "decoder.pth")
        print(f"Exported encoder + decoder weights → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dcfg  = config["data"]
    tcfg  = config["training"]
    ocfg  = config["output"]
    lcfg  = config["logging"]

    # Determine target_dir (optional paired clean images)
    target_dir = dcfg.get("target_dir", None)

    full_ds = MuralRestorationDataset(
        image_dir=dcfg["root"],
        mask_dir=dcfg["mask_dir"],
        target_dir=target_dir,
        image_size=dcfg["image_size"],
        augment=True,
    )
    n_train = int(len(full_ds) * dcfg["train_split"])
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"],
        shuffle=True,  num_workers=dcfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=tcfg["batch_size"],
        shuffle=False, num_workers=dcfg["num_workers"], pin_memory=True,
    )

    model = DAFRNetModule(config)

    callbacks = [
        ModelCheckpoint(
            dirpath=ocfg["checkpoint_dir"],
            filename="dafrnet-{epoch:03d}-psnr{val/psnr:.2f}",
            monitor="val/psnr", mode="max",
            save_top_k=lcfg["save_top_k"],
            save_last=True,
        ),
        EarlyStopping(monitor="val/psnr", patience=20, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = WandbLogger(
        project=lcfg["wandb_project"],
        name=lcfg["wandb_run"],
    )

    trainer = pl.Trainer(
        max_epochs=tcfg["epochs"],
        accelerator="auto",
        devices=2,
        strategy="ddp_notebook",
        precision=tcfg.get("precision", "16-mixed"),
        gradient_clip_val=tcfg.get("gradient_clip", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=lcfg.get("log_every_n_steps", 10),
    )

    trainer.fit(model, train_loader, val_loader)

    best = callbacks[0].best_model_path
    print(f"Best checkpoint: {best}")

    # Export weights for API serving
    export_dir = ocfg.get("export_dir", "models/exports")
    ckpt_model = DAFRNetModule.load_from_checkpoint(best, config=config)
    ckpt_model.export_weights(export_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DAFR-Net dual-branch encoder + inpainting head"
    )
    parser.add_argument("--config", default="configs/encoder.yaml")
    args = parser.parse_args()
    main(args.config)
