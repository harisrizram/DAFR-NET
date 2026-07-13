"""
src/classifier/model.py
Damage Type Classifier — ResNet-50 / ViT backbone
Classes: 0=crack, 1=fade, 2=missing
"""

import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix


DAMAGE_CLASSES = ["crack", "fade", "missing"]


class DamageClassifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = config["model"]["num_classes"]

        # Load pretrained backbone from timm
        self.backbone = timm.create_model(
            config["model"]["backbone"],
            pretrained=config["model"]["pretrained"],
            num_classes=self.num_classes,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_cm = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self._freeze_epochs = config["model"].get("freeze_backbone_epochs", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_cm(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        cm = self.val_cm.compute()
        self.logger.experiment.log({"val/confusion_matrix": cm.cpu().numpy()})
        self.val_cm.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["training"]["epochs"],
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        # Freeze backbone for first N epochs, then unfreeze.
        # Use timm's get_classifier() rather than name-matching "head"/"classifier" —
        # e.g. ResNet's final layer is named "fc", which those substrings miss,
        # silently freezing the whole model (no params left with requires_grad=True).
        if self.current_epoch == 0 and self._freeze_epochs > 0:
            classifier_param_ids = {id(p) for p in self.backbone.get_classifier().parameters()}
            for param in self.backbone.parameters():
                if id(param) not in classifier_param_ids:
                    param.requires_grad = False
        elif self.current_epoch == self._freeze_epochs:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"[Epoch {self.current_epoch}] Backbone unfrozen — full fine-tuning.")

    def predict_damage_type(self, image_tensor: torch.Tensor) -> str:
        """Inference helper — returns human-readable damage class."""
        self.eval()
        with torch.no_grad():
            logits = self(image_tensor.unsqueeze(0))
            idx = logits.argmax(dim=1).item()
        return DAMAGE_CLASSES[idx]
