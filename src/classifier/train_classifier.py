"""
src/classifier/train_classifier.py
Training entry point for the Damage Type Classifier.

Usage:
    python src/classifier/train_classifier.py --config configs/classifier.yaml
"""

import argparse
import shutil
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from model import DamageClassifier


class MuralDamageDataset(Dataset):
    def __init__(self, label_csv: str, image_size: int = 256, augment: bool = True):
        self.df = pd.read_csv(label_csv)

        # Labels were generated on Windows (backslash paths under data\processed\...).
        # Normalize to the actual image directory next to this CSV (e.g. Kaggle's
        # /kaggle/input/.../processed) regardless of the machine that wrote the CSV.
        root = Path(label_csv).parent / "processed"
        self.df["filename"] = self.df["filename"].apply(
            lambda p: str(root / Path(p.replace("\\", "/")).name)
        )

        aug_list = [T.Resize((image_size, image_size))]
        if augment:
            aug_list += [
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
            ]
        aug_list += [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = T.Compose(aug_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filename"]).convert("RGB")
        x = self.transform(img)
        y = int(row["label_id"])
        return x, y


def main(config_path: str, resume: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Dataset
    full_ds = MuralDamageDataset(
        config["data"]["label_csv"],
        image_size=config["data"]["image_size"],
        augment=True,
    )
    n_train = int(len(full_ds) * config["data"]["train_split"])
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True,  num_workers=config["data"]["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=config["training"]["batch_size"],
                              shuffle=False, num_workers=config["data"]["num_workers"])

    # Model
    model = DamageClassifier(config)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=config["output"]["checkpoint_dir"],
        filename="classifier-{epoch:02d}-{val/acc:.3f}",
        monitor="val/acc", mode="max",
        save_top_k=config["logging"]["save_top_k"],
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val/acc", patience=10, mode="max")

    # Logger
    logger = WandbLogger(
        project=config["logging"]["wandb_project"],
        name=config["logging"]["wandb_run"],
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        callbacks=[ckpt_cb, early_stop],
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
    )

    last_ckpt = Path(config["output"]["checkpoint_dir"]) / "last.ckpt"
    ckpt_path = str(last_ckpt) if resume and last_ckpt.exists() else None
    if resume and ckpt_path is None:
        print(f"--resume passed but no checkpoint found at {last_ckpt}; starting fresh.")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    best = ckpt_cb.best_model_path
    print(f"Best model: {best}")

    # Export a Lightning checkpoint under a fixed name for API serving —
    # api/main.py loads DamageClassifier.load_from_checkpoint("models/exports/classifier.ckpt", ...)
    export_dir = Path(config["output"].get("export_dir", "models/exports"))
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, export_dir / "classifier.ckpt")
    print(f"Exported classifier checkpoint → {export_dir / 'classifier.ckpt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from checkpoint_dir/last.ckpt if it exists")
    args = parser.parse_args()
    main(args.config, resume=args.resume)
