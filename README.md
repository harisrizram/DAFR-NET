# DAFR-Net: Damage-Aware Frequency-Guided Restoration Network for Ancient Murals

**Author:** K Anirudh | S Harish Siddharth | M.Tech AI & DS | SASTRA University
**Project type:** SASTRA Summer Research Project 2024–25
**Status:** Phase 1 — core pipeline implemented, dataset ingested, training-ready

> **For anyone continuing or editing this project:**
> Read [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) before making changes.
> Update it after every significant edit — architecture changes, new files, dataset updates, config changes, or known bugs discovered.

---

## Title

**DAFR-Net: A Damage-Aware Frequency-Guided Dual-Branch Diffusion Network for Ancient Mural Restoration**

---

## One-line description

A deep learning pipeline that classifies mural damage type (crack, fade, missing region), routes it through a dual-branch encoder combining structural (Swin Transformer) and frequency-domain (FFT) features, then restores it via a UNet inpainting decoder — with a ControlNet diffusion decoder and super-resolution head planned for Phase 2.

---

## Research novelties

| #   | Novelty                                                  | Gap it fills                                                          |
| --- | -------------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | Damage-type-aware routing                                | All prior work treats every damage type identically                   |
| 2   | Dual-branch encoder: structure + frequency jointly fused | Freq-guided and structure-guided models exist separately, never fused |
| 3   | Joint inpainting + SR from one decoder (Phase 2)         | Every prior model produces one output type, not both                  |

---

## Current implementation status

### Phase 1 — Core pipeline

| Component                                          | Status          | File                                              |
| -------------------------------------------------- | --------------- | ------------------------------------------------- |
| MuralDH dataset                                    | **Downloaded**  | `MuralDH/` (961 annotated + 5096 unannotated)     |
| Data ingestion pipeline                            | **Done**        | `src/utils/prepare_mural_dh.py`                   |
| Damage classifier                                  | **Done**        | `src/classifier/model.py` + `train_classifier.py` |
| Dual-branch encoder (Swin + FFT + cross-attention) | **Done**        | `src/encoder/dual_branch.py`                      |
| Inpainting decoder (UNet)                          | **Done**        | `src/decoder/inpainting_head.py`                  |
| Perceptual loss (VGG16)                            | **Done**        | `src/encoder/train_encoder.py`                    |
| Full Lightning training module                     | **Done**        | `src/encoder/train_encoder.py`                    |
| Heuristic mask generation                          | **Done**        | `src/utils/label_damage.py`                       |
| Synthetic data generator (fallback)                | **Done**        | `src/utils/generate_synthetic_data.py`            |
| Unit tests (5 files)                               | **Done**        | `tests/`                                          |
| FastAPI demo server                                | **Done**        | `api/main.py`                                     |
| Classifier training                                | **Not started** | Awaiting dataset ingestion                        |
| Encoder + decoder training                         | **Not started** | Awaiting dataset ingestion                        |
| Baseline comparisons                               | **Not started** | —                                                 |

### Phase 2 — Thesis completion

| Component                                          | Status      |
| -------------------------------------------------- | ----------- |
| ControlNet diffusion decoder                       | Not started |
| Super-resolution head (`Mural_SR` pairs available) | Not started |
| Frequency-guided loss                              | Not started |
| Ablation study setup                               | Not started |
| Paper draft                                        | Not started |

---

## Dataset: MuralDH

**Source:** [tearsheaven/MuralDH](https://github.com/tearsheaven/MuralDH) | DOI: 10.5061/dryad.bnzs7h4jd
**Location:** `MuralDH/` (project root)

| Folder                          | Contents                                    | Count        |
| ------------------------------- | ------------------------------------------- | ------------ |
| `Mural_seg/train/`              | 512×512 images + binary pixel masks (0/255) | 760 pairs    |
| `Mural_seg/test/`               | Held-out annotated test set                 | 201 pairs    |
| `Mural512/`                     | High-res crops, unannotated                 | 5 096 images |
| `Mural_SR/Mural_DataSet/`       | HR mural images for SR training             | 512 images   |
| `Mural_SR/Mural_DataSet_LR/X2/` | ×2 downscaled LR counterparts               | 512 images   |

Masks are **single-channel PNG, uint8, 0 = clean pixel, 255 = damaged pixel**.

---

## Architecture

```
Input (damaged mural image)
        │
        ▼
┌───────────────────┐
│  DamageClassifier │  ResNet-50 or ViT — outputs: crack | fade | missing
└───────────────────┘
        │ damage type label
        ▼
┌─────────────────────────────────────────────────┐
│              DualBranchEncoder                  │
│  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  StructureBranch │  │   FrequencyBranch    │ │
│  │  Swin-B (timm)   │  │  FFT → real+imag     │ │
│  │  32× downsample  │  │  → 4 Conv layers     │ │
│  │  → (B,256,8,8)   │  │  → (B,256,H,W)       │ │
│  └──────────────────┘  └──────────────────────┘ │
│           │                      │               │
│           └────── CrossAttention ┘               │
│                  (512-dim, 8 heads)              │
│                        │                         │
│                  (B, 512, 8, 8)                  │
└─────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│  InpaintingHead   │  4 × UpBlock (ConvTranspose2d 2×)
│  UNet decoder     │  → (B, 3, 128, 128) → bilinear → (B, 3, 256, 256)
└───────────────────┘
        │
        ▼
   Restored image

Loss = L1 + SSIM + VGG16 perceptual (relu2_2, relu3_3)
```

**Key note:** Swin downsamples 32× (patch=4 × 3 PatchMerge stages), giving `(B, 512, 8, 8)` for 256 input. Four 2× UpBlocks reach `(B, 3, 128, 128)` — a bilinear upsample in `DAFRNetModule.forward()` brings it back to 256.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare MuralDH data

```bash
# Resize all images to 256×256, generate masks for unannotated images,
# write data/processed/, data/masks/, data/damage_labels.csv
python src/utils/prepare_mural_dh.py --dataset_dir MuralDH --out_dir data
```

For annotated images only (fast, ~1 min):

```bash
python src/utils/prepare_mural_dh.py --dataset_dir MuralDH --out_dir data --skip_mural512
```

### 3. Run unit tests

```bash
pytest tests/ -v
```

Uses `swin_tiny` + `pretrained=False` — passes without GPU or dataset.

### 4. Train damage classifier

```bash
python src/classifier/train_classifier.py --config configs/classifier.yaml
```

### 5. Train encoder + inpainting head

```bash
python src/encoder/train_encoder.py --config configs/encoder.yaml
```

### 6. Run API demo

```bash
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/docs
```

---

## Colab / Kaggle setup (T4 GPU)

```python
# 1. Mount Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

# 3. Patch config for T4 memory limits
import yaml

with open("configs/encoder.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["batch_size"] = 4               # reduce from 8
cfg["data"]["num_workers"]    = 2               # reduce from 4
cfg["model"]["structure_branch"]["backbone"] = "swin_tiny_patch4_window7_224"  # ~80 MB vs 350 MB

with open("configs/encoder.yaml", "w") as f:
    yaml.dump(cfg, f)

# 4. Disable perceptual loss if still OOM
# In configs/encoder.yaml: loss.perceptual_weight: 0.0
```

---

## Evaluation targets

| Metric    | Baseline (MuralDH paper) | DAFR-Net Phase 1 target |
| --------- | ------------------------ | ----------------------- |
| PSNR (dB) | > 28                     | > 30                    |
| SSIM      | > 0.85                   | > 0.87                  |
| LPIPS     | —                        | < 0.15                  |

---

## Folder structure

```
DAFR-Net/
├── MuralDH/                  # Dataset (downloaded)
│   ├── Mural_seg/            #   Annotated images + binary masks
│   ├── Mural512/             #   5096 unannotated crops
│   └── Mural_SR/             #   HR/LR pairs for super-resolution
├── data/
│   ├── processed/            # 256×256 resized images (generated)
│   ├── masks/                # Binary damage masks (generated)
│   └── damage_labels.csv     # damage_type + label_id per image (generated)
├── src/
│   ├── classifier/
│   │   ├── model.py          # DamageClassifier (ResNet-50 / ViT)
│   │   └── train_classifier.py
│   ├── encoder/
│   │   ├── dual_branch.py    # StructureBranch + FrequencyBranch + CrossAttentionFusion
│   │   └── train_encoder.py  # DAFRNetModule (Lightning) + PerceptualLoss
│   ├── decoder/
│   │   └── inpainting_head.py  # InpaintingHead (UNet) + InpaintingLoss
│   └── utils/
│       ├── prepare_mural_dh.py   # MuralDH ingestion (resize + masks + CSV)
│       ├── prepare_data.py       # Generic fallback ingestion
│       ├── label_damage.py       # Heuristic mask generation (Canny/HSV)
│       ├── generate_synthetic_data.py  # Synthetic training data generator
│       └── metrics.py            # PSNR / SSIM
├── models/
│   ├── checkpoints/          # Lightning checkpoints during training
│   └── exports/              # encoder.pth, decoder.pth (for API)
├── api/
│   └── main.py               # FastAPI: /classify /restore /health
├── configs/
│   ├── classifier.yaml
│   └── encoder.yaml
├── tests/
│   ├── test_classifier.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_metrics.py
│   └── test_pipeline.py      # End-to-end encoder → decoder → loss → backward
├── docs/
│   └── SESSION_HANDOFF.md    # Living doc — read before editing, update after
├── notebooks/
├── results/
└── logs/
```

---

## Tech stack

| Tool                    | Version | Role                              |
| ----------------------- | ------- | --------------------------------- |
| Python                  | 3.10+   | Primary language                  |
| PyTorch                 | 2.2+    | All model training                |
| torchvision             | 0.17+   | Transforms, VGG16 perceptual loss |
| timm                    | 0.9+    | Swin Transformer, ViT             |
| PyTorch Lightning       | 2.2+    | Training loops, checkpointing     |
| diffusers (HuggingFace) | 0.27+   | Phase 2 ControlNet decoder        |
| einops                  | 0.7+    | Tensor reshaping for attention    |
| opencv-python           | 4.9+    | Mask generation, image ops        |
| wandb                   | 0.16+   | Experiment tracking               |
| FastAPI + uvicorn       | —       | REST API demo                     |

---

## Supervisor talking points

1. Three clearly separated novelties, each ablatable independently
2. Built directly on MuralDH benchmark — results are directly comparable to published numbers
3. Phase 1 deliverable (working inpainting + API demo) is training-ready now that dataset is ingested
4. Phase 2 (ControlNet diffusion + SR + paper) is thesis completion; SR data (`Mural_SR/`) is already available
5. Stack matches existing PyTorch/FastAPI experience

---

## Development notes

- **Before changing any model, config, or file:** read [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md)
- **After any significant change:** update `SESSION_HANDOFF.md` — architecture decisions, new files, bugs found, config changes
- **Windows DataLoader:** set `data.num_workers: 0` in both YAMLs if training hangs at epoch start
- **Offline training:** set `WANDB_MODE=offline` or replace `WandbLogger` with `CSVLogger` in both train scripts
