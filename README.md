# DAFR-Net: Damage-Aware Frequency-Guided Restoration Network for Ancient Murals

**Author:** K Anirudh | S Harish Siddharth | M.Tech AI & DS | SASTRA University
**Project type:** SASTRA Summer Research Project 2024–25
**Status:** Phase 1 — **COMPLETE**. Both models trained end-to-end on MuralDH and served live via a FastAPI demo (classifier val/acc = 74.0%, restoration val/PSNR = 24.36 dB). Phase 2 (ControlNet diffusion decoder + SR head) not started.

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
| FastAPI demo server                                | **Done — smoke tested** | `api/main.py`                             |
| Classifier training                                | **Done** — val/acc 0.740 @ epoch 10/50 (early-stopped) | `models/exports/classifier.ckpt` |
| Encoder + decoder training                         | **Done** — val/PSNR 24.36 dB @ epoch 149/150 (full run) | `models/exports/encoder.pth`, `decoder.pth` |
| Baseline comparisons                               | **Partial** — PSNR compared to MuralDH paper baseline (>28 dB); SSIM/LPIPS not yet computed | —          |

Both models were trained on Kaggle's free 2×T4 GPU tier. Memory constraints forced two fallbacks from the original design: the structure branch uses **Swin-Tiny** instead of Swin-Base, and the **VGG16 perceptual loss is disabled** (`perceptual_weight: 0`). This is why validation PSNR (24.36 dB) sits below the MuralDH baseline (>28 dB) — see [`docs/SUMMER_PROJECT_REPORT.md`](docs/SUMMER_PROJECT_REPORT.md) for the full writeup and [`docs/PPT_CONTENT.md`](docs/PPT_CONTENT.md) / [`docs/DAFR-Net_Summer_Project_PPT.pptx`](docs/DAFR-Net_Summer_Project_PPT.pptx) for presentation material.

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

**Already done** — best checkpoint (val/acc 0.740, epoch 10/50) is exported to `models/exports/classifier.ckpt`. Re-run only to retrain.

### 5. Train encoder + inpainting head

```bash
python src/encoder/train_encoder.py --config configs/encoder.yaml
```

**Already done** — full 150-epoch run (val/PSNR 24.36 dB, epoch 149) is exported to `models/exports/encoder.pth` and `models/exports/decoder.pth`. Re-run only to retrain.

> `configs/classifier.yaml` and `configs/encoder.yaml` still have their `data`/`output` paths set to Kaggle's `/kaggle/input/...` and `/kaggle/working/...` (from the training run) — this does **not** block the API demo below, since `api/main.py` only reads the `model:` section. Only revert these paths if you intend to resume/retrain locally.

### 6. Run the API demo

```bash
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/docs
```

See **[Live demo guide for the panel](#live-demo-guide-for-the-panel)** below for the exact walkthrough.

---

## Colab / Kaggle setup (T4 GPU)

> This is the exact setup already used to complete Phase 1 training (see [`docs/SUMMER_PROJECT_REPORT.md`](docs/SUMMER_PROJECT_REPORT.md) for the full run history). Kept here for reference if you resume training (e.g. a Swin-Base follow-up run) — not required for the API demo.

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

| Metric        | Baseline (MuralDH paper) | DAFR-Net Phase 1 target | **Achieved (2026-07-14)** |
| ------------- | ------------------------ | ------------------------ | -------------------------- |
| Classifier acc | —                        | —                         | **74.0%** (val, epoch 10)  |
| PSNR (dB)     | > 28                     | > 30                      | **24.36** (val, epoch 149) |
| SSIM          | > 0.85                   | > 0.87                    | Not yet tracked             |
| LPIPS         | —                        | < 0.15                    | Not yet tracked             |

The PSNR gap to baseline is attributed to the Swin-Tiny/no-perceptual-loss fallback required by Kaggle's free-tier GPU memory budget — see [`docs/SUMMER_PROJECT_REPORT.md`](docs/SUMMER_PROJECT_REPORT.md) §7 for details and the planned Swin-Base follow-up run.

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

## Live demo guide for the panel

Everything needed is already trained and sitting in `models/exports/` — no training or internet access required at demo time.

### A. One-time pre-demo check (do this before you walk into the room)

```bash
# 1. Make sure the exported weights exist (should already be true)
ls models/exports/
#   -> classifier.ckpt  encoder.pth  decoder.pth

# 2. Install deps if this is a fresh machine
pip install -r requirements.txt

# 3. Start the server and confirm all three models load
uvicorn api.main:app --port 8000
# Watch the console for: [DAFR-Net] Models loaded on cuda   (or "cpu" if no GPU)
```

If it prints `Checkpoints not found — run training first.`, you're either in the wrong directory or `models/exports/` is missing — check paths before the demo, not during it.

### B. During the demo — three ways to show it live

**1. Swagger UI (easiest, no terminal typing in front of the panel)**

- Open `http://localhost:8000/docs` in a browser.
- Expand `POST /classify` → *Try it out* → upload a damaged mural image (anything from `MuralDH/Mural_seg/test/` works well) → *Execute*.
  → Returns `{"damage_type": "crack" | "fade" | "missing", ...}` — good for explaining Stage 1.
- Expand `POST /restore` → *Try it out* → upload the same image → *Execute*.
  → Returns the restored PNG directly in the browser, plus an `X-Damage-Type` response header showing what the classifier detected before restoration ran.

**2. curl (if the panel wants to see the raw API contract)**

```bash
# Health check — confirms all 3 models loaded
curl http://localhost:8000/health

# Classify damage type
curl -X POST http://localhost:8000/classify \
  -F "file=@MuralDH/Mural_seg/test/<some_image>.jpg"

# Restore an image, save the PNG output
curl -X POST http://localhost:8000/restore \
  -F "file=@MuralDH/Mural_seg/test/<some_image>.jpg" \
  -o restored_output.png
```

**3. Before/after slide (prepared, no live risk)**

If you don't want to risk a live network/GPU hiccup in front of the panel, pre-generate 2–3 before/after pairs beforehand using the curl command above, and drop them into `results/` — Slide 9 of `docs/DAFR-Net_Summer_Project_PPT.pptx` already has a placeholder image area sized for this.

### C. Talking points while demoing

- Point out the `X-Damage-Type` header on `/restore` — it proves the classifier's output is actually routed into the restoration stage, not just decorative (Novelty #1).
- Mention the whole thing runs on a laptop CPU/single GPU at inference time even though training used 2×T4 GPUs — the trained artifacts are only ~530 MB total (`encoder.pth` 124 MB + `decoder.pth` 9 MB + `classifier.ckpt` 283 MB).
- If asked about accuracy: be upfront — 74.0% classifier val accuracy and 24.36 dB PSNR, below the >28 dB MuralDH baseline, and explain why (Swin-Tiny + disabled perceptual loss, due to free-tier GPU memory limits) — this is covered in detail in `docs/SUMMER_PROJECT_REPORT.md` §7 (Merits and Demerits).

### D. If something breaks right before the demo

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Checkpoints not found` on startup | Wrong working directory, or `models/exports/` missing/moved | `cd` to repo root; verify the 3 files listed in step A exist |
| `ModuleNotFoundError` (timm, einops, torchmetrics, pytorch-lightning) | Fresh machine, deps not installed | `pip install -r requirements.txt` |
| `/restore` 500 error | Corrupt/non-image upload, or CUDA OOM on a small GPU | Retry with a clean JPEG/PNG; add `DEVICE = "cpu"` override in `api/main.py` if the demo machine's GPU is too small |
| Server slow to start | Loading `classifier.ckpt` (283 MB Lightning checkpoint) on CPU | Normal — takes a few seconds; start the server 1–2 minutes before the panel arrives |

---

## Supervisor talking points

1. Three clearly separated novelties, each ablatable independently
2. Built directly on MuralDH benchmark — results are directly comparable to published numbers
3. Phase 1 deliverable is **done**: both models trained end-to-end (classifier val/acc 0.740, restoration val/PSNR 24.36 dB) and served live through a working FastAPI demo
4. Current PSNR trails the MuralDH baseline (>28 dB) because of free-tier GPU constraints (Swin-Tiny fallback, perceptual loss disabled) — a full Swin-Base run is the natural next step to close that gap
5. Phase 2 (ControlNet diffusion + SR + paper) is thesis completion; SR data (`Mural_SR/`) is already available
6. Stack matches existing PyTorch/FastAPI experience

---

## Development notes

- **Before changing any model, config, or file:** read [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md)
- **After any significant change:** update `SESSION_HANDOFF.md` — architecture decisions, new files, bugs found, config changes
- **Windows DataLoader:** set `data.num_workers: 0` in both YAMLs if training hangs at epoch start
- **Offline training:** set `WANDB_MODE=offline` or replace `WandbLogger` with `CSVLogger` in both train scripts
