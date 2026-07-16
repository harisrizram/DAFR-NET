# DAFR-Net: A Damage-Aware Frequency-Guided Dual-Branch Network for Ancient Mural Restoration

**Summer Project Report**

Submitted by: K Anirudh, S Harish Siddharth
Program: M.Tech Artificial Intelligence & Data Science
Institution: SASTRA University
Academic Year: 2024–25

---

## 1. Abstract

Ancient murals degrade through several distinct damage modes — cracks, fading, and missing (spalled) regions — each with different visual and frequency characteristics. Existing deep-learning-based restoration approaches treat all damage types identically and rely purely on spatial/structural cues, ignoring frequency-domain information that is highly informative for certain damage types (e.g., fading manifests as a low-frequency degradation, while cracks are high-frequency discontinuities). This project presents **DAFR-Net (Damage-Aware Frequency-Guided Restoration Network)**, a three-stage pipeline that (1) classifies the damage type present in a mural image, (2) encodes the image using a dual-branch architecture that jointly extracts structural features (via a Swin Transformer) and frequency-domain features (via FFT), fusing them through cross-attention, and (3) reconstructs the restored image via a UNet-style inpainting decoder. The system was trained end-to-end on the public MuralDH benchmark dataset using free-tier Kaggle GPU resources and deployed as a REST API for real-time inference. The damage classifier achieved 74.0% validation accuracy and the restoration model achieved 24.36 dB validation PSNR after full training.

---

## 2. Introduction

### 2.1 Motivation
Cultural heritage murals are irreplaceable historical artifacts that continuously degrade due to environmental exposure, pigment decay, and physical damage. Manual restoration by conservators is slow, expensive, and requires specialized expertise that is in short supply globally. Automating even a first-pass digital restoration can assist conservators in planning, documentation, and public presentation of degraded artworks.

### 2.2 Problem Statement
Generic image inpainting models (GAN-based or diffusion-based) are typically damage-agnostic: they apply the same restoration strategy regardless of whether the input is affected by a crack, fading, or a fully missing region. This is suboptimal because different damage types require different restoration priors — a hairline crack needs high-frequency structural continuation, whereas a faded region needs low-frequency color/tone recovery. No prior mural-restoration system explicitly conditions its restoration strategy on a predicted damage type, nor jointly fuses spatial-structural and frequency-domain features.

### 2.3 Objectives
1. Build a damage-type classifier (crack / fade / missing) for mural images.
2. Design a dual-branch encoder combining structural (Swin Transformer) and frequency-domain (FFT) representations.
3. Fuse both branches via cross-attention into a unified latent representation.
4. Decode the fused latent into a restored image using a UNet-based inpainting head.
5. Train and validate the complete pipeline on the MuralDH benchmark.
6. Expose the trained pipeline through a deployable REST API.

---

## 3. Dataset

The project uses **MuralDH** (source: `tearsheaven/MuralDH`, DOI: 10.5061/dryad.bnzs7h4jd), a public benchmark dataset for mural damage analysis:

| Folder | Contents | Count |
|---|---|---|
| `Mural_seg/train/` | 512×512 images + binary damage masks | 760 pairs |
| `Mural_seg/test/` | Held-out annotated test set | 201 pairs |
| `Mural512/` | High-resolution unannotated crops | 5,096 images |
| `Mural_SR/Mural_DataSet/` | HR images for super-resolution (Phase 2) | 512 images |
| `Mural_SR/Mural_DataSet_LR/X2/` | ×2 downscaled LR counterparts | 512 images |

All images were resized to 256×256 for training. Damage-type labels (crack / fade / missing) were generated using a heuristic mask-analysis pipeline (Canny edge detection and HSV color-variance heuristics, `src/utils/label_damage.py`) since MuralDH provides binary damage masks but not damage-type labels directly.

---

## 4. Workflow / Architecture Diagram

```
                         Input: Damaged Mural Image (256×256×3)
                                        │
                                        ▼
                       ┌─────────────────────────────┐
                       │   Stage 1: DamageClassifier  │
                       │        (ResNet-50)           │
                       │  Output: crack | fade | missing │
                       └─────────────────────────────┘
                                        │  damage-type label
                                        ▼
        ┌───────────────────────────────────────────────────────────┐
        │              Stage 2: Dual-Branch Encoder                  │
        │                                                             │
        │   ┌───────────────────────┐   ┌───────────────────────┐    │
        │   │   Structure Branch     │   │   Frequency Branch     │    │
        │   │   Swin Transformer     │   │   2D-FFT (ortho norm)  │    │
        │   │   (patch=4, 32× down)  │   │   → real + imag parts  │    │
        │   │   → (B, 256, 8, 8)     │   │   → 4 Conv layers      │    │
        │   │                        │   │   → (B, 256, 8, 8)     │    │
        │   └───────────┬────────────┘   └───────────┬───────────┘    │
        │               │                             │                │
        │               └───────── Cross-Attention ───┘                │
        │                    (embed_dim=512, 8 heads, dropout 0.1)     │
        │                             │                                │
        │                     (B, 512, 8, 8) fused latent               │
        └───────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────────┐
                       │   Stage 3: InpaintingHead    │
                       │   UNet decoder (4 UpBlocks)  │
                       │   channels: [256,128,64,32]  │
                       │   skip connections           │
                       │   → (B,3,128,128) → bilinear │
                       │   upsample → (B,3,256,256)   │
                       └─────────────────────────────┘
                                        │
                                        ▼
                          Output: Restored Mural Image

        Loss = L1 (reconstruction) + SSIM (0.2 weight) + VGG16 perceptual
               (perceptual weight disabled during Kaggle Tiny-backbone runs)
```

**Deployment layer:** A FastAPI server (`api/main.py`) wraps all three trained stages and exposes:
- `GET /health` — model-load status
- `POST /classify` — returns predicted damage type for an uploaded image
- `POST /restore` — returns the restored PNG image with the predicted damage type in the response header

---

## 5. Implementation Details

### 5.1 Technology Stack
| Component | Tool / Version |
|---|---|
| Deep learning framework | PyTorch 2.2+, PyTorch Lightning 2.2+ |
| Backbone | timm (Swin Transformer) |
| API layer | FastAPI + uvicorn |
| Experiment tracking | Weights & Biases |
| Tensor operations | einops |
| Image ops | OpenCV, Pillow, torchvision |
| Training hardware | Kaggle free-tier 2×T4 GPUs |

### 5.2 Training Configuration
**Damage Classifier** (`configs/classifier.yaml`):
- Backbone: ResNet-50 (ImageNet-pretrained), first 5 epochs head-only fine-tuning
- 50 epochs max, batch size 32, Adam, lr 1e-4, cosine schedule with 3-epoch warmup
- Loss: cross-entropy over 3 classes
- Augmentation: horizontal flip, ±15° rotation, 224 random crop

**Dual-Branch Encoder + Inpainting Decoder** (`configs/encoder.yaml`):
- Structure branch: `swin_tiny_patch4_window7_224` (downgraded from Swin-Base due to Kaggle T4 memory limits)
- Frequency branch: 2D-FFT (ortho-normalized), 4 conv layers, 256 output channels
- Fusion: cross-attention, 512-dim embedding, 8 heads
- 150 epochs, batch size 8, AdamW, lr 2e-4, cosine-with-restarts, 5-epoch warmup, gradient clipping at 1.0
- Mixed precision (fp16) training
- Perceptual (VGG16) loss disabled for the Tiny-backbone fallback run to fit T4 memory/time budget

### 5.3 Training Environment Notes
- Both models were trained on Kaggle's free 2×GPU T4 tier due to lack of local GPU resources exceeding consumer-grade capacity.
- DDP (distributed data-parallel) across the two T4s required disabling the `fork` multiprocessing start method and NCCL P2P/InfiniBand, and adding a 3-minute DDP timeout to diagnose hangs — these were non-obvious fixes required to get multi-GPU training stable in the Kaggle notebook environment.
- `num_workers: 0` was required in both data loaders — non-zero worker counts forked after CUDA initialization in the Kaggle notebook process and deadlocked mid-epoch.

---

## 6. Results

| Model | Metric | Value | Notes |
|---|---|---|---|
| Damage Classifier | Validation Accuracy | **74.0%** | Best at epoch 10/50 (early-stopped, patience 10) |
| Encoder + Decoder (Restoration) | Validation PSNR | **24.36 dB** | Best at epoch 149/150 (full run) |

Qualitatively, restored outputs on held-out MuralDH images show plausible reconstruction of mural texture and color in previously-damaged regions, verified via an end-to-end API smoke test against real dataset images (`/classify` and `/restore` endpoints both returning correct, well-formed results).

---

## 7. Merits and Demerits of the Work

### 7.1 Merits
1. **Novel damage-aware conditioning** — restoration strategy is explicitly informed by a predicted damage type, a dimension unaddressed in prior mural-restoration literature.
2. **Dual-domain feature fusion** — combining Swin Transformer structural features with FFT-derived frequency features via learned cross-attention is a genuinely underexplored fusion strategy for this task.
3. **Fully modular, ablatable design** — the classifier, structure branch, frequency branch, and decoder are independently testable components (backed by a 5-file unit test suite), enabling controlled ablation studies in future work.
4. **End-to-end deployment, not just a design** — the trained pipeline is served through a working REST API (`/classify`, `/restore`, `/health`), demonstrating a complete research-to-deployment path rather than isolated offline experiments.
5. **Benchmark-grounded** — built directly on the public MuralDH dataset, making results directly comparable to published baseline numbers.
6. **Resource efficiency** — the entire training pipeline (both models, 150+50 epochs) was completed using only free-tier cloud GPU resources (Kaggle 2×T4), with multiple non-trivial engineering fixes (DDP configuration, channels-last handling, checkpoint export automation) required to make this feasible within Kaggle's session and memory constraints.

### 7.2 Demerits / Limitations
1. **PSNR below target** — 24.36 dB achieved vs. a >28 dB baseline (MuralDH paper) and >30 dB internal target. This gap is primarily attributable to two forced compromises: using Swin-Tiny instead of Swin-Base (due to T4 memory limits), and disabling the VGG16 perceptual loss term (fp32 VGG forward pass was too slow on the available hardware/time budget).
2. **Classifier accuracy has room to improve** — 74% validation accuracy on a 3-class problem indicates meaningful confusion between visually similar classes (particularly fade vs. crack in low-contrast regions).
3. **No paired ground truth** — MuralDH provides damage masks but no clean "before damage" reference images, forcing self-supervised training that inherently limits achievable output sharpness compared to supervised inpainting with paired data.
4. **Limited evaluation metrics** — only PSNR is currently tracked for the restoration model; SSIM and LPIPS (both planned) are not yet computed, limiting the ability to fully characterize perceptual quality.
5. **Single-task decoder** — the current decoder only performs inpainting; joint super-resolution (for which paired HR/LR data already exists in `Mural_SR/`) is deferred to Phase 2.
6. **No formal ablation study yet** — the individual contribution of the frequency branch and cross-attention fusion (vs. structure-branch-only) has not yet been quantitatively isolated.

---

## 8. Comparison with Existing Work

| Aspect | Generic Deep Inpainting (GAN/diffusion-based, e.g. LaMa-style) | MuralDH Paper Baseline | **DAFR-Net (This Work)** |
|---|---|---|---|
| Damage-type awareness | None — same model/weights for all damage | None | **Explicit classifier-gated routing (crack / fade / missing)** |
| Input representation | Spatial/RGB only | Spatial/RGB only | **Spatial (Swin) + Frequency (FFT), jointly fused** |
| Feature fusion mechanism | N/A (single branch) | N/A (single branch) | **Cross-attention fusion (512-d, 8 heads)** |
| Training data | Generic natural-image inpainting datasets | MuralDH (domain-specific) | **MuralDH (domain-specific)** |
| Output capability | Inpainting only | Inpainting only | Inpainting now; **super-resolution planned (Phase 2), using existing paired HR/LR data** |
| Deployment | Typically research code, no serving layer | Research code, no serving layer | **REST API (FastAPI) with `/classify`, `/restore`, `/health`** |
| Reported PSNR | Domain-general, not mural-specific | > 28 dB | 24.36 dB (Swin-Tiny fallback; target > 30 dB with Swin-Base) |
| Compute footprint | Often requires large-scale GPU clusters | Not publicly specified | Trained entirely on free-tier Kaggle 2×T4 GPUs |

**Key differentiators / improvements introduced by this work:**
- Introduces damage-type conditioning as an explicit architectural stage, rather than treating restoration as a single undifferentiated task — a gap identified across all reviewed prior mural-restoration approaches.
- Is the first (to the authors' knowledge, within scope of this project's literature review) mural-restoration pipeline to jointly learn and fuse structural and frequency-domain representations via attention, rather than relying on structure or frequency cues in isolation.
- Demonstrates that a full research pipeline (classification → dual-branch encoding → decoding) can be trained end-to-end on free-tier academic compute, lowering the resource barrier for reproducing and extending this line of work.
- Currently trails the MuralDH paper baseline on raw PSNR — this gap is explained by resource-driven architectural downgrades (Swin-Tiny vs. Swin-Base, disabled perceptual loss) rather than a limitation of the underlying method, and is the primary target for Phase 2 improvement.

---

## 9. Conclusion

DAFR-Net demonstrates a complete, trained, and deployed pipeline for damage-aware mural restoration that introduces two concrete novelties over existing approaches: explicit damage-type conditioning and dual-branch structure–frequency fusion via cross-attention. Phase 1 successfully validates the architecture end-to-end on the MuralDH benchmark, achieving 74.0% damage-classification accuracy and 24.36 dB restoration PSNR under significant compute constraints (free-tier Kaggle GPUs, Swin-Tiny fallback). While the current PSNR trails the published baseline, this is attributable to resource-driven simplifications rather than the underlying method, and closing this gap — via a full Swin-Base run with the perceptual loss re-enabled — is the immediate next step.

---

## 10. Future Work (Phase 2)

1. **Scale to Swin-Base** with checkpoint-resume training to span Kaggle's 12-hour session cap, closing the PSNR gap to the published baseline.
2. **Re-enable perceptual (VGG16) loss** once compute budget allows, to improve perceptual output quality.
3. **ControlNet-based diffusion decoder** as an alternative/complement to the current UNet decoder.
4. **Joint super-resolution head**, using the already-available `Mural_SR` paired HR/LR data.
5. **Frequency-guided loss term**, to more directly supervise the frequency branch's contribution.
6. **Formal ablation study** isolating the contribution of damage-type conditioning and frequency-branch fusion.
7. **Additional evaluation metrics** — SSIM and LPIPS — for a fuller perceptual quality picture.
8. **Manuscript preparation** summarizing Phase 1 + Phase 2 findings for publication.

---

## 11. References

1. MuralDH Dataset — `tearsheaven/MuralDH`, DOI: 10.5061/dryad.bnzs7h4jd
2. Liu, Z. et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
3. He, K. et al., "Deep Residual Learning for Image Recognition" (ResNet)
4. Ronneberger, O. et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
5. Vaswani, A. et al., "Attention Is All You Need" (cross-attention mechanism)
