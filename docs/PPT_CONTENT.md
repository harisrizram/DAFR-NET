# DAFR-Net — Summer Project PPT Content

**Damage-Aware Frequency-Guided Restoration Network for Ancient Mural Restoration**
K Anirudh | S Harish Siddharth | M.Tech AI & DS, SASTRA University | Summer Project 2024–25

---

## Slide 1 — Title
- **DAFR-Net: A Damage-Aware Frequency-Guided Dual-Branch Network for Ancient Mural Restoration**
- K Anirudh, S Harish Siddharth — M.Tech AI & DS, SASTRA University
- Summer Research Project, Phase 1 (Core pipeline — trained & deployed)

---

## Slide 2 — Problem Statement
- Ancient murals degrade through **cracks, fading, and missing (spalled) regions** — each with different visual statistics.
- Manual restoration is slow, expensive, and requires rare conservator expertise.
- Existing deep-learning restoration models (generic image inpainting GANs/diffusion) treat **all damage types identically** and rely only on spatial/structural cues — they ignore frequency-domain degradation patterns (e.g., fading is a low-frequency phenomenon, cracks are high-frequency).
- **Goal:** build a damage-aware, frequency-guided restoration pipeline that classifies damage first, then restores it using both structural and frequency information.

---

## Slide 3 — Objectives
1. Classify mural damage into crack / fade / missing before restoration.
2. Design a dual-branch encoder that jointly learns structural features (Swin Transformer) and frequency-domain features (FFT).
3. Fuse both branches with cross-attention and decode to a restored image via a UNet inpainting head.
4. Train and validate end-to-end on the MuralDH benchmark dataset.
5. Serve the trained pipeline through a REST API for real-time demonstration.

---

## Slide 4 — Dataset: MuralDH
- Source: MuralDH (tearsheaven/MuralDH), DOI 10.5061/dryad.bnzs7h4jd
- `Mural_seg/` — 760 train + 201 test annotated image–mask pairs (512×512, binary damage masks)
- `Mural512/` — 5,096 unannotated high-resolution crops
- `Mural_SR/` — 512 HR/LR pairs reserved for Phase 2 super-resolution
- All images resized to 256×256; damage labels (`crack`/`fade`/`missing`) generated via heuristic mask analysis (`label_damage.py`)

---

## Slide 5 — System Architecture (Diagram Slide)
*(Use the diagram in the report / render the Mermaid block below directly, or redraw as boxes-and-arrows in PowerPoint)*

```
Damaged Mural Image
        │
        ▼
 [1] DamageClassifier (ResNet-50)
        │  → damage type: crack | fade | missing
        ▼
 [2] Dual-Branch Encoder
     ┌────────────────────┬─────────────────────┐
     │ Structure Branch    │ Frequency Branch     │
     │ Swin Transformer     │ FFT → real + imag   │
     │ (32× downsample)      │ → 4 Conv layers     │
     └────────────────────┴─────────────────────┘
                    │  Cross-Attention Fusion (512-d, 8 heads)
                    ▼
 [3] InpaintingHead (UNet decoder, skip connections)
        │
        ▼
   Restored Mural Image
```
- Three clearly separated, independently-ablatable novelties (see Slide 6).

---

## Slide 6 — Key Novelties
| # | Novelty | Gap Filled |
|---|---------|------------|
| 1 | Damage-type-aware routing | Prior work restores all damage types identically |
| 2 | Dual-branch encoder (structure + frequency, jointly fused) | Structure-guided and frequency-guided approaches exist separately, never fused via cross-attention |
| 3 | Joint inpainting + super-resolution from one decoder (Phase 2) | Prior models produce only one output type |

---

## Slide 7 — Implementation Stack
- **PyTorch 2.2 + PyTorch Lightning** — training loops, checkpointing, mixed precision
- **timm** — Swin Transformer backbone
- **FastAPI + uvicorn** — REST inference server (`/classify`, `/restore`, `/health`)
- **Weights & Biases** — experiment tracking
- Trained on **Kaggle's free 2×T4 GPU tier** (Swin-Tiny fallback for memory constraints)

---

## Slide 8 — Training Setup
- **Classifier:** ResNet-50, 50 epochs (early-stopped), batch size 32, cross-entropy loss
- **Encoder + Decoder:** Swin-Tiny + FFT branch + UNet decoder, 150 epochs, batch size 8, L1 + SSIM loss (perceptual/VGG loss disabled for Kaggle memory budget)
- Both trained end-to-end, self-supervised (no paired clean reference in MuralDH)

---

## Slide 9 — Results
| Model | Metric | Value |
|-------|--------|-------|
| Damage Classifier | Validation Accuracy | **74.0%** (epoch 10/50, early-stopped) |
| Restoration (Encoder+Decoder) | Validation PSNR | **24.36 dB** (epoch 149/150, full run) |

- Restored outputs on held-out MuralDH images are visually coherent with plausible mural content restored in damaged regions.
- API smoke test confirms both models load and serve correctly end-to-end.

*(Insert before/after restoration image pairs here from `results/`)*

---

## Slide 10 — Merits
- Modular 3-stage pipeline — each component independently testable/ablatable
- Damage-type conditioning is a genuinely underexplored idea in mural restoration literature
- Dual-branch fusion captures both spatial structure and frequency-domain degradation cues
- End-to-end trained and deployed — not just a paper design, a working, callable API
- Built directly on a public benchmark (MuralDH) — results are directly comparable to published baselines
- Resource-efficient: trained fully on free-tier Kaggle GPUs (no paid compute)

---

## Slide 11 — Limitations
- PSNR (24.36 dB) is currently below the MuralDH baseline target (>28 dB) — Swin-Tiny fallback and disabled perceptual loss (due to T4 memory limits) cap fidelity
- Classifier accuracy (74%) leaves room for improvement — 3-way damage classification confuses visually similar fade/crack cases
- Self-supervised training (no paired clean ground truth) limits achievable sharpness
- No SSIM/LPIPS numbers yet — only PSNR currently tracked
- Phase 2 components (ControlNet diffusion decoder, super-resolution head) not yet implemented

---

## Slide 12 — Comparison with Existing Work
| Aspect | Generic Inpainting (e.g. LaMa, GAN-based) | MuralDH Baseline | **DAFR-Net (this work)** |
|--------|------|------|------|
| Damage-type awareness | No | No | **Yes — explicit classifier-gated routing** |
| Frequency-domain features | No | No | **Yes — dedicated FFT branch fused via cross-attention** |
| Structure + frequency fusion | N/A | N/A | **Yes — cross-attention (512-d, 8 heads)** |
| Deployment | Research code only | Research code only | **REST API demo (FastAPI)** |
| Output types | Inpainting only | Inpainting only | Inpainting now; **+ SR planned (Phase 2)** |
| PSNR | Varies, domain-general | >28 dB (paper) | 24.36 dB (Swin-Tiny fallback, target >30 dB) |

---

## Slide 13 — Conclusion & Future Work
- Phase 1 delivers a fully working, trained, and API-served damage-aware restoration pipeline validated on MuralDH.
- **Phase 2 roadmap:** ControlNet diffusion decoder, joint super-resolution head, frequency-guided loss, ablation study, full Swin-Base run, paper draft.
- Immediate next step: scale to Swin-Base with checkpoint-resume across Kaggle's 12-hour session cap to close the PSNR gap to baseline.

---

## Slide 14 — Thank You / Q&A
- Contact: K Anirudh, S Harish Siddharth — M.Tech AI & DS, SASTRA University
