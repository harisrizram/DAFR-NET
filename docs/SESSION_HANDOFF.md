# DAFR-Net — Session Handoff

**Project:** Damage-Aware Frequency-Guided Restoration Network for Ancient Murals
**Author:** K Anirudh | S Harish Siddharth | M.Tech AI & DS | SASTRA University
**Last Updated:** 2026-06-28 (Session 2)

---

## Dataset: MuralDH (NOW AVAILABLE)

Placed at `e:\DAFR-Net\MuralDH\`. Three components:

| Folder                              | Contents                                        | Count        |
| ----------------------------------- | ----------------------------------------------- | ------------ |
| `Mural_seg/train/images` + `labels` | 760 RGB images + binary masks (512×512, 0/255)  | 760 pairs    |
| `Mural_seg/test/images` + `labels`  | Held-out annotated test set                     | 201 pairs    |
| `Mural512/`                         | Unannotated high-resolution crops (512×512 PNG) | 5 096 images |
| `Mural_SR/Mural_DataSet/`           | HR images for super-resolution                  | 512 images   |
| `Mural_SR/Mural_DataSet_LR/X2/`     | LR counterparts at ×2 scale                     | 512 images   |

**Mask format:** single-channel uint8 PNG, values 0 (clean) or 255 (damaged). Confirmed via PIL.

---

## Data Ingestion (Session 2 — in progress / complete)

**Script:** [`src/utils/prepare_mural_dh.py`](../src/utils/prepare_mural_dh.py)

Run command:

```bash
cd e:\DAFR-Net
python src/utils/prepare_mural_dh.py --dataset_dir MuralDH --out_dir data
```

**What it does:**

1. Resizes all images to 256×256 → `data/processed/`
2. Copies / resizes `Mural_seg` binary masks → `data/masks/`
3. Generates heuristic masks for unannotated `Mural512` images → `data/masks/`
4. Infers damage type (crack / fade / missing) per image
5. Writes `data/damage_labels.csv` with columns: `filename, mask_path, damage_type, label_id, split, source`

**Output (completed 2026-06-28):**

```
data/
  processed/      — 6 057 images (256×256 PNG)
                    961  from Mural_seg  (prefix: seg_*.png)
                    5096 from Mural512   (prefix: img_*crop_*.png)
  masks/          — 6 057 matching binary masks (same naming)
  damage_labels.csv — 6 057 rows, all files verified on disk
```

**Naming note:** Mural*seg output files are prefixed `seg*`(e.g.`seg*001216.png`, `seg_img_0crop_0_1.png`) to avoid collision with Mural512 files that share the `img*_crop\__\_\*` naming convention.

**Damage type distribution:**

```
crack   — 1425 (727 seg + 698 mural512)
fade    —  868 (138 seg + 730 mural512)
missing — 3764 ( 96 seg + 3668 mural512)
```

Missing is over-represented in heuristic masks — consider class-weighted loss for classifier.

**Re-run flag:** `--skip_mural512` processes only the 961 annotated images (fast, ~20 sec).

---

## Files Changed This Session

| File                            | Status        | Notes                                                                                        |
| ------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| `src/utils/prepare_mural_dh.py` | **NEW**       | MuralDH-specific ingestion script                                                            |
| `configs/encoder.yaml`          | modified      | Minor comment update on`target_dir` and `num_workers`                                        |
| `configs/classifier.yaml`       | modified      | Comment on`label_csv` path                                                                   |
| `README.md`                     | **rewritten** | Full status tables, architecture diagram, correct quick-start commands, SESSION_HANDOFF note |

---

## Architecture Notes (carry-forward from Session 1)

### Resolution mismatch

Swin downsamples 32×: encoder output for 256 input = **(B, 512, 8, 8)**.
4 UpBlocks → decoder output = **(B, 3, 128, 128)**.
Fixed in `train_encoder.py:DAFRNetModule.forward()` via bilinear upsample to input resolution.

### Self-supervised training (no clean reference)

MuralDH does **not** include clean/undamaged counterparts.
`target_dir: null` in `encoder.yaml` → training reconstructs full image using unmasked regions as signal.
This is valid for Phase 1 inpainting; Phase 2 may use synthetic clean pairs.

---

## Next Steps (Priority Order)

### 1. Wait for ingestion to complete, then verify

```bash
# Count output files
(Get-ChildItem data/processed -File).Count   # expect ~6057
(Get-ChildItem data/masks     -File).Count   # expect ~6057

# Spot-check the CSV
python -c "import pandas as pd; df=pd.read_csv('data/damage_labels.csv'); print(df.groupby(['damage_type','source']).size())"
```

### 2. Run unit tests (no dataset required)

```bash
pytest tests/ -v
```

Uses tiny swin_tiny + pretrained=False — should pass in ~2 min without GPU.

### 3. Train Classifier (on MuralDH labels)

```bash
python src/classifier/train_classifier.py --config configs/classifier.yaml
```

Expects `data/damage_labels.csv` with `filename` and `label_id` columns.
**Colab T4 note:** set `training.batch_size: 16`, `data.num_workers: 2`.

### 4. Train Encoder + Decoder (restoration)

```bash
python src/encoder/train_encoder.py --config configs/encoder.yaml
```

**Colab T4 note:** set `training.batch_size: 4`, `data.num_workers: 2`,
`model.structure_branch.backbone: swin_tiny_patch4_window7_224` to save VRAM.

### 5. Export weights + launch API

After training the encoder, weights are auto-exported to `models/exports/`.
Classifier must be exported manually:

```python
import shutil
shutil.copy("models/checkpoints/classifier/best.ckpt", "models/exports/classifier.ckpt")
```

Then: `uvicorn api.main:app --reload --port 8000`

---

## Colab Setup (T4 GPU)

```python
# Mount and set up
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

# Patch configs for T4
import yaml
with open("configs/encoder.yaml") as f: cfg = yaml.safe_load(f)
cfg["training"]["batch_size"] = 4
cfg["data"]["num_workers"] = 2
cfg["model"]["structure_branch"]["backbone"] = "swin_tiny_patch4_window7_224"
with open("configs/encoder.yaml", "w") as f: yaml.dump(cfg, f)
```

---

## Phase 2 Items (Not Started)

| Component                    | Description                                          | Estimated LoC |
| ---------------------------- | ---------------------------------------------------- | ------------- |
| ControlNet diffusion decoder | Replace InpaintingHead with diffusion-guided decoder | 250–350       |
| Super-resolution head        | Use`Mural_SR` HR/LR pairs for ×2/×4 SR training      | 150–200       |
| FFT-based loss               | Frequency-domain reconstruction loss                 | 50–80         |
| Ablation study setup         | Config variants + comparison scripts                 | 100–150       |

---

## Key File Map

```
src/
  classifier/
    model.py             DamageClassifier (ResNet-50 / ViT)
    train_classifier.py  Training entry point
  encoder/
    dual_branch.py       StructureBranch + FrequencyBranch + CrossAttentionFusion
    train_encoder.py     ★ DAFRNetModule Lightning + PerceptualLoss
  decoder/
    inpainting_head.py   InpaintingHead (UNet) + InpaintingLoss
  utils/
    prepare_mural_dh.py  ★ NEW (Session 2) — MuralDH-specific ingestion
    prepare_data.py      Generic fallback (filename-based label inference)
    label_damage.py      Pixel-level mask generation (heuristic)
    generate_synthetic_data.py  Synthetic paired data fallback
    metrics.py           PSNR / SSIM
api/
  main.py                FastAPI: /classify /restore /health
configs/
  classifier.yaml        label_csv → data/damage_labels.csv
  encoder.yaml           data.root → data/processed, mask_dir → data/masks
MuralDH/                 ★ Dataset (added Session 2)
  Mural_seg/             Annotated 512×512 images + binary masks
  Mural512/              5096 unannotated crops
  Mural_SR/              HR/LR pairs for super-resolution
tests/
  test_classifier.py
  test_encoder.py
  test_decoder.py
  test_metrics.py
  test_pipeline.py       end-to-end shape + gradient test
```

---

## Known Issues / Watch Out For

1. **wandb required for training** — both train scripts use `WandbLogger`.Offline workaround: replace with `CSVLogger` or set env `WANDB_MODE=offline`.
2. **Windows DataLoader** — `num_workers > 0` can hang on Windows without `if __name__ == '__main__'` guard.Set `data.num_workers: 0` if DataLoader freezes.
3. **Swin / VGG16 weight download** — first run downloads ~700 MB total.Set `model.structure_branch.pretrained: false` and `loss.perceptual_weight: 0.0` for offline/test.
4. **API at startup without checkpoints** — expected to fail until after training. `/health` reports status.
5. **`test_encoder_batch_consistency`** — may be flaky (`atol=1e-5`). Loosen to `atol=1e-4` if needed.
6. **Damage type imbalance** — heuristic classifier tends to over-predict "crack" (Canny fires on texture).
   Check `data/damage_labels.csv` distribution after ingestion; may need class weights in classifier loss.
