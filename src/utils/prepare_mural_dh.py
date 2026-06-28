"""
src/utils/prepare_mural_dh.py
Ingest the MuralDH dataset into DAFR-Net's data/ layout.

Processes three dataset components:
  Mural_seg/train+test — 961 images with binary pixel masks
  Mural512/            — 5 096 images; heuristic masks generated inline

Output:
  data/processed/       — 256×256 RGB images (all sources)
  data/masks/           — 256×256 binary masks  (0=clean, 255=damaged)
  data/damage_labels.csv — rows: filename, mask_path, damage_type, label_id, split, source

Usage:
    cd e:\\DAFR-Net
    python src/utils/prepare_mural_dh.py --dataset_dir MuralDH --out_dir data
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

TARGET_SIZE  = 256
DAMAGE_TYPES = ["crack", "fade", "missing"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_img(img: Image.Image) -> Image.Image:
    return img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)


def _resize_mask(mask: Image.Image) -> Image.Image:
    return mask.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)


# ---------------------------------------------------------------------------
# Damage type inference
# ---------------------------------------------------------------------------

def infer_damage_type(img_bgr: np.ndarray, mask_arr: np.ndarray) -> str:
    """
    Classify damage type from the damaged region of an image.
    mask_arr is a uint8 array (0=clean, 255=damaged).
    """
    mask_bool = mask_arr > 128
    if not mask_bool.any():
        return "missing"

    h, w = mask_arr.shape
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8
    )
    n_comp = n_labels - 1  # exclude background

    # Many disconnected fragments → crack
    if n_comp > 8:
        return "crack"

    # Low saturation in damaged zone → fade / flaking
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1][mask_bool].mean())
    if mean_sat < 60:
        return "fade"

    # Large contiguous hole → missing region
    if n_comp > 0:
        max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        if max_area > h * w * 0.03:
            return "missing"

    return "crack"


# ---------------------------------------------------------------------------
# Heuristic mask generation (for unlabelled Mural512 images)
# ---------------------------------------------------------------------------

def generate_heuristic_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Produce a binary damage mask from an unlabelled mural image.
    Combines crack (Canny), fade (HSV saturation), and missing (luminance)
    detection into a single uint8 mask (0 / 255).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    # -- Crack: multi-scale Canny -------------------------------------------
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    lo  = max(0.0,   med * 0.5)
    hi  = min(255.0, med * 1.5)
    edges      = cv2.Canny(blurred, lo, hi)
    crack_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # -- Fade: low-saturation high-value patches ----------------------------
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    score = (1.0 - s / 255.0) * (v / 255.0)
    _, fade_mask = cv2.threshold(
        (score * 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU
    )
    fade_mask = cv2.morphologyEx(fade_mask, cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8))
    fade_mask = cv2.morphologyEx(fade_mask, cv2.MORPH_CLOSE,
                                 np.ones((7, 7), np.uint8))

    # -- Missing: extreme-luminance outliers --------------------------------
    raw_missing  = ((gray < 30) | (gray > 220)).astype(np.uint8) * 255
    raw_missing  = cv2.morphologyEx(raw_missing, cv2.MORPH_CLOSE,
                                    np.ones((9, 9), np.uint8))
    min_area     = h * w * 0.001
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_missing)
    miss_mask = np.zeros_like(raw_missing)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            miss_mask[labels == i] = 255

    # Union of all signals
    union = np.clip(
        crack_mask.astype(np.int32)
        + fade_mask.astype(np.int32)
        + miss_mask.astype(np.int32),
        0, 255
    ).astype(np.uint8)
    return union


# ---------------------------------------------------------------------------
# Mural_seg processing (annotated images with ground-truth masks)
# ---------------------------------------------------------------------------

def process_seg(dataset_dir: Path, out_img: Path, out_mask: Path) -> list:
    rows = []
    seg_dir = dataset_dir / "Mural_seg"

    for split in ("train", "test"):
        img_dir   = seg_dir / split / "images"
        lbl_dir   = seg_dir / split / "labels"
        img_files = sorted(img_dir.glob("*.png"))

        for img_path in tqdm(img_files, desc=f"Mural_seg/{split}", unit="img"):
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                continue

            img  = _resize_img(Image.open(img_path).convert("RGB"))
            mask = _resize_mask(Image.open(lbl_path).convert("L"))

            # Prefix with "seg_" to avoid collision: some Mural_seg images
            # share the img_*crop_*_* naming convention with Mural512.
            stem          = f"seg_{img_path.stem}"
            out_img_path  = out_img  / f"{stem}.png"
            out_mask_path = out_mask / f"{stem}.png"
            img.save(out_img_path)
            mask.save(out_mask_path)

            img_bgr  = cv2.cvtColor(np.array(img),  cv2.COLOR_RGB2BGR)
            mask_arr = np.array(mask)
            dtype    = infer_damage_type(img_bgr, mask_arr)

            rows.append({
                "filename":    str(out_img_path),
                "mask_path":   str(out_mask_path),
                "damage_type": dtype,
                "label_id":    DAMAGE_TYPES.index(dtype),
                "split":       split,
                "source":      "mural_seg",
            })

    return rows


# ---------------------------------------------------------------------------
# Mural512 processing (unannotated — heuristic masks)
# ---------------------------------------------------------------------------

def process_mural512(dataset_dir: Path, out_img: Path, out_mask: Path) -> list:
    rows    = []
    src_dir = dataset_dir / "Mural512"
    img_files = sorted(src_dir.glob("*.png"))

    for img_path in tqdm(img_files, desc="Mural512", unit="img"):
        img     = _resize_img(Image.open(img_path).convert("RGB"))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_arr = generate_heuristic_mask(img_bgr)

        stem          = img_path.stem
        out_img_path  = out_img  / f"{stem}.png"
        out_mask_path = out_mask / f"{stem}.png"
        img.save(out_img_path)
        Image.fromarray(mask_arr).save(out_mask_path)

        dtype = infer_damage_type(img_bgr, mask_arr)
        rows.append({
            "filename":    str(out_img_path),
            "mask_path":   str(out_mask_path),
            "damage_type": dtype,
            "label_id":    DAMAGE_TYPES.index(dtype),
            "split":       "all",
            "source":      "mural512",
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(dataset_dir: str, out_dir: str, skip_mural512: bool = False):
    dataset_dir = Path(dataset_dir)
    out_dir     = Path(out_dir)

    out_img  = out_dir / "processed"
    out_mask = out_dir / "masks"
    out_img.mkdir(parents=True,  exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    rows: list = []
    rows += process_seg(dataset_dir, out_img, out_mask)
    if not skip_mural512:
        rows += process_mural512(dataset_dir, out_img, out_mask)

    csv_path = out_dir / "damage_labels.csv"
    fieldnames = ["filename", "mask_path", "damage_type", "label_id", "split", "source"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    by_type   = Counter(r["damage_type"] for r in rows)
    by_source = Counter(r["source"]      for r in rows)
    print(f"\nDone. {len(rows)} images -> {out_dir}")
    print(f"CSV:  {csv_path}")
    print("\nBy damage type:  ", dict(by_type))
    print("By source:       ", dict(by_source))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ingest MuralDH into DAFR-Net data/ layout"
    )
    p.add_argument("--dataset_dir",    default="MuralDH",
                   help="Path to MuralDH root (contains Mural512, Mural_seg, Mural_SR)")
    p.add_argument("--out_dir",        default="data",
                   help="Output root — creates data/processed and data/masks")
    p.add_argument("--skip_mural512",  action="store_true",
                   help="Skip the large Mural512 set (process only annotated Mural_seg)")
    args = p.parse_args()
    main(args.dataset_dir, args.out_dir, args.skip_mural512)
