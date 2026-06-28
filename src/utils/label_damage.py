"""
src/utils/label_damage.py
Automatic pixel-level damage mask generation for mural images.

Strategy per damage type:
  crack   — Multi-scale Canny edge detection + morphological refinement
  fade    — HSV saturation/brightness anomaly (bleached = low-S, high-V)
  missing — Luminance outlier detection (near-black or near-white blobs)

When damage type is "auto", it is inferred from the filename using the
same keyword rules as prepare_data.py.

Usage:
    python src/utils/label_damage.py --image_dir data/processed --out_dir data/masks
    python src/utils/label_damage.py --image_dir data/processed --out_dir data/masks --type crack
    python src/utils/label_damage.py --image_dir data/processed --out_dir data/masks --force
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-type mask generators
# ---------------------------------------------------------------------------

def mask_crack(img_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-scale Canny on CLAHE-enhanced grayscale, dilated to fill thin lines.
    Large connected components (> 5 % of image area) are removed — they are
    structural features, not cracks.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    edges_list = []
    for sigma in (0.33, 0.5, 0.66):
        median  = float(np.median(gray))
        lo = int(max(0,   (1 - sigma) * median))
        hi = int(min(255, (1 + sigma) * median))
        edges_list.append(cv2.Canny(gray, lo, hi))

    combined = np.max(np.stack(edges_list, axis=0), axis=0)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.dilate(combined, kernel, iterations=2)

    # Drop large blobs (not cracks)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    area_thresh = combined.size * 0.05
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > area_thresh:
            combined[labels == lbl] = 0

    return (combined > 0).astype(np.uint8) * 255


def mask_fade(img_bgr: np.ndarray) -> np.ndarray:
    """
    Faded pixels are desaturated and bright (high V, low S in HSV).
    Score = (1 - S_norm) * V_norm → Otsu threshold → morphological cleanup.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    s_norm = hsv[:, :, 1] / 255.0
    v_norm = hsv[:, :, 2] / 255.0

    score    = ((1 - s_norm) * v_norm * 255).astype(np.uint8)
    _, mask  = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def mask_missing(img_bgr: np.ndarray) -> np.ndarray:
    """
    Missing regions appear as near-black (exposed substrate) or near-white
    (plaster fill). Connected-component filter keeps only blobs > 0.1 % of
    image area to suppress isolated noise pixels.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, dark   = cv2.threshold(gray, 30,  255, cv2.THRESH_BINARY_INV)
    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    combined  = cv2.bitwise_or(dark, bright)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    min_area = combined.size * 0.001
    out      = np.zeros_like(combined)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > min_area:
            out[labels == lbl] = 255
    return out


MASK_FNS = {
    "crack":   mask_crack,
    "fade":    mask_fade,
    "missing": mask_missing,
}

_CRACK_KW   = {"crack", "fracture", "fissure"}
_FADE_KW    = {"fade", "peel", "color", "discolor"}
_MISSING_KW = {"missing", "loss", "hole", "gap"}


def infer_damage_type(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in _CRACK_KW):
        return "crack"
    if any(k in name for k in _FADE_KW):
        return "fade"
    return "missing"


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def generate_masks(image_dir: str, out_dir: str,
                   damage_type: str = "auto",
                   force: bool = False) -> None:
    image_dir = Path(image_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts  = {".jpg", ".jpeg", ".png"}
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]

    if not paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Generating masks for {len(paths)} images → {out_dir}")
    generated = skipped = errors = 0

    for img_path in tqdm(paths, desc="Masking"):
        out_path = out_dir / img_path.with_suffix(".png").name

        if out_path.exists() and not force:
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            errors += 1
            continue

        dtype   = damage_type if damage_type != "auto" else infer_damage_type(img_path.name)
        mask_fn = MASK_FNS.get(dtype, MASK_FNS["missing"])

        try:
            mask = mask_fn(img_bgr)
            Image.fromarray(mask).save(out_path)
            generated += 1
        except Exception as exc:
            print(f"  Failed [{img_path.name}]: {exc}")
            errors += 1

    print(f"\nDone — generated: {generated} | skipped: {skipped} | errors: {errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pixel-level damage masks from mural images"
    )
    parser.add_argument("--image_dir", default="data/processed",
                        help="Input image directory")
    parser.add_argument("--out_dir",   default="data/masks",
                        help="Output mask directory")
    parser.add_argument("--type", dest="damage_type", default="auto",
                        choices=["auto", "crack", "fade", "missing"],
                        help="Damage type (auto = infer from filename)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing masks")
    args = parser.parse_args()
    generate_masks(args.image_dir, args.out_dir, args.damage_type, args.force)
