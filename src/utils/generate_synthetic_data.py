"""
src/utils/generate_synthetic_data.py
Generate synthetic paired (clean, damaged, mask) triplets for DAFR-Net.

Used when the MuralDH dataset is not yet available — enables:
  • Unit-test validation (shape / dtype / range checks)
  • Quick sanity-training runs to verify the full pipeline

Damage simulations:
  crack   — random branching polyline strokes (dark, thin)
  fade    — elliptical bleach patches (low-saturation, bright)
  missing — rectangular / elliptical filled holes

Usage:
    python src/utils/generate_synthetic_data.py --n 300 --out_dir data/synthetic
    python src/utils/generate_synthetic_data.py --n 100 --out_dir data/synthetic --size 256
"""

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Procedural mural texture generator
# ---------------------------------------------------------------------------

def make_mural_texture(size: int = 256) -> np.ndarray:
    """
    Procedural warm-palette texture (ochre / sienna) with multi-scale noise
    and simple geometric motifs. Returns uint8 BGR array.
    """
    hue = random.randint(10, 40)
    sat = random.randint(40, 120)
    val = random.randint(100, 200)

    img = np.full((size, size, 3), [hue, sat, val], dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Multi-scale noise (Perlin-like via blended Gaussians)
    noise = np.zeros((size, size), dtype=np.float32)
    for scale in (4, 8, 16, 32, 64):
        s = size // scale + 2
        chunk = np.random.randn(s, s).astype(np.float32)
        up    = cv2.resize(chunk, (size, size), interpolation=cv2.INTER_LINEAR)
        noise += up / scale

    ptp = noise.max() - noise.min()
    if ptp > 0:
        noise = (noise - noise.min()) / ptp * 60 - 30
    img = np.clip(img.astype(np.int16) + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Simple geometric motifs (circles / rectangles)
    for _ in range(random.randint(2, 8)):
        color  = tuple(int(random.randint(50, 200)) for _ in range(3))
        cx, cy = random.randint(20, size - 20), random.randint(20, size - 20)
        r      = random.randint(5, 30)
        thick  = random.choice([-1, 1, 2])
        cv2.circle(img, (cx, cy), r, color, thick)

    return img


# ---------------------------------------------------------------------------
# Damage simulators
# ---------------------------------------------------------------------------

def apply_crack(img: np.ndarray):
    """Random branching polyline strokes. Returns (damaged, mask)."""
    damaged = img.copy()
    mask    = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w    = img.shape[:2]

    for _ in range(random.randint(3, 8)):
        x, y  = random.randint(0, w), random.randint(0, h)
        angle = random.uniform(0, 2 * math.pi)
        pts   = [(x, y)]
        for _ in range(5):
            angle += random.uniform(-0.4, 0.4)
            step   = random.randint(5, 20)
            pts.append((
                int(pts[-1][0] + step * math.cos(angle)),
                int(pts[-1][1] + step * math.sin(angle)),
            ))
        arr   = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        thick = random.randint(1, 3)
        color = (random.randint(5, 30),) * 3
        cv2.polylines(damaged, [arr], False, color, thick)
        cv2.polylines(mask,    [arr], False, 255,   thick + 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    return damaged, mask


def apply_fade(img: np.ndarray):
    """Elliptical bleach patches. Returns (damaged, mask)."""
    damaged = img.copy().astype(np.float32)
    mask    = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w    = img.shape[:2]

    for _ in range(random.randint(2, 5)):
        cx, cy = random.randint(30, w - 30), random.randint(30, h - 30)
        rx, ry = random.randint(15, 60),      random.randint(15, 60)
        Y, X   = np.ogrid[:h, :w]
        ellipse = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        alpha   = np.where(ellipse, random.uniform(0.3, 0.7), 0.0)[:, :, np.newaxis]
        damaged += alpha * (255 - damaged)
        mask     = np.clip(mask + (ellipse * 255).astype(np.uint8), 0, 255)

    damaged = np.clip(damaged, 0, 255).astype(np.uint8)
    noise   = (np.random.randn(h, w, 3) * 12).astype(np.int16)
    damaged = np.clip(damaged.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return damaged, mask.astype(np.uint8)


def apply_missing(img: np.ndarray):
    """Rectangular / elliptical holes. Returns (damaged, mask)."""
    damaged = img.copy()
    mask    = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w    = img.shape[:2]

    for _ in range(random.randint(1, 4)):
        cx, cy  = random.randint(20, w - 20), random.randint(20, h - 20)
        rx, ry  = random.randint(10, 50),      random.randint(10, 50)
        fill    = tuple(random.randint(200, 240) for _ in range(3))
        if random.random() < 0.5:
            x1, y1 = max(0, cx - rx), max(0, cy - ry)
            x2, y2 = min(w, cx + rx), min(h, cy + ry)
            cv2.rectangle(damaged, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(mask,    (x1, y1), (x2, y2), 255,  -1)
        else:
            cv2.ellipse(damaged, (cx, cy), (rx, ry), 0, 0, 360, fill, -1)
            cv2.ellipse(mask,    (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    return damaged, mask


DAMAGE_FNS = {
    "crack":   apply_crack,
    "fade":    apply_fade,
    "missing": apply_missing,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(n: int = 200, out_dir: str = "data/synthetic",
             size: int = 256) -> None:
    out     = Path(out_dir)
    clean_d = out / "clean"
    dam_d   = out / "processed"
    mask_d  = out / "masks"
    for d in (clean_d, dam_d, mask_d):
        d.mkdir(parents=True, exist_ok=True)

    types = list(DAMAGE_FNS.keys())
    rows: list[dict] = []

    print(f"Generating {n} synthetic triplets ({size}×{size}) → {out}")
    for i in tqdm(range(n), desc="Synthesizing"):
        dtype       = types[i % len(types)]
        clean_bgr   = make_mural_texture(size)
        damaged_bgr, mask = DAMAGE_FNS[dtype](clean_bgr)

        stem = f"{dtype}_{i:04d}"
        cv2.imwrite(str(clean_d / f"{stem}.png"), clean_bgr)
        cv2.imwrite(str(dam_d   / f"{stem}.png"), damaged_bgr)
        cv2.imwrite(str(mask_d  / f"{stem}.png"), mask)

        rows.append({
            "filename":    str(dam_d / f"{stem}.png"),
            "damage_type": dtype,
            "label_id":    types.index(dtype),
        })

    csv_path = out / "damage_labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "damage_type", "label_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {out}/")
    print(f"  clean/           — {n} clean reference images")
    print(f"  processed/       — {n} damaged inputs  (point data.root here)")
    print(f"  masks/           — {n} binary masks    (point data.mask_dir here)")
    print(f"  damage_labels.csv")
    print()
    print("Quick-start (update configs/classifier.yaml first):")
    print(f"  data:")
    print(f"    root:      {dam_d}")
    print(f"    mask_dir:  {mask_d}")
    print(f"    label_csv: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic mural training data for DAFR-Net"
    )
    parser.add_argument("--n",       type=int, default=200,
                        help="Number of triplets to generate")
    parser.add_argument("--out_dir", default="data/synthetic",
                        help="Output root directory")
    parser.add_argument("--size",    type=int, default=256,
                        help="Image size (pixels)")
    args = parser.parse_args()
    generate(args.n, args.out_dir, args.size)
