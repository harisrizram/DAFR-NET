"""
src/utils/prepare_data.py
Preprocess MuralDH dataset: resize, normalize, generate damage label CSV.

Usage:
    python src/utils/prepare_data.py --data_dir data/raw --out_dir data/processed
"""

import os
import argparse
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm


DAMAGE_KEYWORDS = {
    "crack":   ["crack", "fracture", "fissure"],
    "fade":    ["fade", "peel", "color", "discolor"],
    "missing": ["missing", "loss", "hole", "gap"],
}

TARGET_SIZE = (256, 256)


def infer_damage_type(filename: str) -> str:
    """Infer damage type from filename convention in MuralDH dataset."""
    name = filename.lower()
    for dtype, keywords in DAMAGE_KEYWORDS.items():
        if any(k in name for k in keywords):
            return dtype
    return "missing"   # default for unannotated


def prepare(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_rows = []
    image_files = list(data_dir.rglob("*.jpg")) + \
                  list(data_dir.rglob("*.png")) + \
                  list(data_dir.rglob("*.jpeg"))

    print(f"Found {len(image_files)} images in {data_dir}")

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)

            rel_path = img_path.relative_to(data_dir)
            out_path = out_dir / rel_path.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)

            damage_type = infer_damage_type(img_path.name)
            label_rows.append({
                "filename": str(out_path),
                "damage_type": damage_type,
                "label_id": list(DAMAGE_KEYWORDS.keys()).index(damage_type),
            })
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")

    label_csv = Path("data/damage_labels.csv")
    with open(label_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "damage_type", "label_id"])
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nDone. {len(label_rows)} images processed.")
    print(f"Labels saved to: {label_csv}")

    # Summary
    from collections import Counter
    counts = Counter(r["damage_type"] for r in label_rows)
    for dtype, count in counts.items():
        print(f"  {dtype}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir",  default="data/processed")
    args = parser.parse_args()
    prepare(args.data_dir, args.out_dir)
