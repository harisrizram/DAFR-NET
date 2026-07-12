"""One-off: build dafrnet_data.zip with forward-slash entry names (zip spec
compliant) so Kaggle's Linux unzip accepts it. PowerShell's Compress-Archive
writes backslash-separated names, which Kaggle rejects as "forbidden character".
"""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "dafrnet_data.zip"

items = [
    (root / "data" / "processed", "processed"),
    (root / "data" / "masks", "masks"),
    (root / "data" / "damage_labels.csv", "damage_labels.csv"),
]

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for src, arc_prefix in items:
        if src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    arcname = f"{arc_prefix}/{f.relative_to(src).as_posix()}"
                    zf.write(f, arcname)
        else:
            zf.write(src, arc_prefix)

print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
