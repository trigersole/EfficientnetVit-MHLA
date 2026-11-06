"""Verify a Cityscapes folder before any R0/training run.

Checks: expected image counts (train 2975, val 500), a *_gtFine_labelIds.png mask for every
image, matching image/mask sizes on a sample, and that the masks hold raw labelIds (0-33, 255 ok).
Usage: python scripts/tc2/check_cityscapes.py ~/datasets/cityscapes
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED = {"train": 2975, "val": 500}


def main(root: Path) -> int:
    ok = True
    for split, expected in EXPECTED.items():
        images = sorted((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
        masks = [
            root / "gtFine" / split / p.parent.name / p.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            for p in images
        ]
        missing = [m for m in masks if not m.is_file()]
        print(f"{split}: images={len(images)} (expected {expected}), missing labelIds masks={len(missing)}")
        if len(images) != expected or missing:
            ok = False
            for m in missing[:5]:
                print("  missing:", m)
        random.seed(0)
        values = set()
        for image_path, mask_path in random.sample(list(zip(images, masks)), min(20, len(images))):
            if not mask_path.is_file():
                continue
            image, mask = Image.open(image_path), np.asarray(Image.open(mask_path))
            if image.size != (mask.shape[1], mask.shape[0]) or mask.ndim != 2:
                print(f"  size mismatch: {image_path.name} {image.size} vs mask {mask.shape}")
                ok = False
            values.update(np.unique(mask).tolist())
        print(f"  sampled mask values: min={min(values) if values else None} max={max(values) if values else None} "
              f"n_unique={len(values)}")
        if values and (max(v for v in values if v != 255) > 33):
            print("  masks do not look like raw labelIds (values > 33)")
            ok = False
    print("DATASET_OK" if ok else "DATASET_PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).expanduser()))
