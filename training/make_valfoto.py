"""Create the fixed 'val-foto' set: val-50 pages degraded once with the strong phone-photo augmentation.

python3 make_valfoto.py  -> /content/valfoto/<id>/<id>.png (+ .txt copied from GT)
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/content/ocr-model")
from PIL import Image  # noqa: E402

from bench.augment import augment  # noqa: E402

SRC = Path("/content/bressay/bressay/data/pages")
OUT = Path("/content/valfoto")
ids = Path("/content/ocr-model/bench/val50.txt").read_text().split()
for pid in ids:
    rng = random.Random(f"valfoto-{pid}")
    im = Image.open(SRC / pid / f"{pid}.png").convert("RGB")
    cfg_im = augment(im, "strong", rng, force=True)
    (OUT / pid).mkdir(parents=True, exist_ok=True)
    cfg_im.save(OUT / pid / f"{pid}.png")
    shutil.copy(SRC / pid / f"{pid}.txt", OUT / pid / f"{pid}.txt")
print(f"{len(ids)} pages -> {OUT}")
