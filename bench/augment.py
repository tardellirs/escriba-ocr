"""Image prep + 'phone photo' style augmentation (PIL only, deterministic given a Random)."""
from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

LEVELS = {
    # prob, rot_deg, persp, scale, blur, jpeg_q (min,max), bright/contrast, shadow
    "none": None,
    "light": dict(p=0.5, rot=1.0, persp=0.02, scale=(0.95, 1.05), blur=0.6, jpeg=(60, 95), bc=0.10, shadow=0.15),
    "strong": dict(p=0.9, rot=2.5, persp=0.05, scale=(0.85, 1.10), blur=1.2, jpeg=(35, 85), bc=0.25, shadow=0.35),
}


def upscale(im: Image.Image, long_side: int) -> Image.Image:
    """Resize so the longer side == long_side (up or down). Processors only downscale."""
    if not long_side:
        return im
    f = long_side / max(im.size)
    if abs(f - 1) < 1e-3:
        return im
    return im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)


def _perspective(im, rng, mag):
    w, h = im.size
    d = lambda: rng.uniform(-mag, mag)  # noqa: E731
    src = [(d() * w, d() * h), (w + d() * w, d() * h), (w + d() * w, h + d() * h), (d() * w, h + d() * h)]
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    # solve homography coefficients for PIL (maps output -> input)
    import numpy as np

    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    coeffs = np.linalg.solve(np.array(A, float), np.array(B, float))
    return im.transform((w, h), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=(255, 255, 255))


def _shadow(im, rng, strength):
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)
    x0 = rng.uniform(-0.5, 0.8) * w
    dr.polygon([(x0, 0), (x0 + rng.uniform(0.3, 0.9) * w, 0), (x0 + rng.uniform(0.1, 0.9) * w, h), (x0 - 0.2 * w, h)],
               fill=int(255 * strength))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.08))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, im, mask)


def augment(im: Image.Image, level: str, rng: random.Random, force: bool = False) -> Image.Image:
    cfg = LEVELS.get(level)
    if not cfg or (not force and rng.random() > cfg["p"]):
        return im
    im = im.convert("RGB")
    if cfg["rot"]:
        im = im.rotate(rng.uniform(-cfg["rot"], cfg["rot"]), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if cfg["persp"] and rng.random() < 0.7:
        im = _perspective(im, rng, cfg["persp"])
    s = rng.uniform(*cfg["scale"])
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BICUBIC)
    if rng.random() < 0.6:
        im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-cfg["bc"], cfg["bc"]))
        im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-cfg["bc"], cfg["bc"]))
    if rng.random() < 0.5:
        im = _shadow(im, rng, rng.uniform(0.05, cfg["shadow"]))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, cfg["blur"])))
    if rng.random() < 0.8:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.randint(*cfg["jpeg"]))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return im
