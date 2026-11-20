"""Minimal client for Escriba-OCR-1B served by vLLM (OpenAI-compatible API). Standard library + Pillow only.

Start the server with:
    vllm serve tardellirs/Escriba-OCR-1B --port 8010 --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}'
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import DEFAULT_BASE_URL, DEFAULT_MODEL, LONG_SIDE


@dataclass
class Transcription:
    text: str
    seconds: float
    output_tokens: int
    finish_reason: str | None


def prepare_image(data: bytes, long_side: int = LONG_SIDE) -> bytes:
    """Fix EXIF rotation, convert to RGB and resize so the longer side is `long_side` (up or down). Returns PNG bytes."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    f = long_side / max(im.size)
    im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def transcribe_bytes(data: bytes, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                     timeout: float = 300.0, max_tokens: int = 2048) -> Transcription:
    """Send one page image to the server and return its error-preserving transcription."""
    url = base_url.rstrip("/") + "/chat/completions"
    image = base64.b64encode(prepare_image(data)).decode()
    body = {
        "model": model,
        # Image only: the empty prompt is the transcription mode used in training.
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}],
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"Could not reach the model server at {base_url} ({e}). "
                              f"Start it with: vllm serve {model} --port 8010") from e
    choice = resp["choices"][0]
    return Transcription(text=(choice["message"].get("content") or "").strip(),
                         seconds=round(time.perf_counter() - t0, 2),
                         output_tokens=resp.get("usage", {}).get("completion_tokens", 0),
                         finish_reason=choice.get("finish_reason"))


def transcribe_file(path: str | Path, **kw) -> Transcription:
    return transcribe_bytes(Path(path).read_bytes(), **kw)
