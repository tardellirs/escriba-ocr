"""Transcribe BRESSAY pages with open VLMs through OpenRouter (OpenAI-compatible API).

uv run python -m bench.run_openrouter --model google/gemma-3-27b-it [--split test] [--limit 5] [--workers 4]

Key: macOS Keychain service `openrouter-api-key` (or env OPENROUTER_API_KEY).
Pages are upscaled to 1540 px on the long side (same preprocessing as our fine-tuned model) and sent as JPEG q92.
Outputs (resumable): results/or-<model>/pred/<page>.txt and meta.jsonl (latency, tokens, cost, provider).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from .augment import upscale
from .data import ROOT, load_split
from .prompts import PROMPTS

URL = "https://openrouter.ai/api/v1/chat/completions"


def api_key() -> str:
    if k := os.environ.get("OPENROUTER_API_KEY"):
        return k
    return subprocess.run(["security", "find-generic-password", "-s", "openrouter-api-key", "-w"],
                          capture_output=True, text=True, check=True).stdout.strip()


def image_url(path, long_side):
    im = upscale(Image.open(path).convert("RGB"), long_side)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt", default="dipl", choices=list(PROMPTS))
    ap.add_argument("--long-side", type=int, default=1540)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--no-reasoning", action="store_true", help="ask the provider to disable thinking")
    ap.add_argument("--run")
    a = ap.parse_args()

    run = a.run or "or-" + a.model.split("/")[-1]
    out = ROOT / "results" / run
    (out / "pred").mkdir(parents=True, exist_ok=True)
    pages = load_split(a.split)[: a.limit]
    todo = [p for p in pages if not (out / "pred" / f"{p.id}.txt").exists()]
    print(f"{run}: {len(pages)} pages, {len(todo)} to do", flush=True)
    key, lock = api_key(), threading.Lock()

    def one(p):
        body = {
            "model": a.model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(p.image, a.long_side)}},
                {"type": "text", "text": PROMPTS[a.prompt]}]}],
            "temperature": 0, "max_tokens": a.max_tokens, "usage": {"include": True},
        }
        if a.no_reasoning:
            body["reasoning"] = {"enabled": False}
        data = json.dumps(body).encode()
        for attempt in range(6):
            t0 = time.perf_counter()
            req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                                                    "X-Title": "ocr-bench"})
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    resp = json.loads(r.read())
                dt = time.perf_counter() - t0
                if "error" in resp:
                    raise RuntimeError(json.dumps(resp["error"])[:300])
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, ConnectionError) as e:
                msg = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
                if attempt == 5 or (isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 402, 403, 404)):
                    raise RuntimeError(msg)
                time.sleep(3 * 2 ** attempt)
        ch = resp["choices"][0]
        text = ch["message"].get("content") or ""
        u = resp.get("usage", {})
        rec = dict(page=p.id, model=a.model, provider=resp.get("provider"), latency_s=round(dt, 3),
                   prompt_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                   thinking_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                   cost_usd=u.get("cost"), finish=ch.get("finish_reason"), long_side=a.long_side)
        rec["output_tokens"] -= rec["thinking_tokens"]
        (out / "pred" / f"{p.id}.txt").write_text(text, encoding="utf-8")
        with lock, open(out / "meta.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, p): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                r = f.result()
                print(f"[{i}/{len(todo)}] {p.id} {r['latency_s']}s out={r['output_tokens']} think={r['thinking_tokens']} ${r['cost_usd']} {r['finish']}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {p.id} ERROR {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
