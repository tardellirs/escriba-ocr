"""Transcribe the BFL sample with Escriba-OCR-1B and its base model (runs ON the Colab GPU VM).

Expects /content/sample40/*.png (original BFL scans, grayscale). Writes /content/bfl_out/<run>/pred/*.txt
and run_info.json, then /content/bfl_out.tgz. Settings follow the paper: image-only prompt, greedy,
repetition penalty 1.05; pages resized so the longer side is 1200 px (Escriba, its training size)
or 1600 px (base model, as in the BRESSAY baselines).
"""
import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

if len(sys.argv) == 1:  # driver: install, then one subprocess per model (frees GPU memory between models)
    subprocess.run("pip install -q uv && uv pip install --system -q vllm", shell=True, check=True)
    for i in range(2):
        subprocess.run([sys.executable, __file__, str(i)], check=True)
    subprocess.run("cd /content && tar czf bfl_out.tgz bfl_out", shell=True, check=True)
    print("DONE", flush=True)
    sys.exit(0)

from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

SRC = Path("/content/sample40")
OUT = Path("/content/bfl_out")
RUNS = [("escriba-ocr-1b", "tardellirs/Escriba-OCR-1B", 1200), ("lightonocr3-1b-base", "lightonai/LightOnOCR-3-1B", 1600)]


def png_b64(path: Path, long_side: int) -> str:
    im = Image.open(path).convert("RGB")
    f = long_side / max(im.size)
    im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


pages = sorted(SRC.glob("*.png"))
run, model, long_side = RUNS[int(sys.argv[1])]
out = OUT / run
(out / "pred").mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
llm = LLM(model=model, max_model_len=8192, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85, seed=0)
sp = SamplingParams(temperature=0.0, max_tokens=2048, repetition_penalty=1.05)
msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(p, long_side)}"}}]}] for p in pages]
res = llm.chat(msgs, sp, use_tqdm=False)
for p, r in zip(pages, res):
    (out / "pred" / f"{p.stem}.txt").write_text(r.outputs[0].text.strip(), encoding="utf-8")
info = dict(model=model, run=run, long_side=long_side, pages=len(pages), seconds=round(time.perf_counter() - t0, 1),
            truncated=sum(r.outputs[0].finish_reason == "length" for r in res))
(out / "run_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info), flush=True)
