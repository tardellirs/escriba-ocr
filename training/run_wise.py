"""WiSE-FT: interpolate Escriba-OCR-1B with its base model and transcribe BRESSAY val/test and BFL (runs ON the Colab GPU VM).

theta(alpha) = (1 - alpha) * theta_escriba + alpha * theta_base, for every weight (same hub key layout).
alpha = 0 is Escriba, alpha = 1 the base model. Every model gets Escriba's interface: image-only prompt, greedy,
repetition penalty 1.05, page resized so its longer side is 1200 px (BRESSAY pages are upscaled, BFL scans downscaled).

Selection rule, fixed before any result is seen: alpha* is the largest alpha whose BRESSAY validation CER
is within 0.10 point of alpha = 0. BFL and the BRESSAY test set are reported for every alpha but never used to choose.

Inputs:  /content/bressay/bressay/{sets,data/pages}, /content/sample40/*.png
Outputs: /content/wise_out/a<alpha>/{val,test,bfl}/pred/*.txt, run_info.json; /content/wise_out.tgz
Usage:   python3 run_wise.py            (driver: install, one subprocess per alpha, tar)
         python3 run_wise.py <alpha>
"""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ALPHAS = [float(x) for x in os.environ.get("WISE_ALPHAS", "0,0.1,0.2,0.3,0.4,0.5,0.7,1").split(",")]
ESCRIBA, BASE = "tardellirs/Escriba-OCR-1B", "lightonai/LightOnOCR-3-1B"
OUT = Path("/content/wise_out")
BRESSAY = Path("/content/bressay/bressay")
LONG_SIDE = 1200


def driver():
    subprocess.run("python3 -c 'import vllm' 2>/dev/null || (pip install -q uv && uv pip install --system -q vllm)",
                   shell=True, check=True)
    for a in ALPHAS:
        if (OUT / f"a{a:.1f}" / "run_info.json").exists():
            print(f"[skip] alpha={a}", flush=True)
            continue
        subprocess.run([sys.executable, __file__, str(a)], check=True)
    subprocess.run("cd /content && tar czf wise_out.tgz wise_out", shell=True, check=True)
    print("WISE_DONE", flush=True)


def merged_dir(alpha: float) -> str:
    if alpha == 0.0:
        return ESCRIBA
    if alpha == 1.0:
        return BASE
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    def state(repo):
        d = Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]))
        sd = {}
        for f in sorted(d.glob("*.safetensors")):
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    sd[k] = fh.get_tensor(k)
        return d, sd

    edir, e = state(ESCRIBA)
    _, b = state(BASE)
    assert set(e) == set(b), f"key mismatch: {sorted(set(e) ^ set(b))[:5]}"
    m = {k: ((1 - alpha) * e[k].float() + alpha * b[k].float()).to(e[k].dtype).contiguous() for k in e}
    out = Path(f"/content/wise_models/a{alpha:.1f}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(m, out / "model.safetensors", metadata={"format": "pt"})
    for f in edir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.endswith(".index.json"):
            shutil.copy(f, out / f.name)
    print(f"[merge] alpha={alpha} tensors={len(m)}", flush=True)
    return str(out)


def png_b64(path: Path) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    f = LONG_SIDE / max(im.size)
    im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run(alpha: float):
    from vllm import LLM, SamplingParams

    sets = {
        "val": [(p, BRESSAY / "data/pages" / p / f"{p}.png") for p in (BRESSAY / "sets/validation.txt").read_text().split()],
        "test": [(p, BRESSAY / "data/pages" / p / f"{p}.png") for p in (BRESSAY / "sets/test.txt").read_text().split()],
        "bfl": [(p.stem, p) for p in sorted(Path("/content/sample40").glob("*.png"))],
    }
    t0 = time.perf_counter()
    model = merged_dir(alpha)
    llm = LLM(model=model, max_model_len=8192, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=2048, repetition_penalty=1.05)
    root = OUT / f"a{alpha:.1f}"
    info = dict(alpha=alpha, model=model, long_side=LONG_SIDE)
    for name, items in sets.items():
        (root / name / "pred").mkdir(parents=True, exist_ok=True)
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(p)}"}}]}]
                for _, p in items]
        res = llm.chat(msgs, sp, use_tqdm=False)
        for (pid, _), r in zip(items, res):
            (root / name / "pred" / f"{pid}.txt").write_text(r.outputs[0].text.strip(), encoding="utf-8")
        info[f"{name}_pages"] = len(items)
        info[f"{name}_truncated"] = sum(r.outputs[0].finish_reason == "length" for r in res)
        print(f"[run] alpha={alpha} {name} done", flush=True)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    (root / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    if alpha not in (0.0, 1.0):
        shutil.rmtree(model, ignore_errors=True)


if __name__ == "__main__":
    driver() if len(sys.argv) == 1 else run(float(sys.argv[1]))
