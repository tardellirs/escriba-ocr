"""Run open VLMs on BRESSAY test pages with vLLM (runs ON the Colab GPU VM).

python3 run_vllm.py <hf_model_id> <run_name> [--prompt dipl|empty|ocr|glm] [--seq 20] [--limit N]

Writes /content/results/<run>/pred/*.txt, meta.jsonl, run_info.json
Speed: (1) throughput = all pages in one batched call; (2) latency = first --seq pages one at a time.
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content")
from prompts import PROMPTS  # uploaded copy of bench/prompts.py

BRESSAY = Path("/content/bressay/bressay")
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. Just return the plain text representation of this document "
    "as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the primary_language, "
    "is_rotation_valid, rotation_correction, is_table, is_diagram, and is_html fields."
)
EXTRA_PROMPTS = {"empty": "", "ocr": "OCR:", "glm": "Text Recognition:", "olm": OLMOCR_PROMPT}

def load_png(path, long_side):
    """BRESSAY pages are ~550 px wide; upscale (LANCZOS) so small VLMs get enough visual tokens."""
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if long_side and max(im.size) < long_side:
        f = long_side / max(im.size)
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("run")
    ap.add_argument("--prompt", default="dipl")
    ap.add_argument("--seq", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ids-file", help="file with page ids (overrides --split)")
    ap.add_argument("--image-root", default=str(BRESSAY / "data" / "pages"), help="dir with <id>/<id>.png")
    ap.add_argument("--out-root", default="/content/results")
    ap.add_argument("--rep-penalty", type=float, default=1.05)
    ap.add_argument("--long-side", type=int, default=1600, help="upscale page so its longer side is at least this (0 = off)")
    ap.add_argument("--no-think", action="store_true", help="pass enable_thinking=False to chat template")
    a = ap.parse_args()

    prompt = PROMPTS.get(a.prompt, EXTRA_PROMPTS.get(a.prompt))
    split_file = {"train": "training.txt", "val": "validation.txt", "test": "test.txt"}[a.split]
    ids = (Path(a.ids_file) if a.ids_file else BRESSAY / "sets" / split_file).read_text().split()[: a.limit]
    out = Path(a.out_root) / a.run
    (out / "pred").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    from vllm import LLM, SamplingParams  # noqa: E402

    t0 = time.perf_counter()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.88, trust_remote_code=True, seed=0)
    load_s = time.perf_counter() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, repetition_penalty=a.rep_penalty)
    ctk = {"enable_thinking": False} if a.no_think else None


    def msg(pid):
        b64 = base64.b64encode(load_png(Path(a.image_root) / pid / f"{pid}.png", a.long_side)).decode()
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]


    def chat(msgs):
        kw = {"chat_template_kwargs": ctk} if ctk else {}
        return llm.chat(msgs, sp, use_tqdm=False, **kw)


    def clean(text):
        """OCR models emit markdown; drop heading markers / bold but keep BRESSAY ##x## markup."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"\A\s*---\n.*?\n---\n", "", text, flags=re.S)  # olmOCR YAML front matter
        text = re.sub(r"(?m)^#{1,6} +", "", text)
        text = text.replace("**", "")
        return text.strip()


    # warmup
    chat([msg(ids[0])])

    # sequential latency
    lat = {}
    for pid in ids[: a.seq]:
        t = time.perf_counter()
        chat([msg(pid)])
        lat[pid] = time.perf_counter() - t

    # batched throughput (also produces the outputs we score)
    t = time.perf_counter()
    res = chat([msg(pid) for pid in ids])
    batch_s = time.perf_counter() - t

    with open(out / "meta.jsonl", "w") as f:
        for pid, r in zip(ids, res):
            o = r.outputs[0]
            (out / "raw" / f"{pid}.txt").write_text(o.text, encoding="utf-8")
            (out / "pred" / f"{pid}.txt").write_text(clean(o.text), encoding="utf-8")
            f.write(json.dumps(dict(page=pid, model=a.model, latency_s=round(lat[pid], 3) if pid in lat else None,
                                    prompt_tokens=len(r.prompt_token_ids), output_tokens=len(o.token_ids),
                                    thinking_tokens=0, finish=o.finish_reason, prompt=a.prompt)) + "\n")

    import torch  # noqa: E402

    info = dict(model=a.model, run=a.run, prompt=a.prompt, rep_penalty=a.rep_penalty, long_side=a.long_side, pages=len(ids), gpu=torch.cuda.get_device_name(0),
                load_s=round(load_s, 1), batch_s=round(batch_s, 1), pages_per_s=round(len(ids) / batch_s, 3),
                seq_median_s=round(sorted(lat.values())[len(lat) // 2], 2) if lat else None,
                truncated=sum(r.outputs[0].finish_reason == "length" for r in res))
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
