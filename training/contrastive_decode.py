"""Greedy decoding with an LM-prior penalty ("contrastive" against the image-free model).

score(t) = log p(t | image, prefix) - alpha * log p(t | prefix)   [no image]
restricted to tokens with p(t | image, prefix) >= beta * max_t p(t | image, prefix)

alpha = 0 is plain greedy (reference). Larger alpha penalises tokens the decoder would
predict from language alone (e.g. the normalised spelling), favouring what is drawn.

python3 contrastive_decode.py --model /content/ft/<run>/final --ids-file ids.txt --alphas 0,0.5 --out-root /content/ft_eval/cd
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content/ocr-model")
sys.path.insert(0, "/content")
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench.augment import upscale  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ids-file", required=True)
ap.add_argument("--image-root", default="/content/bressay/bressay/data/pages")
ap.add_argument("--alphas", default="0,0.5")
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--long-side", type=int, default=1200)
ap.add_argument("--max-new", type=int, default=1400)
ap.add_argument("--out-root", default="/content/ft_eval/cd")
ap.add_argument("--tag", default="cd")
a = ap.parse_args()

from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor  # noqa: E402
from lightonocr_io import hub_to_model_key  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

proc = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-3-1B")
model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-3-1B", dtype=torch.bfloat16, attn_implementation="sdpa")
sd = {hub_to_model_key(k): v for k, v in load_file(str(Path(a.model) / "model.safetensors")).items()}
res = model.load_state_dict(sd, strict=False)
assert not [k for k in res.missing_keys if k != "lm_head.weight"], res.missing_keys[:5]
if "lm_head.weight" in res.missing_keys:
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
model = model.cuda().eval()
eos = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")

pfx_img = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True, tokenize=False)
pfx_txt = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": ""}]}], add_generation_prompt=True, tokenize=False)
ids = Path(a.ids_file).read_text().split()


@torch.no_grad()
def decode(pid, alpha):
    im = upscale(Image.open(Path(a.image_root) / pid / f"{pid}.png").convert("RGB"), a.long_side)
    enc = {k: v.cuda() for k, v in proc(text=[pfx_img], images=[im], return_tensors="pt").items()}
    out = model(**enc, use_cache=True)
    past_c, lc = out.past_key_values, out.logits[0, -1].float()
    if alpha:
        enc_u = proc.tokenizer(pfx_txt, return_tensors="pt").to("cuda")
        out_u = model.model.language_model(input_ids=enc_u["input_ids"], use_cache=True)
        past_u = out_u.past_key_values
        lu = model.lm_head(out_u.last_hidden_state[:, -1]).float()[0]
    gen = []
    for _ in range(a.max_new):
        lpc = torch.log_softmax(lc, -1)
        if alpha:
            lpu = torch.log_softmax(lu, -1)
            ok = lpc >= (lpc.max() + torch.log(torch.tensor(a.beta, device=lpc.device)))
            score = torch.where(ok, lpc - alpha * lpu, torch.full_like(lpc, -1e9))
            t = int(score.argmax())
        else:
            t = int(lpc.argmax())
        if t == eos:
            break
        gen.append(t)
        tt = torch.tensor([[t]], device="cuda")
        o = model(input_ids=tt, past_key_values=past_c, use_cache=True)
        past_c, lc = o.past_key_values, o.logits[0, -1].float()
        if alpha:
            ou = model.model.language_model(input_ids=tt, past_key_values=past_u, use_cache=True)
            past_u = ou.past_key_values
            lu = model.lm_head(ou.last_hidden_state[:, -1]).float()[0]
    return proc.tokenizer.decode(gen, skip_special_tokens=True), len(gen)


for alpha in [float(x) for x in a.alphas.split(",")]:
    run = f"{a.tag}-a{alpha:g}"
    out = Path(a.out_root) / run
    (out / "pred").mkdir(parents=True, exist_ok=True)
    meta = open(out / "meta.jsonl", "w")
    t0 = time.time()
    for i, pid in enumerate(ids):
        t = time.time()
        txt, n = decode(pid, alpha)
        (out / "pred" / f"{pid}.txt").write_text(txt, encoding="utf-8")
        meta.write(json.dumps(dict(page=pid, model=a.model, latency_s=round(time.time() - t, 2), prompt_tokens=0,
                                   output_tokens=n, thinking_tokens=0, alpha=alpha)) + "\n")
        meta.flush()
        if i % 10 == 0:
            print(f"[cd] {run} {i + 1}/{len(ids)} {time.time() - t:.1f}s", flush=True)
    (out / "run_info.json").write_text(json.dumps(dict(alpha=alpha, beta=a.beta, pages=len(ids), minutes=round((time.time() - t0) / 60, 1))))
    print(f"[cd] done {run} in {(time.time() - t0) / 60:.1f} min", flush=True)
