"""Score a model's own transcriptions with and without the page image (runs ON a GPU VM).

For every token of the saved output of a run (results/<run>/pred/<page>.txt), teacher-forced:
  lp_img    log p(token | page image, prefix)
  lp_blank  log p(token | blank white page of the same size, prefix)
  lp_none   log p(token | no image at all, prefix)
  lp_alt    log-prob of the best *other* token at that position (image pass), for the top-2 margin
Tokens whose probability hardly depends on the image come from the language prior, not from the page;
bench/prior_gap_eval.py tests whether that gap flags silent rewrites better than confidence does.
The generation-time repetition penalty (1.05) is not applied here.

Usage (repo root on the VM, BRESSAY in data/raw/bressay/bressay):
    python3 training/prior_gap.py --model tardellirs/Escriba-OCR-1B --run lightonocr3-1b-ft-r2 --long-side 1200
    python3 training/prior_gap.py --model lightonai/LightOnOCR-3-1B --run lightonocr3-1b --long-side 1600
Output: results/<run>/prior_gap.json.gz
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from bench.augment import upscale  # noqa: E402
from bench.data import load_split  # noqa: E402
from lightonocr_io import hub_to_model_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--long-side", type=int, required=True)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor  # noqa: E402

BASE = "lightonai/LightOnOCR-3-1B"
proc = LightOnOcrProcessor.from_pretrained(BASE)
model = LightOnOcrForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
sd = {}
for f in sorted(Path(snapshot_download(a.model)).glob("*.safetensors")):
    with safe_open(f, "pt") as fh:
        sd.update({hub_to_model_key(k): fh.get_tensor(k).to(torch.bfloat16) for k in fh.keys()})
res = model.load_state_dict(sd, strict=False)  # transformers 5.22 maps the decoder keys wrongly; see lightonocr_io.py
assert not [k for k in res.missing_keys if k != "lm_head.weight"] and not res.unexpected_keys, res
if "lm_head.weight" in res.missing_keys:
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
model = model.cuda().eval()
tok = proc.tokenizer
pfx_img = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True, tokenize=False)
pfx_txt = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": ""}]}], add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def score(prefix_enc: dict, out_ids: torch.Tensor):
    ids = torch.cat([prefix_enc["input_ids"], out_ids[None]], 1)
    kw = {k: v for k, v in prefix_enc.items() if k not in ("input_ids", "attention_mask")}
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **kw).logits[0]
    p = prefix_enc["input_ids"].shape[1]
    lp = torch.log_softmax(logits[p - 1:p - 1 + len(out_ids)].float(), -1)
    chosen = lp.gather(1, out_ids[:, None])[:, 0]
    lp.scatter_(1, out_ids[:, None], -1e9)
    return chosen.cpu().tolist(), lp.max(1).values.cpu().tolist()


def enc_image(im):
    return {k: v.cuda() for k, v in proc(text=[pfx_img], images=[im], return_tensors="pt").items()}


pred = ROOT / "results" / a.run / "pred"
pages = load_split("test")[: a.limit]
out, t0, top1 = {}, time.time(), [0, 0]
for i, pg in enumerate(pages):
    text = (pred / f"{pg.id}.txt").read_text(encoding="utf-8")
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor(e["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")], device="cuda")
    im = upscale(Image.open(pg.image).convert("RGB"), a.long_side)
    lp_img, lp_alt = score(enc_image(im), ids)
    lp_blank, _ = score(enc_image(Image.new("RGB", im.size, "white")), ids)
    lp_none, _ = score({k: v.cuda() for k, v in tok(pfx_txt, return_tensors="pt").items()}, ids)
    top1[0] += sum(x >= y for x, y in zip(lp_img, lp_alt))
    top1[1] += len(lp_img)
    r = lambda xs: [round(x, 3) for x in xs]  # noqa: E731
    out[pg.id] = dict(text=text, offsets=[list(o) for o in e["offset_mapping"]] + [[len(text), len(text)]],
                      lp_img=r(lp_img), lp_alt=r(lp_alt), lp_blank=r(lp_blank), lp_none=r(lp_none))
    if i % 20 == 0:
        print(f"[prior_gap] {a.run} {i + 1}/{len(pages)} {time.time() - t0:.0f}s, output = top-1 token for {100 * top1[0] / top1[1]:.1f}% (sanity: should be ~95%+)", flush=True)
with gzip.open(ROOT / "results" / a.run / "prior_gap.json.gz", "wt", encoding="utf-8") as f:
    json.dump(dict(model=a.model, long_side=a.long_side, gpu=torch.cuda.get_device_name(0), pages=out), f)
print(f"[prior_gap] done {a.run}: {len(out)} pages in {(time.time() - t0) / 60:.1f} min", flush=True)
