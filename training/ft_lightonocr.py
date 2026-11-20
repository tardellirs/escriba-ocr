"""Fine-tune LightOnOCR-3-1B on BRESSAY (diplomatic transcription). Runs on the Colab GPU VM.

python3 ft_lightonocr.py --run NAME [--lr 2e-5] [--mode decoder|lora|all] [--mix pages|pages_par|all]
       [--long-side 1540] [--prompt empty|dipl] [--aug none|light|strong] [--examples 2400] [--frac 1.0]
       [--accum 8] [--seed 0] [--save-every 0]

- Mixed precision: fp32 master weights + bf16 autocast (pure-bf16 AdamW would swallow lr~2e-5 updates).
- Loss only on the assistant text (+ <|im_end|>).
- Saves a vLLM-loadable checkpoint to /content/ft/<run>/final (bf16 weights + original config/processor files).
- Logs /content/ft/<run>/train_log.jsonl and config.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content/ocr-model")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench.augment import augment, upscale  # noqa: E402
from bench.prompts import PROMPTS  # noqa: E402

MODEL_ID = "lightonai/LightOnOCR-3-1B"
B = Path("/content/bressay/bressay")

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--mode", default="decoder", choices=["decoder", "lora", "all"])
ap.add_argument("--mix", default="all", choices=["pages", "pages_par", "all"])
ap.add_argument("--long-side", type=int, default=1540)
ap.add_argument("--prompt", default="empty", choices=["empty", "dipl"])
ap.add_argument("--aug", default="none", choices=["none", "light", "strong"])
ap.add_argument("--examples", type=int, default=2400, help="total training examples (micro-batches of 1)")
ap.add_argument("--frac", type=float, default=1.0, help="fraction of training PAGES to use (data-scaling curve)")
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--max-len", type=int, default=4608)
ap.add_argument("--lora-r", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save-every", type=int, default=0, help="save intermediate checkpoint every N optimizer steps")
ap.add_argument("--ckpt-steps", default="", help="comma list of optimizer steps to save checkpoints at (curve eval)")
ap.add_argument("--precision", default="bf16m", choices=["bf16m", "amp32"],
                help="bf16m: bf16 model + fp32 master copy in the optimizer; amp32: fp32 weights + bf16 autocast")
ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing (faster, more memory)")
ap.add_argument("--err-boost", type=float, default=1.0,
                help="sampling weight multiplier for paragraph/line items whose GT has a likely student misspelling")
ap.add_argument("--max-minutes", type=float, default=0, help="stop early after N minutes (0 = no limit)")
a = ap.parse_args()

out = Path("/content/ft") / a.run
out.mkdir(parents=True, exist_ok=True)
import transformers  # noqa: E402

(out / "config.json").write_text(json.dumps({**vars(a), "transformers": transformers.__version__, "torch": torch.__version__}, indent=1))
random.seed(a.seed)
torch.manual_seed(a.seed)

# ---------------- data ----------------
train_pages = (B / "sets" / "training.txt").read_text().split()
rng_split = random.Random(1234)
train_pages = sorted(train_pages)
rng_split.shuffle(train_pages)
train_pages = sorted(train_pages[: max(1, round(len(train_pages) * a.frac))])
page_set = set(train_pages)
(out / "train_pages.txt").write_text("\n".join(train_pages) + "\n")


def items(level):
    res = []
    for d in sorted((B / "data" / level).iterdir()):
        if d.name not in page_set:
            continue
        for png in sorted(d.glob("*.png")):
            res.append((png, png.with_suffix(".txt")))
    return res


pools = {"pages": items("pages")}
if a.mix in ("pages_par", "all"):
    pools["paragraphs"] = items("paragraphs")
if a.mix == "all":
    pools["lines"] = items("lines")
weights = {"pages": {"pages": 1.0}, "pages_par": {"pages": 0.5, "paragraphs": 0.5},
           "all": {"pages": 0.4, "paragraphs": 0.3, "lines": 0.3}}[a.mix]
print({k: len(v) for k, v in pools.items()}, weights, flush=True)

rng = random.Random(a.seed)
# Deterministic sample order (same across runs with the same seed/mix): cycle each pool in shuffled order.
order = {k: [] for k in pools}


item_w = {}
if a.err_boost != 1.0:
    from bench.metrics import is_suspect, words  # Hunspell pt_BR via spylls

    for k in pools:
        if k == "pages":
            continue
        ws = []
        from bench import textnorm as tn

        for _, txt in pools[k]:
            raw = tn.dehyphenate(tn.nfc(txt.read_text(encoding="utf-8")))
            lines = [ln for ln in tn.plain(raw, keep_lines=True).split("\n") if ln.strip()]
            cand = []
            for i, ln in enumerate(lines):
                w = words(ln)
                if i == len(lines) - 1 and ln.rstrip().endswith("-") and w:
                    w = w[:-1]  # fragment broken at the end of the crop
                if i == 0 and w and w[0][:1].islower():
                    w = w[1:]  # may be the tail of a word broken on the previous line
                cand += w
            ws.append(a.err_boost if any(is_suspect(w) for w in cand) else 1.0)
        item_w[k] = ws
        print(f"err-boost {k}: {sum(w > 1 for w in ws)}/{len(ws)} items boosted", flush=True)


def draw():
    k = rng.choices(list(weights), weights=list(weights.values()))[0]
    if k in item_w:  # weighted sampling with replacement
        return k, pools[k][rng.choices(range(len(pools[k])), weights=item_w[k])[0]]
    if not order[k]:
        order[k] = list(range(len(pools[k])))
        rng.shuffle(order[k])
    return k, pools[k][order[k].pop()]


# ---------------- model ----------------
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor  # noqa: E402

proc = LightOnOcrProcessor.from_pretrained(MODEL_ID)
sys.path.insert(0, "/content")
from lightonocr_io import load_model, save_model  # noqa: E402  (transformers 5.22 misloads the decoder otherwise)

model = load_model(torch.bfloat16 if a.precision == "bf16m" else torch.float32).cuda()
model.config.use_cache = False
if not a.no_gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

names = [n for n, _ in model.named_parameters()]
is_vision = lambda n: "vision" in n  # noqa: E731  (vision_tower / vision_encoder)
if a.mode == "lora":
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    lm_mods = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, target_modules=lm_mods))
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}]
else:
    vis, rest = [], []
    for n, p in model.named_parameters():
        if is_vision(n):
            p.requires_grad = a.mode == "all"
            if p.requires_grad:
                vis.append(p)
        else:
            p.requires_grad = True
            rest.append(p)
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": vis, "lr": a.lr * 0.1}] if vis else [])
n_train = sum(p.numel() for g in groups for p in g["params"])
print(f"trainable params: {n_train / 1e6:.1f}M ({a.mode})", flush=True)
model_params = [p for g in groups for p in g["params"]]
if a.precision == "bf16m":
    masters = {id(p): p.detach().float().clone() for p in model_params}
    groups = [{"params": [masters[id(p)] for p in g["params"]], "lr": g["lr"]} for g in groups]
opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
steps = math.ceil(a.examples / a.accum)  # 0 -> save the untouched base model (round-trip check)
warm = max(1, int(steps * a.warmup))
base_lrs = [g["lr"] for g in groups]


def set_lr(step):
    f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * f


prompt_text = PROMPTS["dipl"] if a.prompt == "dipl" else None
user_content = [{"type": "image"}] + ([{"type": "text", "text": prompt_text}] if prompt_text else [])
prefix = proc.apply_chat_template([{"role": "user", "content": user_content}], add_generation_prompt=True, tokenize=False)


def encode(png, txt):
    im = Image.open(png).convert("RGB")
    im = augment(im, a.aug, rng)
    im = upscale(im, a.long_side)
    target = txt.read_text(encoding="utf-8").strip() + "\n<|im_end|>"
    enc = proc(text=[prefix + target], images=[im], return_tensors="pt")
    n_prefix = len(proc(text=[prefix], images=[im], return_tensors="pt")["input_ids"][0])
    return enc, n_prefix


def save(tag):
    d = out / tag
    m = model.merge_and_unload() if a.mode == "lora" else model
    save_model(m, d)
    print(f"saved {d}", flush=True)
    return m


log = open(out / "train_log.jsonl", "a")
model.train()
t0 = time.time()
tok_count, run_loss, run_n = 0, 0.0, 0  # run_loss = summed token CE, run_n = target tokens
kinds = {}
ckpt_steps = {int(x) for x in a.ckpt_steps.split(",") if x.strip()}
params = model_params
ce = torch.nn.CrossEntropyLoss(reduction="sum")
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "amp32")  # noqa: E731
step = -1
for step in range(steps):
    set_lr(step)
    n_tgt_total = 0
    for _ in range(a.accum):
        k, (png, txt) = draw()
        kinds[k] = kinds.get(k, 0) + 1
        enc, n_prefix = encode(png, txt)
        L = enc["input_ids"].shape[1]
        if L > a.max_len:
            continue
        n_t = L - n_prefix  # target tokens (text + <|im_end|>)
        enc = {kk: v.cuda() for kk, v in enc.items()}
        with amp():
            # logits only for positions that predict the target (saves the 4k x 152k logits tensor)
            logits = model(**enc, logits_to_keep=n_t + 1).logits[0, :-1].float()
        tgt = enc["input_ids"][0, -n_t:]
        loss_sum = ce(logits, tgt)
        loss_sum.backward()  # normalized by total target tokens of the window below
        n_tgt_total += n_t
        run_loss += loss_sum.item()
        run_n += n_t
        tok_count += L
    if a.precision == "bf16m":
        for p in params:
            mp = masters[id(p)]
            mp.grad = None if p.grad is None else p.grad.float().div_(max(1, n_tgt_total))
            p.grad = None
        torch.nn.utils.clip_grad_norm_([masters[id(p)] for p in params], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p in params:
                p.copy_(masters[id(p)])
    else:
        if n_tgt_total:
            for p in params:
                if p.grad is not None:
                    p.grad.div_(n_tgt_total)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    el = time.time() - t0
    if step % 5 == 0 or step == steps - 1:
        rec = dict(step=step + 1, of=steps, loss=round(run_loss / max(1, run_n), 4), lr=opt.param_groups[0]["lr"],
                   tok_s=round(tok_count / el, 1), min=round(el / 60, 2), kinds=kinds,
                   mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        run_loss, run_n = 0.0, 0
    if ((a.save_every and (step + 1) % a.save_every == 0) or (step + 1) in ckpt_steps) and step + 1 < steps and a.mode != "lora":
        save(f"step{step + 1}")
    if a.max_minutes and el / 60 > a.max_minutes:
        print(f"time limit reached at step {step + 1}", flush=True)
        break
model.eval()
torch.cuda.empty_cache()
save("final")
(out / "done.json").write_text(json.dumps(dict(minutes=round((time.time() - t0) / 60, 1), steps_done=step + 1)))
