"""Build the open-model comparison table for the Escriba-OCR-1B model card from results/*/scores.json.

uv run python -m bench.make_card_table [--card models/Escriba-OCR-1B/README.md]
Replaces the <!-- OPEN_MODELS_TABLE --> block (between the marker and <!-- /OPEN_MODELS_TABLE -->).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import ROOT

# run dir -> (display name, params, how it was run)
MODELS = {
    "lightonocr3-1b-ft-r2": ("**Escriba-OCR-1B (fine-tuned)**", "**1B**", "local"),
    "or-qwen3-vl-235b-a22b-instruct": ("Qwen3-VL 235B-A22B", "235B (22B active)", "OpenRouter"),
    "or-llama-4-maverick": ("Llama 4 Maverick", "400B (17B active)", "OpenRouter"),
    "or-qwen2.5-vl-72b-instruct": ("Qwen2.5-VL 72B", "72B", "OpenRouter"),
    "or-gemma-4-31b-it": ("Gemma 4 31B", "31B", "OpenRouter"),
    "or-qwen3-vl-30b-a3b-instruct": ("Qwen3-VL 30B-A3B", "30B (3B active)", "OpenRouter"),
    "or-qwen3.5-27b": ("Qwen3.5 27B", "27B", "OpenRouter"),
    "or-gemma-3-27b-it": ("Gemma 3 27B", "27B", "OpenRouter"),
    "or-mistral-small-3.2-24b-instruct": ("Mistral Small 3.2", "24B", "OpenRouter"),
    "or-llama-4-scout": ("Llama 4 Scout", "109B (17B active)", "OpenRouter"),
    "or-gemma-3-12b-it": ("Gemma 3 12B", "12B", "OpenRouter"),
    "or-qwen3.5-9b": ("Qwen3.5 9B", "9B", "OpenRouter"),
    "or-qwen3-vl-8b-instruct": ("Qwen3-VL 8B", "8B", "OpenRouter"),
    "or-gemma-3-4b-it": ("Gemma 3 4B", "4B", "OpenRouter"),
    "lightonocr3-4b": ("LightOnOCR-3 4B", "4B", "local (vLLM)"),
    "qwen3.5-4b": ("Qwen3.5 4B", "4B", "local (vLLM)"),
    "qwen3.5-2b": ("Qwen3.5 2B", "2B", "local (vLLM)"),
    "qwen3-vl-2b": ("Qwen3-VL 2B", "2B", "local (vLLM)"),
    "lightonocr3-1b": ("LightOnOCR-3 1B (base)", "1B", "local (vLLM)"),
    "paddleocr-vl-1.6": ("PaddleOCR-VL 1.6", "0.9B", "local (vLLM)"),
    "glm-ocr": ("GLM-OCR", "0.9B", "local (vLLM)"),
    "lightonocr3-0.8b": ("LightOnOCR-3 0.8B", "0.8B", "local (vLLM)"),
}
START, END = "<!-- OPEN_MODELS_TABLE -->", "<!-- /OPEN_MODELS_TABLE -->"


def pct(x):
    return "—" if x is None or x != x else f"{x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=str(ROOT / "models" / "Escriba-OCR-1B" / "README.md"))
    a = ap.parse_args()
    rows = []
    for run, (name, params, how) in MODELS.items():
        f = ROOT / "results" / run / "scores.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        meta = ROOT / "results" / run / "meta.jsonl"
        trunc = 0
        if meta.exists():
            trunc = sum(json.loads(l).get("finish") in ("length", "max_tokens") for l in meta.read_text().splitlines() if l.strip())
        rows.append((s["cer_cap"], name, params, how, s, trunc))
    rows.sort(key=lambda r: r[0])
    lines = [
        "| Model | Params | Run via | CER | WER | Strike recall | Insert recall | Lines | Truncated pages |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cer, name, params, how, s, trunc in rows:
        n = f" ({s['pages']}/199 pages)" if s["pages"] < 199 else ""
        lines.append(f"| {name}{n} | {params} | {how} | {cer:.2f}% | {s['wer_ci']:.1f}% | {pct(s.get('strike_R'))} | "
                     f"{pct(s.get('insert_R'))} | {s['lines_ok']:.0f}% | {trunc} |")
    note = ("\nOpen models were prompted with the same Portuguese error-preserving instructions (local OCR-specialised models "
            "LightOnOCR, PaddleOCR-VL and GLM-OCR with their native prompt, as they do not follow instructions). "
            "OpenRouter models: page upscaled to 1540 px, temperature 0, reasoning disabled, 3,000 max tokens. "
            "Local models: vLLM 0.34 on an NVIDIA L4, page upscaled to 1600 px, greedy decoding, repetition penalty 1.05. "
            "\"Truncated\" counts pages where generation hit the token limit, usually a repetition loop.\n")
    block = START + "\n\n" + "\n".join(lines) + "\n" + note + "\n" + END
    card = Path(a.card).read_text()
    if END in card:
        card = card[: card.index(START)] + block + card[card.index(END) + len(END):]
    else:
        card = card.replace(START, block)
    Path(a.card).write_text(card)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
