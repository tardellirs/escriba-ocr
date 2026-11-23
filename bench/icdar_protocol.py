"""Score page transcriptions with the metric used for published BRESSAY results.

The ICDAR 2024 BRESSAY competition ranks systems by case-sensitive CER/WER with the
edit markup kept and no per-page cap. Its scoring script is not public, so we compute
the plausible variants and report all of them:
  - line breaks kept as "\n" or replaced by spaces;
  - micro-averaged (total edits / total reference characters) or averaged over pages;
  - markup kept verbatim or mapped to single symbols, as in the BRESSAY authors'
    toolkit (github.com/arthurflor23/handwritten-text-recognition, sarah/data/source/bressay.py).

Usage: uv run python -m bench.icdar_protocol results/<run> [results/<run> ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from . import textnorm as tn
from .data import load_split

_MAP = [
    (r"##@@\?\?\?@@##", "↑◬↑"), (r"\$\$@@\?\?\?@@\$\$", "↓◬↓"), (r"@@\?\?\?@@", "◬"),
    (r"##--xxx--##", "↑◬↑"), (r"\$\$--xxx--\$\$", "↓◬↓"), (r"--xxx--", "◬"),
    (r"##--(.*?)--##", "↑↔{}↔↑"), (r"\$\$--(.*?)--\$\$", "↓↔{}↔↓"),
    (r"##(.*?)##", "↑{}↑"), (r"\$\$(.*?)\$\$", "↓{}↓"), (r"--(.*?)--", "↔{}↔"),
]


def symbols(t: str) -> str:
    for rx, rep in _MAP:
        t = re.sub(rx, lambda m, rep=rep: rep.format(*m.groups()) if m.groups() else rep, t)
    return t


def lines(t: str) -> str:
    t = unicodedata.normalize("NFC", t).replace("\r\n", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n") if ln.strip())


def flat(t: str) -> str:
    return re.sub(r"\s+", " ", lines(t)).strip()


VIEWS = {
    "lines": lines,
    "flat": flat,
    "symbols": lambda t: symbols(lines(t)),
}


def score(run_dir: Path) -> dict:
    pages = load_split("test")
    out = {}
    for name, f in VIEWS.items():
        d_sum = n_sum = dw_sum = nw_sum = 0
        per_page = []
        for pg in pages:
            hyp = tn.strip_fences((run_dir / "pred" / f"{pg.id}.txt").read_text(encoding="utf-8"))
            r, h = f(pg.text), f(hyp)
            d = Levenshtein.distance(r, h)
            d_sum += d
            n_sum += len(r)
            per_page.append(d / max(1, len(r)))
            dw_sum += Levenshtein.distance(r.split(), h.split())
            nw_sum += len(r.split())
        out[name] = {
            "cer_micro": 100 * d_sum / n_sum,
            "cer_page_mean": 100 * statistics.mean(per_page),
            "wer_micro": 100 * dw_sum / nw_sum,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = {str(r): score(r) for r in a.runs}
    for run, views in res.items():
        print(run)
        for v, m in views.items():
            print(f"  {v:8s} CER {m['cer_micro']:.2f} (page mean {m['cer_page_mean']:.2f})  WER {m['wer_micro']:.2f}")
    main_cer = [v["lines"]["cer_micro"] for v in res.values()]
    main_wer = [v["lines"]["wer_micro"] for v in res.values()]
    if len(main_cer) > 1:
        print(f"mean ± SD over runs: CER {statistics.mean(main_cer):.2f} ± {statistics.stdev(main_cer):.2f}, "
              f"WER {statistics.mean(main_wer):.2f} ± {statistics.stdev(main_wer):.2f}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
