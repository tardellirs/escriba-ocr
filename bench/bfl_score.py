"""Score the BFL out-of-domain test (40 letters, one reviewer) against both reference versions.

References: bfl/references/pass1/ (first review, done before any model output was seen) and
bfl/references/adjudicated/ (after adjudicating the points where both models agreed against the reference,
plus every markup span Escriba produced). Predictions: results/bfl/<run>/pred/*.txt.

Normalization, applied to reference and hypothesis alike:
- line-end hyphenation is undone in the hypothesis (references have no line breaks); a split
  before a clitic pronoun keeps its hyphen ("candidatar-\\nme" -> "candidatar-me");
- "S. A" and "S.A" are the same (the gap is not readable in handwriting; left out of the review).
CER/WER as in the paper: diplomatic view with markup, whitespace collapsed, case-insensitive,
micro-averaged over characters with each page capped at 100%.

    uv run python -m bench.bfl_score [--json results/bfl/scores.json]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from bench import textnorm as tn
from bench.metrics import cer, wer

ROOT = Path(__file__).resolve().parents[1]
IDS = (ROOT / "bfl/sample40.txt").read_text().split()
REFS = {"pass1": ROOT / "bfl/references/pass1", "adjudicated": ROOT / "bfl/references/adjudicated"}
RUNS = ["escriba-ocr-1b", "lightonocr3-1b-base"]

_CLITIC = r"(?:me|te|se|lhe|lhes|nos|vos|o|a|os|as|lo|la|los|las)\b"
_SA = re.compile(r"\bS\.\s*A\b")


_NOTE = re.compile(r"\n\s*[*_]*Note\b.*", re.S)


def norm_hyp(s: str, strip_notes: bool = False) -> str:
    """strip_notes drops the English comment the base model appends ("Note: The handwritten text..."),
    a post-hoc correction that favours the base model; Escriba never appends one."""
    s = tn.nfc(tn.strip_fences(s))
    if strip_notes:
        s = _NOTE.sub("", s)
    s = re.sub(rf"(\w)-[ \t]*\n\s*({_CLITIC})", r"\1-\2", s)
    s = re.sub(r"(\w)-[ \t]*\n\s*(\w)", r"\1\2", s)
    return _SA.sub("S.A", s)


def norm_ref(s: str) -> str:
    return _SA.sub("S.A", tn.nfc(s))


def spans(s: str) -> Counter:
    return Counter(k for k, _ in tn.spans(s))


def page_cer(run: str, ref_dir: Path, strip_notes: bool = False) -> list[tuple[int, int]]:
    out = []
    for i in IDS:
        r = norm_ref((ref_dir / f"{i}.txt").read_text(encoding="utf-8"))
        h = norm_hyp((ROOT / "results/bfl" / run / "pred" / f"{i}.txt").read_text(encoding="utf-8"), strip_notes)
        d, n = cer(tn.diplomatic(r).casefold(), tn.diplomatic(h).casefold())
        out.append((min(d, n), n))
    return out


def paired_bootstrap(a: list, b: list, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference in micro CER (a - b, points) with a 95% page-level bootstrap interval."""
    rng = random.Random(seed)
    k = len(a)
    diff = lambda idx: 100 * (sum(a[j][0] for j in idx) / sum(a[j][1] for j in idx) - sum(b[j][0] for j in idx) / sum(b[j][1] for j in idx))  # noqa: E731
    ds = sorted(diff([rng.randrange(k) for _ in range(k)]) for _ in range(n_boot))
    return round(diff(range(k)), 2), round(ds[int(0.025 * n_boot)], 2), round(ds[int(0.975 * n_boot) - 1], 2)


def score(run: str, ref_dir: Path, strip_notes: bool = False) -> dict:
    c = [0, 0]; w = [0, 0]; t = [0, 0]; per = []
    tp = fp = fn = 0
    for i in IDS:
        r = norm_ref((ref_dir / f"{i}.txt").read_text(encoding="utf-8"))
        h = norm_hyp((ROOT / "results/bfl" / run / "pred" / f"{i}.txt").read_text(encoding="utf-8"), strip_notes)
        rd, hd = tn.diplomatic(r).casefold(), tn.diplomatic(h).casefold()
        d, n = cer(rd, hd); d = min(d, n); c[0] += d; c[1] += n; per.append(d / n)
        d, n = wer(rd, hd); w[0] += min(d, n); w[1] += n
        d, n = cer(tn.plain(r).casefold(), tn.plain(h).casefold()); t[0] += min(d, n); t[1] += n
        rs, hs = spans(r), spans(h)  # markup counted per page and kind
        tp += sum((rs & hs).values()); fp += sum((hs - rs).values()); fn += sum((rs - hs).values())
    return dict(cer=round(100 * c[0] / c[1], 2), wer=round(100 * w[0] / w[1], 2), cer_text=round(100 * t[0] / t[1], 2),
                cer_page_median=round(100 * statistics.median(per), 2),
                markup=dict(ref_spans=tp + fn, predicted=tp + fp, matched=tp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = {}
    for ref, d in REFS.items():
        out[ref] = {run: score(run, d) for run in RUNS}
        out[ref]["lightonocr3-1b-base (comments removed)"] = score(RUNS[1], d, strip_notes=True)
        e, b = page_cer(RUNS[0], d), page_cer(RUNS[1], d, strip_notes=True)
        delta, lo, hi = paired_bootstrap(e, b)
        out[ref]["escriba_minus_base_without_comments"] = dict(
            delta=delta, ci95=[lo, hi], escriba_better=sum(x[0] < y[0] for x, y in zip(e, b)),
            escriba_worse=sum(x[0] > y[0] for x, y in zip(e, b)))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
