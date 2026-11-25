"""Can the image-vs-prior gap flag silent rewrites? Evaluates results/<run>/prior_gap.json.gz (training/prior_gap.py).

Each output word that aligns to a reference word (bench/token_types.py alignment) is labelled
  1  silent rewrite: a different known word, or a different number, than the one on the page
  0  correct (same word, case-insensitive)
Garbled outputs and deletions are left out: the question is whether *clean-looking* errors can be caught.

Word scores (over the word's tokens; lower = more suspicious):
  conf        min log p(token | image)                          plain confidence
  margin      min [log p(token | image) - log p(best other token | image)]
  gap_blank   min [log p(token | image) - log p(token | blank page)]   < 0: the image argued against the token
  gap_none    same against no image at all
Reported: AUROC, and the share of silent rewrites caught when the lowest-scoring 5% / 10% of words are flagged.

    uv run python -m bench.prior_gap_eval [--runs lightonocr3-1b-ft-r2 lightonocr3-1b]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein as L

from bench import textnorm as tn
from bench.data import ROOT, load_split
from bench.token_types import band, kind, tokens, train_freq, well_formed

SCORES = ["conf", "margin", "gap_blank", "gap_none"]


def word_spans(raw: str, words: list[str]) -> list[tuple[int, int] | None]:
    """Locate each plain-text output word in the raw output, left to right (None if not found nearby)."""
    out, cur = [], 0
    for w in words:
        i = raw.find(w, cur, cur + 300)
        if i < 0:
            out.append(None)
            continue
        out.append((i, i + len(w)))
        cur = i + len(w)
    return out


def word_scores(pg: dict, span) -> dict | None:
    s, e = span
    idx = [k for k, (a, b) in enumerate(pg["offsets"]) if a < e and b > s]
    if not idx:
        return None
    f = lambda key: np.array([pg[key][k] for k in idx])  # noqa: E731
    img = f("lp_img")
    return dict(conf=img.min(), margin=(img - f("lp_alt")).min(),
                gap_blank=(img - f("lp_blank")).min(), gap_none=(img - f("lp_none")).min())


def collect(run: str, freq) -> list[dict]:
    data = json.load(gzip.open(ROOT / "results" / run / "prior_gap.json.gz", "rt", encoding="utf-8"))["pages"]
    rows = []
    for p in load_split("test"):
        if p.id not in data:
            continue
        pg = data[p.id]
        raw = pg["text"]
        if tn.nfc(raw) != raw:
            continue
        r = tokens(p.text)
        h = [t for t, _, _ in tokens(raw)]
        spans = word_spans(raw, h)
        for tag, i1, i2, j1, j2 in L.opcodes([t.lower() for t, _, _ in r], [t.lower() for t in h]):
            if tag not in ("equal", "replace"):
                continue
            for k in range(i1, i2):
                j = j1 + (k - i1)
                if j >= j2 or spans[j] is None:
                    continue
                tok, ini, line = r[k]
                kd = kind(tok, ini, line)
                if kd is None:
                    continue
                if tag == "equal":
                    y = 0
                elif well_formed(kd, h[j], freq):
                    y = 1
                else:
                    continue
                sc = word_scores(pg, spans[j])
                if sc:
                    rows.append(dict(page=p.id, ref=tok, hyp=h[j], kind=kd, band=band(freq[tok.lower()]), y=y, **sc))
    return rows


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score of a rewrite < score of a correct word): ties count half."""
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    if not len(pos) or not len(neg):
        return float("nan")
    above = len(neg) - np.searchsorted(neg, pos, side="right")
    ties = np.searchsorted(neg, pos, side="right") - np.searchsorted(neg, pos, side="left")
    return float((above + 0.5 * ties).sum() / (len(pos) * len(neg)))


def caught(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    k = max(1, int(round(frac * len(s))))
    flagged = np.argsort(s, kind="stable")[:k]
    return 100 * y[flagged].sum() / max(y.sum(), 1)


def report(rows: list[dict], title: str) -> dict:
    y = np.array([r["y"] for r in rows])
    out = {"n_words": len(rows), "n_silent": int(y.sum())}
    print(f"\n{title}: {len(rows)} words, {int(y.sum())} silent rewrites ({100 * y.mean():.1f}%)")
    print(f"  {'score':10} {'AUROC':>6} {'caught@5%':>10} {'caught@10%':>11}")
    for sname in SCORES:
        s = np.array([r[sname] for r in rows])
        out[sname] = dict(auroc=auroc(y, s), at5=caught(y, s, 0.05), at10=caught(y, s, 0.10))
        print(f"  {sname:10} {out[sname]['auroc']:6.3f} {out[sname]['at5']:9.1f}% {out[sname]['at10']:10.1f}%")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["lightonocr3-1b-ft-r2", "lightonocr3-1b"])
    ap.add_argument("--json")
    a = ap.parse_args()
    freq = train_freq()
    res = defaultdict(dict)
    for run in a.runs:
        rows = collect(run, freq)
        res[run]["all"] = report(rows, f"{run}: all words")
        for kd in ("number", "proper", "misspelt", "word"):
            sub = [r for r in rows if r["kind"] == kd]
            if sum(r["y"] for r in sub) >= 5:
                res[run][kd] = report(sub, f"{run}: {kd}")
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
