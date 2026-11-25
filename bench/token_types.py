"""Where do silent rewrites fall? Word-level outcomes on BRESSAY test, by token type and training frequency.

Each reference token (markup removed, line-end hyphenation undone) is aligned to the model output
(case-insensitive word alignment) and classed as kept, substituted or deleted. A substitution is
*silent* when the output is itself a known form: a dictionary word or a word of the training
transcriptions, or, for numbers, another number. Nothing downstream (spell checker, format check) flags those.

Token types (reference side):
  number   contains a digit
  proper   capitalised, not sentence-initial, not the first line (titles), lowercase form not in the dictionary
  misspelt lowercase, >=3 letters, not in the pt_BR dictionary (the paper's "likely misspellings")
  word     lowercase, >=3 letters, in the dictionary
Frequency band of the lowercase form in the 647 training transcriptions: frequent (>=10), rare (1-9), unseen (0).

    uv run python -m bench.token_types [--runs RUN ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein as L

from bench import textnorm as tn
from bench.data import ROOT, load_split
from bench.metrics import in_dict

RUNS = {
    "escriba": "lightonocr3-1b-ft-r2",
    "lighton-base": "lightonocr3-1b",
    "qwen3-vl-235b": "or-qwen3-vl-235b-a22b-instruct",
    "qwen3.5-27b": "or-qwen3.5-27b",
}
_TOK = re.compile(r"\w+(?:[-./,]\w+)*", re.U)


def tokens(s: str) -> list[tuple[str, bool, int]]:
    """(token, sentence_initial, line_no) over the plain text, hyphenation undone."""
    s = tn.plain(tn.dehyphenate(tn.nfc(tn.strip_fences(s))), keep_lines=True)
    out, prev = [], "."
    for ln_no, ln in enumerate(l for l in s.split("\n") if l.strip()):
        pos = 0
        for m in _TOK.finditer(ln):
            gap = ln[pos:m.start()]
            punct = [c for c in gap if not c.isspace()]
            if punct:
                prev = punct[-1]
            out.append((m.group(), prev in ".!?:", ln_no))
            prev, pos = "w", m.end()
    return out


def kind(tok: str, initial: bool, line: int) -> str | None:
    if any(c.isdigit() for c in tok):
        return "number"
    if tok[0].isupper():
        if not initial and line > 0 and not in_dict(tok.lower()):
            return "proper"
        return None
    if len(tok) < 3:
        return None
    return "word" if in_dict(tok) else "misspelt"


def well_formed(k: str, h: str, freq: Counter) -> bool:
    if k == "number":
        return any(c.isdigit() for c in h)
    return in_dict(h) or freq[h.lower()] > 0


def train_freq() -> Counter:
    c: Counter = Counter()
    for p in load_split("train"):
        c.update(t.lower() for t, _, _ in tokens(p.text))
    return c


def band(n: int) -> str:
    return "frequent" if n >= 10 else "rare" if n else "unseen"


def page_counts(ref: str, hyp: str, freq: Counter) -> Counter:
    r, h = tokens(ref), [t for t, _, _ in tokens(hyp)]
    c: Counter = Counter()
    rl, hl = [t.lower() for t, _, _ in r], [t.lower() for t in h]
    for tag, i1, i2, j1, j2 in L.opcodes(rl, hl):
        for k in range(i1, i2):
            tok, ini, line = r[k]
            kd = kind(tok, ini, line)
            if kd is None:
                continue
            groups = [kd]
            if kd in ("word", "proper"):
                groups.append(f"{kd}:{band(freq[tok.lower()])}")
            if tag == "equal":
                out = "kept"
            elif tag == "replace" and j1 + (k - i1) < j2:
                hw = h[j1 + (k - i1)]
                silent = well_formed(kd, hw, freq)
                near = L.distance(tok.lower(), hw.lower()) <= 2
                out = "silent" + ("_near" if near else "_far") if silent else "garbled"
            else:
                out = "deleted"
            for g in groups:
                c[(g, "n")] += 1
                c[(g, out)] += 1
    return c


def boot_ci(pages: list[Counter], key, num_outs, rng, B=2000):
    n = np.array([p[(key, "n")] for p in pages], float)
    x = np.array([sum(p[(key, o)] for o in num_outs) for p in pages], float)
    if n.sum() == 0:
        return None
    idx = rng.integers(0, len(pages), (B, len(pages)))
    bs = x[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return 100 * x.sum() / n.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), int(n.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument("--json")
    a = ap.parse_args()
    freq = train_freq()
    test = load_split("test")
    rng = np.random.default_rng(0)
    groups = ["number", "proper", "proper:frequent", "proper:rare", "proper:unseen", "misspelt",
              "word", "word:frequent", "word:rare", "word:unseen"]
    res = defaultdict(dict)
    for name in a.runs:
        pred = ROOT / "results" / RUNS.get(name, name) / "pred"
        pages = [page_counts(p.text, (pred / f"{p.id}.txt").read_text(encoding="utf-8"), freq) for p in test]
        for g in groups:
            silent = boot_ci(pages, g, ["silent_near", "silent_far"], rng)
            if silent is None:
                continue
            res[name][g] = {
                "n": silent[3],
                "silent": silent[:3],
                "silent_near": boot_ci(pages, g, ["silent_near"], rng)[:3],
                "any_error": boot_ci(pages, g, ["silent_near", "silent_far", "garbled", "deleted"], rng)[:3],
            }
    print(f"{'group':18}" + "".join(f"{n:>26}" for n in a.runs))
    for g in groups:
        row = [res[n].get(g) for n in a.runs]
        if not any(row):
            continue
        cells = [f"{r['silent'][0]:5.1f} [{r['silent'][1]:4.1f},{r['silent'][2]:5.1f}] /{r['any_error'][0]:5.1f}" if r else "" for r in row]
        print(f"{g + ' (n=' + str(row[0]['n']) + ')':18}" + "".join(f"{c:>26}" for c in cells))
    print("cells: silent-rewrite % [95% CI] / any-error %")
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
