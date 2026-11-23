"""Points re-examined in the second BFL review pass (bfl_review.py shows them next to each page).

A point is a word span where the first-pass reference differs from Escriba's output and either
- both models read the span the same way (case-insensitive), or
- Escriba's reading contains BRESSAY markup (a possible deletion or illegible word).
Differences only in spacing, and "S. A." against "S.A.", are left out. The review tool does not say
which model produced a reading.

    uv run python -m bench.bfl_points [--out bfl/review_points.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rapidfuzz.distance import Levenshtein as L

from bench import textnorm as tn
from bench.bfl_score import IDS, REFS, ROOT

RUNS = ROOT / "results/bfl"
_MK = re.compile(r"--.+?--|##.+?##|@@\?*@@")


def norm(t: str) -> str:
    return tn.diplomatic(tn.dehyphenate(tn.nfc(tn.strip_fences(t))))


def diffs(ref: str, hyp: str) -> dict[tuple[int, int], str]:
    rw, hw = ref.split(), hyp.split()
    out = {}
    for op in L.opcodes([w.casefold() for w in rw], [w.casefold() for w in hw]):
        if op.tag != "equal":
            out[(op.src_start, op.src_end)] = " ".join(hw[op.dest_start:op.dest_end])
    return out


def points(ref_dir: Path = REFS["pass1"]) -> dict:
    pts = {}
    for i in IDS:
        r = norm((ref_dir / f"{i}.txt").read_text(encoding="utf-8"))
        rw = r.split()
        e = diffs(r, norm((RUNS / "escriba-ocr-1b/pred" / f"{i}.txt").read_text(encoding="utf-8")))
        b = diffs(r, norm((RUNS / "lightonocr3-1b-base/pred" / f"{i}.txt").read_text(encoding="utf-8")))
        items = []
        for k, v in e.items():
            refw = " ".join(rw[k[0]:k[1]])
            ctx = " ".join(rw[max(0, k[0] - 3):k[1] + 3])
            if refw.startswith("S.") and v.upper().startswith("S.A"):
                continue
            if refw.replace(" ", "").casefold() == v.replace(" ", "").casefold():
                continue
            if _MK.search(v):
                items.append({"ref": refw or "(nada)", "alt": v, "ctx": ctx, "why": "marcação"})
            elif k in b and b[k].casefold() == v.casefold():
                items.append({"ref": refw or "(nada)", "alt": v or "(nada)", "ctx": ctx, "why": "leitura"})
        if items:
            pts[i] = items
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    pts = points()
    kinds = [x["why"] for v in pts.values() for x in v]
    print(f"{len(kinds)} points on {len(pts)} pages: {kinds.count('leitura')} readings, {kinds.count('marcação')} markup spans")
    if a.out:
        a.out.write_text(json.dumps(pts, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
