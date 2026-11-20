"""Paired page-level bootstrap between two scored runs (scores.json written by bench.evaluate).

uv run python -m bench.compare results/finetune/val154/A@final results/finetune/val154/B@final [--n 2000]
Reports micro-averaged CER(ci) and error-preservation for both, the difference B-A and a 95% CI.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(d):
    return json.loads((Path(d) / "scores.json").read_text())["per_page"]


METRIC = "cer_ci"


def agg(pp, ids):
    e = sum(min(pp[i][METRIC][0], pp[i][METRIC][1]) for i in ids)  # capped per page
    n = sum(pp[i][METRIC][1] for i in ids)
    pres = sum(pp[i]["err"].get("preserved", 0) for i in ids)
    sus = sum(pp[i]["err"].get("suspect", 0) for i in ids)
    return 100 * e / n, (100 * pres / sus if sus else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--metric", default="cer_ci", help="cer_ci | cer_dipl (case-sensitive) | cer_plain (markup-free)")
    x = ap.parse_args()
    global METRIC
    METRIC = x.metric
    A, Bp = load(x.a), load(x.b)
    ids = sorted(set(A) & set(Bp))
    ca, pa = agg(A, ids)
    cb, pb = agg(Bp, ids)
    rng = random.Random(0)
    dc, dp = [], []
    for _ in range(x.n):
        s = [rng.choice(ids) for _ in ids]
        a1, a2 = agg(A, s)
        b1, b2 = agg(Bp, s)
        dc.append(b1 - a1)
        dp.append(b2 - a2)
    dc.sort()
    dp.sort()
    lo, hi = int(0.025 * x.n), int(0.975 * x.n) - 1
    print(f"pages={len(ids)}")
    print(f"CERcap  A={ca:.2f}  B={cb:.2f}  B-A={cb - ca:+.2f}  95%CI [{dc[lo]:+.2f}, {dc[hi]:+.2f}]")
    print(f"preserv A={pa:.1f}  B={pb:.1f}  B-A={pb - pa:+.1f}  95%CI [{dp[lo]:+.1f}, {dp[hi]:+.1f}]")


if __name__ == "__main__":
    main()
