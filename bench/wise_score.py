"""Score the WiSE-FT sweep (bench/colab/run_wise.py): BRESSAY val/test with the paper's metrics, BFL with bench.bfl_score.

alpha* = largest alpha whose BRESSAY validation CER is within 0.10 point of alpha = 0 (rule fixed before the run).

    uv run python -m bench.wise_score [--dir results/wise] [--json results/wise/scores.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from bench import bfl_score as B
from bench import textnorm as tn
from bench.evaluate import score as bressay_score
from bench.metrics import cer, wer

TOL = 0.10
_TRAIN_VOCAB: Counter | None = None


def train_vocab() -> Counter:
    global _TRAIN_VOCAB
    if _TRAIN_VOCAB is None:
        base = B.ROOT / "data/raw/bressay/bressay"
        ids = set((base / "sets/training.txt").read_text().split())
        _TRAIN_VOCAB = Counter()
        for d in (base / "data/pages").iterdir():
            if d.name in ids:
                for f in d.glob("*.txt"):
                    _TRAIN_VOCAB.update(re.findall(r"\w+(?:-\w+)*", f.read_text(encoding="utf-8").casefold()))
    return _TRAIN_VOCAB


def _key(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w)


def bfl(pred_dir: Path) -> dict:
    """CER/WER against the adjudicated BFL references, and word error split by BRESSAY-training vocabulary."""
    V = train_vocab()
    c = [0, 0]; w = [0, 0]; words = {"seen": [0, 0], "unseen": [0, 0]}; notes = 0
    for i in B.IDS:
        ref = B.norm_ref((B.REFS["adjudicated"] / f"{i}.txt").read_text(encoding="utf-8"))
        raw = (pred_dir / f"{i}.txt").read_text(encoding="utf-8")
        notes += bool(re.search(r"\n\s*[*_]*Note\b", raw))
        hyp = B.norm_hyp(raw)
        rd, hd = tn.diplomatic(ref).casefold(), tn.diplomatic(hyp).casefold()
        d, n = cer(rd, hd); c[0] += min(d, n); c[1] += n
        d, n = wer(rd, hd); w[0] += min(d, n); w[1] += n
        r = [_key(x) for x in tn.plain(ref).casefold().split()]
        h = [_key(x) for x in tn.plain(hyp).casefold().split()]
        ok = set()
        for a, _, n in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks():
            ok.update(range(a, a + n))
        for j, k in enumerate(r):
            if k and not k.isdigit():
                g = "seen" if V[k] else "unseen"
                words[g][1] += 1; words[g][0] += j not in ok
    return dict(cer=round(100 * c[0] / c[1], 2), wer=round(100 * w[0] / w[1], 2), pages_with_note=notes,
                wer_seen=round(100 * words["seen"][0] / words["seen"][1], 1),
                wer_unseen=round(100 * words["unseen"][0] / words["unseen"][1], 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=B.ROOT / "results/wise")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = {}
    for d in sorted(a.dir.glob("a*/")):
        alpha = float(d.name[1:])
        row = {"alpha": alpha}
        for split in ("val", "test"):
            s = bressay_score(f"{d.name}/{split}", split, res=a.dir)
            row[split] = {k: round(s[k], 2) for k in ("cer_cap", "cer_text", "wer_ci", "lines_ok", "err_preserved",
                                                       "err_corrected", "strike_R", "strike_x_R", "insert_R", "illegible_R")}
        row["bfl"] = bfl(d / "bfl" / "pred")
        rows[alpha] = row
    ref = rows[0.0]["val"]["cer_cap"]
    star = max(al for al, r in rows.items() if r["val"]["cer_cap"] <= ref + TOL)
    print(f"{'alpha':>5} | {'val CER':>7} | {'test CER':>8} | {'test pres':>9} | {'strike R':>8} | {'illeg R':>7} | {'BFL CER':>7} | {'BFL seen':>8} | {'BFL unseen':>10} | notes")
    for al, r in sorted(rows.items()):
        v, t, b = r["val"], r["test"], r["bfl"]
        print(f"{al:5.1f}{'*' if al == star else ' '}| {v['cer_cap']:7.2f} | {t['cer_cap']:8.2f} | {t['err_preserved']:9.1f} | {t['strike_R']:8.1f} | "
              f"{t['illegible_R']:7.1f} | {b['cer']:7.2f} | {b['wer_seen']:8.1f} | {b['wer_unseen']:10.1f} | {b['pages_with_note']}")
    print(f"alpha* = {star} (val CER within {TOL} of {ref})")
    if a.json:
        a.json.write_text(json.dumps(dict(alpha_star=star, tol=TOL, rows=list(rows.values())), indent=1) + "\n")


if __name__ == "__main__":
    main()
