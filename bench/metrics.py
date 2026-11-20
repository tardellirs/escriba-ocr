"""CER/WER, student-error preservation and markup precision/recall."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from . import textnorm as tn

DICT = Path(__file__).resolve().parent.parent / "data" / "dict" / "pt_BR"
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.U)


def edit_rate(ref, hyp) -> tuple[int, int]:
    """(edit distance, reference length) — sums across pages give micro-averaged rates."""
    return Levenshtein.distance(ref, hyp), len(ref)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    return edit_rate(ref, hyp)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    return edit_rate(ref.split(), hyp.split())


# ---------- student-error preservation ----------

@lru_cache(maxsize=1)
def _dict():
    from spylls.hunspell import Dictionary

    return Dictionary.from_files(str(DICT))


@lru_cache(maxsize=200_000)
def in_dict(w: str) -> bool:
    d = _dict()
    parts = w.split("-")
    return all(d.lookup(p) or d.lookup(p.lower()) for p in parts if p)


def words(s: str) -> list[str]:
    return _WORD.findall(s)


def _error_view(s: str) -> list[str]:
    """Words for error analysis: markup removed, line-break splits rejoined.

    Hyphenated splits ('rela-\\nção') are always joined; unhyphenated ones
    ('infor\\nmem') are joined only when the concatenation is a dictionary word.
    """
    lines = [words(ln) for ln in tn.plain(tn.dehyphenate(tn.nfc(s)), keep_lines=True).split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln and out and not in_dict(out[-1]) and in_dict(out[-1] + ln[0]):
            out[-1] += ln[0]
            ln = ln[1:]
        out.extend(ln)
    return out


def is_suspect(w: str) -> bool:
    """Likely student misspelling: lowercase-initial, >=3 letters, not in pt_BR dictionary.

    Capitalized words are skipped (proper nouns dominate OOV there).
    """
    return len(w) >= 3 and w[0].islower() and not in_dict(w)


def error_preservation(ref: str, hyp: str) -> Counter:
    """Classify each suspect (non-dictionary) reference word by what the model output at that position.

    preserved: same word (case-insensitive)
    corrected: model produced a *dictionary* word instead (normalized the student's error)
    misread:   model produced another non-dictionary word
    deleted:   no aligned word
    """
    r, h = _error_view(ref), _error_view(hyp)
    rl, hl = [w.lower() for w in r], [w.lower() for w in h]
    c: Counter = Counter()
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(rl, hl):
        for k in range(i1, i2):
            w = r[k]
            if not is_suspect(w):
                continue
            c["suspect"] += 1
            if tag == "equal":
                c["preserved"] += 1
            elif tag == "replace" and (j1 + (k - i1)) < j2:
                hw = h[j1 + (k - i1)]
                c["corrected" if in_dict(hw) else "misread"] += 1
            else:
                c["deleted"] += 1
    # invented errors: reference dictionary words the model turned into non-dictionary words
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(rl, hl):
        if tag != "replace":
            continue
        for k in range(i1, i2):
            w = r[k]
            if len(w) >= 3 and w[0].islower() and in_dict(w):
                c["dict_words_replaced"] += 1
                if (j1 + (k - i1)) < j2 and not in_dict(h[j1 + (k - i1)]):
                    c["invented"] += 1
    c["dict_words"] += sum(1 for w in r if len(w) >= 3 and w[0].islower() and in_dict(w))
    return c


# ---------- markup ----------

def markup_counts(ref: str, hyp: str) -> Counter:
    """Per-kind true positives / ref totals / hyp totals via multiset intersection of (kind, content)."""
    rs, hs = Counter(tn.spans(ref)), Counter(tn.spans(hyp))
    c: Counter = Counter()
    for (kind, _), n in rs.items():
        c[f"{kind}_ref"] += n
    for (kind, _), n in hs.items():
        c[f"{kind}_hyp"] += n
    for key in rs.keys() & hs.keys():
        c[f"{key[0]}_tp"] += min(rs[key], hs[key])
    return c


def page_metrics(ref: str, hyp: str) -> dict:
    hyp = tn.strip_fences(hyp)
    m: dict = {}
    lines = lambda t: "\n".join(tn.collapse(ln) for ln in tn.nfc(t).casefold().split("\n") if ln.strip())  # noqa: E731
    views = (("dipl", tn.diplomatic), ("ci", lambda t: tn.diplomatic(t).casefold()), ("plain", tn.plain), ("text", lambda t: tn.plain(t).casefold()), ("lines", lines))
    for view, f in views:
        r, h = f(ref), f(hyp)
        m[f"cer_{view}"] = cer(r, h)
        m[f"wer_{view}"] = wer(r, h)
    m["n_lines"] = (len([l for l in tn.nfc(ref).split("\n") if l.strip()]), len([l for l in tn.nfc(hyp).split("\n") if l.strip()]))
    m["err"] = error_preservation(ref, hyp)
    m["markup"] = markup_counts(ref, hyp)
    return m
