"""Text views over BRESSAY-style transcriptions.

- diplomatic(): keeps BRESSAY markup, collapses whitespace (line breaks -> space).
- plain():      removes struck-through and illegible spans, unwraps insertions.
- dehyphenate(): joins words split by an end-of-line hyphen (used only for word-level error analysis).
- spans():      extracts marked spans as (kind, content) for markup precision/recall.
"""
from __future__ import annotations

import re
import unicodedata

# Order matters: insertions may contain strike/illegible markup.
_INS = re.compile(r"(##|\$\$)(.*?)\1", re.S)
_STRIKE = re.compile(r"--(.+?)--", re.S)
_ILLEG = re.compile(r"@@\?*@@")
_WS = re.compile(r"\s+")


# Visually indistinguishable in handwriting -> same symbol.
_GLYPHS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "°": "º", "–": "-", "—": "-", "…": "..."})


def nfc(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    return s.translate(_GLYPHS)


def strip_fences(s: str) -> str:
    """Remove ``` fences a model may wrap the answer in."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s


def collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def diplomatic(s: str) -> str:
    return collapse(nfc(s))


def plain(s: str, keep_lines: bool = False) -> str:
    s = nfc(s)
    s = _INS.sub(lambda m: " " + m.group(2) + " ", s)  # unwrap insertions
    s = _STRIKE.sub(" ", s)  # drop struck-through (incl. --xxx--)
    s = _ILLEG.sub(" ", s)  # drop illegible
    s = s.replace("##", " ").replace("$$", " ")  # stray markers
    if keep_lines:
        return "\n".join(_WS.sub(" ", ln).strip() for ln in s.split("\n"))
    return collapse(s)


def dehyphenate(s: str) -> str:
    """'con-\\nceito' -> 'conceito' (expects line breaks preserved)."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)


def spans(s: str) -> list[tuple[str, str]]:
    """Marked spans as (kind, normalized content).

    kinds: strike (legible), strike_x (illegible strike), insert, illegible.
    Strikes/illegibles nested in an insertion count once with their own kind.
    """
    s = nfc(s)
    out: list[tuple[str, str]] = []
    for m in _INS.finditer(s):
        inner = m.group(2)
        if _STRIKE.fullmatch(inner.strip()) or _ILLEG.fullmatch(inner.strip()):
            continue  # counted below as strike/illegible
        out.append(("insert", collapse(inner).lower()))
    for m in _STRIKE.finditer(s):
        c = collapse(m.group(1)).lower()
        out.append(("strike_x", "") if c == "xxx" else ("strike", c))
    for _ in _ILLEG.finditer(s):
        out.append(("illegible", ""))
    return out
