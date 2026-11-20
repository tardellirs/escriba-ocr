"""BRESSAY page-level loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRESSAY = ROOT / "data" / "raw" / "bressay" / "bressay"
SPLITS = {"train": "training.txt", "val": "validation.txt", "test": "test.txt"}


@dataclass(frozen=True)
class Page:
    id: str
    image: Path
    text: str


def load_split(split: str = "test", root: Path = BRESSAY) -> list[Page]:
    ids = (root / "sets" / SPLITS[split]).read_text().split()
    pages = []
    for pid in ids:
        d = root / "data" / "pages" / pid
        pages.append(Page(pid, d / f"{pid}.png", (d / f"{pid}.txt").read_text(encoding="utf-8")))
    return pages
