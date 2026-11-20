"""Convert HTR-Flor line-level evaluations.json into page-level predictions for bench.evaluate.

uv run python -m bench.htr_to_pages <evaluations.json> <run_name> [--latency-s-per-line X]

Line images are named <page>-<paragraph>-<line>.png; lines are ordered by name and a blank
line is inserted between paragraphs (same layout as BRESSAY page transcriptions).
NOTE: uses ground-truth line segmentation (an advantage over page-level models).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .data import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("evaluations")
ap.add_argument("run")
ap.add_argument("--info", help="JSON string merged into run_info.json (e.g. gpu, pages_per_s)")
a = ap.parse_args()

ev = json.loads(Path(a.evaluations).read_text())
items = ev["data"] if isinstance(ev, dict) else ev
pages: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
for it in items:
    name = Path(it["image"]).stem
    m = re.match(r"(\d{4}-\d{3})-(\d+)-(\d+)$", name)
    if not m:
        continue
    pred = it["predictions"][0]
    pred = pred["text"] if isinstance(pred, dict) else pred
    pages[m.group(1)].append((m.group(2), m.group(3), pred))

out = ROOT / "results" / a.run / "pred"
out.mkdir(parents=True, exist_ok=True)
for pid, lines in pages.items():
    lines.sort(key=lambda x: (int(x[0]), int(x[1])))
    buf, prev = [], None
    for para, _, text in lines:
        if prev is not None and para != prev:
            buf.append("")
        buf.append(text)
        prev = para
    (out / f"{pid}.txt").write_text("\n".join(buf), encoding="utf-8")
if a.info:
    (out.parent / "run_info.json").write_text(json.dumps(json.loads(a.info), indent=1))
print(f"{len(pages)} pages written to {out}")
