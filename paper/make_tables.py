"""LaTeX tables for the Escriba-OCR paper, generated from results/*/scores.json.

uv run python paper/make_tables.py  ->  paper/tables/{main,open}.tex
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.make_card_table import MODELS  # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def s(run):
    return json.loads((ROOT / "results" / run / "scores.json").read_text())["summary"]


def trunc(run):
    m = ROOT / "results" / run / "meta.jsonl"
    if not m.exists():
        return 0
    return sum(json.loads(l).get("finish") in ("length", "max_tokens") for l in m.read_text().splitlines() if l.strip())


def capped(run, key):
    pp = json.loads((ROOT / "results" / run / "scores.json").read_text())["per_page"]
    e = sum(min(v[key][0], v[key][1]) for v in pp.values()); n = sum(v[key][1] for v in pp.values())
    return 100 * e / n


def f(x, nd=1):
    return "--" if x is None or x != x else f"{x:.{nd}f}"


# ---- main comparison (open-weight systems only) ----
MAIN = [
    ("lightonocr3-1b-ft-r2", r"\textbf{Escriba-OCR-1B (ours)}", "1B"),
    ("or-qwen3-vl-235b-a22b-instruct", "Qwen3-VL 235B-A22B", "235B"),
    ("or-qwen3.5-27b", "Qwen3.5 27B", "27B"),
    ("lightonocr3-4b", "LightOnOCR-3 4B", "4B"),
    ("or-qwen2.5-vl-72b-instruct", "Qwen2.5-VL 72B", "72B"),
    ("lightonocr3-1b", "LightOnOCR-3 1B (base)", "1B"),
    ("htr-flor-gtlines", "HTR-Flor (trained)$^\\dagger$", "$<$1M"),
]
rows = []
for run, name, params in MAIN:
    m = s(run)
    sx = f"{f(m.get('strike_x_R'),0)} ({f(m.get('strike_x_P'),0)})"
    rows.append(f"{name} & {params} & {f(m['cer_cap'],2)} & {f(capped(run,'cer_dipl'),2)} & {f(m['cer_text'],2)} & {f(m['wer_ci'])} & "
                f"{f(m.get('strike_R'),0)} & {sx} & {f(m.get('insert_R'),0)} & {f(m['lines_ok'],0)} \\\\")
(OUT / "main.tex").write_text("\n".join(rows) + "\n")

# ---- open / zero-shot models ----
rows = []
items = []
for run, (name, params, how) in MODELS.items():
    if run == "lightonocr3-1b-ft-r2" or not (ROOT / "results" / run / "scores.json").exists():
        continue
    m = s(run)
    items.append((m["cer_cap"], name, params, "API" if how == "OpenRouter" else "local", m, trunc(run)))
for cer, name, params, how, m, t in sorted(items, key=lambda r: r[0]):
    part = f"$^\\ddagger$" if m["pages"] < 199 else ""
    rows.append(f"{name}{part} & {params.replace(' active', '')} & {f(cer,2)} & {f(m['wer_ci'])} & {f(m['err_preserved'])} & "
                f"{f(m.get('insert_R'),0)} & {f(m['lines_ok'],0)} & {t} \\\\")
(OUT / "open.tex").write_text("\n".join(rows) + "\n")
print((OUT / "main.tex").read_text())
print((OUT / "open.tex").read_text())
