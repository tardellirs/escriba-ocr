"""Qualitative example block (LaTeX) with line crops and diff-highlighted transcriptions."""
import json, sys
from pathlib import Path
from PIL import Image
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.data import BRESSAY
from bench import textnorm as tn
from bench.data import load_split
SYSR = [("Escriba-OCR-1B", "lightonocr3-1b-ft-r2"), ("Qwen3-VL 235B-A22B", "or-qwen3-vl-235b-a22b-instruct"), ("Qwen2.5-VL 72B", "or-qwen2.5-vl-72b-instruct")]
GT = {p.id: p for p in load_split("test")}
PRED = {r: {f.stem: f.read_text() for f in Path(f"results/{r}/pred").glob("*.txt")} for _, r in SYSR}
def best_line(line, text):
    """Closest output line; if the model wrote the page without line breaks, the closest span instead."""
    c = [l for l in tn.nfc(text).split("\n") if l.strip()]
    if not c: return ""
    b = min(c, key=lambda x: Levenshtein.normalized_distance(line, x))
    if Levenshtein.normalized_distance(line, b) <= 0.5: return b
    flat = " ".join(c); a = fuzz.partial_ratio_alignment(line, flat)
    s, e = a.dest_start, a.dest_end
    while s > 0 and flat[s - 1] != " ": s -= 1  # extend to word boundaries
    while e < len(flat) and flat[e] not in " ": e += 1
    w = flat[s:e].split()
    d = lambda ws: Levenshtein.distance(line, " ".join(ws))
    while len(w) > 1 and d(w[1:]) < d(w): w = w[1:]  # drop words the alignment over-reached
    while len(w) > 1 and d(w[:-1]) < d(w): w = w[:-1]
    return " ".join(w)
pick = [("A", "0248-040", 29, "Edit markup"), ("B", "7870-046", 9, "A preserved misspelling (\\emph{esteriotipando})"), ("C", "7605-061", 2, "A normalised misspelling (\\emph{entreternimento})")]
FIG = Path("paper/figures"); FIG.mkdir(exist_ok=True)
def esc(s):
    rep = {"\\": r"\textbackslash{}", "#": r"\#", "$": r"\$", "%": r"\%", "&": r"\&", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s).replace("--", "-{}-")
def diff(ref, hyp):
    out = []
    for op in Levenshtein.opcodes(ref, hyp):
        seg = hyp[op.dest_start:op.dest_end]
        if op.tag == "equal": out.append(esc(seg))
        elif op.tag in ("replace", "insert"): out.append(r"\err{" + esc(seg) + "}")
        else: out.append(r"\miss{}")
    return "".join(out)
def line_png(pid, idx, name):
    files = sorted((BRESSAY / "data" / "lines" / pid).glob("*.png"), key=lambda p: tuple(int(x) for x in p.stem.split("-")[2:]))
    im = Image.open(files[idx]).convert("L"); f = 4
    im.resize((im.width * f, im.height * f), Image.LANCZOS).save(FIG / name)
    return files[idx].stem
blocks = []
for key, pid, idx, title in pick:
    ref = [l for l in tn.nfc(GT[pid].text).split("\n") if l.strip()][idx]
    stem = line_png(pid, idx, f"ex_{pid}_{idx}.png")
    rows = [("Reference", esc(ref))] + [(name, diff(ref, best_line(ref, PRED[run].get(pid, "")))) for name, run in SYSR]
    body = "\n".join(f"\\textsf{{\\scriptsize {n}}} & \\texttt{{\\scriptsize {t}}} \\\\" for n, t in rows)
    blocks.append(f"\\textbf{{\\small ({chr(97+len(blocks))}) {title}}} \\hfill {{\\scriptsize\\texttt{{{stem}}}}}\\\\[2pt]\n\\includegraphics[width=\\linewidth]{{figures/ex_{pid}_{idx}.png}}\\\\[2pt]\n"
                  f"\\begin{{tabular}}{{@{{}}p{{2.3cm}}p{{\\dimexpr\\linewidth-2.3cm-2\\tabcolsep}}@{{}}}}\n{body}\n\\end{{tabular}}")
Path("paper/tables/examples.tex").write_text("\n\\par\\vspace{8pt}\n".join(blocks) + "\n")
print(Path("paper/tables/examples.tex").read_text()[:1500])
