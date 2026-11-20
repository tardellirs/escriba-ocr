"""Dataset statistics + error decomposition + per-page analyses for the paper. Writes paper/analysis/*.json"""
import json, re, statistics, unicodedata, sys
from collections import Counter
from pathlib import Path
from PIL import Image
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.data import load_split, BRESSAY
from bench import textnorm as tn, metrics as M

OUT = Path(__file__).parent
# ---------- dataset statistics ----------
st = {}
for split in ("train", "val", "test"):
    pages = load_split(split)
    lines = words = chars = 0; mk = Counter(); widths = []; heights = []; sus = 0
    for p in pages:
        t = tn.nfc(p.text)
        ls = [l for l in t.split("\n") if l.strip()]
        lines += len(ls); words += len(tn.plain(t).split()); chars += len(tn.diplomatic(t))
        for k, _ in tn.spans(t): mk[k] += 1
        w, h = Image.open(p.image).size; widths.append(w); heights.append(h)
        sus += sum(1 for w_ in M._error_view(p.text) if M.is_suspect(w_))
    st[split] = dict(pages=len(pages), lines=lines, words=words, chars=chars, lines_per_page=round(lines/len(pages),1),
                     words_per_page=round(words/len(pages),1), chars_per_page=round(chars/len(pages)), markup=dict(mk),
                     img_w_median=statistics.median(widths), img_h_median=statistics.median(heights), suspect_words=sus,
                     px_per_line=round(statistics.median(heights)/(lines/len(pages)),1))
(OUT / "dataset_stats.json").write_text(json.dumps(st, indent=1))
print(json.dumps(st, indent=1))

# ---------- error decomposition ----------
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
PUNCT = re.compile(r"[^\w\s]", re.U)
def views(t):
    d = tn.diplomatic(t).casefold()
    return d
SYS = {"escriba": "lightonocr3-1b-ft-r2", "qwen235": "or-qwen3-vl-235b-a22b-instruct", "qwen27": "or-qwen3.5-27b",
       "qwen72": "or-qwen2.5-vl-72b-instruct", "lo4b": "lightonocr3-4b", "base": "lightonocr3-1b", "htr": "htr-flor-gtlines"}
gt = {p.id: p for p in load_split("test")}
dec = {}; perpage = {}
for k, run in SYS.items():
    preds = {f.stem: f.read_text(encoding="utf-8") for f in (Path("results") / run / "pred").glob("*.txt")}
    tot = Counter(); pp = {}
    for pid, p in gt.items():
        r, h = views(p.text), views(tn.strip_fences(preds.get(pid, "")))
        e = Levenshtein.distance(r, h); n = len(r)
        ops = Counter(op.tag for op in Levenshtein.editops(r, h)) if e < 5000 else Counter()
        e_noacc = Levenshtein.distance(strip_acc(r), strip_acc(h))
        e_nopunct = Levenshtein.distance(PUNCT.sub("", r), PUNCT.sub("", h))
        rc, hc = tn.diplomatic(p.text), tn.diplomatic(tn.strip_fences(preds.get(pid, "")))
        e_case = Levenshtein.distance(rc, hc)
        tot["n"] += n; tot["e"] += min(e, n); tot["e_noacc"] += min(e_noacc, n); tot["e_nopunct"] += min(e_nopunct, n)
        tot["e_case"] += min(e_case, n)
        for t_, c in ops.items(): tot[t_] += c
        pp[pid] = round(100 * min(e, n) / n, 2)
    dec[k] = {"cer": round(100*tot["e"]/tot["n"], 2), "cer_case": round(100*tot["e_case"]/tot["n"], 2),
              "cer_no_accents": round(100*tot["e_noacc"]/tot["n"], 2), "cer_no_punct": round(100*tot["e_nopunct"]/tot["n"], 2),
              "sub": tot["replace"], "ins": tot["insert"], "del": tot["delete"]}
    perpage[k] = pp
(OUT / "error_decomposition.json").write_text(json.dumps(dec, indent=1))
(OUT / "perpage_cer.json").write_text(json.dumps(perpage))
print(json.dumps(dec, indent=1))
