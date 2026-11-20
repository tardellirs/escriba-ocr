"""Score runs in results/ against BRESSAY ground truth.

uv run python -m bench.evaluate [run ...]   (default: all runs)
Writes results/summary.md and results/<run>/scores.json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter

from .data import ROOT, load_split
from .metrics import page_metrics

RES = ROOT / "results"
KINDS = ["strike", "strike_x", "insert", "illegible"]
PRICES: dict = {}  # USD per 1M tokens (input, output) for API runs without reported cost


def pct(n, d):
    return 100 * n / d if d else float("nan")


def score(run: str, split: str = "test", res=None) -> dict:
    d = (res or RES) / run
    gt = {p.id: p for p in load_split(split)}
    preds = {f.stem: f.read_text(encoding="utf-8") for f in (d / "pred").glob("*.txt") if f.stem in gt}
    meta = {}
    if (d / "meta.jsonl").exists():
        for line in (d / "meta.jsonl").read_text().splitlines():
            r = json.loads(line)
            meta[r["page"]] = r
    sums: Counter = Counter()
    err: Counter = Counter()
    mk: Counter = Counter()
    per_page = {}
    for pid, hyp in preds.items():
        m = page_metrics(gt[pid].text, hyp)
        for k in ("cer_dipl", "wer_dipl", "cer_ci", "wer_ci", "cer_plain", "wer_plain", "cer_lines", "wer_lines"):
            sums[k + "_e"] += m[k][0]
            sums[k + "_n"] += m[k][1]
        err += m["err"]
        mk += m["markup"]
        per_page[pid] = {"cer_ci": list(m["cer_ci"]), "cer_dipl": list(m["cer_dipl"]), "cer_lines": list(m["cer_lines"]),
                         "cer_plain": list(m["cer_plain"]), "cer_text": list(m["cer_text"]),
                         "err": dict(m["err"]), "markup": dict(m["markup"]), "n_lines": list(m["n_lines"])}
        sums["lines_ok"] += int(m["n_lines"][0] == m["n_lines"][1])
        # capped CER: a page counts at most 100% (one repetition loop can't dominate the micro-average)
        sums["cer_text_e"] += min(m["cer_text"][0], m["cer_text"][1])
        sums["cer_text_n"] += m["cer_text"][1]
        sums["cer_cap_e"] += min(m["cer_ci"][0], m["cer_ci"][1])
        sums["cer_cap_n"] += m["cer_ci"][1]
    lat = [meta[p]["latency_s"] for p in preds if p in meta and meta[p].get("latency_s") is not None]
    out_tok = [meta[p]["output_tokens"] + meta[p]["thinking_tokens"] for p in preds if p in meta]
    in_tok = [meta[p]["prompt_tokens"] for p in preds if p in meta]
    price = PRICES.get(next(iter(meta.values()))["model"]) if meta else None
    cost = price and (sum(in_tok) * price[0] + sum(out_tok) * price[1]) / 1e6 / len(in_tok) * 1000
    reported = [meta[p]["cost_usd"] for p in preds if p in meta and meta[p].get("cost_usd") is not None]
    if reported:  # OpenRouter returns the real billed cost per call
        cost = sum(reported) / len(reported) * 1000
    info = json.loads((d / "run_info.json").read_text()) if (d / "run_info.json").exists() else {}
    s = dict(
        pages_per_s=info.get("pages_per_s"), gpu=info.get("gpu"),
        run=run, pages=len(preds), of=len(gt),
        **{k: pct(sums[k + "_e"], sums[k + "_n"]) for k in ("cer_dipl", "wer_dipl", "cer_ci", "wer_ci", "cer_plain", "wer_plain", "cer_lines", "wer_lines")},
        invented=pct(err["invented"], err["dict_words"]),
        cer_cap=pct(sums["cer_cap_e"], sums["cer_cap_n"]),
        cer_text=pct(sums["cer_text_e"], sums["cer_text_n"]),
        lines_ok=pct(sums["lines_ok"], len(preds)),
        err_suspect=err["suspect"],
        err_preserved=pct(err["preserved"], err["suspect"]),
        err_corrected=pct(err["corrected"], err["suspect"]),
        err_misread=pct(err["misread"], err["suspect"]),
        err_deleted=pct(err["deleted"], err["suspect"]),
        **{f"{k}_P": pct(mk[f"{k}_tp"], mk[f"{k}_hyp"]) for k in KINDS},
        **{f"{k}_R": pct(mk[f"{k}_tp"], mk[f"{k}_ref"]) for k in KINDS},
        lat_median=statistics.median(lat) if lat else None,
        lat_p90=statistics.quantiles(lat, n=10)[-1] if len(lat) >= 10 else None,
        in_tok_mean=statistics.mean(in_tok) if in_tok else None,
        out_tok_mean=statistics.mean(out_tok) if out_tok else None,
        usd_per_1k_pages=cost or None,
    )
    (d / "scores.json").write_text(json.dumps(dict(summary=s, per_page=per_page), indent=1))
    return s


def fmt(x, nd=2):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=str(RES))
    a = ap.parse_args()
    res = Path(a.root)
    runs = a.runs or sorted(p.name for p in res.iterdir() if (p / "pred").is_dir())
    rows = [score(r, a.split, res) for r in runs]
    lines = [
        "| run | pages | CER dipl | WER dipl | CER dipl s/ caixa | WER dipl s/ caixa | CER texto | WER texto | erros preservados | corrigidos | lidos errado | omitidos | riscado P/R | riscado-xxx P/R | inserção P/R | ilegível P/R | s/página (mediana) | s/página (p90) | tokens in/out | US$/1000 pág. | pág/s em lote (GPU) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in rows:
        lines.append(
            f"| {s['run']} | {s['pages']}/{s['of']} | {fmt(s['cer_dipl'])} | {fmt(s['wer_dipl'])} | {fmt(s['cer_ci'])} | {fmt(s['wer_ci'])} | {fmt(s['cer_plain'])} | {fmt(s['wer_plain'])} "
            f"| {fmt(s['err_preserved'],1)}% (n={s['err_suspect']}) | {fmt(s['err_corrected'],1)}% | {fmt(s['err_misread'],1)}% | {fmt(s['err_deleted'],1)}% "
            + "".join(f"| {fmt(s[k+'_P'],0)}/{fmt(s[k+'_R'],0)} " for k in KINDS)
            + f"| {fmt(s['lat_median'],1)} | {fmt(s['lat_p90'],1)} | {fmt(s['in_tok_mean'],0)}/{fmt(s['out_tok_mean'],0)} | {fmt(s['usd_per_1k_pages'])} | {fmt(s['pages_per_s'])} {s['gpu'] or ''} |"
        )
    table = "\n".join(lines)
    (res / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
