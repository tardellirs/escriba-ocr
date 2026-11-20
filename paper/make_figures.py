"""Figures for the Escriba-OCR paper (vector PDF, print/light theme).

uv run python paper/make_figures.py
Reads results/*/scores.json and results/finetune/**/summary.jsonl; writes paper/figures/*.pdf
Palette: validated categorical set (dataviz validate_palette.js, light: all checks pass; contrast WARN ->
every series also has a distinct marker shape and direct labels).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8f8d87"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def score(run):
    return json.loads((ROOT / "results" / run / "scores.json").read_text())["summary"]


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------- Figure 1: CER vs model size ----------
OPEN = {  # run: (label, total params in B, label it?)
    "or-qwen3-vl-235b-a22b-instruct": ("Qwen3-VL 235B-A22B", 235, True),
    "or-llama-4-maverick": ("Llama 4 Maverick", 400, True),
    "or-qwen2.5-vl-72b-instruct": ("Qwen2.5-VL 72B", 72, True),
    "or-gemma-4-31b-it": ("Gemma 4 31B", 31, False),
    "or-qwen3-vl-30b-a3b-instruct": ("Qwen3-VL 30B-A3B", 30, False),
    "or-qwen3.5-27b": ("Qwen3.5 27B", 27, True),
    "or-gemma-3-27b-it": ("Gemma 3 27B", 27, True),
    "or-mistral-small-3.2-24b-instruct": ("Mistral Small 3.2", 24, False),
    "or-llama-4-scout": ("Llama 4 Scout", 109, False),
    "or-gemma-3-12b-it": ("Gemma 3 12B", 12, False),
    "or-qwen3.5-9b": ("Qwen3.5 9B", 9, False),
    "or-qwen3-vl-8b-instruct": ("Qwen3-VL 8B", 8, False),
    "or-gemma-3-4b-it": ("Gemma 3 4B", 4, False),
    "qwen3.5-4b": ("Qwen3.5 4B", 4, False),
    "qwen3.5-2b": ("Qwen3.5 2B", 2, False),
    "qwen3-vl-2b": ("Qwen3-VL 2B", 2, False),
}
OCR = {
    "lightonocr3-4b": ("LightOnOCR-3 4B", 4, False),
    "lightonocr3-1b": ("LightOnOCR-3 1B (base)", 1, True),
    "lightonocr3-0.8b": ("LightOnOCR-3 0.8B", 0.8, False),
    "paddleocr-vl-1.6": ("PaddleOCR-VL", 0.9, False),
    "glm-ocr": ("GLM-OCR", 0.9, False),
}
fig, ax = plt.subplots(figsize=(3.3, 2.55))
grid(ax, "both")
for group, color, marker, name in ((OPEN, AQUA, "o", "Open VLMs, zero-shot"), (OCR, YELLOW, "s", "OCR VLMs, zero-shot")):
    xs, ys = [], []
    for run, (label, p, lab) in group.items():
        s = score(run)
        xs.append(p); ys.append(s["cer_cap"])
        if lab:
            off = {"LightOnOCR-3 1B (base)": (0, -9, "center"), "Qwen3.5 27B": (0, -9, "center")}.get(label, (3, 2, "left"))
            ax.annotate(label, (p, s["cer_cap"]), xytext=off[:2], textcoords="offset points", fontsize=6, color=INK2, ha=off[2])
    ax.scatter(xs, ys, s=16, c=color, marker=marker, edgecolors="white", linewidths=0.6, zorder=3, label=name)
h = score("htr-flor-gtlines")
ax.scatter([0.001], [h["cer_cap"]], s=22, c=VIOLET, marker="D", edgecolors="white", linewidths=0.6, zorder=3, label="HTR-Flor (trained)")
ax.annotate("HTR-Flor", (0.001, h["cer_cap"]), xytext=(4, 2), textcoords="offset points", fontsize=6, color=INK2)
e = score("lightonocr3-1b-ft-r2")
ax.scatter([1], [e["cer_cap"]], s=70, c=BLUE, marker="*", edgecolors="white", linewidths=0.6, zorder=4, label="Escriba-OCR-1B")
ax.annotate("Escriba-OCR-1B (ours)", (1, e["cer_cap"]), xytext=(8, -3), textcoords="offset points", fontsize=7, color=INK, weight="bold", ha="left")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.0005, 700); ax.set_ylim(1.2, 110)
ax.set_xlabel("Parameters (billions, log scale; MoE: total)")
ax.set_ylabel("Page CER, % (log scale)")
ax.set_yticks([1.5, 2, 3, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1.5", "2", "3", "5", "10", "20", "50", "100"])
ax.legend(loc="upper left", fontsize=6, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "cer_vs_params.pdf"); fig.savefig(FIG / "cer_vs_params.png", dpi=220)
plt.close(fig)

# ---------- Figure 2: data scaling and training length / augmentation ----------
summ = {}
for f in (ROOT / "results" / "finetune" / "lambda" / "summary.jsonl", ROOT / "results" / "finetune" / "summary.jsonl"):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        for k, v in (r.get("evals") or {}).items():
            if isinstance(v, dict) and isinstance(v.get("cer_ci"), float):
                summ[(r["run"], k)] = v["cer_ci"]
# use per-page-capped CER (same definition as the tables)
cap = json.loads((ROOT / "paper" / "analysis" / "ablation_capped.json").read_text())
for key, v in cap.items():
    if key.startswith("foto:"):
        run, ck = key[5:].split("@"); summ[(run, f"{ck}/valfoto")] = v
    else:
        run, ck = key.split("@"); summ[(run, f"{ck}/val154")] = v["cer"]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.85))
ax = axs[0]; grid(ax)
pages = [162, 324, 647]
cer = [summ[("r3-scale-25", "final/val154")], summ[("r3-scale-50", "final/val154")], summ[("r1-all-6e-5-8k", "final/val154")]]
ax.plot(pages, cer, color=GRAY, marker="s", markersize=4, linewidth=1.5)
for x, y in zip(pages, cer):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
ax.set_xscale("log", base=2); ax.set_xticks(pages); ax.set_xticklabels(["162\n(25%)", "324\n(50%)", "647\n(100%)"])
ax.set_ylim(1.8, 3.8); ax.set_xlim(130, 800); ax.set_ylabel("Validation CER, %"); ax.set_title("(a) Training pages (8k ex., no aug.)", fontsize=8, loc="left")
for ax, key, title, ylim in ((axs[1], "val154", "(b) Clean scans", (1.8, 3.0)), (axs[2], "valfoto", "(c) Simulated phone photos", (8, 23))):
    grid(ax)
    r1 = [(2400, summ[("r1-all-6e-5-8k", f"step300/{key}")]), (4800, summ[("r1-all-6e-5-8k", f"step600/{key}")]), (8000, summ[("r1-all-6e-5-8k", f"final/{key}")])]
    r2 = [(4800, summ[("r2-final-aug-12k", f"step600/{key}")]), (8000, summ[("r2-final-aug-12k", f"step1000/{key}")]), (12000, summ[("r2-final-aug-12k", f"final/{key}")])]
    ax.plot(*zip(*r1), color=GRAY, marker="s", markersize=3.5, linewidth=1.5, label="no augmentation")
    ax.plot(*zip(*r2), color=BLUE, marker="o", markersize=3.5, linewidth=1.5, label="photo augmentation")
    ax.annotate(f"{r2[-1][1]:.2f}" if key == "val154" else f"{r2[-1][1]:.1f}", r2[-1], xytext=(-2, 5), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks([2400, 4800, 8000, 12000]); ax.set_xticklabels(["2.4k", "4.8k", "8k", "12k"])
    ax.set_xlabel("Training examples"); ax.set_ylim(*ylim); ax.set_title(title, fontsize=8, loc="left")
axs[0].set_xlabel("Pages (log scale)")
axs[1].legend(fontsize=6, loc="upper right")
fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig(FIG / "scaling_training.pdf"); fig.savefig(FIG / "scaling_training.png", dpi=220)
plt.close(fig)

# ---------- Figure 3: what models do with student misspellings ----------
SYS = [("lightonocr3-1b-ft-r2", "Escriba-OCR-1B"), ("or-qwen3-vl-235b-a22b-instruct", "Qwen3-VL 235B-A22B"),
       ("or-qwen3.5-27b", "Qwen3.5 27B"), ("or-qwen2.5-vl-72b-instruct", "Qwen2.5-VL 72B"), ("lightonocr3-4b", "LightOnOCR-3 4B"),
       ("lightonocr3-1b", "LightOnOCR-3 1B (base)"), ("htr-flor-gtlines", "HTR-Flor")]
cats = [("err_preserved", "copied verbatim", BLUE), ("err_corrected", "replaced by a dictionary word", ORANGE),
        ("err_misread", "replaced by another non-word", YELLOW), ("err_deleted", "deleted", GRAY)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 1.95), gridspec_kw={"width_ratios": [3.2, 1]})
for i, (run, name) in enumerate(SYS):
    s = score(run); left = 0
    for key, lab, col in cats:
        v = s[key]
        ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=1, height=0.68, label=lab if i == 0 else None, zorder=3)
        if v >= 7:
            ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if col in (BLUE, ORANGE) else INK)
        left += v
ax.set_yticks(range(len(SYS))); ax.set_yticklabels([n for _, n in SYS]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("% of the 496 test words absent from the pt-BR dictionary")
ax.legend(ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.45, 1.0), handlelength=1, columnspacing=0.8)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("(a) Automatic analysis", fontsize=8, loc="left", pad=14)
review = [("Real error,\nfixed by\nmodel", 43, ORANGE), ("Ground\ntruth\nwrong", 45, BLUE), ("Ambi-\nguous", 12, GRAY)]
for j, (lab, v, col) in enumerate(review):
    ax2.bar(j, v, color=col, width=0.68, zorder=3)
    ax2.text(j, v + 1.5, f"{v}%", ha="center", fontsize=6.5, color=INK)
ax2.set_xticks(range(3)); ax2.set_xticklabels([r[0] for r in review], fontsize=5.6)
ax2.set_ylim(0, 55); ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
ax2.set_title("(b) Human review (n = 89)", fontsize=8, loc="left", pad=14)
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(FIG / "error_handling.pdf"); fig.savefig(FIG / "error_handling.png", dpi=220)
plt.close(fig)
print("figures written to", FIG)

# ---------- Figure 4: per-page CER distribution + difficulty ----------
pp = json.loads((ROOT / "paper" / "analysis" / "perpage_cer.json").read_text())
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
grid(ax, "both")
for key, label, col, ls in (("escriba", "Escriba-OCR-1B", BLUE, "-"), ("qwen235", "Qwen3-VL 235B-A22B", AQUA, "-"),
                            ("qwen27", "Qwen3.5 27B", AQUA, "--"), ("qwen72", "Qwen2.5-VL 72B", AQUA, ":"), ("base", "LightOnOCR-3 1B (base)", YELLOW, "-")):
    v = sorted(pp[key].values())
    ys = [(i + 1) / len(v) * 100 for i in range(len(v))]
    ax.step(v, ys, where="post", color=col, linestyle=ls, linewidth=1.4, label=label)
ax.set_xscale("log"); ax.set_xlim(0.2, 100)
ax.set_xticks([0.3, 1, 3, 10, 30, 100]); ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
ax.set_xlabel("Page CER, % (log scale)"); ax.set_ylabel("Pages with CER ≤ x (%)")
ax.set_title("(a) Distribution over the 199 test pages", fontsize=8, loc="left")
ax.legend(fontsize=6, loc="lower right")
grid(ax2, "both")
b, e2 = pp["base"], pp["escriba"]
ids = sorted(b)
ax2.scatter([b[i] for i in ids], [e2[i] for i in ids], s=9, c=BLUE, edgecolors="white", linewidths=0.4, zorder=3)
lim = (0.3, 100)
ax2.plot(lim, lim, color=GRAY, linewidth=0.8, linestyle="--", zorder=2)
ax2.annotate("no change", (40, 40), xytext=(-4, 4), textcoords="offset points", fontsize=6, color=INK2, ha="right", rotation=38)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlim(*lim); ax2.set_ylim(0.2, 100)
for a_ in (ax2,):
    a_.set_xticks([1, 3, 10, 30, 100]); a_.set_xticklabels(["1", "3", "10", "30", "100"])
    a_.set_yticks([0.3, 1, 3, 10, 30, 100]); a_.set_yticklabels(["0.3", "1", "3", "10", "30", "100"])
ax2.set_xlabel("Base model page CER, %"); ax2.set_ylabel("Escriba page CER, %")
ax2.set_title("(b) Gain from fine-tuning, per page", fontsize=8, loc="left")
fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig(FIG / "perpage.pdf"); fig.savefig(FIG / "perpage.png", dpi=220)
plt.close(fig)

# ---------- Figure 5: training dynamics of the final run ----------
log = [json.loads(l) for l in (ROOT / "results" / "finetune" / "lambda" / "runs" / "r2-final-aug-12k" / "train_log.jsonl").read_text().splitlines() if l.strip()]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
grid(ax)
steps = [r["step"] * 8 for r in log]
loss = [r["loss"] for r in log]
win = 10
smooth = [sum(loss[max(0, i - win + 1):i + 1]) / len(loss[max(0, i - win + 1):i + 1]) for i in range(len(loss))]
ax.plot(steps, loss, color=GRAY, linewidth=0.5, alpha=0.6)
ax.plot(steps, smooth, color=BLUE, linewidth=1.5)
ax.set_yscale("log"); ax.set_xlabel("Training examples"); ax.set_ylabel("Token cross-entropy (log)")
ax.set_xticks([0, 4000, 8000, 12000]); ax.set_xticklabels(["0", "4k", "8k", "12k"])
ax.set_title("Training loss of the final run", fontsize=8, loc="left")
fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig(FIG / "training.pdf"); fig.savefig(FIG / "training.png", dpi=220)
plt.close(fig)
print("extra figures written")
