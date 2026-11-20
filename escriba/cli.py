"""Command line: `escriba transcribe` and `escriba viewer`."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import DEFAULT_BASE_URL, DEFAULT_MODEL

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def cmd_transcribe(a: argparse.Namespace) -> int:
    from .client import transcribe_file

    files = []
    for p in map(Path, a.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
    if not files:
        print("No images found.", file=sys.stderr)
        return 1
    run = Path(a.out) / a.name
    run.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        try:
            t = transcribe_file(f, base_url=a.base_url, model=a.model)
        except ConnectionError as e:
            print(e, file=sys.stderr)
            return 2
        shutil.copy(f, run / f.name)
        (run / (f.stem + ".txt")).write_text(t.text + "\n", encoding="utf-8")
        (run / (f.stem + ".json")).write_text(json.dumps({"seconds": t.seconds, "output_tokens": t.output_tokens,
                                                          "finish_reason": t.finish_reason, "model": a.model}))
        warn = "  (stopped at the token limit)" if t.finish_reason == "length" else ""
        print(f"[{i}/{len(files)}] {f.name}: {t.seconds}s, {t.output_tokens} tokens{warn}")
        if a.print:
            print(t.text, "\n")
    print(f"Saved to {run}/  —  browse with: escriba viewer --out {a.out}")
    return 0


def cmd_viewer(a: argparse.Namespace) -> int:
    from .viewer import serve

    serve(out=Path(a.out), bressay=Path(a.bressay) if a.bressay else None, results=Path(a.results) if a.results else None,
          host=a.host, port=a.port, base_url=a.base_url, model=a.model, open_browser=not a.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="escriba", description="Error-preserving transcription of handwritten essays with Escriba-OCR-1B.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible server (default {DEFAULT_BASE_URL})")
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--out", default="out", help="output folder (default: out/)")

    t = sub.add_parser("transcribe", parents=[common], help="transcribe page images (files or folders)")
    t.add_argument("inputs", nargs="+")
    t.add_argument("--name", default="pages", help="run name: results go to <out>/<name>/")
    t.add_argument("--print", action="store_true", help="also print each transcription")
    t.set_defaults(func=cmd_transcribe)

    v = sub.add_parser("viewer", parents=[common], help="browse transcriptions in the browser and transcribe new pages")
    v.add_argument("--bressay", help="BRESSAY root (e.g. data/raw/bressay/bressay) to browse the benchmark")
    v.add_argument("--results", help="results/ folder with <system>/pred/<page>.txt to compare systems")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(func=cmd_viewer)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
