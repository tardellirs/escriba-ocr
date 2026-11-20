"""Local viewer: browse transcriptions next to their page images, compare systems, transcribe new pages.

    escriba viewer                                   # browse out/ ; drop images to transcribe them
    escriba viewer --bressay data/raw/bressay/bressay --results results   # browse the benchmark

Standard library HTTP server; binds to 127.0.0.1 by default.
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
STATIC = Path(__file__).parent / "static"
SAFE = re.compile(r"[^A-Za-z0-9._-]+")
BENCH = "bressay-test"


class Library:
    def __init__(self, out: Path, bressay: Path | None, results: Path | None):
        self.out, self.bressay, self.results = out, bressay, results
        self.lock = threading.Lock()

    # ---- listing ----
    def runs(self) -> list[dict]:
        runs = []
        if self.out.exists():
            for d in sorted(p for p in self.out.iterdir() if p.is_dir()):
                pages = sorted(f.stem for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
                if pages:
                    runs.append({"id": d.name, "label": d.name, "kind": "out", "pages": pages})
        if self.bressay:
            ids = (self.bressay / "sets" / "test.txt").read_text().split()
            runs.append({"id": BENCH, "label": "BRESSAY test (199 pages)", "kind": "bench", "pages": ids})
        return runs

    def systems(self) -> list[str]:
        if not self.results or not self.results.exists():
            return []
        return sorted(p.parent.name for p in self.results.glob("*/pred") if p.is_dir())

    # ---- one page ----
    def image_path(self, run: str, page: str) -> Path | None:
        if run == BENCH and self.bressay:
            p = self.bressay / "data" / "pages" / page / f"{page}.png"
            return p if p.exists() else None
        d = self._run_dir(run)
        for ext in IMAGE_EXT:
            for cand in (d / f"{page}{ext}", d / f"{page}{ext.upper()}"):
                if cand.exists():
                    return cand
        return None

    def page(self, run: str, page: str, systems: list[str]) -> dict:
        res: dict = {"run": run, "page": page, "text": None, "reference": None, "systems": {}, "meta": None}
        if run == BENCH and self.bressay:
            ref = self.bressay / "data" / "pages" / page / f"{page}.txt"
            res["reference"] = ref.read_text(encoding="utf-8") if ref.exists() else None
            for s in systems:
                f = self.results / s / "pred" / f"{page}.txt" if self.results else None
                if f and f.exists():
                    res["systems"][s] = f.read_text(encoding="utf-8")
            return res
        d = self._run_dir(run)
        for name, key in ((f"{page}.txt", "text"), (f"{page}.ref.txt", "reference")):
            f = d / name
            if f.exists():
                res[key] = f.read_text(encoding="utf-8")
        meta = d / f"{page}.json"
        if meta.exists():
            res["meta"] = json.loads(meta.read_text())
        return res

    def save_upload(self, filename: str, data: bytes, text: str, meta: dict) -> tuple[str, str]:
        stem = SAFE.sub("_", Path(filename).stem)[:80] or "page"
        ext = Path(filename).suffix.lower() if Path(filename).suffix.lower() in IMAGE_EXT else ".png"
        d = self.out / "uploads"
        with self.lock:
            d.mkdir(parents=True, exist_ok=True)
            name, i = stem, 2
            while (d / f"{name}.txt").exists():
                name, i = f"{stem}-{i}", i + 1
            (d / f"{name}{ext}").write_bytes(data)
            (d / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            (d / f"{name}.json").write_text(json.dumps(meta))
        return "uploads", name

    def _run_dir(self, run: str) -> Path:
        d = (self.out / SAFE.sub("_", run)).resolve()
        if self.out.resolve() not in d.parents and d != self.out.resolve():
            raise PermissionError(run)
        return d


def make_handler(lib: Library, base_url: str, model: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EscribaViewer/1.0"

        def log_message(self, fmt, *args):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/state":
                    return self._json({"runs": lib.runs(), "systems": lib.systems(), "model": model, "base_url": base_url,
                                       "server_ok": server_ok(base_url)})
                if u.path == "/api/page":
                    systems = [s for s in q.get("systems", "").split(",") if s]
                    return self._json(lib.page(q["run"], q["page"], systems))
                if u.path == "/image":
                    p = lib.image_path(q["run"], q["page"])
                    if not p:
                        return self._json({"error": "image not found"}, 404)
                    return self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            except (KeyError, PermissionError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/api/transcribe":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", "0"))
            if not n or n > 40 * 1024 * 1024:
                return self._json({"error": "Send one image of at most 40 MB."}, 400)
            data = self.rfile.read(n)
            filename = urllib.parse.unquote(self.headers.get("X-Filename", "page.png"))
            from .client import transcribe_bytes

            try:
                t = transcribe_bytes(data, base_url=base_url, model=model)
            except ConnectionError as e:
                return self._json({"error": str(e)}, 502)
            except Exception as e:  # unreadable image, server error, ...
                return self._json({"error": f"Transcription failed: {e}"}, 500)
            run, page = lib.save_upload(filename, data, t.text,
                                        {"seconds": t.seconds, "output_tokens": t.output_tokens, "finish_reason": t.finish_reason, "model": model})
            self._json({"run": run, "page": page})

    return Handler


def server_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def serve(out: Path, bressay: Path | None, results: Path | None, host: str, port: int, base_url: str, model: str,
          open_browser: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lib = Library(out, bressay, results)
    httpd = ThreadingHTTPServer((host, port), make_handler(lib, base_url, model))
    url = f"http://{host}:{port}"
    print(f"Escriba viewer on {url}  (Ctrl+C to stop)")
    print(f"  transcriptions: {out.resolve()}")
    if bressay:
        print(f"  benchmark: {bressay}  systems: {len(lib.systems())}")
    print(f"  model server: {base_url} ({'reachable' if server_ok(base_url) else 'not reachable: uploads disabled until it starts'})")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
