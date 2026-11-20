"""Escriba-OCR-1B: transcribe a handwritten essay page (diplomatic transcription).

Recommended: vLLM (loads the checkpoint as-is).
    pip install "vllm>=0.34" pillow
    python inference_example.py page.png

IMPORTANT: upscale the page so its longer side is 1200 px. The processor only downscales,
and small scans (~550 px, like BRESSAY) give the model too few visual tokens.
"""
import base64
import io
import sys

from PIL import Image
from vllm import LLM, SamplingParams

MODEL = "tardellirs/Escriba-OCR-1B"
LONG_SIDE = 1200


def to_data_url(path: str) -> str:
    im = Image.open(path).convert("RGB")
    f = LONG_SIDE / max(im.size)
    im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    llm = LLM(model=MODEL, max_model_len=8192, limit_mm_per_prompt={"image": 1})
    params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=2048)
    # Empty prompt = transcription mode (image only), exactly as in training.
    messages = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": to_data_url(p)}}]}]
                for p in sys.argv[1:]]
    for path, out in zip(sys.argv[1:], llm.chat(messages, params)):
        print(f"===== {path}\n{out.outputs[0].text}\n")


if __name__ == "__main__":
    main()
