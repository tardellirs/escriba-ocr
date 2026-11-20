"""Correct load/save of LightOnOCR-3-1B with transformers 5.22.

transformers 5.22 `LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-3-1B")`
silently leaves all 311 decoder weights randomly initialised (checkpoint keys are `language_model.model.*`,
the class expects `model.language_model.*`) -> garbage output, loss ~12. We load the hub safetensors
ourselves with an explicit key map, and save back in the hub layout (which vLLM loads correctly).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_ID = "lightonai/LightOnOCR-3-1B"


def hub_dir() -> Path:
    return Path(snapshot_download(MODEL_ID))


def hub_state() -> dict[str, torch.Tensor]:
    sd = {}
    for f in sorted(hub_dir().glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                sd[k] = fh.get_tensor(k)
    return sd


def hub_to_model_key(k: str) -> str:
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model."):]
    if k.startswith("language_model.lm_head."):
        return "lm_head." + k[len("language_model.lm_head."):]
    return k  # model.vision_encoder.*, model.vision_projection.*


def model_to_hub_key(k: str) -> str:
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith("lm_head."):
        return "language_model.lm_head." + k[len("lm_head."):]
    return k


def load_model(dtype=torch.float32, attn="sdpa"):
    from transformers import LightOnOcrForConditionalGeneration

    model = LightOnOcrForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation=attn)
    hub = hub_state()
    mapped = {hub_to_model_key(k): v.to(dtype) for k, v in hub.items()}
    res = model.load_state_dict(mapped, strict=False)
    missing = [k for k in res.missing_keys if k != "lm_head.weight"]
    assert not missing, f"still missing after remap: {missing[:5]}"
    assert not res.unexpected_keys, f"unexpected after remap: {res.unexpected_keys[:5]}"
    if "lm_head.weight" in res.missing_keys:  # tied to the input embeddings in the checkpoint
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
    return model


def save_model(model, out: Path, dtype=torch.bfloat16) -> None:
    """Write hub-layout weights + the hub's config/processor files (vLLM-loadable)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hub_keys = set(hub_state().keys()) if not hasattr(save_model, "_keys") else save_model._keys
    save_model._keys = hub_keys
    sd = {}
    for k, v in model.state_dict().items():
        hk = model_to_hub_key(k)
        if hk not in hub_keys:  # tied lm_head not stored in the checkpoint
            continue
        sd[hk] = v.detach().to("cpu", dtype).contiguous()
    missing = hub_keys - set(sd)
    assert not missing, f"save: missing {sorted(missing)[:5]}"
    save_file(sd, out / "model.safetensors", metadata={"format": "pt"})
    for f in hub_dir().iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.endswith(".safetensors.index.json"):
            if f.suffix in (".json", ".jinja", ".txt", ".model", ".py"):
                shutil.copy(f, out / f.name)
