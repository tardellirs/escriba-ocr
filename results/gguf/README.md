# GGUF (llama.cpp) results

Outputs of the GGUF conversions of Escriba-OCR-1B on 30 BRESSAY test pages drawn at random (`ids.txt`).
Folder names: `<machine>_<language model>[-mm<vision projector>]`. Without `-mm…` the vision projector is F16.
`q16_*` ran on a 16-vCPU VM (quality only); `turin4_*` on a 4-vCPU AMD EPYC Turin VM, with per-page latency in
`meta.jsonl`. `summary.json` has the CER of each variant and the median latencies reported in the paper (Table 6)
and on the [GGUF model card](https://huggingface.co/tardellirs/Escriba-OCR-1B-GGUF).
