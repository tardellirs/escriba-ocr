# GGUF (llama.cpp) results

Outputs of the GGUF conversions of Escriba-OCR-1B on 30 BRESSAY test pages drawn at random (`ids.txt`).
Folder names: `<machine>_<language model>[-mm<vision projector>]`. Without `-mm…` the vision projector is F16.
`turin4_v3/` is the run reported in the paper (Table 6) and on the [GGUF model card](https://huggingface.co/tardellirs/Escriba-OCR-1B-GGUF):
all five variants on the same 4-vCPU AMD EPYC Turin VM (16 GB RAM), llama.cpp commit in `llama_commit.txt`, `llama-server -np 1 -t 4
-c 8192 --cache-ram 0`, one page at a time. Its `meta.jsonl` files hold, per page, the end-to-end latency, llama.cpp's own timings
(image encoding = `prompt_ms`, generation speed = `predicted_per_second`) and the server's peak RSS. `bench.py` and `runall.sh`
reproduce it and `med.py` prints the medians. Subfolders: `f16` (F16 + F16), `q8` (Q8_0 + Q8_0), `q4f16` (Q4_K_M + F16),
`q4q8` (Q4_K_M + Q8_0), `q4q4` (Q4_K_M + Q4_K_M).

`q16_*` ran on a 16-vCPU VM (quality only) and `turin4_*` on an earlier llama.cpp build (5-page latencies, superseded).
`summary.json` has the CER of each variant and the medians.
