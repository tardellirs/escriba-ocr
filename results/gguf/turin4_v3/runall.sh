#!/bin/bash
exec > /opt/cpubench/runall.log 2>&1
cd /opt/gguf
for f in Escriba-OCR-1B-F16.gguf Escriba-OCR-1B-Q8_0.gguf mmproj-Escriba-OCR-1B-F16.gguf mmproj-Escriba-OCR-1B-Q4_K_M.gguf; do
  [ -s $f ] || curl -sSL -o $f https://huggingface.co/tardellirs/Escriba-OCR-1B-GGUF/resolve/main/$f; done
ls -la /opt/gguf
cd /opt/cpubench
pkill -f llama-server; sleep 3
for v in "q4q4 Q4_K_M Q4_K_M" "q4f16 Q4_K_M F16" "q8 Q8_0 Q8_0" "f16 F16 F16"; do
  set -- $v; rm -rf out/$1; echo "START $1 $(date +%T)"; python3 bench.py $1 $2 $3 > bench_$1.log 2>&1; pkill -f llama-server; sleep 3; echo "END $1 $(date +%T) $(grep -c '"page"' bench_$1.log)"; done
echo RUNALL_DONE
