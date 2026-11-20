#!/bin/bash
# Download BRESSAY (CC BY 4.0 on Zenodo; the dataset README restricts use to non-commercial research and teaching)
# into data/raw/bressay/bressay, the layout expected by bench/data.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/bressay
curl -L -o data/raw/bressay/bressay.zip "https://zenodo.org/records/11637681/files/bressay.zip?download=1"
echo "e4a4093304efff8322b605c8b79aa957  data/raw/bressay/bressay.zip" | md5sum -c - 2>/dev/null || md5 data/raw/bressay/bressay.zip
unzip -q data/raw/bressay/bressay.zip -d data/raw/bressay
# Hunspell pt-BR dictionary used by the error-preservation metric (LibreOffice dictionaries)
mkdir -p data/dict
for e in aff dic; do curl -sSL -o data/dict/pt_BR.$e "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_BR/pt_BR.$e"; done
echo "BRESSAY and dictionary ready."
