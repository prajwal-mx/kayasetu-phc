#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Verifying Kaggle CLI Authentication ==="
kaggle config view

mkdir -p data/raw/pad_ufes_20 models

echo "=== [2/4] Downloading PAD-UFES-20 (Bare Smartphone Skin + Clinical Metadata) ==="
kaggle datasets download -d mahdavi1202/skin-cancer -p data/raw/pad_ufes_20 --unzip

echo "=== [3/4] Extracting Real Sample Cases into data/samples/ ==="
python3 -m backend.app.sample_generator

echo "=== [4/4] Running Multimodal PyTorch + Tabular Fusion Training & ONNX Export ==="
python3 kaggle_kernel/train_kayasetu_multimodal.py \
  --data-dir data/raw/pad_ufes_20 \
  --output-dir models \
  --epochs 12 \
  --batch-size 32 \
  --lr 3e-4

echo "=== DONE! Model exported to models/kayasetu_multimodal.onnx ==="
