#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Resolving Kaggle username ==="
KAGGLE_USER=$(kaggle config view | grep -i "username" | awk -F': ' '{print $2}' | tr -d '[:space:]')
echo "Kaggle user: $KAGGLE_USER"

echo "=== [2/4] Pushing training kernel to Kaggle Free GPU ==="
kaggle kernels push -p kaggle_kernel/

echo "=== [3/4] Polling kernel status until complete ==="
while true; do
  STATUS=$(kaggle kernels status "$KAGGLE_USER/kayasetu-phc-multimodal-trainer")
  echo "$STATUS"
  if echo "$STATUS" | grep -qi "complete"; then
    break
  fi
  if echo "$STATUS" | grep -qiE "error|cancel"; then
    echo "Kernel failed or was cancelled. Check the Kaggle kernel logs."
    exit 1
  fi
  sleep 30
done

echo "=== [4/4] Downloading trained ONNX model and validation report into models/ ==="
mkdir -p models
kaggle kernels output "$KAGGLE_USER/kayasetu-phc-multimodal-trainer" -p models/

echo "=== DONE! Check models/kayasetu_multimodal.onnx and models/validation_report.json ==="
