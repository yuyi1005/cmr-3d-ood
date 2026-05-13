#!/usr/bin/env bash
set -euo pipefail

python3 tools/data/cmr-3d-ood/data_conversion.py \
  --input-path MnMs2/dataset/trainval \
  --output-path data/cmr-3d-ood/mnms2/trainval \
  --label-value 1 \
  --num-samples-per-case 20 \
  --slice-height 256 \
  --slice-width 256 \
  --max-angle-deg 70 \
  --center-jitter-mm 10 \
  --filename-contains _SA_E \
  --visualize
