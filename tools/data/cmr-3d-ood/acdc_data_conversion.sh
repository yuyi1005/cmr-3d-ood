#!/usr/bin/env bash
set -euo pipefail

python3 tools/data/cmr-3d-ood/data_conversion.py \
  --input-path ACDC/database/training \
  --output-path data/cmr-3d-ood/acdc/trainval \
  --label-value 3 \
  --num-samples-per-case 20 \
  --slice-height 256 \
  --slice-width 256 \
  --max-angle-deg 70 \
  --center-jitter-mm 10 \
  --multiply-affine-by-zooms \
  --visualize
