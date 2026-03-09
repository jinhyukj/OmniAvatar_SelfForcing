#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Train OmniAvatar Self-Forcing: distill 14B I2V teacher into 1.3B causal V2V student.
#
# Usage:
#   bash scripts/train_omniavatar_sf.sh
#
# Required environment variables:
#   DATA_LIST_PATH       - Text file with one video directory per line
#   LATENTSYNC_MASK_PATH - Path to LatentSync spatial mask PNG
#
# Optional environment variables:
#   CKPT_ROOT_DIR        - Root for pretrained model weights (default: pretrained_models)
#   OMNIAVATAR_ROOT      - OmniAvatar LoRA checkpoint dir (default: $CKPT_ROOT_DIR/OmniAvatar_LoRA)
#   FASTGEN_OUTPUT_ROOT  - Output directory (default: FASTGEN_OUTPUT)
#   NUM_GPUS             - Number of GPUs (default: 4)
#   MASTER_PORT          - Distributed master port (default: 29501)

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Validate required env vars
if [ -z "${DATA_LIST_PATH:-}" ]; then
    echo "ERROR: DATA_LIST_PATH must be set (text file with one video directory per line)"
    exit 1
fi
if [ -z "${LATENTSYNC_MASK_PATH:-}" ]; then
    echo "ERROR: LATENTSYNC_MASK_PATH must be set (path to LatentSync spatial mask PNG)"
    exit 1
fi

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" train.py \
    --config=fastgen/configs/experiments/OmniAvatar/config_sf.py \
    - trainer.ddp=True \
      dataloader_train.datatags="['${DATA_LIST_PATH}']" \
      dataloader_train.latentsync_mask_path="${LATENTSYNC_MASK_PATH}" \
      log_config.name=omniavatar_sf_v2v_1.3B
