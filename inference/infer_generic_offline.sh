#!/usr/bin/env bash
# Generic OFFLINE track: run inference with the submitted checkpoints.
#
# Usage:
#   bash inference/infer_generic_offline.sh /path/to/MARS
# or edit DATA_DIR below and run the script without arguments.
# DATA_DIR must contain a test/ split (see inference/README.md).
set -euo pipefail

DATA_DIR=${1:-/path/to/MARS}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_data_dir
require_pretrained_models generic
download_checkpoints "generic/offline/*"

# Link the downloaded weights into the layout expected by resume_id=pretrained
SAVE="$ROOT/generic/code/save/motion_diffusion/react_2025/offline/checkpoints"
mkdir -p "$SAVE"
ln -sfn "$CKPT_DIR/generic/offline" "$SAVE/pretrained"

cd "$ROOT/generic/code"
python main.py \
    --config-name generic_offline/motion_diffusion \
    stage=test \
    data_dir="$DATA_DIR" \
    trainer.batch_size=1 \
    resume_id=pretrained \
    run_id=infer_generic_offline \
    trainer.generic.eval_eeg=false \
    trainer.model.diff_model.eeg_head.enabled=true

echo "[inference] done. Predictions (results.pt) and metrics are in:"
echo "  $ROOT/generic/code/outputs/motion_diffusion/react_2025/offline/infer_generic_offline/"
