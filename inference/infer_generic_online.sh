#!/usr/bin/env bash
# Generic ONLINE track: run inference with the submitted checkpoints.
#
# Usage:
#   bash inference/infer_generic_online.sh /path/to/MARS
# or edit DATA_DIR below and run the script without arguments.
# DATA_DIR must contain a test/ split (see inference/README.md).
set -euo pipefail

DATA_DIR=${1:-/path/to/MARS}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_data_dir
require_pretrained_models generic
download_checkpoints "generic/online/*"

# Link the downloaded weights into the layout expected by resume_id=pretrained
SAVE="$ROOT/generic/code/save/motion_diffusion/react_2025/online/checkpoints"
mkdir -p "$SAVE"
ln -sfn "$CKPT_DIR/generic/online" "$SAVE/pretrained"

cd "$ROOT/generic/code"
python main.py \
    --config-name generic_online/motion_diffusion \
    stage=test \
    data_dir="$DATA_DIR" \
    trainer.batch_size=1 \
    resume_id=pretrained \
    run_id=infer_generic_online \
    trainer.generic.eval_eeg=false \
    trainer.model.diff_model.eeg_head.enabled=true \
    trainer.model.diff_model.diffusion_decoder.args.past_l_emotion_drop_prob=0.2

echo "[inference] done. Predictions (results.pt) and metrics are in:"
echo "  $ROOT/generic/code/outputs/motion_diffusion/react_2025/online/infer_generic_online/"
