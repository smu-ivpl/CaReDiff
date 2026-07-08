#!/usr/bin/env bash
# Personalised OFFLINE track: run inference with the submitted checkpoints.
# Submitted listener condition: "both" (personality + LHFB, gated fusion).
#
# Usage:
#   bash inference/infer_personalised_offline.sh /path/to/MARS
# or edit DATA_DIR below and run the script without arguments.
# DATA_DIR must contain a test/ split (see inference/README.md).
set -euo pipefail

DATA_DIR=${1:-/path/to/MARS}

# Other available adapters: personality (personality_only), lhfb (3dmm_only)
CONDITION=both
CONDITION_MODE=3dmm_personality

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_data_dir
require_pretrained_models personalised
download_checkpoints "personalised/offline/*"
PKG="$CKPT_DIR/personalised"

cd "$ROOT/personalised/code"
python main.py \
    --config-name g2p_delta \
    stage=test \
    task=offline \
    data_dir="$DATA_DIR" \
    run_id=infer_personalised_offline_$CONDITION \
    trainer.batch_size=4 \
    num_gts=10 \
    trainer.generic.eval_condition_mode=matched \
    trainer.generic.eval_eeg=false \
    trainer.main_model.args.personal_condition_mode=$CONDITION_MODE \
    resume_id=$CONDITION \
    trainer.ckpt_dir="$PKG/offline/adapters" \
    trainer.pretrained.diffusion_decoder="$PKG/offline/backbone/CausalTransformerDenoiser/checkpoint_120.pth" \
    trainer.pretrained.diffusion_prior="$PKG/offline/backbone/DiffusionPriorNetwork/checkpoint_120.pth" \
    trainer.pretrained.eeg_head_checkpoint="$PKG/offline/backbone/EEGPredictionHead/checkpoint_120.pth"

echo "[inference] done. Predictions (results.pt) and metrics are in the run's"
echo "output directory (printed at startup, under personalised/code/outputs/)."
