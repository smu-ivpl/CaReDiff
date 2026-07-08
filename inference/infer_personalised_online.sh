#!/usr/bin/env bash
# Personalised ONLINE track: run inference with the submitted checkpoints.
# Submitted listener condition: "personality" (Big-Five, 5-d).
#
# Usage:
#   bash inference/infer_personalised_online.sh /path/to/MARS
# or edit DATA_DIR below and run the script without arguments.
# DATA_DIR must contain a test/ split (see inference/README.md).
set -euo pipefail

DATA_DIR=${1:-/path/to/MARS}

# Other available adapters: lhfb (3dmm_only), both (3dmm_personality)
CONDITION=personality
CONDITION_MODE=personality_only

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_data_dir
require_pretrained_models personalised
download_checkpoints "personalised/online/*"
PKG="$CKPT_DIR/personalised"

cd "$ROOT/personalised/code"
# past_l_emotion_drop_prob=0.2 enables the past-listener conditioning pathway
# and is required to reproduce the submitted results (see personalised/README.md).
python main.py \
    --config-name g2p_delta_online \
    stage=test \
    task=online \
    data_dir="$DATA_DIR" \
    run_id=infer_personalised_online_$CONDITION \
    trainer.batch_size=4 \
    num_gts=10 \
    trainer.generic.eval_condition_mode=matched \
    trainer.generic.eval_eeg=false \
    trainer.main_model.args.personal_condition_mode=$CONDITION_MODE \
    trainer.model.diff_model.diffusion_decoder.args.past_l_emotion_drop_prob=0.2 \
    resume_id=$CONDITION \
    trainer.ckpt_dir="$PKG/online/adapters" \
    trainer.pretrained.diffusion_decoder="$PKG/online/backbone/CausalTransformerDenoiser/checkpoint_120.pth" \
    trainer.pretrained.diffusion_prior="$PKG/online/backbone/DiffusionPriorNetwork/checkpoint_120.pth" \
    trainer.pretrained.eeg_head_checkpoint="$PKG/online/backbone/EEGPredictionHead/checkpoint_120.pth"

echo "[inference] done. Predictions (results.pt) and metrics are in the run's"
echo "output directory (printed at startup, under personalised/code/outputs/)."
