# CaReDiff — Personalised Facial Reaction Generation

Submission for the [REACT 2026 Challenge](https://sites.google.com/view/react2026/home) (Personalised Online & Offline tracks).

Given speaker behaviour (audio, facial attributes, 3DMM coefficients) and a
personalised condition of the target listener, the model generates multiple
appropriate personalised listener facial reactions (25-d: 15 AUs +
valence/arousal + 8 expressions) with an auxiliary EEG prediction head.
Each track ships three models, one per listener condition (personality /
LHFB / both); all three share the same frozen generic backbone of that track.

## Architecture

**Frozen generic backbone + Personalised Residual Adapter (PRA)** — the
generic CaReDiff backbone of the same track (see `../generic/`) is kept
completely frozen; personalisation is added as a lightweight adapter:

- **PRA**: zero-initialised, gated low-rank residual adapter injected into the
  denoiser feature pathway, plus a FiLM adapter on the coarse GRU branch.
  Only the adapter (and a fine-tuned EEG head) is trained.
- **Listener conditions**: Big-Five personality (5-d), listener historical
  facial behaviour (LHFB, 3DMM sequence), or both with gated fusion.
- **Counterfactual listener-swap loss**: a margin loss that penalises the
  model when a mismatched listener condition reconstructs the reaction as
  well as the matched one (weight 0.5, margin 0.05).
- **Online track**: each online adapter is warm-started from its offline
  counterpart of the same condition and adapted with scheduled sampling
  (probability ramping to 0.5 over 25 epochs). Generation is autoregressive
  over 30-frame windows; each of the 10 parallel predictions conditions on
  its own previous window.

## Repository Structure

```
├── README.md
├── code/
│   ├── main.py                  # Entry point (Hydra-based)
│   ├── requirements.txt
│   ├── configs/
│   │   ├── g2p_delta.yaml        # Personalised offline
│   │   └── g2p_delta_online.yaml # Personalised online
│   ├── dataset/                  # Personalised MARS loading (incl. LHFB pairing)
│   ├── trainer/
│   │   ├── g2p_delta.py          # Adapter training / evaluation
│   │   └── perfrdiff_rewrite_weight.py
│   ├── framework/
│   │   ├── g2p_delta/            # PRA adapter modules
│   │   ├── motion_diffusion/     # Frozen backbone (prior + causal denoiser)
│   │   └── metrics/              # FRC, FRD, S_MSE(FRDiv), FRVar, TLCC, FID
│   └── external/                 # FaceVerse (3DMM, code only)
└── checkpoints/
    └── README.md                 # Download links & placement instructions
```

## Setup

Same environment as the generic tracks:

```bash
conda create -n react python=3.10 && conda activate react
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r code/requirements.txt
```

### External Dependencies

Download and place in `code/external/` and `code/pretrained_models/` (shared
with the generic tracks; not duplicated here):
- **[pretrained_models](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb)** → extract to project root. **Required for evaluation** — the post-processor (EmotionVAE) checkpoint at `pretrained_models/post_processor/checkpoint.pth` is loaded on every `stage=test` run to length-align ground-truth sequences with predictions before computing metrics.
- [FaceVerse v2 model](https://github.com/LizhenWangT/FaceVerse) → `external/FaceVerse/data/`. Needed for 3DMM coefficient processing (LHFB conditions) and FRRea rendering.
- [PIRender checkpoint](https://1drv.ms/u/c/4c787027becb2e91/EclM8oNvDeBKgI4I2lO95zkBXbTgRxuyGerKJ_EhYBuEtA?e=40O0Wc) → `external/PIRender/cur_model_fold.pth`. Needed only for FRRea rendering.

### Model Checkpoints

Download from HuggingFace, keeping the directory structure, and set `PKG`
to the absolute path of the downloaded `personalised` folder:
- **https://huggingface.co/IVPL/CaReDiff/tree/main/personalised**

See [`checkpoints/README.md`](checkpoints/README.md) for the layout, SHA-256
checksums and placement instructions. The MARS dataset is not included and
must be obtained through the challenge organisers.

## Conditions

| Adapter | Listener condition | `personal_condition_mode` |
|---|---|---|
| `personality` | Big-Five personality (5-d) | `personality_only` |
| `lhfb` | Listener historical facial behaviour (3DMM) | `3dmm_only` |
| `both` | Both, gated fusion | `3dmm_personality` |

## Training

AdamW, learning rate 2e-4, weight decay 1e-4, gradient clipping 1.0,
30 epochs, batch size 32, seed 1234. The backbone stays frozen throughout.
Example for the personality condition (for `lhfb` / `both`, change
`personal_condition_mode` according to the table above):

```bash
cd code
PKG=<absolute path of the downloaded personalised folder>

# Personalised Offline
python main.py --config-name g2p_delta stage=fit data_dir=<MARS_ROOT> \
  run_id=train_offline_personality trainer.batch_size=32 \
  trainer.generic.epochs=30 trainer.generic.save_period=10 trainer.generic.val_period=1 \
  trainer.main_model.args.personal_condition_mode=personality_only \
  trainer.main_model.args.counterfactual_weight=0.5 \
  trainer.main_model.args.counterfactual_margin=0.05 \
  trainer.pretrained.diffusion_decoder=$PKG/offline/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/offline/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/offline/backbone/EEGPredictionHead/checkpoint_120.pth

# Personalised Online (warm-started from the offline adapter of the same condition)
python main.py --config-name g2p_delta_online stage=fit data_dir=<MARS_ROOT> \
  run_id=train_online_personality trainer.batch_size=32 \
  trainer.generic.save_period=10 trainer.generic.val_period=1 \
  +trainer.generic.scheduled_sampling=true +trainer.generic.ss_p_max=0.5 \
  +trainer.generic.ss_ramp_epochs=25 \
  trainer.model.diff_model.diffusion_decoder.args.past_l_emotion_drop_prob=0.2 \
  trainer.main_model.args.personal_condition_mode=personality_only \
  trainer.pretrained.modifier_warmstart=$PKG/offline/adapters/personality/ModifierNetwork/checkpoint_best.pth \
  trainer.pretrained.diffusion_decoder=$PKG/online/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/online/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/online/backbone/EEGPredictionHead/checkpoint_120.pth
```

## Evaluation

Evaluation loads the adapter from `<trainer.ckpt_dir>/<resume_id>/ModifierNetwork/`,
which maps directly onto the `adapters/` layout on HuggingFace. The loader
verifies that the checkpoint was trained with the configured condition mode
and stops with an error on a mismatch. Predictions (25-d attribute sequences)
are also saved to `results.pt` in the run's output directory.

```bash
cd code
PKG=<absolute path of the downloaded personalised folder>

# Personalised Offline (personality condition)
python main.py --config-name g2p_delta stage=test task=offline \
  data_dir=<MARS_ROOT> run_id=eval_offline_personality \
  trainer.batch_size=4 num_gts=10 \
  trainer.generic.eval_condition_mode=matched \
  trainer.generic.eval_eeg=false \
  trainer.main_model.args.personal_condition_mode=personality_only \
  resume_id=personality \
  trainer.ckpt_dir=$PKG/offline/adapters \
  trainer.pretrained.diffusion_decoder=$PKG/offline/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/offline/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/offline/backbone/EEGPredictionHead/checkpoint_120.pth

# Personalised Online (personality condition)
python main.py --config-name g2p_delta_online stage=test task=online \
  data_dir=<MARS_ROOT> run_id=eval_online_personality \
  trainer.batch_size=4 num_gts=10 \
  trainer.generic.eval_condition_mode=matched \
  trainer.generic.eval_eeg=false \
  trainer.main_model.args.personal_condition_mode=personality_only \
  trainer.model.diff_model.diffusion_decoder.args.past_l_emotion_drop_prob=0.2 \
  resume_id=personality \
  trainer.ckpt_dir=$PKG/online/adapters \
  trainer.pretrained.diffusion_decoder=$PKG/online/backbone/CausalTransformerDenoiser/checkpoint_120.pth \
  trainer.pretrained.diffusion_prior=$PKG/online/backbone/DiffusionPriorNetwork/checkpoint_120.pth \
  trainer.pretrained.eeg_head_checkpoint=$PKG/online/backbone/EEGPredictionHead/checkpoint_120.pth
```

For the `lhfb` / `both` conditions, change `resume_id` (`lhfb` or `both`) and
`personal_condition_mode` (`3dmm_only` or `3dmm_personality`). For the online
track, keep `past_l_emotion_drop_prob=0.2`; this flag enables the
past-listener conditioning pathway and is required to reproduce the reported
numbers.

## Results (MARS test set, official evaluation code, num_gts=10)

### Personalised Offline

| Condition | FRCorr ↑ | FRDist ↓ | FRDiv ↑ | FRVar ↑ | FRRea ↓ | FRSyn ↓ |
|---|---|---|---|---|---|---|
| personality | 0.7786 | 173.63 | 0.1221 | 0.0782 | 50.91 | 48.37 |
| lhfb | 0.7824 | 173.11 | 0.1200 | 0.0766 | 51.23 | 48.26 |
| both | 0.7822 | 171.41 | 0.1187 | 0.0761 | 50.82 | 48.28 |

### Personalised Online

| Condition | FRCorr ↑ | FRDist ↓ | FRDiv ↑ | FRVar ↑ | FRRea ↓ | FRSyn ↓ |
|---|---|---|---|---|---|---|
| personality | 0.6485 | 185.17 | 0.1521 | 0.0831 | 50.58 | 47.92 |
| lhfb | 0.6481 | 191.88 | 0.1521 | 0.0828 | 50.89 | 47.92 |
| both | 0.6355 | 181.11 | 0.1451 | 0.0790 | 52.09 | 48.16 |

FRRea is the FID between rendered generated frames and ground-truth frames
(56,100 frames per side, frame stride 30).

## Metrics

| Metric | Description |
|--------|-------------|
| FRCorr ↑ | Facial Reaction Correlation (CCC against GT) |
| FRDist ↓ | Facial Reaction Distance (DTW against GT) |
| FRDiv ↑ | Diversity across the 10 generated predictions (pairwise MSE) |
| FRVar ↑ | Temporal variance within a generated reaction |
| FRRea ↓ | Realism (FID on rendered frames) |
| FRSyn ↓ | Synchrony (Time-Lagged Cross-Correlation) |
