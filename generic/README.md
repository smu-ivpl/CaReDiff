# CaReDiff — Generic Facial Reaction Generation

Submission for the [REACT 2026 Challenge](https://sites.google.com/view/react2026/home) (Generic Online & Offline tracks).

Given speaker behaviour (audio, video, 3DMM coefficients, facial attributes), the model generates multiple appropriate listener facial reactions (25-d: 15 AUs + valence/arousal + 8 expressions) with an auxiliary EEG prediction head.

## Architecture

**PerFRDiff + EEG** — Diffusion-based generation with:
- **Prior Network**: Transformer encoder that encodes speaker context (audio + facial attributes) into a latent distribution.
- **Denoiser**: Causal transformer decoder that generates listener reactions via iterative denoising, conditioned on the prior output.
- **EEG Prediction Head**: Auxiliary head that predicts listener EEG signals from the shared latent representation.
- Cosine noise schedule, DDIM sampling (50 steps at inference), classifier-free guidance.

## Repository Structure

```
├── README.md
├── code/
│   ├── main.py                  # Entry point (Hydra-based)
│   ├── requirements.txt
│   ├── configs/                  # Hydra config tree
│   │   ├── generic_online/       # Online task configs
│   │   └── generic_offline/      # Offline task configs
│   ├── dataset/                  # Data loading (ReactionDataset)
│   ├── trainer/                  # Training loops
│   │   ├── motion_diffusion.py   # PerFRDiff trainer
│   │   └── motion_transvae.py    # TransVAE trainer
│   ├── framework/                # Model architectures & metrics
│   │   ├── motion_diffusion/     # Diffusion prior + denoiser
│   │   ├── motion_transvae/      # VAE baseline
│   │   └── metrics/              # FRC, FRD, TLCC, S_MSE, FRVar
│   ├── external/                 # FaceVerse (3DMM) & PIRender (code only)
│   └── pretrained_models/        # wav2vec config (weights not included)
└── checkpoints/
    └── README.md                 # Download links & placement instructions
```

## Setup

```bash
conda create -n react python=3.10 && conda activate react
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r code/requirements.txt
```

### External Dependencies

Download and place in `code/external/` and `code/pretrained_models/`:
- **[pretrained_models](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb)** → extract to project root. **Required for evaluation** — the post-processor (EmotionVAE) checkpoint at `pretrained_models/post_processor/checkpoint.pth` is loaded on every `stage=test` run to length-align ground-truth sequences with predictions before computing metrics.
- [FaceVerse v2 model](https://github.com/LizhenWangT/FaceVerse) → `external/FaceVerse/data/`. Needed only for FRREa rendering.
- [PIRender checkpoint](https://1drv.ms/u/c/4c787027becb2e91/EclM8oNvDeBKgI4I2lO95zkBXbTgRxuyGerKJ_EhYBuEtA?e=40O0Wc) → `external/PIRender/cur_model_fold.pth`. Needed only for FRREa rendering.

### Model Checkpoints

Download from HuggingFace and place under `code/save/`:
- **https://huggingface.co/IVPL/CaReDiff/tree/main/generic**

See [`checkpoints/README.md`](checkpoints/README.md) for detailed placement instructions.

## Training

### Generic Online (PerFRDiff + EEG)

```bash
python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=8 \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.model.diff_model.eeg_head.enabled=true \
    trainer.generic.train_eeg_head_only=false
```

### Generic Offline (TransVAE + EEG)

```bash
python main.py \
    --config-name generic_offline/motion_diffusion \
    trainer.batch_size=4 \
    trainer.max_seq_len=750 \
    trainer.window_size=8 \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.train_eeg_head_only=false \
    trainer.model.eeg_head.enabled=true
```

## Evaluation

### Generic Online

```bash
python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=1 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    resume_id=<train-experiment-id> \
    trainer.generic.eval_eeg=true \
    trainer.model.diff_model.eeg_head.enabled=true
```

### Generic Offline

```bash
python main.py \
    --config-name generic_offline/motion_diffusion \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    trainer.batch_size=1 \
    trainer.max_seq_len=750 \
    trainer.window_size=8 \
    trainer.data_transform=zero_center \
    resume_id=<train-experiment-id> \
    trainer.eval_eeg=true \
    trainer.eval_eeg_metrics=true \
    trainer.eval_facial_metrics=true \
    trainer.save_results=true \
    trainer.renderer.do_render=false
```

## Metrics

| Metric | Description |
|--------|-------------|
| FRCorr ↑ | Facial Reaction Correlation (CCC against GT) |
| FRDist ↓ | Facial Reaction Distance (DTW against GT) |
| FRDiv ↑ | Diversity across the 10 generated predictions (pairwise MSE) |
| FRVar ↑ | Temporal variance within a generated reaction |
| FRRea ↓ | Realism (FID on rendered frames) |
| FRSyn ↓ | Synchrony (Time-Lagged Cross-Correlation) |
