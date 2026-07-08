# Inference on a New Test Set

This folder contains one script per task. Each script downloads the matching
checkpoints from [HuggingFace](https://huggingface.co/IVPL/CaReDiff)
automatically (skipped if already downloaded), places them where the code
expects them, and runs inference with all required flags set. The only input
you need to provide is the dataset path.

## 1. One-time setup

Environment (shared by all four tasks):

```bash
conda create -n react python=3.10 && conda activate react
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r generic/code/requirements.txt
```

Post-processor weights (required for every test run; hosted on OneDrive, so
this step is manual): download the `pretrained_models` archive linked in the
root README under "External Dependencies" and extract it into both code
directories:

```
generic/code/pretrained_models/
personalised/code/pretrained_models/
```

## 2. Expected dataset layout

`DATA_DIR` must point to a dataset root that contains a `test/` split in the
standard MARS layout:

```
DATA_DIR/
└── test/
    ├── audio-features/      # precomputed wav2vec features
    ├── video-face-crop/     # <speaker|listener>/session*/<clip>.mp4
    ├── facial-attributes/   # 25-d listener/speaker attribute sequences
    ├── coefficients/        # 3DMM coefficients
    └── eeg_processed/       # only needed if EEG evaluation is enabled
```

If only raw audio is available, the wav2vec features can be precomputed with
`personalised/code/dataset/data_preprocess/audio_feature_extraction_test.py`.

## 3. Run a task

From the repository root, pass the dataset root as the first argument
(alternatively, edit the `DATA_DIR` line at the top of each script):

```bash
bash inference/infer_generic_offline.sh       /path/to/MARS
bash inference/infer_generic_online.sh        /path/to/MARS
bash inference/infer_personalised_offline.sh  /path/to/MARS   # condition: both
bash inference/infer_personalised_online.sh   /path/to/MARS   # condition: personality
```

To pin a GPU, prefix the command with `CUDA_VISIBLE_DEVICES=<id>`.

| Script | Task | Submitted model |
|---|---|---|
| `infer_generic_offline.sh` | Generic Offline | PerFRDiff + EEG head |
| `infer_generic_online.sh` | Generic Online | PerFRDiff + EEG head |
| `infer_personalised_offline.sh` | Personalised Offline | frozen backbone + PRA adapter (`both`) |
| `infer_personalised_online.sh` | Personalised Online | frozen backbone + PRA adapter (`personality`) |

For the personalised tracks, the other listener conditions can be selected by
editing the two `CONDITION*` lines at the top of the script (the mapping is
given in a comment there).

## 4. Output

Each run writes to its own output directory (path printed at startup and at
the end of the run):

- `results.pt`: a dict with key `PRED` holding the generated reactions, a list
  with one entry per test sample, each a tensor of shape
  `[10, seq_len, 25]` (10 parallel predictions; 25-d = 15 AUs +
  valence/arousal + 8 expressions). `GT` holds the length-aligned ground
  truth. The generic runs additionally store `PRED_EEG` / `GT_EEG`.
- The evaluation metrics (FRCorr, FRDist, FRDiv, FRVar, FRSyn) are printed in
  the log for reference.

FRRea (FID on rendered frames) is not part of these scripts, as it requires a
separate rendering pass with extra dependencies (FaceVerse assets and the
PIRender checkpoint); we are happy to provide the exact procedure on request.

## Troubleshooting

- `pretrained_models/post_processor/checkpoint.pth not found`: step 1 has not
  been completed for that track.
- If your copy of the test set has no `eeg_processed/` folder, append
  `trainer.generic.eval_eeg=false` to the `python main.py` call in the two
  generic scripts. EEG evaluation is auxiliary and does not affect the facial
  reaction predictions or metrics.
- Checkpoints are cached under `inference/checkpoints/`. Delete that folder to
  force a re-download. SHA-256 checksums are listed in
  `generic/checkpoints/README.md` and `personalised/checkpoints/README.md`.
