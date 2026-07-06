# CaReDiff

**Causal Reaction Diffusion** — Submission for the [REACT 2026 Challenge](https://sites.google.com/view/react2026/home) (ACM MM 2026, MAFRG).

Given speaker behaviour (audio, video, 3DMM coefficients, facial attributes), CaReDiff generates multiple appropriate listener facial reactions (25-d: 15 AUs + valence/arousal + 8 expressions) with an auxiliary EEG prediction head.

## Tracks

| Track | Architecture | Description |
|-------|-------------|-------------|
| [Generic Online](generic/) | PerFRDiff + EEG | Diffusion-based generation over autoregressive windows |
| [Generic Offline](generic/) | PerFRDiff + EEG | Diffusion-based full-sequence generation |
| [Personalised Online](personalised/online/) | PerFRDiff + PRA + EEG | Frozen generic backbone + Personalised Residual Adapter (autoregressive windows) |
| [Personalised Offline](personalised/offline/) | PerFRDiff + PRA + EEG | Frozen generic backbone + Personalised Residual Adapter (full-sequence) |

## Repository Structure

```
CaReDiff/
├── README.md                          ← this file
├── generic/
│   ├── README.md                      ← generic track details
│   ├── code/                          ← source code (Hydra-based)
│   └── checkpoints/README.md          ← checkpoint download, placement & SHA-256
└── personalised/
    ├── README.md                      ← personalised track details
    ├── code/                          ← source code (Hydra-based)
    └── checkpoints/README.md          ← checkpoint download, placement & SHA-256
```

## Checkpoints

All model checkpoints are hosted on HuggingFace:

**https://huggingface.co/IVPL/CaReDiff/tree/main**

| Track | HuggingFace Path | Contents |
|-------|-----------------|----------|
| Generic Online | [`generic/online/`](https://huggingface.co/IVPL/CaReDiff/tree/main/generic/online) | prior + denoiser + EEG head|
| Generic Offline | [`generic/offline/`](https://huggingface.co/IVPL/CaReDiff/tree/main/generic/offline) | prior + denoiser + EEG head|
| Personalised Online | [`personalised/online/`](https://huggingface.co/IVPL/CaReDiff/tree/main/personalised/online) | shared backbone + 3 adapters (personality / lhfb / both) |
| Personalised Offline | [`personalised/offline/`](https://huggingface.co/IVPL/CaReDiff/tree/main/personalised/offline) | shared backbone + 3 adapters (personality / lhfb / both) |

Placement instructions and SHA-256 checksums are in each variant's checkpoints README:
[`generic/checkpoints/README.md`](generic/checkpoints/README.md) and
[`personalised/checkpoints/README.md`](personalised/checkpoints/README.md).

## Setup

```bash
conda create -n react python=3.10 && conda activate react
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r <track>/code/requirements.txt
```

### External Dependencies

- **[pretrained_models](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb)** → extract to project root. **Required for evaluation** — the post-processor (EmotionVAE) checkpoint at `pretrained_models/post_processor/checkpoint.pth` is loaded on every `stage=test` run to length-align ground-truth sequences with predictions before computing metrics.
- [FaceVerse v2 model](https://github.com/LizhenWangT/FaceVerse) → `external/FaceVerse/data/`. Needed only for FRREa rendering.
- [PIRender checkpoint](https://1drv.ms/u/c/4c787027becb2e91/EclM8oNvDeBKgI4I2lO95zkBXbTgRxuyGerKJ_EhYBuEtA?e=40O0Wc) → `external/PIRender/cur_model_fold.pth`. Needed only for FRREa rendering.

### Data

The [MARS dataset](https://sites.google.com/view/react2026/home) must be obtained through the challenge organisers.

## Metrics

| Metric | Description |
|--------|-------------|
| FRCorr ↑ | Facial Reaction Correlation (CCC against GT) |
| FRDist ↓ | Facial Reaction Distance (DTW against GT) |
| FRDiv ↑ | Diversity across the 10 generated predictions (pairwise MSE) |
| FRVar ↑ | Temporal variance within a generated reaction |
| FRRea ↓ | Realism (FID on rendered frames) |
| FRSyn ↓ | Synchrony (Time-Lagged Cross-Correlation) |

## Citation

```bibtex
@article{song2023multiple,
  title={Multiple Appropriate Facial Reaction Generation in Dyadic Interaction Settings: What, Why and How?},
  author={Song, Siyang and Spitale, Micol and Luo, Yiming and Bal, Batuhan and Gunes, Hatice},
  journal={arXiv preprint arXiv:2302.06514},
  year={2023}
}

@inproceedings{song2025react,
  title={React 2025: the third multiple appropriate facial reaction generation challenge},
  author={Song, Siyang and Spitale, Micol and Kong, Xiangyu and Zhu, Hengde and Luo, Cheng and Palmero, Cristina and Barquero, German and others},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={13979--13984},
  year={2025}
}
```

## Acknowledgement

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)
- [REACT 2026 Baseline](https://github.com/reactmultimodalchallenge/baseline_react2026)

## License

See [LICENSE](generic/code/LICENSE).
