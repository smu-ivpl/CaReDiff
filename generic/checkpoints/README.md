# Checkpoints

## Download

All checkpoints are hosted on HuggingFace:

**https://huggingface.co/IVPL/CaReDiff/tree/main/generic**

```
generic/
├── online/
│   ├── DiffusionPriorNetwork/checkpoint_120.pth
│   ├── CausalTransformerDenoiser/checkpoint_120.pth
│   └── EEGPredictionHead/checkpoint_120.pth
└── offline/
    ├── DiffusionPriorNetwork/checkpoint_120.pth
    ├── CausalTransformerDenoiser/checkpoint_120.pth
    └── EEGPredictionHead/checkpoint_120.pth
```

## Placement

After downloading, place the checkpoint files under `save/` in the project root.
The folder name (e.g., `pretrained`) can be anything — pass it as `resume_id` when running evaluation.

```
save/motion_diffusion/react_2025/
├── online/checkpoints/pretrained/
│   ├── DiffusionPriorNetwork/checkpoint_120.pth
│   ├── CausalTransformerDenoiser/checkpoint_120.pth
│   └── EEGPredictionHead/checkpoint_120.pth
└── offline/checkpoints/pretrained/
    ├── DiffusionPriorNetwork/checkpoint_120.pth
    ├── CausalTransformerDenoiser/checkpoint_120.pth
    └── EEGPredictionHead/checkpoint_120.pth
```

## Evaluation

```bash
# Generic Online
python main.py \
    --config-name generic_online/motion_diffusion \
    stage=test data_dir=./datasets/REACT2026/ \
    trainer.batch_size=1 resume_id=pretrained

# Generic Offline
python main.py \
    --config-name generic_offline/motion_transvae \
    stage=test data_dir=./datasets/REACT2026/ \
    trainer.batch_size=1 resume_id=pretrained
```

## SHA-256

### Online

| File | SHA-256 |
|------|---------|
| `DiffusionPriorNetwork/checkpoint_120.pth` | `8b717d619cd37fc793f80f37a4af607bda5e9709c83b82f17916b2467a4380a6` |
| `CausalTransformerDenoiser/checkpoint_120.pth` | `f4fc53506fc94a65e86b52bfe1491669a73ca3429ad0b1ab51c62488854242f0` |
| `EEGPredictionHead/checkpoint_120.pth` | `60c7a7ae4e6a233fdb59c0ee1e099daf1158931d876a5f46386a781fa2a52a52` |

### Offline

| File | SHA-256 |
|------|---------|
| `DiffusionPriorNetwork/checkpoint_120.pth` | `d1b66e87f51afd9bb93bdcef1b9e350e6366aa8f995920e400d7e7dd4e299357` |
| `CausalTransformerDenoiser/checkpoint_120.pth` | `68faca9700415c949eecbe7bd3e381877a76b5e1b24bdab9c30e6fd5b628faa2` |
| `EEGPredictionHead/checkpoint_120.pth` | `750c49999a180cda330b88d771f99d1dca0fd94a810470ea77a45561cfd58780` |
